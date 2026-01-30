from pathlib import Path
from typing import Tuple, Optional, Union, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor as T
from torchvision.transforms.functional import to_pil_image
from transformers import Owlv2Processor, Owlv2ForObjectDetection, Owlv2ImageProcessor

from .utils import RawDetectionOutput


class OwlV2:
    def __init__(
            self,
            model_name: str = "google/owlv2-large-patch14-ensemble",
            cache_dir: str = None,
            device: str = "cuda"
    ):
        self.device = device
        self.processor = Owlv2Processor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name, cache_dir=cache_dir).to(self.device).eval()

        self.config = self.model.config
        self.num_patches = (self.config.vision_config.image_size // self.config.vision_config.patch_size) ** 2

    def unnormalize_image(self, pixel_values: T) -> Image.Image:
        """
        Unnormalize the pixel values to get the original image

        Args:
            pixel_values: The pixel values of the image. Shape: [B, C, H, W]

        Returns:
            The original image as a PIL Image
        """
        mean = torch.tensor(self.processor.image_processor.image_mean)
        std = torch.tensor(self.processor.image_processor.image_std)

        pixel_values = pixel_values.squeeze(0).detach().cpu()
        img = (pixel_values * std.view(-1, 1, 1)) + mean.view(-1, 1, 1)
        img = (img * 255).clamp(0, 255).byte()
        return to_pil_image(img)

    def open_image(
            self,
            path: Union[List[Union[str, Path]], Union[str, Path]]
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Open the image from the path or list of paths

        Args:
            path: The path to the image or

        Returns:
            The PIL Image
        """
        if isinstance(path, (list, tuple)):
            return [Image.open(p).convert("RGB") for p in path]
        return Image.open(path).convert("RGB")

    def process_text(
            self,
            text: Union[str, List[str]],
            transfer_to_device: bool = True
    ):
        """
        Process the text to get the tokenized input

        Args:
            text: The text or list of texts

        Returns:
            The tokenized input
        """
        text = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        if transfer_to_device:
            text = text.to(self.device)
        return text

    def process_image(
            self,
            image: Union[Union[str, Path], T],
            transfer_to_device: bool = True
    ) -> T:
        """
        Process the image to get the processed pixel values. The input is padded to a square image.

        Args:
            image: The path to the image or the PIL Image

        Returns:
            The pixel values of the image. Shape: [B, C, H, W]
        """
        if isinstance(image, (Path, str)):
            image = self.open_image(image)

        image = self.processor(images=image, return_tensors="pt").pixel_values
        if transfer_to_device:
            image = image.to(self.device)

        return image

    @torch.no_grad()
    def get_image_features(
            self,
            pixel_values: T,
            return_boxes: bool = False,
            return_objectnesses: bool = True
    ) -> Tuple[T, Optional[T], Optional[T]]:
        """
        Get the query embeddings for the image.

        Args:
            pixel_values: The pixel values of the image. Shape: [B, C, H, W]
            return_boxes: Whether to return the bounding boxes
            return_objectnesses: Whether to return the objectnesses

        Returns:
            The query embeddings, objectnesses and bounding boxes
        """
        objectnesses = None
        boxes = None

        feature_map = self.model.image_embedder(pixel_values)[0]  # B x H x W x C

        b, h, w, d = feature_map.shape
        image_features = feature_map.reshape(b, h * w, d)

        query = self.model.class_predictor(image_features)[1]

        if return_objectnesses:
            objectnesses = F.sigmoid(self.model.objectness_predictor(image_features))
        if return_boxes:
            boxes = self.model.box_predictor(image_features, feature_map=feature_map)
        return query, objectnesses, boxes

    @torch.no_grad()
    def get_topk_image_features(
            self,
            pixel_values: T,
            top_k: int = 1,
            return_boxes: bool = False,
    ) -> Tuple[T, T, Optional[T]]:
        """
        Get the top-k query embeddings for the image ranked by objectness.

        Args:
            pixel_values: The pixel values of the image. Shape: [B, C, H, W]
            top_k: The number of top-k queries to return
            return_boxes: Whether to return the bounding boxes

        Returns:
            The top-k query embeddings, objectnesses and bounding boxes
        """
        query, objectnesses, boxes = self.get_image_features(pixel_values, return_boxes, True)

        top_k_obj = objectnesses.topk(k=top_k, dim=-1, largest=True)
        batch_idx = torch.arange(query.shape[0], device=query.device).unsqueeze(-1)

        best_query = query[batch_idx, top_k_obj.indices]
        best_objectnesses = top_k_obj.values
        best_boxes = boxes[batch_idx, top_k_obj.indices] if return_boxes else None

        return best_query, best_objectnesses, best_boxes

    @torch.no_grad()
    def get_text_features(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        text_embeds = self.model.owlv2.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_embeds / torch.linalg.norm(text_embeds, ord=2, dim=-1, keepdim=True)
        return text_embeds

    @torch.no_grad()
    def query_guided_detection(
            self,
            pixel_values: T,
            query_embedding: T,
            query_mask: Optional[T] = None,
    ) -> RawDetectionOutput:
        feature_map = self.model.image_embedder(pixel_values)[0]  # B x H x W x C

        b, h, w, d = feature_map.shape
        feature_map_flat = feature_map.reshape(b, h * w, d)  # B x HW x C

        boxes = self.model.box_predictor(
            image_feats=feature_map_flat, feature_map=feature_map
        )  # B x HW x 4

        class_predictions, class_embeds = self.model.class_predictor(
            image_feats=feature_map_flat,
            query_embeds=query_embedding,  # B x num_classes x D
            query_mask=query_mask,
        ) # B x HW x num_classes

        return RawDetectionOutput(logits=class_predictions, boxes=boxes)

    @torch.no_grad()
    def text_guided_detection(
            self,
            input_ids: torch.Tensor,
            pixel_values: torch.FloatTensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> RawDetectionOutput:
        out = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
        return RawDetectionOutput(logits=out.logits, boxes=out.pred_boxes)
