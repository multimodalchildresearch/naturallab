from dataclasses import asdict
from typing import Tuple

import torch
from pathlib import Path
from PIL import Image
from torch import Tensor as T
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.detection import MeanAveragePrecision, IntersectionOverUnion
from torchvision.ops import box_convert
from torchvision.transforms.functional import crop
from tqdm import tqdm
from transformers import AutoModel, AutoImageProcessor

from naturallab.gaze_analysis.object_detection import post_process_batch
from naturallab.object_detection.utils import convert_boxes_absolute, scale_bounding_boxes, DetectionOutput
from naturallab.utils.h5 import H5Container, agg_prototypes


class PrototypePredictor(torch.nn.Module):
    def __init__(self,
                 dataset,
                 label_prototype_mapping,
                 prototype_path,
                 hf_model_name: str = "openai/clip-vit-large-patch14-336",
                 cache_dir=None,
                 agg="mean"
                 ):
        super().__init__()
        self.dataset = dataset
        self.hf_model_name = hf_model_name

        self.model = AutoModel.from_pretrained(hf_model_name, cache_dir=cache_dir).eval().to("cuda")
        self.processor = AutoImageProcessor.from_pretrained(hf_model_name, cache_dir=cache_dir)

        self.proto_container = H5Container(prototype_path)
        self.prototypes, self.categories = agg_prototypes(self.proto_container, agg=agg)

        main_cmd = "has_prototype and "
        self.prototype_sets = {}
        for label, subset in label_prototype_mapping.items():
            query = main_cmd + "toyset.isin(@subset)"
            object_subset = self.dataset.meta.query(query).object.values
            idx = [self.categories.index(cat) for cat in sorted(set(object_subset))]
            subcat = [self.categories[i] for i in idx]
            subproto = self.prototypes[idx].clone().contiguous()
            self.prototype_sets[label] = (subproto, subcat)

    @torch.no_grad()
    def predict_clip(self, img: Image.Image, proto: T) -> Tuple[T, T]:
        image_embeds = self.model.get_image_features(
            **self.processor(img, return_tensors="pt").to(self.model.device)).cpu()

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        proto_embeds = proto / proto.norm(p=2, dim=-1, keepdim=True)

        logits_per_text = torch.matmul(image_embeds,
                                       proto_embeds.t().to(proto_embeds.device)) * self.model.logit_scale.exp().to(
            proto_embeds.device
        )
        probs = torch.softmax(logits_per_text, dim=-1)

        preds = probs.max(dim=-1)
        return preds.indices, preds.values

    @torch.no_grad()
    def predict_image_model(self, img: Image.Image, proto: T) -> Tuple[T, T]:
        image_inputs = self.processor(img, return_tensors="pt").to(self.model.device)
        image_embeds = self.model(**image_inputs).last_hidden_state[:, 0].cpu()

        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        proto_embeds = proto / proto.norm(p=2, dim=-1, keepdim=True)

        probs = torch.matmul(image_embeds, proto_embeds.t())

        preds = probs.max(dim=-1)
        return preds.indices, preds.values

    def forward(self, img: Image.Image, proto: T) -> Tuple[T, T]:
        if "clip" in self.hf_model_name:
            return self.predict_clip(img=img, proto=proto)
        else:
            return self.predict_image_model(img=img, proto=proto)


class TwoStageDetector:
    def __init__(self,
                 first_stage_labels,
                 second_stage_labels,
                 detection_model,
                 prototype_path,
                 classification_model_name,
                 dataset,
                 prototype_agg="mean",
                 cache_dir=None
                 ):
        self.first_stage_labels = first_stage_labels
        self.second_stage_labels = second_stage_labels
        self.dataset = dataset

        self.detection_model = detection_model
        self.classification_model = PrototypePredictor(dataset=dataset,
                                                       prototype_path=prototype_path,
                                                       label_prototype_mapping=second_stage_labels,
                                                       hf_model_name=classification_model_name,
                                                       agg=prototype_agg,
                                                       cache_dir=cache_dir)

        self.vocab = None
        self.vocab_inputs = None
        self.vocab_to_cat = None
        self.prototype_sets = None

        self.meta = dict(classification_model=Path(classification_model_name).stem,
                         prototype=Path(prototype_path).stem,
                         first_stage_labels=self.first_stage_labels,
                         second_stage_labels=self.second_stage_labels,
                         prototype_agg=prototype_agg)

        self.setup()

    def setup(self):
        self.vocab_to_cat = {}
        for label, text_out_label in self.first_stage_labels.items():
            for v in text_out_label:
                self.vocab_to_cat[v] = label
        self.vocab = list(self.vocab_to_cat.keys()) + list(self.second_stage_labels.keys())
        self.vocab_inputs = self.detection_model.process_text(self.vocab)

    def first_stage(self,
                    image: Image.Image,
                    nms_threshold: float = 0.2,
                    score_threshold: float = 0.05) -> DetectionOutput:
        pixel_values = self.detection_model.process_image(image)
        raw = self.detection_model.text_guided_detection(**self.vocab_inputs, pixel_values=pixel_values)

        post = post_process_batch(raw, nms_threshold=nms_threshold, score_threshold=score_threshold)[0].to("cpu")

        post.boxes = convert_boxes_absolute(
            box_convert(
                scale_bounding_boxes(
                    post.boxes.clone(),
                    original_size=image.size, model_size=self.detection_model.config.vision_config.image_size
                ),
                in_fmt='cxcywh', out_fmt="xywh"
            ),
            img_size=image.size
        )

        return post

    def second_stage(self,
                     image: Image.Image,
                     first_stage_output: DetectionOutput,
                     offset: int = 4,
                     classification_threshold: float = 0.2) -> DetectionOutput:
        predictions = [None for _ in range(len(first_stage_output.labels))]
        scores = first_stage_output.scores.clone()
        remove_idx = []

        # for each class in the OwlV2 prediction
        for i, cat in enumerate(self.vocab):
            # for each prediction of that class
            for box_idx in torch.where(first_stage_output.labels == i)[0]:
                # classify if needed. Some classes like "hand" are already enough
                if cat in self.classification_model.prototype_sets:
                    prototypes, categories = self.classification_model.prototype_sets[cat]
                    x, y, w, h = first_stage_output.boxes[box_idx].numpy()  # get box
                    # crop the object out of the image and perform classification
                    cropped_frame = crop(image, top=y - offset, left=x - offset, width=w + offset * 2,
                                         height=h + offset * 2)
                    pred_idx, pred_val = self.classification_model(cropped_frame, prototypes)

                    if pred_val.item() > classification_threshold:
                        predictions[box_idx] = categories[pred_idx]
                        scores[box_idx] = pred_val.item()
                    else:
                        remove_idx.append(box_idx.item())
                else:
                    # if we dont need a second stage, we convert the text label back to class
                    predictions[box_idx] = self.vocab_to_cat[cat]

        for i in sorted(remove_idx, reverse=True):
            del predictions[i]

        mask = torch.ones(first_stage_output.boxes.shape[0], dtype=torch.bool)
        mask[remove_idx] = False

        labels = torch.tensor([self.dataset.cat_to_id[cat] for cat in predictions])
        return DetectionOutput(boxes=first_stage_output.boxes[mask], scores=scores[mask], labels=labels)

    def forward(self,
                image: Image.Image,
                nms_threshold: float = 0.2,
                score_threshold: float = 0.05,
                offset: int = 4,
                classification_threshold: float = 0.2
                ) -> DetectionOutput:
        stage_1_out = self.first_stage(image=image, nms_threshold=nms_threshold, score_threshold=score_threshold)
        return self.second_stage(image=image, first_stage_output=stage_1_out, offset=offset,
                                 classification_threshold=classification_threshold)


def evaluate(dataset, pipeline,
             nms_threshold: float = 0.2,
             score_threshold: float = 0.1,
             offset: int = 2,
             classification_threshold: float = 0.1):
    metrics = MetricCollection([
        IntersectionOverUnion(box_format='xywh'),
        MeanAveragePrecision(box_format='xywh', iou_type='bbox', class_metrics=True)
    ])
    dl = DataLoader(dataset, batch_size=None, num_workers=4)

    methode_kwargs = dict(nms_threshold=nms_threshold, score_threshold=score_threshold,
                          offset=offset, classification_threshold=classification_threshold)

    predictions, targets = [], []
    for dp in tqdm(dl, total=len(dl)):
        pred = pipeline.forward(dp['img'], **methode_kwargs)
        predictions.append(asdict(pred))
        
        # Convert normalized boxes to absolute pixels (targets are in [0,1] xywh)
        img = dp['img']
        w, h = img.size
        gt_boxes = dp['boxes'].clone().float()
        gt_boxes[:, 0] *= w  # x
        gt_boxes[:, 1] *= h  # y
        gt_boxes[:, 2] *= w  # w
        gt_boxes[:, 3] *= h  # h
        targets.append(dict(boxes=gt_boxes, labels=dp['labels']))

    metrics.update(predictions, targets)
    res = metrics.compute()
    return {**methode_kwargs, **res}