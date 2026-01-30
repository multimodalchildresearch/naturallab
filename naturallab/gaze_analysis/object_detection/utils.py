from dataclasses import dataclass
from typing import List, Dict, Union, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor as T
from torchvision.ops import batched_nms, box_convert, clip_boxes_to_image

COLORS = [
    "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow", "pink", "lime",
    "brown", "gray", "olive", "teal", "navy"
]


@dataclass
class RawDetectionOutput:
    logits: torch.FloatTensor = None
    boxes: torch.FloatTensor = None


@dataclass
class DetectionOutput:
    scores: torch.FloatTensor = None
    boxes: torch.FloatTensor = None
    labels: torch.IntTensor = None

    def to(self, device: str) -> "DetectionOutput":
        self.scores = self.scores.to(device)
        self.boxes = self.boxes.to(device)
        self.labels = self.labels.to(device) if hasattr(self.labels, "to") else self.labels
        return self

    def cpu(self) -> "DetectionOutput":
        return self.to("cpu")

    def cuda(self) -> "DetectionOutput":
        return self.to("cuda")


def convert_boxes_absolute(
        boxes: T,
        img_size: Tuple[int, int],
) -> T:
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be of shape (N, 4)"
    boxes[:, 0] *= img_size[0]
    boxes[:, 1] *= img_size[1]
    boxes[:, 2] *= img_size[0]
    boxes[:, 3] *= img_size[1]
    return boxes


def convert_boxes_relative(
        boxes: T,
        img_size: Tuple[int, int],
) -> T:
    assert boxes.ndim == 2 and boxes.shape[1] == 4, "boxes must be of shape (N, 4)"
    boxes[:, 0] /= img_size[0]
    boxes[:, 1] /= img_size[1]
    boxes[:, 2] /= img_size[0]
    boxes[:, 3] /= img_size[1]
    return boxes


def scale_bounding_boxes(boxes: T, original_size: Tuple[int, int], model_size: int) -> T:
    """
    Scale bounding boxes from model space back to original image space.

    Args:
        boxes (torch.Tensor): Tensor of shape (N, 4) in relative [cx, cy, w, h] format
        original_size (tuple): Original image dimensions as (width, height)
        model_size (int): Size of the square model input

    Returns:
        torch.Tensor: Bounding boxes scaled to original image in same [cx, cy, w, h] format
    """
    orig_w, orig_h = original_size
    scale = min(model_size / orig_w, model_size / orig_h)
    scaled_w, scaled_h = orig_w * scale, orig_h * scale

    wh_scale = torch.tensor(
        [model_size / scaled_w, model_size / scaled_h, model_size / scaled_w, model_size / scaled_h],
        device=boxes.device
    )

    return boxes * wh_scale


def plot_bbox(
        image: Image.Image,
        output: DetectionOutput = None,
        boxes: T = None,
        labels: T = None,
        scores: T = None,
        class_mapping: Union[List, Dict] = None,
        box_format: str = "cxcywh",
        engine: str = "matplotlib"
):
    assert engine in ["matplotlib", "pillow"], "engine must be either 'matplotlib' or 'pillow'"
    assert box_format in ["cxcywh", "xyxy", "xywh"], "box_format must be 'cxcywh', 'xywh' or 'xyxy'"
    assert output is not None or boxes is not None, "either output or boxes must be provided"

    if engine == "pillow":
        image = image.copy()
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype(None  # Use default font, 18)
        except Exception as e:
            font = ImageFont.load_default()
        required_bbox_format = "absolute"
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(image, extent=(0, 1, 1, 0))
        ax.set_axis_off()
        required_bbox_format = "relative"

    if output is not None:
        boxes = output.boxes
        labels = output.labels
        scores = output.scores

    boxes = box_convert(boxes, in_fmt=box_format, out_fmt="xyxy")

    if required_bbox_format == "absolute":
        boxes = convert_boxes_absolute(boxes, image.size)
        boxes = clip_boxes_to_image(boxes, image.size)

    if labels is not None:
        labels = [i.item() if isinstance(i, torch.Tensor) else i for i in labels]
        if class_mapping is None:
            unique_labels = sorted(set(labels)) if labels is not None else []
        else:
            unique_labels = list(range(len(class_mapping))) if isinstance(class_mapping, list) else class_mapping.keys()
        cmap = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique_labels)}

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = cmap.get(labels[i], "lime") if labels is not None else "lime"

            string = ""
            if labels is not None:
                label = labels[i]
                if class_mapping is not None:
                    label = class_mapping[label]

                string += f"{label}"
            if scores is not None:
                string += f"({scores[i]:1.2f})"

            if engine == "pillow":
                draw.rectangle((x1, y1, x2, y2), outline=color, width=5)
                if string:
                    draw.text((x1 + 3, y1 + 3), string, fill="white", stroke_fill="white",
                              stroke_width=0, font=font)
            elif engine == "matplotlib":

                ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color)
                if string:
                    ax.text(x1, y1, string, ha='left', va='bottom', color='black', fontsize=8,
                            bbox={
                                'facecolor': 'white',
                                'edgecolor': color,
                                'boxstyle': 'square,pad=.3',
                            })
    if engine == "pillow":
        return image
    elif engine == "matplotlib":
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        plt.tight_layout()
        return fig


def post_process(
        logits: T,
        boxes: T,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.98,
        apply_sigmoid: bool = True,
        scale_score: bool = False

) -> DetectionOutput:
    """
    Post-processes the bounding boxes and logits by applying a score threshold and batched non-maximum suppression (NMS).

    Args:
        logits (torch.Tensor): Tensor of shape (predictions, num_classes) containing classification scores.
        boxes (torch.Tensor): Tensor of shape (predictions, 4) containing bounding box coordinates.
        nms_threshold (float): IoU threshold for NMS.
        score_threshold (float): Minimum score required to keep a detection.
        apply_sigmoid (bool): Whether to apply sigmoid to the logits.
        scale_score (bool): Whether to scale the scores based on the maximum score.

    Returns:
    """
    assert logits.ndim == 2
    assert boxes.ndim == 2
    scores, labels = torch.max(logits, dim=-1)
    if apply_sigmoid:
        scores = torch.sigmoid(scores)

    if scale_score:
        max_scores = torch.max(scores) + 1e-6
        alphas = (scores - (max_scores * 0.1)) / (max_scores * 0.9)
        alphas = torch.clip(alphas, 0, 1)
        scores = alphas * scores

    mask = scores > score_threshold

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    boxes = box_convert(filtered_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    keep_indices = batched_nms(
        boxes=boxes,
        scores=filtered_scores,
        idxs=filtered_labels,
        iou_threshold=nms_threshold
    )

    return DetectionOutput(
        scores=filtered_scores[keep_indices],
        boxes=filtered_boxes[keep_indices],
        labels=filtered_labels[keep_indices]
    )


def post_process_batch(
        output: RawDetectionOutput = None,
        logits: T = None,
        boxes: T = None,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.98
) -> List[DetectionOutput]:
    """
    Post-processes the bounding boxes and logits by applying a score threshold and batched non-maximum suppression (NMS).

    Args:
        output (RawDetectionOutput): Raw detection output containing logits and boxes.
        logits (torch.Tensor): Tensor of shape (batch, predictions, num_classes) containing classification scores.
        boxes (torch.Tensor): Tensor of shape (batch, predictions, 4) containing bounding box coordinates.
        nms_threshold (float): IoU threshold for NMS.
        score_threshold (float): Minimum score required to keep a detection.
    """
    assert output is not None or (
            logits is not None and boxes is not None), "either output or logits and boxes must be provided"

    if output is not None:
        logits = output.logits
        boxes = output.boxes

    batch_size = logits.shape[0]
    results = []
    for i in range(batch_size):
        result = post_process(logits[i], boxes[i], nms_threshold, score_threshold)
        results.append(result)

    return results
