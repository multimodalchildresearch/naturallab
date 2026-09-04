from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from naturallab.gaze_analysis.object_detection import owlv2
from naturallab.gaze_analysis.object_detection.utils import DetectionOutput
from scripts.detect_custom_objects import (
    IMAGE_SUFFIXES,
    discover_media_files,
)


class FakeOwlBackend:
    config = SimpleNamespace(
        vision_config=SimpleNamespace(image_size=100),
    )

    def process_image(self, image):
        assert image.size == (100, 100)
        return "pixels"

    def process_text(self, queries):
        assert queries == ["toy", "book"]
        return {"input_ids": "ids", "attention_mask": "mask"}

    def text_guided_detection(self, **inputs):
        assert inputs == {
            "input_ids": "ids",
            "attention_mask": "mask",
            "pixel_values": "pixels",
        }
        return object()


def test_owlv2_compatibility_wrapper_returns_absolute_xywh(monkeypatch) -> None:
    detector = owlv2.OWLv2Detector.__new__(owlv2.OWLv2Detector)
    detector.backend = FakeOwlBackend()
    detector.device = "cpu"
    monkeypatch.setattr(
        owlv2,
        "post_process_batch",
        lambda *args, **kwargs: [
            DetectionOutput(
                boxes=torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
                scores=torch.tensor([0.8]),
                labels=torch.tensor([1]),
            )
        ],
    )

    result = detector.detect(
        Image.new("RGB", (100, 100)),
        ["toy", "book"],
    )

    assert result["boxes"][0] == pytest.approx([25.0, 25.0, 50.0, 50.0])
    assert result["scores"] == pytest.approx([0.8])
    assert result["labels"] == ["book"]


def test_object_image_discovery_is_case_insensitive_and_natural(
    tmp_path,
) -> None:
    for name in ("frame10.JPG", "frame2.png", "frame1.jpeg", "notes.txt"):
        (tmp_path / name).write_bytes(b"placeholder")

    assert [
        path.name
        for path in discover_media_files(tmp_path, IMAGE_SUFFIXES)
    ] == ["frame1.jpeg", "frame2.png", "frame10.JPG"]
