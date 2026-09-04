from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from naturallab.gaze_analysis.object_detection import owlv2
from naturallab.gaze_analysis.object_detection.utils import DetectionOutput
from scripts.detect_custom_objects import (
    BASE_DETECTION_COLUMNS,
    IMAGE_SUFFIXES,
    VIDEO_DETECTION_COLUMNS,
    annotated_image_filename,
    classify_detection_input,
    discover_media_files,
    draw_detection_preview,
    parse_category_queries,
    require_empty_output_directory,
    write_detection_tables,
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


def test_detection_preview_draws_without_mutating_source() -> None:
    source = Image.new("RGB", (100, 100), "white")
    detection = {
        "x": 20,
        "y": 20,
        "width": 40,
        "height": 30,
        "category": "ball",
        "match_score": 0.75,
    }

    preview = draw_detection_preview(source, [detection])

    assert source.getpixel((20, 20)) == (255, 255, 255)
    assert preview.getpixel((20, 20)) != (255, 255, 255)


def test_detection_input_rejects_empty_or_unsupported_sources(tmp_path) -> None:
    with pytest.raises(ValueError, match="no supported images"):
        classify_detection_input(tmp_path)

    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported input file type"):
        classify_detection_input(unsupported)


def test_detector_refuses_to_mix_outputs_from_multiple_runs(tmp_path) -> None:
    output = tmp_path / "existing-run"
    output.mkdir()
    (output / "detections.csv").write_text("old result", encoding="utf-8")

    with pytest.raises(ValueError, match="output directory is not empty"):
        require_empty_output_directory(output)


def test_image_preview_names_do_not_collide_across_suffixes() -> None:
    first = annotated_image_filename(Path("frame.jpg"), 1)
    second = annotated_image_filename(Path("frame.png"), 2)

    assert first != second
    assert first.endswith("_frame_detections.jpg")


def test_zero_detection_tables_keep_stable_headers(tmp_path) -> None:
    image_output = tmp_path / "images"
    image_output.mkdir()
    image_frame, image_summary = write_detection_tables(
        [],
        image_output,
        "images",
    )

    assert image_frame.empty
    assert image_summary is None
    assert list(pd.read_csv(image_output / "detections.csv").columns) == [
        *BASE_DETECTION_COLUMNS,
        "image",
    ]

    video_output = tmp_path / "video"
    video_output.mkdir()
    video_frame, video_summary = write_detection_tables(
        [],
        video_output,
        "video",
    )

    assert video_frame.empty
    assert video_summary.empty
    assert list(pd.read_csv(video_output / "detections.csv").columns) == list(
        VIDEO_DETECTION_COLUMNS
    )
    assert list(
        pd.read_csv(video_output / "detection_summary.csv").columns
    ) == ["category", "count", "avg_confidence"]


def test_category_queries_reject_an_empty_mapping() -> None:
    with pytest.raises(ValueError, match="non-empty list"):
        parse_category_queries("{}")
