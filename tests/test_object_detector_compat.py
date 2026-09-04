from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
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
    create_prototypes,
    detect_objects,
    discover_media_files,
    draw_detection_preview,
    parse_category_queries,
    process_video,
    require_empty_output_directory,
    write_prototypes_atomically,
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


def test_prototype_creation_fails_if_any_reference_image_is_bad(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    category = tmp_path / "references" / "component"
    category.mkdir(parents=True)
    Image.new("RGB", (8, 8), "white").save(category / "01-valid.png")
    (category / "02-broken.jpg").write_bytes(b"not an image")
    output = tmp_path / "prototypes.h5"
    output.write_bytes(b"previous complete prototype")

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

        def get_image_features(self, **_inputs):
            return torch.ones((1, 4))

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return FakeModel()

    class FakeProcessor:
        def __call__(self, _image, *, return_tensors):
            assert return_tensors == "pt"
            return {"pixel_values": torch.ones((1, 3, 8, 8))}

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(_model_name):
            return FakeProcessor()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=FakeAutoModel,
            AutoImageProcessor=FakeAutoImageProcessor,
        ),
    )

    result = create_prototypes(
        SimpleNamespace(
            images=str(tmp_path / "references"),
            output=str(output),
            model="test-model",
            device="cpu",
        )
    )

    assert result == 1
    assert output.read_bytes() == b"previous complete prototype"


def test_atomic_prototype_write_preserves_previous_file_on_failure(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prototypes.h5"
    output.write_bytes(b"previous complete prototype")

    class BrokenFile:
        attrs = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def create_dataset(self, _name, *, data):
            assert data is not None
            raise RuntimeError("simulated HDF5 write failure")

        def flush(self):
            return None

    broken_h5py = SimpleNamespace(File=lambda *_args, **_kwargs: BrokenFile())

    with pytest.raises(RuntimeError, match="simulated HDF5"):
        write_prototypes_atomically(
            output,
            {"component": np.ones(4)},
            "test-model",
            broken_h5py,
        )

    assert output.read_bytes() == b"previous complete prototype"
    assert not list(tmp_path.glob(".prototypes.h5.*.tmp"))


def test_video_detection_rejects_truncated_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    import naturallab.media
    import scripts.detect_custom_objects as detector_script

    class ProbeCapture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 4.0
            if property_id == cv2.CAP_PROP_FPS:
                return 25.0
            raise AssertionError(f"unexpected property: {property_id}")

        def read(self):
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            return None

    class TruncatedSource:
        def __init__(self, _path, *, step):
            assert step == 2

        def __iter__(self):
            yield SimpleNamespace(
                frame_index=0,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                source_timestamp=0.0,
                timestamp_ns=0,
                metadata={"timestamp_source": "test"},
            )

    monkeypatch.setattr(cv2, "VideoCapture", ProbeCapture)
    monkeypatch.setattr(naturallab.media, "VideoFileSource", TruncatedSource)
    monkeypatch.setattr(detector_script, "process_image", lambda *_args: [])
    video_path = tmp_path / "truncated.mp4"
    video_path.write_bytes(b"fixture")
    output_path = tmp_path / "results"
    output_path.mkdir()

    with pytest.raises(RuntimeError, match="1 of 2 requested sampled frames"):
        process_video(
            video_path,
            output_path,
            object(),
            object(),
            object(),
            object(),
            ["component"],
            ["object"],
            SimpleNamespace(
                frame_skip=2,
                save_frames=False,
                frame_interval=100,
            ),
        )

    assert not (output_path / "detections.csv").exists()


def test_failed_video_detection_does_not_publish_staged_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import h5py

    from naturallab.gaze_analysis.object_detection import owlv2 as owl_module
    import scripts.detect_custom_objects as detector_script

    prototype_path = tmp_path / "prototypes.h5"
    with h5py.File(prototype_path, "w") as prototype_file:
        prototype_file.create_dataset("component", data=np.ones(4))
        prototype_file.attrs["model"] = "test-model"
    video_path = tmp_path / "truncated.mp4"
    video_path.write_bytes(b"fixture")
    output_path = tmp_path / "detection-run"

    class FakeModel:
        def eval(self):
            return self

        def to(self, _device):
            return self

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_model_name):
            return FakeModel()

    class FakeAutoImageProcessor:
        @staticmethod
        def from_pretrained(_model_name):
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=FakeAutoModel,
            AutoImageProcessor=FakeAutoImageProcessor,
        ),
    )
    monkeypatch.setattr(
        owl_module,
        "OWLv2Detector",
        lambda **_kwargs: object(),
    )

    def fail_after_partial_write(_input, staging_path, *_args):
        (staging_path / "partial.jpg").write_bytes(b"partial")
        raise RuntimeError("truncated decode")

    monkeypatch.setattr(detector_script, "process_video", fail_after_partial_write)

    result = detect_objects(
        SimpleNamespace(
            input=str(video_path),
            prototypes=str(prototype_path),
            output=str(output_path),
            categories=None,
            threshold=0.1,
            match_threshold=0.3,
            frame_skip=1,
            save_frames=True,
            frame_interval=1,
            device="cpu",
        )
    )

    assert result == 1
    assert not output_path.exists()
    assert not list(tmp_path.glob(".detection-run.staging-*"))
