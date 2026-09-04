from typing import List

import cv2
import numpy as np
import pytest

from naturallab.spatial_tracking.detection import QwenDetectorModule
from naturallab.spatial_tracking.tracking.kalman_tracker import (
    KalmanPersonTracker,
)
from naturallab.spatial_tracking.vlm import (
    EvidenceImage,
    InferenceProvenance,
    NormalizedXYXY,
    PersonGroundingDetection,
    PersonGroundingResult,
)


class FakeGrounder:
    def __init__(self, result: PersonGroundingResult):
        self.result = result
        self.images: List[EvidenceImage] = []

    def ground(self, image: EvidenceImage) -> PersonGroundingResult:
        self.images.append(image)
        return self.result


class SequenceGrounder:
    def __init__(self, results):
        self.results = iter(results)

    def ground(self, image: EvidenceImage) -> PersonGroundingResult:
        return next(self.results)


def grounding_result() -> PersonGroundingResult:
    return PersonGroundingResult(
        detections=(
            PersonGroundingDetection(
                bbox=NormalizedXYXY(0.1, 0.2, 0.8, 0.9),
                confidence=0.75,
            ),
            PersonGroundingDetection(
                bbox=NormalizedXYXY(0.0, 0.0, 0.5, 0.5),
                confidence=None,
            ),
        ),
        provenance=InferenceProvenance(
            model_id="Qwen/Qwen3.6-27B",
            prompt_version="test-grounding/v1",
            endpoint_identity="http://vlm.internal/v1",
            precision="fp8",
            detection_cadence_frames=3,
        ),
    )


def empty_grounding_result() -> PersonGroundingResult:
    result = grounding_result()
    return PersonGroundingResult(
        detections=(),
        provenance=result.provenance,
    )


def test_process_encodes_bgr_jpeg_and_converts_boxes_to_pixel_coordinates():
    grounder = FakeGrounder(grounding_result())
    detector = QwenDetectorModule(
        grounder=grounder,
        cadence_frames=1,
        jpeg_quality=100,
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[:, :, 0] = 240

    output = detector.process({"frame": frame, "source": "external-video"})

    assert output["frame"] is frame
    assert output["source"] == "external-video"
    assert output["detections"] == [
        [20.0, 20.0, 160.0, 90.0, 0.75],
        [0.0, 0.0, 100.0, 50.0, None],
    ]
    assert len(grounder.images) == 1
    evidence = grounder.images[0]
    assert evidence.mime_type == "image/jpeg"
    assert evidence.data.startswith(b"\xff\xd8")

    decoded = cv2.imdecode(
        np.frombuffer(evidence.data, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape == frame.shape
    assert decoded[50, 100, 0] > 220
    assert decoded[50, 100, 2] < 20

    assert output["detection_provenance"] == {
        "model_id": "Qwen/Qwen3.6-27B",
        "prompt_version": "test-grounding/v1",
        "endpoint_identity": "http://vlm.internal/v1",
        "precision": "fp8",
        "detection_cadence_frames": 1,
    }
    assert output["detection_metadata"]["skipped"] is False
    assert output["detection_metadata"]["frame_index"] == 0
    assert output["detection_metadata"]["confidence_threshold"] is None
    assert output["detection_metadata"]["inference_seconds"] >= 0.0


def test_cadence_marks_skips_explicitly_without_reusing_stale_detections():
    grounder = FakeGrounder(grounding_result())
    detector = QwenDetectorModule(grounder=grounder, cadence_frames=3)
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    outputs = [detector.process({"frame": frame}) for _ in range(4)]

    assert len(grounder.images) == 2
    assert [output["detection_metadata"]["skipped"] for output in outputs] == [
        False,
        True,
        True,
        False,
    ]
    assert outputs[1]["detections"] == []
    assert outputs[1]["detection_provenance"] is None
    assert outputs[1]["detection_metadata"] == {
        "backend": "qwen",
        "frame_index": 1,
        "cadence_frames": 3,
        "confidence_threshold": None,
        "skipped": True,
        "skip_reason": "cadence",
        "inference_seconds": None,
    }


def test_kalman_tracker_returns_predictions_between_qwen_calls():
    grounder = FakeGrounder(grounding_result())
    detector = QwenDetectorModule(grounder=grounder, cadence_frames=3)
    tracker = KalmanPersonTracker(max_age=2, min_hits=1)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    track_counts = []
    first_track_prediction_flags = []
    for _ in range(7):
        detection_data = detector.process({"frame": frame})
        output = tracker.process(detection_data)
        track_counts.append(len(output["tracks"]))
        first_track_prediction_flags.append(
            output["tracks"][0]["is_prediction"]
        )

    assert track_counts == [2] * 7
    assert first_track_prediction_flags == [
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert len(grounder.images) == 3


def test_cadence_skip_does_not_resurrect_a_track_after_real_misses():
    grounder = SequenceGrounder(
        [
            grounding_result(),
            empty_grounding_result(),
            empty_grounding_result(),
        ]
    )
    detector = QwenDetectorModule(grounder=grounder, cadence_frames=2)
    tracker = KalmanPersonTracker(max_age=3, min_hits=1)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    track_counts = []
    for _ in range(6):
        track_counts.append(
            len(
                tracker.process(detector.process({"frame": frame}))[
                    "tracks"
                ]
            )
        )

    assert track_counts == [2, 2, 2, 2, 0, 0]


def test_reset_restarts_cadence():
    grounder = FakeGrounder(grounding_result())
    detector = QwenDetectorModule(grounder=grounder, cadence_frames=2)
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    detector.process({"frame": frame})
    detector.process({"frame": frame})
    detector.reset()
    output = detector.process({"frame": frame})

    assert len(grounder.images) == 2
    assert output["detection_metadata"]["frame_index"] == 0
    assert output["detection_metadata"]["skipped"] is False


@pytest.mark.parametrize("cadence", [0, -1, 1.5, True])
def test_cadence_must_be_a_positive_integer(cadence):
    with pytest.raises(ValueError, match="positive integer"):
        QwenDetectorModule(
            grounder=FakeGrounder(grounding_result()),
            cadence_frames=cadence,
        )


@pytest.mark.parametrize(
    "frame,exception",
    [
        (None, TypeError),
        (np.zeros((10, 10), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 4), dtype=np.uint8), ValueError),
        (np.zeros((10, 10, 3), dtype=np.float32), ValueError),
    ],
)
def test_process_rejects_non_bgr_uint8_frames(frame, exception):
    detector = QwenDetectorModule(
        grounder=FakeGrounder(grounding_result()),
        cadence_frames=1,
    )

    with pytest.raises(exception):
        detector.process({"frame": frame})


def test_confidence_threshold_keeps_nullable_scores() -> None:
    detector = QwenDetectorModule(
        grounder=FakeGrounder(grounding_result()),
        cadence_frames=1,
        confidence_threshold=0.8,
    )

    output = detector.process(
        {"frame": np.zeros((100, 200, 3), dtype=np.uint8)}
    )

    assert output["detections"] == [
        [0.0, 0.0, 100.0, 50.0, None],
    ]
    assert output["detection_metadata"]["confidence_threshold"] == 0.8


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, float("nan"), float("inf"), True],
)
def test_confidence_threshold_must_be_a_finite_probability(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        QwenDetectorModule(
            grounder=FakeGrounder(grounding_result()),
            confidence_threshold=threshold,
        )
