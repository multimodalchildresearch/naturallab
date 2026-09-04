"""Tracker-pipeline adapter for Qwen person grounding."""

from __future__ import annotations

import math
from numbers import Real
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from naturallab.spatial_tracking.base import TrackerModule
from naturallab.spatial_tracking.vlm import (
    EvidenceImage,
    PersonGrounder,
    QwenPersonGrounder,
)


class QwenDetectorModule(TrackerModule):
    """Expose a Qwen person grounder through the ``TrackerModule`` contract.

    The module runs on frame zero and then once every ``cadence_frames`` frames.
    Frames between inference calls return an empty ``detections`` list and are
    marked as skipped in ``detection_metadata``. This lets downstream code
    distinguish an intentional cadence skip from an inference result containing
    no people.
    """

    def __init__(
        self,
        grounder: Optional[PersonGrounder] = None,
        cadence_frames: Optional[int] = None,
        jpeg_quality: int = 95,
        confidence_threshold: Optional[float] = None,
    ) -> None:
        """Create a detector adapter.

        Args:
            grounder: Injectable grounding backend. Defaults to
                :class:`QwenPersonGrounder`.
            cadence_frames: Positive number of frames between inference calls.
                When omitted, the backend configuration value is used when
                available, otherwise every frame is processed.
            jpeg_quality: JPEG quality in the inclusive range 1 through 100.
            confidence_threshold: Drop detections with a reported confidence
                below this value. Detections with nullable confidence are kept
                rather than assigned an invented score.
        """

        super().__init__(name="QwenDetector")
        self.grounder = grounder or QwenPersonGrounder()
        if cadence_frames is None:
            config = getattr(self.grounder, "config", None)
            cadence_frames = getattr(config, "detection_cadence_frames", 1)
        self.cadence_frames = self._validate_positive_integer(
            cadence_frames,
            "cadence_frames",
        )
        self.jpeg_quality = self._validate_jpeg_quality(jpeg_quality)
        self.confidence_threshold = self._validate_confidence_threshold(
            confidence_threshold
        )
        self.frame_count = 0
        self.last_inference_time = 0.0

    @staticmethod
    def _validate_positive_integer(value: Any, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field_name} must be a positive integer")
        return value

    @classmethod
    def _validate_jpeg_quality(cls, value: Any) -> int:
        quality = cls._validate_positive_integer(value, "jpeg_quality")
        if quality > 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return quality

    @staticmethod
    def _validate_confidence_threshold(
        value: Optional[float],
    ) -> Optional[float]:
        if value is None:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or not 0 <= value <= 1
        ):
            raise ValueError(
                "confidence_threshold must be between 0 and 1 when provided"
            )
        return float(value)

    @staticmethod
    def _validate_frame(frame: Any) -> np.ndarray:
        if not isinstance(frame, np.ndarray):
            raise TypeError("data['frame'] must be a numpy.ndarray")
        if (
            frame.ndim != 3
            or frame.shape[0] < 1
            or frame.shape[1] < 1
            or frame.shape[2] != 3
        ):
            raise ValueError("data['frame'] must have shape (height, width, 3)")
        if frame.dtype != np.uint8:
            raise ValueError("data['frame'] must use uint8 BGR pixels")
        return frame

    def _encode_frame(self, frame: np.ndarray) -> EvidenceImage:
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not success:
            raise RuntimeError("OpenCV could not encode the BGR frame as JPEG")
        return EvidenceImage(encoded.tobytes(), mime_type="image/jpeg")

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Ground people in ``data['frame']`` when cadence calls for it."""

        if "frame" not in data:
            raise KeyError("data must contain a 'frame' entry")
        frame = self._validate_frame(data["frame"])
        frame_index = self.frame_count
        self.frame_count += 1

        if frame_index % self.cadence_frames != 0:
            return {
                **data,
                "detections": [],
                "detection_provenance": None,
                "detection_metadata": {
                    "backend": "qwen",
                    "frame_index": frame_index,
                    "cadence_frames": self.cadence_frames,
                    "confidence_threshold": self.confidence_threshold,
                    "skipped": True,
                    "skip_reason": "cadence",
                    "inference_seconds": None,
                },
            }

        start_time = time.perf_counter()
        result = self.grounder.ground(self._encode_frame(frame))
        self.last_inference_time = time.perf_counter() - start_time

        height, width = frame.shape[:2]
        detections: List[List[Optional[float]]] = []
        for detection in result.detections:
            if (
                self.confidence_threshold is not None
                and detection.confidence is not None
                and detection.confidence < self.confidence_threshold
            ):
                continue
            x1, y1, x2, y2 = detection.bbox.as_tuple()
            detections.append(
                [
                    x1 * width,
                    y1 * height,
                    x2 * width,
                    y2 * height,
                    detection.confidence,
                ]
            )

        detection_provenance = result.provenance.as_dict()
        # Cadence is owned by this adapter, and may intentionally override the
        # backend's default. Persist the value that was actually used.
        detection_provenance["detection_cadence_frames"] = self.cadence_frames

        return {
            **data,
            "detections": detections,
            "detection_provenance": detection_provenance,
            "detection_metadata": {
                "backend": "qwen",
                "frame_index": frame_index,
                "cadence_frames": self.cadence_frames,
                "confidence_threshold": self.confidence_threshold,
                "skipped": False,
                "skip_reason": None,
                "inference_seconds": self.last_inference_time,
            },
        }

    def reset(self) -> None:
        """Restart cadence accounting and timing statistics."""

        self.frame_count = 0
        self.last_inference_time = 0.0
        self.log_info("Qwen detector reset")
