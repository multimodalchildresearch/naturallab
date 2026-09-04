"""Timestamp-aware gaze-to-object assignment with explicit abstention."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple


CoordinateSpace = Literal["pixels", "normalized"]
OverlapPolicy = Literal["abstain", "smallest_box"]
GAZE_ASSIGNMENT_ALGORITHM = "naturallab-gaze-object/v1"


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _require_finite(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def _optional_confidence(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    confidence = _require_finite(value, "confidence")
    if not 0 <= confidence <= 1:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


@dataclass(frozen=True)
class GazeSample:
    """One gaze point in a named camera view."""

    sample_id: str
    timestamp_seconds: float
    view_id: str
    x: float
    y: float
    coordinate_space: CoordinateSpace = "pixels"
    confidence: Optional[float] = None
    valid: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_id",
            _require_identifier(self.sample_id, "sample_id"),
        )
        object.__setattr__(
            self,
            "view_id",
            _require_identifier(self.view_id, "view_id"),
        )
        timestamp = _require_finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )
        if timestamp < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "x", _require_finite(self.x, "x"))
        object.__setattr__(self, "y", _require_finite(self.y, "y"))
        if self.coordinate_space not in {"pixels", "normalized"}:
            raise ValueError(
                "coordinate_space must be 'pixels' or 'normalized'"
            )
        object.__setattr__(
            self,
            "confidence",
            _optional_confidence(self.confidence),
        )
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a boolean")


@dataclass(frozen=True)
class ObjectObservation:
    """One object box at one source timestamp in pixel coordinates."""

    observation_id: str
    timestamp_seconds: float
    view_id: str
    bbox_xyxy: Tuple[float, float, float, float]
    category: str
    track_id: Optional[str] = None
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _require_identifier(self.observation_id, "observation_id"),
        )
        object.__setattr__(
            self,
            "view_id",
            _require_identifier(self.view_id, "view_id"),
        )
        object.__setattr__(
            self,
            "category",
            _require_identifier(self.category, "category"),
        )
        timestamp = _require_finite(
            self.timestamp_seconds,
            "timestamp_seconds",
        )
        if timestamp < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        object.__setattr__(self, "timestamp_seconds", timestamp)
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        bbox = tuple(
            _require_finite(value, "bbox_xyxy")
            for value in self.bbox_xyxy
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise ValueError("bbox_xyxy must have positive width and height")
        object.__setattr__(self, "bbox_xyxy", bbox)
        if self.track_id is not None:
            object.__setattr__(
                self,
                "track_id",
                _require_identifier(self.track_id, "track_id"),
            )
        object.__setattr__(
            self,
            "confidence",
            _optional_confidence(self.confidence),
        )

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (x2 - x1) * (y2 - y1)

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.bbox_xyxy
        return x1 <= x <= x2 and y1 <= y <= y2


@dataclass(frozen=True)
class GazeAssignmentProvenance:
    algorithm: str
    timestamp_tolerance_seconds: float
    overlap_policy: OverlapPolicy
    image_sizes: Mapping[str, Tuple[int, int]]

    def as_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "timestamp_tolerance_seconds": (
                self.timestamp_tolerance_seconds
            ),
            "overlap_policy": self.overlap_policy,
            "image_sizes": {
                view_id: list(size)
                for view_id, size in sorted(self.image_sizes.items())
            },
        }


@dataclass(frozen=True)
class GazeObjectAssignment:
    """Assignment result; an absent object is an explicit abstention."""

    gaze_sample_id: str
    gaze_timestamp_seconds: float
    view_id: str
    object_observation_id: Optional[str]
    object_track_id: Optional[str]
    category: Optional[str]
    object_timestamp_seconds: Optional[float]
    time_delta_seconds: Optional[float]
    candidate_object_ids: Tuple[str, ...]
    reason: str
    provenance: GazeAssignmentProvenance

    @property
    def assigned(self) -> bool:
        return self.object_observation_id is not None

    def as_dict(self) -> Dict[str, object]:
        return {
            "gaze_sample_id": self.gaze_sample_id,
            "gaze_timestamp_seconds": self.gaze_timestamp_seconds,
            "view_id": self.view_id,
            "object_observation_id": self.object_observation_id,
            "object_track_id": self.object_track_id,
            "category": self.category,
            "object_timestamp_seconds": self.object_timestamp_seconds,
            "time_delta_seconds": self.time_delta_seconds,
            "candidate_object_ids": list(self.candidate_object_ids),
            "assigned": self.assigned,
            "reason": self.reason,
            "provenance": self.provenance.as_dict(),
        }


def _nearest_timestamp(
    sorted_timestamps: Sequence[float],
    target: float,
) -> Optional[float]:
    if not sorted_timestamps:
        return None
    index = bisect_left(sorted_timestamps, target)
    candidates = []
    if index < len(sorted_timestamps):
        candidates.append(sorted_timestamps[index])
    if index:
        candidates.append(sorted_timestamps[index - 1])
    return min(candidates, key=lambda value: (abs(value - target), value))


def _pixel_point(
    sample: GazeSample,
    image_sizes: Mapping[str, Tuple[int, int]],
) -> Tuple[float, float]:
    if sample.coordinate_space == "pixels":
        return sample.x, sample.y
    if sample.view_id not in image_sizes:
        raise ValueError(
            "normalized gaze requires image_sizes for view "
            f"{sample.view_id!r}"
        )
    width, height = image_sizes[sample.view_id]
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
        or width < 1
        or height < 1
    ):
        raise ValueError("image sizes must contain positive integer dimensions")
    return sample.x * width, sample.y * height


def _validated_image_sizes(
    image_sizes: Optional[Mapping[str, Tuple[int, int]]],
) -> Mapping[str, Tuple[int, int]]:
    if image_sizes is None:
        return MappingProxyType({})
    if not isinstance(image_sizes, Mapping):
        raise ValueError("image_sizes must be a mapping")
    validated: Dict[str, Tuple[int, int]] = {}
    for raw_view_id, raw_size in image_sizes.items():
        view_id = _require_identifier(raw_view_id, "image_sizes view ID")
        if view_id in validated:
            raise ValueError(
                "image_sizes view IDs must be unique after whitespace "
                f"normalization: {view_id!r}"
            )
        if (
            isinstance(raw_size, (str, bytes))
            or not isinstance(raw_size, Sequence)
            or len(raw_size) != 2
        ):
            raise ValueError(
                "image sizes must contain (width, height) pairs"
            )
        width, height = raw_size
        if (
            isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width < 1
            or height < 1
        ):
            raise ValueError(
                "image sizes must contain positive integer dimensions"
            )
        validated[view_id] = (width, height)
    return MappingProxyType(validated)


def assign_gaze_to_objects(
    gaze_samples: Iterable[GazeSample],
    object_observations: Iterable[ObjectObservation],
    *,
    image_sizes: Optional[Mapping[str, Tuple[int, int]]] = None,
    timestamp_tolerance_seconds: float = 0.05,
    overlap_policy: OverlapPolicy = "abstain",
) -> Tuple[GazeObjectAssignment, ...]:
    """Assign gaze to synchronized boxes without guessing ambiguous overlaps."""

    tolerance = _require_finite(
        timestamp_tolerance_seconds,
        "timestamp_tolerance_seconds",
    )
    if tolerance < 0:
        raise ValueError("timestamp_tolerance_seconds must be non-negative")
    if overlap_policy not in {"abstain", "smallest_box"}:
        raise ValueError(
            "overlap_policy must be 'abstain' or 'smallest_box'"
        )
    sizes = _validated_image_sizes(image_sizes)
    provenance = GazeAssignmentProvenance(
        algorithm=GAZE_ASSIGNMENT_ALGORITHM,
        timestamp_tolerance_seconds=tolerance,
        overlap_policy=overlap_policy,
        image_sizes=sizes,
    )

    frames_by_view: Dict[
        str,
        Dict[float, Tuple[ObjectObservation, ...]],
    ] = {}
    mutable_frames: Dict[
        str,
        Dict[float, list[ObjectObservation]],
    ] = {}
    seen_observation_ids = set()
    for observation in object_observations:
        if not isinstance(observation, ObjectObservation):
            raise TypeError(
                "object_observations must contain ObjectObservation values"
            )
        if observation.observation_id in seen_observation_ids:
            raise ValueError(
                "duplicate object observation ID: "
                f"{observation.observation_id}"
            )
        seen_observation_ids.add(observation.observation_id)
        mutable_frames.setdefault(observation.view_id, {}).setdefault(
            observation.timestamp_seconds,
            [],
        ).append(observation)
    for view_id, mutable_timestamp_groups in mutable_frames.items():
        frames_by_view[view_id] = {
            timestamp: tuple(
                sorted(
                    observations,
                    key=lambda value: value.observation_id,
                )
            )
            for timestamp, observations in mutable_timestamp_groups.items()
        }

    assignments = []
    seen_sample_ids = set()
    for sample in gaze_samples:
        if not isinstance(sample, GazeSample):
            raise TypeError("gaze_samples must contain GazeSample values")
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate gaze sample ID: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)

        if not sample.valid:
            assignments.append(
                GazeObjectAssignment(
                    gaze_sample_id=sample.sample_id,
                    gaze_timestamp_seconds=sample.timestamp_seconds,
                    view_id=sample.view_id,
                    object_observation_id=None,
                    object_track_id=None,
                    category=None,
                    object_timestamp_seconds=None,
                    time_delta_seconds=None,
                    candidate_object_ids=(),
                    reason="invalid_gaze_sample",
                    provenance=provenance,
                )
            )
            continue

        timestamp_groups = frames_by_view.get(sample.view_id, {})
        object_timestamp = _nearest_timestamp(
            sorted(timestamp_groups),
            sample.timestamp_seconds,
        )
        if (
            object_timestamp is None
            or abs(object_timestamp - sample.timestamp_seconds) > tolerance
        ):
            assignments.append(
                GazeObjectAssignment(
                    gaze_sample_id=sample.sample_id,
                    gaze_timestamp_seconds=sample.timestamp_seconds,
                    view_id=sample.view_id,
                    object_observation_id=None,
                    object_track_id=None,
                    category=None,
                    object_timestamp_seconds=object_timestamp,
                    time_delta_seconds=(
                        None
                        if object_timestamp is None
                        else object_timestamp - sample.timestamp_seconds
                    ),
                    candidate_object_ids=(),
                    reason="no_synchronized_object_frame",
                    provenance=provenance,
                )
            )
            continue

        gaze_x, gaze_y = _pixel_point(sample, sizes)
        containing = tuple(
            observation
            for observation in timestamp_groups[object_timestamp]
            if observation.contains(gaze_x, gaze_y)
        )
        candidate_ids = tuple(
            observation.observation_id for observation in containing
        )
        if not containing:
            assignments.append(
                GazeObjectAssignment(
                    gaze_sample_id=sample.sample_id,
                    gaze_timestamp_seconds=sample.timestamp_seconds,
                    view_id=sample.view_id,
                    object_observation_id=None,
                    object_track_id=None,
                    category=None,
                    object_timestamp_seconds=object_timestamp,
                    time_delta_seconds=(
                        object_timestamp - sample.timestamp_seconds
                    ),
                    candidate_object_ids=(),
                    reason="gaze_outside_objects",
                    provenance=provenance,
                )
            )
            continue

        if len(containing) > 1 and overlap_policy == "abstain":
            assignments.append(
                GazeObjectAssignment(
                    gaze_sample_id=sample.sample_id,
                    gaze_timestamp_seconds=sample.timestamp_seconds,
                    view_id=sample.view_id,
                    object_observation_id=None,
                    object_track_id=None,
                    category=None,
                    object_timestamp_seconds=object_timestamp,
                    time_delta_seconds=(
                        object_timestamp - sample.timestamp_seconds
                    ),
                    candidate_object_ids=candidate_ids,
                    reason="ambiguous_object_overlap",
                    provenance=provenance,
                )
            )
            continue

        selected = min(
            containing,
            key=lambda observation: (
                observation.area,
                observation.observation_id,
            ),
        )
        assignments.append(
            GazeObjectAssignment(
                gaze_sample_id=sample.sample_id,
                gaze_timestamp_seconds=sample.timestamp_seconds,
                view_id=sample.view_id,
                object_observation_id=selected.observation_id,
                object_track_id=selected.track_id,
                category=selected.category,
                object_timestamp_seconds=object_timestamp,
                time_delta_seconds=(
                    object_timestamp - sample.timestamp_seconds
                ),
                candidate_object_ids=candidate_ids,
                reason="assigned",
                provenance=provenance,
            )
        )
    return tuple(assignments)
