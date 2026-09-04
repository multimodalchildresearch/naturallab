"""Typed contracts shared by vision-language-model tracking backends."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple


class VLMValidationError(ValueError):
    """Raised when a VLM input or structured response violates its contract."""


def _finite_unit_interval(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise VLMValidationError(f"{field_name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise VLMValidationError(f"{field_name} must be between 0 and 1")
    return number


@dataclass(frozen=True)
class NormalizedXYXY:
    """A normalized ``(x1, y1, x2, y2)`` bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        for field_name in ("x1", "y1", "x2", "y2"):
            object.__setattr__(
                self,
                field_name,
                _finite_unit_interval(getattr(self, field_name), field_name),
            )
        if self.x1 >= self.x2:
            raise VLMValidationError("normalized bounding box must satisfy x1 < x2")
        if self.y1 >= self.y2:
            raise VLMValidationError("normalized bounding box must satisfy y1 < y2")

    @classmethod
    def from_sequence(cls, values: Any) -> "NormalizedXYXY":
        if (
            not isinstance(values, (list, tuple))
            or len(values) != 4
        ):
            raise VLMValidationError(
                "bbox must be a four-element [x1, y1, x2, y2] array"
            )
        return cls(*values)

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class EvidenceImage:
    """Image bytes supplied to an OpenAI-compatible multimodal endpoint."""

    data: bytes
    mime_type: str = "image/jpeg"
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise VLMValidationError("evidence image data must be non-empty bytes")
        if (
            not isinstance(self.mime_type, str)
            or not self.mime_type.startswith("image/")
            or any(character.isspace() for character in self.mime_type)
        ):
            raise VLMValidationError(
                "evidence image mime_type must be an image MIME type"
            )
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label.strip()
        ):
            raise VLMValidationError("evidence image label must be non-empty")

    def as_data_url(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True)
class InferenceProvenance:
    """Non-secret metadata needed to reproduce a VLM inference call."""

    model_id: str
    prompt_version: str
    endpoint_identity: str
    precision: str
    detection_cadence_frames: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "endpoint_identity": self.endpoint_identity,
            "precision": self.precision,
            "detection_cadence_frames": self.detection_cadence_frames,
        }


@dataclass(frozen=True)
class PersonGroundingDetection:
    """One grounded person in normalized image coordinates."""

    bbox: NormalizedXYXY
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.bbox, NormalizedXYXY):
            raise VLMValidationError("bbox must be a NormalizedXYXY value")
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_unit_interval(self.confidence, "confidence"),
            )


@dataclass(frozen=True)
class PersonGroundingResult:
    """Grounding detections plus their inference provenance."""

    detections: Tuple[PersonGroundingDetection, ...]
    provenance: InferenceProvenance


@dataclass(frozen=True)
class TrackRoleAssignment:
    """A whitelisted semantic role or an explicit abstention for one track."""

    track_id: str
    role: Optional[str]
    abstained: bool
    confidence: Optional[float]
    reason: Optional[str]
    provenance: InferenceProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise VLMValidationError("track_id must be a non-empty string")
        if not isinstance(self.abstained, bool):
            raise VLMValidationError("abstained must be a boolean")
        if self.abstained and self.role is not None:
            raise VLMValidationError("an abstained assignment cannot contain a role")
        if not self.abstained and (
            not isinstance(self.role, str) or not self.role.strip()
        ):
            raise VLMValidationError(
                "a non-abstained assignment must contain a role"
            )
        if self.confidence is not None:
            object.__setattr__(
                self,
                "confidence",
                _finite_unit_interval(self.confidence, "confidence"),
            )
        if self.reason is not None and not isinstance(self.reason, str):
            raise VLMValidationError("reason must be a string or null")


class PersonGrounder(Protocol):
    """Swappable interface for person-grounding backends."""

    def ground(self, image: EvidenceImage) -> PersonGroundingResult:
        ...


class TrackRoleAssigner(Protocol):
    """Swappable interface for assigning semantic roles to temporal tracks."""

    def assign_role(
        self,
        track_id: str,
        evidence_images: Sequence[EvidenceImage],
    ) -> TrackRoleAssignment:
        ...


class JSONTransport(Protocol):
    """Small injectable HTTP boundary used by OpenAI-compatible clients."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        ...
