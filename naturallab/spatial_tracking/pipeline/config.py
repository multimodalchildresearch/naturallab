"""Validated configuration for researcher-facing spatial tracking pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence

import yaml  # type: ignore[import-untyped]

from naturallab.spatial_tracking.vlm import (
    DEFAULT_QWEN_MODEL_ID,
    GROUNDING_PROMPT_VERSION,
    ROLE_ASSIGNMENT_PROMPT_VERSION,
)

DEFAULT_SPATIAL_PRESET = "qwen36_27b_quality"
REID_MODEL_PATH_ENV = "NATURALLAB_REID_MODEL_PATH"
REID_CACHE_DIR_ENV = "NATURALLAB_REID_CACHE_DIR"
DEFAULT_REID_ARCHITECTURE = "osnet_ain_x1_0"
DEFAULT_REID_REPOSITORY = "kaiyangzhou/osnet"
DEFAULT_REID_REVISION = "01af85e82a9db4f3a4f6ed3a72ed9150bd416d04"
DEFAULT_REID_FILENAME = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_"
    "coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)
DEFAULT_REID_SHA256 = (
    "8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f"
)
DEFAULT_REID_SIZE_BYTES = 17_293_009
_PRESET_NAME = re.compile(r"\A[a-z0-9][a-z0-9_-]*\Z")
_REPOSITORY_NAME = re.compile(
    r"\A[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\Z"
)
_GIT_REVISION = re.compile(r"\A[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")


class PipelineConfigError(ValueError):
    """Raised when a spatial pipeline preset is invalid or unsupported."""


@dataclass(frozen=True)
class VLMServiceConfig:
    """Connection and generation settings for the shared VLM service."""

    base_url_env: str
    api_key_env: str
    default_base_url: str
    model_id: str
    precision: str
    timeout_seconds: float
    max_tokens: int
    enable_thinking: bool


@dataclass(frozen=True)
class QwenDetectorConfig:
    """Qwen person-grounding adapter settings."""

    backend: str
    model_id: str
    prompt_version: str
    detection_cadence_frames: int
    confidence_threshold: Optional[float]
    jpeg_quality: int
    adaptive_redetection: bool


@dataclass(frozen=True)
class ReIDModelConfig:
    """Immutable identity of a downloadable person-ReID checkpoint."""

    architecture: str
    repository: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    auto_download: bool

    @property
    def model_id(self) -> str:
        """Return a compact immutable identifier for provenance."""

        return f"{self.repository}@{self.revision}:{self.filename}"

    @property
    def download_url(self) -> str:
        """Return the pinned Hugging Face single-file download URL."""

        return (
            f"https://huggingface.co/{self.repository}/resolve/"
            f"{self.revision}/{self.filename}"
        )

    def provenance(self) -> dict[str, Any]:
        """Return public artifact identity and integrity metadata."""

        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "repository": self.repository,
            "revision": self.revision,
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "auto_download": self.auto_download,
        }


@dataclass(frozen=True)
class DeepSORTConfig:
    """Settings accepted by the repository's DeepSORT implementation."""

    backend: str
    max_age: int
    min_hits: int
    iou_threshold: float
    reid_model: ReIDModelConfig
    reid_device: str
    reid_threshold: float
    keep_lost_timeout: int
    min_features_for_reid: int
    confidence_growth_rate: float
    enable_diagnostics: bool
    allow_reid_fallback: bool

    def constructor_kwargs(
        self,
        *,
        allow_reid_fallback: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Return only arguments understood by ``DeepSORTTracker``."""

        return {
            "max_age": self.max_age,
            "min_hits": self.min_hits,
            "iou_threshold": self.iou_threshold,
            "reid_model_path": self.reid_model.filename,
            "reid_model_architecture": self.reid_model.architecture,
            "reid_device": self.reid_device,
            "reid_threshold": self.reid_threshold,
            "keep_lost_timeout": self.keep_lost_timeout,
            "min_features_for_reid": self.min_features_for_reid,
            "confidence_growth_rate": self.confidence_growth_rate,
            "enable_diagnostics": self.enable_diagnostics,
            "allow_reid_fallback": (
                self.allow_reid_fallback
                if allow_reid_fallback is None
                else allow_reid_fallback
            ),
        }


@dataclass(frozen=True)
class QwenRoleAssignmentConfig:
    """Post-tracking semantic-role assignment settings."""

    backend: str
    model_id: str
    prompt_version: str
    evidence_images_per_track: int
    allow_abstention: bool
    roles: tuple[str, ...]
    role_descriptions: tuple[tuple[str, str], ...] = ()

    def role_description_mapping(self) -> dict[str, str]:
        """Return role descriptions in the backend's mapping form."""

        return dict(self.role_descriptions)


@dataclass(frozen=True)
class SpatialPipelinePreset:
    """A complete, validated detector/tracker/role configuration."""

    name: str
    version: int
    description: str
    vlm_service: VLMServiceConfig
    detector: QwenDetectorConfig
    tracker: DeepSORTConfig
    role_assignment: QwenRoleAssignmentConfig

    def provenance(self) -> dict[str, Any]:
        """Return reproducibility metadata without resolved credentials."""

        return {
            "preset_name": self.name,
            "preset_version": self.version,
            "model_id": self.vlm_service.model_id,
            "precision": self.vlm_service.precision,
            "vlm_timeout_seconds": self.vlm_service.timeout_seconds,
            "vlm_max_tokens": self.vlm_service.max_tokens,
            "vlm_enable_thinking": self.vlm_service.enable_thinking,
            "detector_backend": self.detector.backend,
            "detector_prompt_version": self.detector.prompt_version,
            "detection_cadence_frames": (
                self.detector.detection_cadence_frames
            ),
            "confidence_threshold": self.detector.confidence_threshold,
            "detector_jpeg_quality": self.detector.jpeg_quality,
            "tracker_backend": self.tracker.backend,
            "tracker_parameters": self.tracker.constructor_kwargs(),
            "reid_model": self.tracker.reid_model.provenance(),
            "role_assignment_backend": self.role_assignment.backend,
            "role_prompt_version": self.role_assignment.prompt_version,
            "role_evidence_images_per_track": (
                self.role_assignment.evidence_images_per_track
            ),
            "roles": list(self.role_assignment.roles),
            "role_descriptions": (
                self.role_assignment.role_description_mapping()
            ),
            "role_allow_abstention": (
                self.role_assignment.allow_abstention
            ),
        }


def _mapping(
    value: Any,
    path: str,
    *,
    required: Sequence[str],
    optional: Sequence[str] = (),
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PipelineConfigError(f"{path} must be a mapping")
    keys = set(value)
    missing = set(required) - keys
    if missing:
        raise PipelineConfigError(
            f"{path} is missing required fields: {', '.join(sorted(missing))}"
        )
    unknown = keys - set(required) - set(optional)
    if unknown:
        raise PipelineConfigError(
            f"{path} has unsupported fields: {', '.join(sorted(unknown))}"
        )
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigError(f"{path} must be non-empty text")
    return value.strip()


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise PipelineConfigError(f"{path} must be true or false")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PipelineConfigError(f"{path} must be a positive integer")
    return value


def _positive_number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise PipelineConfigError(f"{path} must be a positive number")
    return float(value)


def _probability(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise PipelineConfigError(f"{path} must be between 0 and 1")
    return float(value)


def _optional_probability(value: Any, path: str) -> Optional[float]:
    if value is None:
        return None
    return _probability(value, path)


def _roles(value: Any, path: str) -> tuple[str, ...]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or not value
    ):
        raise PipelineConfigError(f"{path} must be a non-empty list")
    roles = tuple(
        _text(role, f"{path}[{index}]")
        for index, role in enumerate(value)
    )
    if len(set(roles)) != len(roles):
        raise PipelineConfigError(f"{path} must not contain duplicate roles")
    return roles


def _role_descriptions(
    value: Any,
    roles: Sequence[str],
    path: str,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    mapping = _mapping(
        value,
        path,
        required=(),
        optional=roles,
    )
    return tuple(
        (
            role,
            _text(mapping[role], f"{path}.{role}"),
        )
        for role in roles
        if role in mapping
    )


def parse_spatial_pipeline_preset(
    document: Mapping[str, Any],
) -> SpatialPipelinePreset:
    """Parse and strictly validate a spatial pipeline preset mapping."""

    root = _mapping(
        document,
        "preset",
        required=(
            "name",
            "version",
            "description",
            "vlm_service",
            "spatial",
        ),
    )
    service = _mapping(
        root["vlm_service"],
        "vlm_service",
        required=(
            "base_url_env",
            "api_key_env",
            "default_base_url",
            "model_id",
            "precision",
            "timeout_seconds",
            "max_tokens",
            "enable_thinking",
        ),
    )
    spatial = _mapping(
        root["spatial"],
        "spatial",
        required=("detector", "tracker", "role_assignment"),
    )
    detector = _mapping(
        spatial["detector"],
        "spatial.detector",
        required=(
            "backend",
            "model_id",
            "prompt_version",
            "detection_cadence_frames",
            "confidence_threshold",
            "jpeg_quality",
            "adaptive_redetection",
        ),
    )
    tracker = _mapping(
        spatial["tracker"],
        "spatial.tracker",
        required=(
            "backend",
            "max_age",
            "min_hits",
            "iou_threshold",
            "reid_model",
            "reid_device",
            "reid_threshold",
            "keep_lost_timeout",
            "min_features_for_reid",
            "confidence_growth_rate",
            "enable_diagnostics",
            "allow_reid_fallback",
        ),
    )
    reid_model = _mapping(
        tracker["reid_model"],
        "spatial.tracker.reid_model",
        required=(
            "architecture",
            "repository",
            "revision",
            "filename",
            "sha256",
            "size_bytes",
            "auto_download",
        ),
    )
    role = _mapping(
        spatial["role_assignment"],
        "spatial.role_assignment",
        required=(
            "backend",
            "model_id",
            "prompt_version",
            "evidence_images_per_track",
            "allow_abstention",
            "roles",
        ),
        optional=("role_descriptions",),
    )

    jpeg_quality = _positive_integer(
        detector["jpeg_quality"],
        "spatial.detector.jpeg_quality",
    )
    if jpeg_quality > 100:
        raise PipelineConfigError(
            "spatial.detector.jpeg_quality must be between 1 and 100"
        )

    parsed_roles = _roles(
        role["roles"],
        "spatial.role_assignment.roles",
    )
    preset = SpatialPipelinePreset(
        name=_text(root["name"], "name"),
        version=_positive_integer(root["version"], "version"),
        description=_text(root["description"], "description"),
        vlm_service=VLMServiceConfig(
            base_url_env=_text(
                service["base_url_env"],
                "vlm_service.base_url_env",
            ),
            api_key_env=_text(
                service["api_key_env"],
                "vlm_service.api_key_env",
            ),
            default_base_url=_text(
                service["default_base_url"],
                "vlm_service.default_base_url",
            ),
            model_id=_text(service["model_id"], "vlm_service.model_id"),
            precision=_text(service["precision"], "vlm_service.precision"),
            timeout_seconds=_positive_number(
                service["timeout_seconds"],
                "vlm_service.timeout_seconds",
            ),
            max_tokens=_positive_integer(
                service["max_tokens"],
                "vlm_service.max_tokens",
            ),
            enable_thinking=_boolean(
                service["enable_thinking"],
                "vlm_service.enable_thinking",
            ),
        ),
        detector=QwenDetectorConfig(
            backend=_text(detector["backend"], "spatial.detector.backend"),
            model_id=_text(detector["model_id"], "spatial.detector.model_id"),
            prompt_version=_text(
                detector["prompt_version"],
                "spatial.detector.prompt_version",
            ),
            detection_cadence_frames=_positive_integer(
                detector["detection_cadence_frames"],
                "spatial.detector.detection_cadence_frames",
            ),
            confidence_threshold=_optional_probability(
                detector["confidence_threshold"],
                "spatial.detector.confidence_threshold",
            ),
            jpeg_quality=jpeg_quality,
            adaptive_redetection=_boolean(
                detector["adaptive_redetection"],
                "spatial.detector.adaptive_redetection",
            ),
        ),
        tracker=DeepSORTConfig(
            backend=_text(tracker["backend"], "spatial.tracker.backend"),
            max_age=_positive_integer(
                tracker["max_age"],
                "spatial.tracker.max_age",
            ),
            min_hits=_positive_integer(
                tracker["min_hits"],
                "spatial.tracker.min_hits",
            ),
            iou_threshold=_probability(
                tracker["iou_threshold"],
                "spatial.tracker.iou_threshold",
            ),
            reid_model=ReIDModelConfig(
                architecture=_text(
                    reid_model["architecture"],
                    "spatial.tracker.reid_model.architecture",
                ),
                repository=_text(
                    reid_model["repository"],
                    "spatial.tracker.reid_model.repository",
                ),
                revision=_text(
                    reid_model["revision"],
                    "spatial.tracker.reid_model.revision",
                ),
                filename=_text(
                    reid_model["filename"],
                    "spatial.tracker.reid_model.filename",
                ),
                sha256=_text(
                    reid_model["sha256"],
                    "spatial.tracker.reid_model.sha256",
                ),
                size_bytes=_positive_integer(
                    reid_model["size_bytes"],
                    "spatial.tracker.reid_model.size_bytes",
                ),
                auto_download=_boolean(
                    reid_model["auto_download"],
                    "spatial.tracker.reid_model.auto_download",
                ),
            ),
            reid_device=_text(
                tracker["reid_device"],
                "spatial.tracker.reid_device",
            ),
            reid_threshold=_probability(
                tracker["reid_threshold"],
                "spatial.tracker.reid_threshold",
            ),
            keep_lost_timeout=_positive_integer(
                tracker["keep_lost_timeout"],
                "spatial.tracker.keep_lost_timeout",
            ),
            min_features_for_reid=_positive_integer(
                tracker["min_features_for_reid"],
                "spatial.tracker.min_features_for_reid",
            ),
            confidence_growth_rate=_probability(
                tracker["confidence_growth_rate"],
                "spatial.tracker.confidence_growth_rate",
            ),
            enable_diagnostics=_boolean(
                tracker["enable_diagnostics"],
                "spatial.tracker.enable_diagnostics",
            ),
            allow_reid_fallback=_boolean(
                tracker["allow_reid_fallback"],
                "spatial.tracker.allow_reid_fallback",
            ),
        ),
        role_assignment=QwenRoleAssignmentConfig(
            backend=_text(
                role["backend"],
                "spatial.role_assignment.backend",
            ),
            model_id=_text(
                role["model_id"],
                "spatial.role_assignment.model_id",
            ),
            prompt_version=_text(
                role["prompt_version"],
                "spatial.role_assignment.prompt_version",
            ),
            evidence_images_per_track=_positive_integer(
                role["evidence_images_per_track"],
                "spatial.role_assignment.evidence_images_per_track",
            ),
            allow_abstention=_boolean(
                role["allow_abstention"],
                "spatial.role_assignment.allow_abstention",
            ),
            roles=parsed_roles,
            role_descriptions=_role_descriptions(
                role.get("role_descriptions"),
                parsed_roles,
                "spatial.role_assignment.role_descriptions",
            ),
        ),
    )
    validate_spatial_pipeline_preset(preset)
    return preset


def _validate_supported_quality_path(preset: SpatialPipelinePreset) -> None:
    model_ids = {
        preset.vlm_service.model_id,
        preset.detector.model_id,
        preset.role_assignment.model_id,
    }
    if model_ids != {DEFAULT_QWEN_MODEL_ID}:
        raise PipelineConfigError(
            "the quality pipeline requires model_id "
            f"{DEFAULT_QWEN_MODEL_ID!r} for service, detection, and roles"
        )
    if preset.detector.backend != "qwen_grounding":
        raise PipelineConfigError(
            "unsupported detector backend "
            f"{preset.detector.backend!r}; expected 'qwen_grounding'"
        )
    if preset.tracker.backend != "deepsort":
        raise PipelineConfigError(
            "unsupported tracker backend "
            f"{preset.tracker.backend!r}; expected 'deepsort'"
        )
    if preset.role_assignment.backend != "qwen_track_role":
        raise PipelineConfigError(
            "unsupported role-assignment backend "
            f"{preset.role_assignment.backend!r}; "
            "expected 'qwen_track_role'"
        )
    if preset.detector.prompt_version != GROUNDING_PROMPT_VERSION:
        raise PipelineConfigError(
            "detector prompt_version is unsupported; expected "
            f"{GROUNDING_PROMPT_VERSION!r}"
        )
    if (
        preset.role_assignment.prompt_version
        != ROLE_ASSIGNMENT_PROMPT_VERSION
    ):
        raise PipelineConfigError(
            "role prompt_version is unsupported; expected "
            f"{ROLE_ASSIGNMENT_PROMPT_VERSION!r}"
        )
    if preset.detector.adaptive_redetection:
        raise PipelineConfigError(
            "adaptive_redetection is not implemented; set it to false"
        )
    if not preset.role_assignment.allow_abstention:
        raise PipelineConfigError(
            "Qwen role assignment requires allow_abstention: true"
        )
    if preset.tracker.allow_reid_fallback:
        raise PipelineConfigError(
            "the quality preset requires allow_reid_fallback: false; fallback "
            "may only be enabled explicitly at runtime"
        )
    expected_reid_identity = (
        DEFAULT_REID_ARCHITECTURE,
        DEFAULT_REID_REPOSITORY,
        DEFAULT_REID_REVISION,
        DEFAULT_REID_FILENAME,
        DEFAULT_REID_SHA256,
        DEFAULT_REID_SIZE_BYTES,
    )
    configured_reid_identity = (
        preset.tracker.reid_model.architecture,
        preset.tracker.reid_model.repository,
        preset.tracker.reid_model.revision,
        preset.tracker.reid_model.filename,
        preset.tracker.reid_model.sha256,
        preset.tracker.reid_model.size_bytes,
    )
    if configured_reid_identity != expected_reid_identity:
        raise PipelineConfigError(
            "the quality pipeline requires the pinned official "
            f"{DEFAULT_REID_ARCHITECTURE} checkpoint"
        )
    if not preset.tracker.reid_model.auto_download:
        raise PipelineConfigError(
            "the quality pipeline requires reid_model.auto_download: true"
        )


def validate_spatial_pipeline_preset(
    preset: SpatialPipelinePreset,
) -> None:
    """Validate all values, including programmatically constructed presets."""

    if not isinstance(preset, SpatialPipelinePreset):
        raise PipelineConfigError(
            "preset must be a SpatialPipelinePreset instance"
        )
    _text(preset.name, "name")
    _positive_integer(preset.version, "version")
    _text(preset.description, "description")

    service = preset.vlm_service
    if not isinstance(service, VLMServiceConfig):
        raise PipelineConfigError(
            "vlm_service must be a VLMServiceConfig instance"
        )
    _text(service.base_url_env, "vlm_service.base_url_env")
    _text(service.api_key_env, "vlm_service.api_key_env")
    _text(service.default_base_url, "vlm_service.default_base_url")
    _text(service.model_id, "vlm_service.model_id")
    _text(service.precision, "vlm_service.precision")
    _positive_number(
        service.timeout_seconds,
        "vlm_service.timeout_seconds",
    )
    _positive_integer(service.max_tokens, "vlm_service.max_tokens")
    _boolean(
        service.enable_thinking,
        "vlm_service.enable_thinking",
    )
    if service.enable_thinking:
        raise PipelineConfigError(
            "the structured Qwen quality path requires enable_thinking: false"
        )

    detector = preset.detector
    if not isinstance(detector, QwenDetectorConfig):
        raise PipelineConfigError(
            "detector must be a QwenDetectorConfig instance"
        )
    _text(detector.backend, "spatial.detector.backend")
    _text(detector.model_id, "spatial.detector.model_id")
    _text(detector.prompt_version, "spatial.detector.prompt_version")
    _positive_integer(
        detector.detection_cadence_frames,
        "spatial.detector.detection_cadence_frames",
    )
    _optional_probability(
        detector.confidence_threshold,
        "spatial.detector.confidence_threshold",
    )
    jpeg_quality = _positive_integer(
        detector.jpeg_quality,
        "spatial.detector.jpeg_quality",
    )
    if jpeg_quality > 100:
        raise PipelineConfigError(
            "spatial.detector.jpeg_quality must be between 1 and 100"
        )
    _boolean(
        detector.adaptive_redetection,
        "spatial.detector.adaptive_redetection",
    )

    tracker = preset.tracker
    if not isinstance(tracker, DeepSORTConfig):
        raise PipelineConfigError(
            "tracker must be a DeepSORTConfig instance"
        )
    _text(tracker.backend, "spatial.tracker.backend")
    _positive_integer(tracker.max_age, "spatial.tracker.max_age")
    _positive_integer(tracker.min_hits, "spatial.tracker.min_hits")
    _probability(
        tracker.iou_threshold,
        "spatial.tracker.iou_threshold",
    )
    model = tracker.reid_model
    if not isinstance(model, ReIDModelConfig):
        raise PipelineConfigError(
            "spatial.tracker.reid_model must be a ReIDModelConfig instance"
        )
    architecture = _text(
        model.architecture,
        "spatial.tracker.reid_model.architecture",
    )
    if architecture != DEFAULT_REID_ARCHITECTURE:
        raise PipelineConfigError(
            "spatial.tracker.reid_model.architecture must be "
            f"{DEFAULT_REID_ARCHITECTURE!r}"
        )
    repository = _text(
        model.repository,
        "spatial.tracker.reid_model.repository",
    )
    if _REPOSITORY_NAME.fullmatch(repository) is None:
        raise PipelineConfigError(
            "spatial.tracker.reid_model.repository must be an owner/name "
            "Hugging Face repository"
        )
    revision = _text(
        model.revision,
        "spatial.tracker.reid_model.revision",
    )
    if _GIT_REVISION.fullmatch(revision) is None:
        raise PipelineConfigError(
            "spatial.tracker.reid_model.revision must be a full 40-character "
            "lowercase commit hash"
        )
    filename = _text(
        model.filename,
        "spatial.tracker.reid_model.filename",
    )
    if Path(filename).name != filename:
        raise PipelineConfigError(
            "spatial.tracker.reid_model.filename must be a filename, not a path"
        )
    if Path(filename).suffix.lower() != ".pth":
        raise PipelineConfigError(
            "spatial.tracker.reid_model.filename must use the canonical .pth "
            "checkpoint extension"
        )
    sha256 = _text(
        model.sha256,
        "spatial.tracker.reid_model.sha256",
    )
    if _SHA256.fullmatch(sha256) is None:
        raise PipelineConfigError(
            "spatial.tracker.reid_model.sha256 must be 64 lowercase "
            "hexadecimal characters"
        )
    _positive_integer(
        model.size_bytes,
        "spatial.tracker.reid_model.size_bytes",
    )
    _boolean(
        model.auto_download,
        "spatial.tracker.reid_model.auto_download",
    )
    reid_device = _text(
        tracker.reid_device,
        "spatial.tracker.reid_device",
    )
    if reid_device not in {"cpu", "cuda", "mps"}:
        raise PipelineConfigError(
            "spatial.tracker.reid_device must be cpu, cuda, or mps"
        )
    _probability(
        tracker.reid_threshold,
        "spatial.tracker.reid_threshold",
    )
    _positive_integer(
        tracker.keep_lost_timeout,
        "spatial.tracker.keep_lost_timeout",
    )
    _positive_integer(
        tracker.min_features_for_reid,
        "spatial.tracker.min_features_for_reid",
    )
    _probability(
        tracker.confidence_growth_rate,
        "spatial.tracker.confidence_growth_rate",
    )
    _boolean(
        tracker.enable_diagnostics,
        "spatial.tracker.enable_diagnostics",
    )
    _boolean(
        tracker.allow_reid_fallback,
        "spatial.tracker.allow_reid_fallback",
    )

    role = preset.role_assignment
    if not isinstance(role, QwenRoleAssignmentConfig):
        raise PipelineConfigError(
            "role_assignment must be a QwenRoleAssignmentConfig instance"
        )
    _text(role.backend, "spatial.role_assignment.backend")
    _text(role.model_id, "spatial.role_assignment.model_id")
    _text(
        role.prompt_version,
        "spatial.role_assignment.prompt_version",
    )
    _positive_integer(
        role.evidence_images_per_track,
        "spatial.role_assignment.evidence_images_per_track",
    )
    _boolean(
        role.allow_abstention,
        "spatial.role_assignment.allow_abstention",
    )
    roles = _roles(role.roles, "spatial.role_assignment.roles")
    if not isinstance(role.role_descriptions, tuple):
        raise PipelineConfigError(
            "spatial.role_assignment.role_descriptions must be immutable "
            "(a tuple of role/description pairs)"
        )
    try:
        description_mapping = dict(role.role_descriptions)
    except (TypeError, ValueError) as error:
        raise PipelineConfigError(
            "spatial.role_assignment.role_descriptions must contain "
            "two-item role/description pairs"
        ) from error
    if len(description_mapping) != len(role.role_descriptions):
        raise PipelineConfigError(
            "spatial.role_assignment.role_descriptions contains duplicates"
        )
    _role_descriptions(
        description_mapping,
        roles,
        "spatial.role_assignment.role_descriptions",
    )
    _validate_supported_quality_path(preset)


def resolve_reid_model_path(
    preset: SpatialPipelinePreset,
    *,
    configured_path: Optional[str | os.PathLike[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Path:
    """Resolve the explicit override or deterministic NaturalLab cache path."""

    validate_spatial_pipeline_preset(preset)
    environment = os.environ if environ is None else environ
    environment_value = environment.get(REID_MODEL_PATH_ENV, "").strip()
    raw_path: str | os.PathLike[str]
    if configured_path is not None:
        raw_path = configured_path
    elif environment_value:
        raw_path = environment_value
    else:
        cache_override = environment.get(REID_CACHE_DIR_ENV, "").strip()
        xdg_cache = environment.get("XDG_CACHE_HOME", "").strip()
        if cache_override:
            cache_directory = Path(cache_override).expanduser()
        elif xdg_cache:
            cache_directory = (
                Path(xdg_cache).expanduser() / "naturallab" / "reid"
            )
        else:
            cache_directory = Path.home() / ".cache" / "naturallab" / "reid"
        raw_path = cache_directory / preset.tracker.reid_model.filename
    try:
        path = Path(raw_path).expanduser()
    except TypeError as error:
        raise PipelineConfigError(
            "configured ReID model path must be path-like"
        ) from error
    if not str(path).strip():
        raise PipelineConfigError(
            "configured ReID model path must not be empty"
        )
    if path.suffix.lower() != ".pth":
        raise PipelineConfigError(
            "configured ReID model path must use the canonical .pth "
            "checkpoint extension"
        )
    base_directory = Path.cwd() if cwd is None else Path(cwd)
    try:
        return (
            path.resolve(strict=False)
            if path.is_absolute()
            else (base_directory / path).resolve(strict=False)
        )
    except (OSError, RuntimeError) as error:
        raise PipelineConfigError(
            "configured ReID model path could not be resolved"
        ) from error


def load_spatial_pipeline_preset(
    name: str = DEFAULT_SPATIAL_PRESET,
) -> SpatialPipelinePreset:
    """Load a validated built-in YAML preset by resource name."""

    if not isinstance(name, str) or _PRESET_NAME.fullmatch(name) is None:
        raise PipelineConfigError(
            "preset name must contain only lowercase letters, digits, "
            "underscores, and hyphens"
        )
    resource = resources.files("naturallab.config.presets").joinpath(
        f"{name}.yaml"
    )
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PipelineConfigError(
            f"unknown built-in spatial pipeline preset: {name!r}"
        ) from exc
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PipelineConfigError(
            f"built-in preset {name!r} is not valid YAML"
        ) from exc
    if not isinstance(document, Mapping):
        raise PipelineConfigError(
            f"built-in preset {name!r} must contain a mapping"
        )
    preset = parse_spatial_pipeline_preset(document)
    if preset.name != name:
        raise PipelineConfigError(
            f"built-in preset name {preset.name!r} does not match {name!r}"
        )
    return preset
