"""Factories for constructing validated spatial tracking workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Callable, Optional
import warnings

from naturallab.spatial_tracking.base import TrackerModule
from naturallab.spatial_tracking.detection import QwenDetectorModule
from naturallab.spatial_tracking.pipeline.config import (
    DEFAULT_SPATIAL_PRESET,
    PipelineConfigError,
    SpatialPipelinePreset,
    load_spatial_pipeline_preset,
    resolve_reid_model_path,
    validate_spatial_pipeline_preset,
)
from naturallab.spatial_tracking.pipeline.reid import (
    ReIDCheckpointError,
    ReIDCheckpointResolution,
    ReIDCheckpointWarning,
    acquire_reid_checkpoint,
    fallback_guidance,
    fallback_resolution,
    reid_checkpoint_source,
)
from naturallab.spatial_tracking.pipeline.tracker_pipeline import (
    TrackerPipeline,
)
from naturallab.spatial_tracking.vlm import (
    JSONTransport,
    QwenBackendConfig,
    QwenPersonGrounder,
    QwenTrackRoleAssigner,
)

DeepSORTFactory = Callable[..., TrackerModule]


class PipelineDependencyError(RuntimeError):
    """Raised when a selected pipeline backend cannot be constructed."""


@dataclass(frozen=True)
class SpatialPipelineComponents:
    """Constructed runtime components plus their validated source preset."""

    preset: SpatialPipelinePreset
    pipeline: TrackerPipeline
    detector: QwenDetectorModule
    tracker: TrackerModule
    role_assigner: QwenTrackRoleAssigner
    qwen_config: QwenBackendConfig
    reid_model_path: Path
    reid_checkpoint: ReIDCheckpointResolution

    def provenance(self) -> dict[str, Any]:
        """Return configured runtime identity without an API key."""

        value = self.preset.provenance()
        value["endpoint_identity"] = self.qwen_config.endpoint_identity
        tracker_parameters = dict(value["tracker_parameters"])
        tracker_parameters["reid_model_path"] = self.reid_model_path.name
        tracker_parameters["allow_reid_fallback"] = (
            self.reid_checkpoint.fallback_allowed
        )
        value["tracker_parameters"] = tracker_parameters
        value["reid_model_filename"] = self.reid_model_path.name
        value["reid_model"] = self.reid_checkpoint.provenance(
            self.preset.tracker.reid_model
        )
        return value


def _resolve_service_config(
    preset: SpatialPipelinePreset,
    *,
    base_url: Optional[str],
    api_key: Optional[str],
) -> QwenBackendConfig:
    service = preset.vlm_service
    resolved_base_url = (
        os.environ.get(service.base_url_env, service.default_base_url)
        if base_url is None
        else base_url
    )
    resolved_api_key = (
        os.environ.get(service.api_key_env)
        if api_key is None
        else api_key
    )
    return QwenBackendConfig(
        model_id=service.model_id,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        precision=service.precision,
        detection_cadence_frames=(
            preset.detector.detection_cadence_frames
        ),
        timeout_seconds=service.timeout_seconds,
        max_tokens=service.max_tokens,
        enable_thinking=service.enable_thinking,
    )


def _default_deepsort_factory() -> DeepSORTFactory:
    try:
        from naturallab.spatial_tracking.tracking.deepsort.tracker import (
            DeepSORTTracker,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise PipelineDependencyError(
            "DeepSORT backend is unavailable. Install the tracking "
            "dependencies, including PyTorch."
        ) from exc
    return DeepSORTTracker


def _build_tracker(
    preset: SpatialPipelinePreset,
    *,
    deep_sort_factory: Optional[DeepSORTFactory],
    feature_extractor: Optional[Any],
    feature_gallery: Optional[Any],
    reid_model_path: Path,
    allow_reid_fallback: bool,
) -> TrackerModule:
    tracker_factory = deep_sort_factory or _default_deepsort_factory()
    kwargs = preset.tracker.constructor_kwargs(
        allow_reid_fallback=allow_reid_fallback,
    )
    kwargs["reid_model_path"] = str(reid_model_path)
    if feature_extractor is not None:
        kwargs["feature_extractor"] = feature_extractor
    if feature_gallery is not None:
        kwargs["feature_gallery"] = feature_gallery
    try:
        tracker = tracker_factory(**kwargs)
    except PipelineDependencyError:
        raise
    except Exception as exc:
        raise PipelineDependencyError(
            "DeepSORT backend could not be initialized with preset "
            f"{preset.name!r}: {exc}"
        ) from exc
    if not isinstance(tracker, TrackerModule):
        raise PipelineDependencyError(
            "DeepSORT factory must return a TrackerModule instance"
        )
    return tracker


def build_spatial_pipeline(
    preset: Optional[SpatialPipelinePreset] = None,
    *,
    preset_name: str = DEFAULT_SPATIAL_PRESET,
    transport: Optional[JSONTransport] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    reid_model_path: Optional[str | os.PathLike[str]] = None,
    deep_sort_factory: Optional[DeepSORTFactory] = None,
    feature_extractor: Optional[Any] = None,
    feature_gallery: Optional[Any] = None,
    allow_reid_fallback: Optional[bool] = None,
    reid_download_opener: Optional[Callable[..., Any]] = None,
) -> SpatialPipelineComponents:
    """Build Qwen detection, DeepSORT tracking, and Qwen role assignment.

    Runtime services and DeepSORT feature models are injectable so callers can
    test construction without network calls. The pinned ReID checkpoint is
    downloaded and verified when absent. Histogram fallback is disabled unless
    the caller explicitly sets ``allow_reid_fallback=True`` for this run.
    """

    if preset is not None and preset_name != DEFAULT_SPATIAL_PRESET:
        raise PipelineConfigError(
            "pass either a preset object or preset_name, not both"
        )
    selected = preset or load_spatial_pipeline_preset(preset_name)
    validate_spatial_pipeline_preset(selected)
    if allow_reid_fallback is not None and not isinstance(
        allow_reid_fallback,
        bool,
    ):
        raise PipelineConfigError(
            "allow_reid_fallback must be true, false, or None"
        )
    effective_allow_reid_fallback = (
        selected.tracker.allow_reid_fallback
        if allow_reid_fallback is None
        else allow_reid_fallback
    )
    qwen_config = _resolve_service_config(
        selected,
        base_url=base_url,
        api_key=api_key,
    )
    resolved_reid_model_path = resolve_reid_model_path(
        selected,
        configured_path=reid_model_path,
    )
    checkpoint = ReIDCheckpointResolution(
        path=resolved_reid_model_path,
        source=(
            "injected-runtime"
            if feature_extractor is not None or deep_sort_factory is not None
            else reid_checkpoint_source(
                configured_path=reid_model_path,
                environ=os.environ,
            )
        ),
        verified=False,
        downloaded=False,
        fallback_allowed=effective_allow_reid_fallback,
    )
    runtime_feature_extractor = feature_extractor
    if feature_extractor is None and deep_sort_factory is None:
        try:
            checkpoint = acquire_reid_checkpoint(
                selected,
                configured_path=reid_model_path,
                opener=reid_download_opener,
            )
            checkpoint = replace(
                checkpoint,
                fallback_allowed=effective_allow_reid_fallback,
            )
        except ReIDCheckpointError as error:
            message = fallback_guidance(str(error))
            if not effective_allow_reid_fallback:
                warnings.warn(
                    message,
                    ReIDCheckpointWarning,
                    stacklevel=2,
                )
                raise PipelineDependencyError(message) from error
            checkpoint = fallback_resolution(
                selected,
                error,
                configured_path=reid_model_path,
            )
            from naturallab.spatial_tracking.tracking.deepsort.feature_extractor import (
                HistogramFeatureExtractor,
            )

            runtime_feature_extractor = HistogramFeatureExtractor(error)

    grounder = QwenPersonGrounder(qwen_config, transport=transport)
    detector = QwenDetectorModule(
        grounder=grounder,
        cadence_frames=selected.detector.detection_cadence_frames,
        jpeg_quality=selected.detector.jpeg_quality,
        confidence_threshold=selected.detector.confidence_threshold,
    )
    tracker = _build_tracker(
        selected,
        deep_sort_factory=deep_sort_factory,
        feature_extractor=runtime_feature_extractor,
        feature_gallery=feature_gallery,
        reid_model_path=resolved_reid_model_path,
        allow_reid_fallback=effective_allow_reid_fallback,
    )
    actual_reid_backend = getattr(
        tracker,
        "reid_backend",
        selected.tracker.reid_model.architecture,
    )
    if actual_reid_backend == "histogram" and not checkpoint.fallback_used:
        checkpoint = replace(
            checkpoint,
            fallback_allowed=True,
            fallback_used=True,
            failure_category=getattr(
                tracker,
                "reid_failure_category",
                "model-load-failed",
            ),
        )
    role_assigner = QwenTrackRoleAssigner(
        roles=selected.role_assignment.roles,
        role_descriptions=(
            selected.role_assignment.role_description_mapping()
        ),
        evidence_images_per_track=(
            selected.role_assignment.evidence_images_per_track
        ),
        config=qwen_config,
        transport=transport,
    )
    pipeline_provenance = selected.provenance()
    tracker_parameters = dict(
        pipeline_provenance["tracker_parameters"]
    )
    tracker_parameters["reid_model_path"] = (
        resolved_reid_model_path.name
    )
    tracker_parameters["allow_reid_fallback"] = (
        effective_allow_reid_fallback
    )
    pipeline_provenance["tracker_parameters"] = tracker_parameters
    pipeline_provenance["reid_model_filename"] = (
        resolved_reid_model_path.name
    )
    pipeline_provenance["reid_model"] = checkpoint.provenance(
        selected.tracker.reid_model
    )
    pipeline_provenance["endpoint_identity"] = (
        qwen_config.endpoint_identity
    )
    pipeline = TrackerPipeline(
        [detector, tracker],
        provenance=pipeline_provenance,
    )
    return SpatialPipelineComponents(
        preset=selected,
        pipeline=pipeline,
        detector=detector,
        tracker=tracker,
        role_assigner=role_assigner,
        qwen_config=qwen_config,
        reid_model_path=resolved_reid_model_path,
        reid_checkpoint=checkpoint,
    )
