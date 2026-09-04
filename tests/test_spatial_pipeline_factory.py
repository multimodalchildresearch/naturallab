from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from naturallab.spatial_tracking.base import TrackerModule
from naturallab.spatial_tracking.pipeline import (
    DEFAULT_REID_ARCHITECTURE,
    DEFAULT_REID_FILENAME,
    DEFAULT_REID_REPOSITORY,
    DEFAULT_REID_REVISION,
    DEFAULT_REID_SHA256,
    DEFAULT_REID_SIZE_BYTES,
    PipelineConfigError,
    PipelineDependencyError,
    ReIDCheckpointWarning,
    build_spatial_pipeline,
    load_spatial_pipeline_preset,
)
from naturallab.spatial_tracking.tracking.deepsort.tracker import (
    DeepSORTTracker,
    DeepSORTUnavailableError,
)
from naturallab.spatial_tracking.tracking.deepsort.feature_extractor import (
    AppearanceFeatureExtractor,
)
from naturallab.spatial_tracking.vlm import (
    DEFAULT_QWEN_MODEL_ID,
    EvidenceImage,
)


class FakeDeepSORT(TrackerModule):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("FakeDeepSORT")
        self.kwargs = kwargs

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {**data, "tracks": []}

    def reset(self) -> None:
        return None


class EmptyGroundingTransport:
    def post_json(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout_seconds: float,
    ) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"detections":[]}',
                    }
                }
            ]
        }


class FakeFeatureExtractor:
    has_model = True
    model_error = None
    last_inference_error = None

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def extract_deep_features(self, image: np.ndarray) -> np.ndarray:
        return np.ones(4, dtype=np.float32)

    def extract_fallback_features(self, image: np.ndarray) -> np.ndarray:
        return np.ones(48, dtype=np.float32)


class FakeFeatureGallery:
    def __init__(self) -> None:
        self.gallery: Dict[str, Any] = {}
        self.frame_count = 0

    def update_frame_count(self) -> None:
        self.frame_count += 1

    def get_feature_count(self, track_id: str) -> int:
        return 0

    def find_matching_id(self, features: np.ndarray) -> None:
        return None

    def add_features(self, track_id: str, features: np.ndarray) -> None:
        self.gallery.setdefault(
            track_id,
            {"features": [], "last_seen": self.frame_count},
        )["features"].append(features)


def test_builtin_quality_preset_is_complete_and_exact() -> None:
    preset = load_spatial_pipeline_preset()

    assert preset.name == "qwen36_27b_quality"
    assert preset.version == 3
    assert preset.vlm_service.model_id == DEFAULT_QWEN_MODEL_ID
    assert preset.vlm_service.max_tokens == 1024
    assert preset.vlm_service.enable_thinking is False
    assert preset.detector.model_id == DEFAULT_QWEN_MODEL_ID
    assert preset.role_assignment.model_id == DEFAULT_QWEN_MODEL_ID
    assert preset.detector.backend == "qwen_grounding"
    assert preset.detector.detection_cadence_frames == 10
    assert preset.detector.confidence_threshold is None
    assert preset.detector.jpeg_quality == 95
    assert preset.detector.adaptive_redetection is False
    assert preset.tracker.backend == "deepsort"
    assert preset.tracker.reid_model.architecture == DEFAULT_REID_ARCHITECTURE
    assert preset.tracker.reid_model.repository == DEFAULT_REID_REPOSITORY
    assert preset.tracker.reid_model.revision == DEFAULT_REID_REVISION
    assert preset.tracker.reid_model.filename == DEFAULT_REID_FILENAME
    assert preset.tracker.reid_model.sha256 == DEFAULT_REID_SHA256
    assert preset.tracker.reid_model.size_bytes == DEFAULT_REID_SIZE_BYTES
    assert preset.tracker.reid_model.auto_download is True
    assert preset.tracker.allow_reid_fallback is False


def test_factory_builds_exact_config_with_injected_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "NATURALLAB_VLM_BASE_URL",
        "https://quality-vlm.internal:8123/v1",
    )
    monkeypatch.setenv("NATURALLAB_VLM_API_KEY", "do-not-export")
    extractor = FakeFeatureExtractor()
    gallery = FakeFeatureGallery()

    components = build_spatial_pipeline(
        transport=EmptyGroundingTransport(),
        deep_sort_factory=FakeDeepSORT,
        feature_extractor=extractor,
        feature_gallery=gallery,
    )

    assert components.qwen_config.model_id == DEFAULT_QWEN_MODEL_ID
    assert components.qwen_config.detection_cadence_frames == 10
    assert components.qwen_config.max_tokens == 1024
    assert components.qwen_config.enable_thinking is False
    assert components.qwen_config.base_url == (
        "https://quality-vlm.internal:8123/v1"
    )
    assert components.qwen_config.api_key == "do-not-export"
    assert components.detector.cadence_frames == 10
    assert components.detector.confidence_threshold is None
    assert components.detector.jpeg_quality == 95
    assert components.role_assigner is None
    assert components.pipeline.modules == [
        components.detector,
        components.tracker,
    ]

    tracker = components.tracker
    assert isinstance(tracker, FakeDeepSORT)
    assert tracker.kwargs["max_age"] == 30
    assert tracker.kwargs["allow_reid_fallback"] is False
    assert tracker.kwargs["reid_model_architecture"] == (
        DEFAULT_REID_ARCHITECTURE
    )
    assert tracker.kwargs["reid_model_path"].endswith(DEFAULT_REID_FILENAME)
    assert tracker.kwargs["feature_extractor"] is extractor
    assert tracker.kwargs["feature_gallery"] is gallery

    provenance = components.provenance()
    assert provenance["model_id"] == DEFAULT_QWEN_MODEL_ID
    assert provenance["detection_cadence_frames"] == 10
    assert provenance["confidence_threshold"] is None
    assert provenance["vlm_max_tokens"] == 1024
    assert provenance["vlm_enable_thinking"] is False
    assert provenance["role_evidence_images_per_track"] == 5
    assert provenance["role_assignment_enabled"] is False
    assert provenance["roles"] == []
    assert provenance["role_descriptions"] == {}
    assert provenance["reid_model"]["architecture"] == (
        DEFAULT_REID_ARCHITECTURE
    )
    assert provenance["reid_model"]["checkpoint_source"] == "injected-runtime"
    assert provenance["reid_model"]["checkpoint_verified"] is False
    assert provenance["reid_model"]["fallback_used"] is False
    assert provenance["endpoint_identity"] == (
        "https://quality-vlm.internal:8123/v1"
    )
    assert "do-not-export" not in repr(provenance)

    success, _, output = components.pipeline.process_frame(
        np.zeros((8, 8, 3), dtype=np.uint8)
    )
    assert success is True
    assert output["pipeline_provenance"] == provenance
    assert "do-not-export" not in repr(output["pipeline_provenance"])


def test_factory_builds_roles_only_from_an_explicit_arbitrary_mapping() -> None:
    components = build_spatial_pipeline(
        transport=EmptyGroundingTransport(),
        deep_sort_factory=FakeDeepSORT,
        feature_extractor=FakeFeatureExtractor(),
        role_descriptions={
            "participant": "the person completing the task",
            "facilitator": "the person presenting the materials",
        },
    )

    assert components.role_assigner is not None
    assert components.role_assigner.roles == (
        "participant",
        "facilitator",
    )
    assert components.role_assigner.role_descriptions == {
        "participant": "the person completing the task",
        "facilitator": "the person presenting the materials",
    }
    assert components.role_assigner.evidence_images_per_track == 5
    provenance = components.provenance()
    assert provenance["role_assignment_enabled"] is True
    assert provenance["roles"] == ["participant", "facilitator"]
    assert provenance["role_descriptions"] == {
        "participant": "the person completing the task",
        "facilitator": "the person presenting the materials",
    }
    assert components.pipeline.provenance == provenance

    with pytest.raises(
        ValueError,
        match="evidence_images_per_track limit",
    ):
        components.role_assigner.assign_role(
            "track-too-many-images",
            [
                # Content is never sent because the limit is checked first.
                EvidenceImage(f"frame-{index}".encode())
                for index in range(6)
            ],
        )


def test_deepsort_diagnostics_are_disabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    tracker = DeepSORTTracker(
        feature_extractor=FakeFeatureExtractor(),
        feature_gallery=FakeFeatureGallery(),
    )

    assert tracker.diagnostics is None
    assert tracker.diagnostics_provenance == {
        "enabled": False,
        "path_policy": "explicit_new_or_empty_directory_required",
        "output_directory_name": None,
        "persisted_content": "none",
        "persists_images": False,
    }
    assert not (tmp_path / "deepsort_diagnostics").exists()


def test_deepsort_diagnostics_require_a_fresh_explicit_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="diagnostics_output_dir"):
        DeepSORTTracker(
            enable_diagnostics=True,
            feature_extractor=FakeFeatureExtractor(),
            feature_gallery=FakeFeatureGallery(),
        )

    reused = tmp_path / "reused-diagnostics"
    reused.mkdir()
    (reused / "prior-run.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        DeepSORTTracker(
            enable_diagnostics=True,
            diagnostics_output_dir=reused,
            feature_extractor=FakeFeatureExtractor(),
            feature_gallery=FakeFeatureGallery(),
        )
    assert (reused / "prior-run.txt").read_text(encoding="utf-8") == "old"


def test_opt_in_deepsort_diagnostics_persist_text_only_and_are_provenanced(
    tmp_path: Path,
) -> None:
    preset = load_spatial_pipeline_preset()
    with_diagnostics = replace(
        preset,
        tracker=replace(preset.tracker, enable_diagnostics=True),
    )
    output_directory = tmp_path / "private-run" / "diagnostics"
    components = build_spatial_pipeline(
        preset=with_diagnostics,
        transport=EmptyGroundingTransport(),
        feature_extractor=FakeFeatureExtractor(),
        feature_gallery=FakeFeatureGallery(),
        diagnostics_output_dir=output_directory,
    )

    components.pipeline.process_frame(
        np.zeros((60, 80, 3), dtype=np.uint8)
    )
    provenance = components.provenance()["diagnostics"]
    assert provenance == {
        "enabled": True,
        "path_policy": "explicit_new_or_empty_directory_required",
        "output_directory_name": "diagnostics",
        "persisted_content": "text_log_only",
        "persists_images": False,
    }
    assert str(tmp_path) not in repr(provenance)
    assert (output_directory / "deepsort_log.txt").is_file()
    assert not list(output_directory.rglob("*.jpg"))


@pytest.mark.parametrize(
    "role_descriptions",
    [
        {},
        {"": "a participant"},
        {"participant": ""},
        {"participant": 3},
        [("participant", "a participant")],
    ],
)
def test_factory_rejects_invalid_explicit_role_descriptions(
    role_descriptions: Any,
) -> None:
    with pytest.raises(PipelineConfigError, match="role"):
        build_spatial_pipeline(
            deep_sort_factory=FakeDeepSORT,
            feature_extractor=FakeFeatureExtractor(),
            role_descriptions=role_descriptions,
        )


def test_factory_rejects_unsupported_backend_without_fallback() -> None:
    preset = load_spatial_pipeline_preset()
    unsupported = replace(
        preset,
        tracker=replace(preset.tracker, backend="kalman"),
    )

    with pytest.raises(
        PipelineConfigError,
        match="unsupported tracker backend 'kalman'",
    ):
        build_spatial_pipeline(
            preset=unsupported,
            deep_sort_factory=FakeDeepSORT,
        )


@pytest.mark.parametrize(
    "tracker_change,match",
    [
        ({"max_age": 0}, "max_age must be a positive integer"),
        ({"reid_device": "bogus"}, "reid_device must be cpu, cuda, or mps"),
        (
            {"allow_reid_fallback": True},
            "requires allow_reid_fallback: false",
        ),
    ],
)
def test_programmatic_preset_cannot_bypass_tracker_validation(
    tracker_change: Dict[str, Any],
    match: str,
) -> None:
    preset = load_spatial_pipeline_preset()
    invalid = replace(
        preset,
        tracker=replace(preset.tracker, **tracker_change),
    )

    with pytest.raises(PipelineConfigError, match=match):
        build_spatial_pipeline(
            preset=invalid,
            deep_sort_factory=FakeDeepSORT,
            feature_extractor=FakeFeatureExtractor(),
        )


def test_programmatic_preset_rejects_noncanonical_reid_filename() -> None:
    preset = load_spatial_pipeline_preset()
    invalid = replace(
        preset,
        tracker=replace(
            preset.tracker,
            reid_model=replace(
                preset.tracker.reid_model,
                filename="osnet_ain_x1_0_msmt17.pt",
            ),
        ),
    )

    with pytest.raises(PipelineConfigError, match="canonical .pth"):
        build_spatial_pipeline(
            preset=invalid,
            deep_sort_factory=FakeDeepSORT,
            feature_extractor=FakeFeatureExtractor(),
        )


def test_factory_surfaces_backend_initialization_failure() -> None:
    def unavailable_deepsort(**kwargs: Any) -> TrackerModule:
        raise ModuleNotFoundError("torch")

    with pytest.raises(
        PipelineDependencyError,
        match="DeepSORT backend could not be initialized",
    ) as caught:
        build_spatial_pipeline(deep_sort_factory=unavailable_deepsort)

    assert isinstance(caught.value.__cause__, ModuleNotFoundError)


def test_default_quality_factory_fails_before_loading_a_missing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NATURALLAB_REID_MODEL_PATH", raising=False)
    monkeypatch.setenv("NATURALLAB_REID_CACHE_DIR", str(tmp_path))
    before = tuple(tmp_path.iterdir())

    def fail_download(*args: Any, **kwargs: Any) -> None:
        raise OSError("offline")

    with pytest.warns(ReIDCheckpointWarning, match="No appearance fallback"):
        with pytest.raises(
            PipelineDependencyError,
            match="allow_reid_fallback=True",
        ):
            build_spatial_pipeline(reid_download_opener=fail_download)

    assert tuple(tmp_path.iterdir()) == before


def test_factory_fallback_requires_opt_in_and_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_checkpoint = tmp_path / "invalid.pth"
    invalid_checkpoint.write_bytes(b"not the pinned model")
    monkeypatch.setattr(
        AppearanceFeatureExtractor,
        "_load_model",
        lambda self, model_path: pytest.fail(
            "an integrity-rejected checkpoint must never be loaded"
        ),
    )

    with pytest.warns(ReIDCheckpointWarning, match="allow_reid_fallback=True"):
        components = build_spatial_pipeline(
            transport=EmptyGroundingTransport(),
            reid_model_path=invalid_checkpoint,
            allow_reid_fallback=True,
        )

    provenance = components.provenance()["reid_model"]
    assert components.tracker.reid_backend == "histogram"
    assert provenance["checkpoint_verified"] is False
    assert provenance["fallback_allowed"] is True
    assert provenance["fallback_used"] is True
    assert provenance["failure_category"] == "configured-checkpoint-invalid"
    assert provenance["reid_backend"] == "histogram"


def test_injected_tracker_cannot_bypass_strict_fallback_policy() -> None:
    class HistogramDeepSORT(FakeDeepSORT):
        reid_backend = "histogram"

    with pytest.raises(
        PipelineDependencyError,
        match="allow_reid_fallback was not enabled",
    ):
        build_spatial_pipeline(
            transport=EmptyGroundingTransport(),
            deep_sort_factory=HistogramDeepSORT,
            feature_extractor=FakeFeatureExtractor(),
        )


def test_unknown_preset_fails_clearly() -> None:
    with pytest.raises(PipelineConfigError, match="unknown built-in"):
        load_spatial_pipeline_preset("not_a_real_preset")


def test_quality_deepsort_requires_an_actual_reid_model() -> None:
    class MissingFeatureExtractor:
        has_model = False
        model_error = RuntimeError("checkpoint missing")

    with pytest.warns(ReIDCheckpointWarning, match="No appearance fallback"):
        with pytest.raises(
            DeepSORTUnavailableError,
            match="ReID model could not be loaded",
        ) as caught:
            DeepSORTTracker(
                allow_reid_fallback=False,
                feature_extractor=MissingFeatureExtractor(),
                feature_gallery=FakeFeatureGallery(),
                enable_diagnostics=False,
            )

    assert caught.value.__cause__ is MissingFeatureExtractor.model_error


def test_deepsort_uses_histograms_only_after_explicit_startup_opt_in() -> None:
    class MissingFeatureExtractor:
        has_model = False
        model_error = RuntimeError("checkpoint unavailable")

        def extract_fallback_features(
            self,
            image: np.ndarray,
        ) -> np.ndarray:
            return np.ones(48, dtype=np.float32)

    gallery = FakeFeatureGallery()
    with pytest.warns(ReIDCheckpointWarning, match="allow_reid_fallback=True"):
        tracker = DeepSORTTracker(
            min_hits=1,
            allow_reid_fallback=True,
            feature_extractor=MissingFeatureExtractor(),
            feature_gallery=gallery,
            enable_diagnostics=False,
        )

    tracker.process(
        {
            "frame": np.zeros((60, 80, 3), dtype=np.uint8),
            "detections": [[5.0, 5.0, 25.0, 45.0, 0.9]],
            "detection_metadata": {"skipped": False},
        }
    )

    assert tracker.reid_backend == "histogram"
    feature = next(iter(gallery.gallery.values()))["features"][0]
    assert feature.shape == (48,)


def test_legacy_false_strict_flag_cannot_enable_fallback() -> None:
    with pytest.warns(DeprecationWarning, match="deprecated"):
        with pytest.raises(
            ValueError,
            match="set allow_reid_fallback=True explicitly",
        ):
            DeepSORTTracker(require_reid_model=False)


def test_quality_deepsort_forbids_runtime_histogram_fallback() -> None:
    class RuntimeFailingFeatureExtractor:
        has_model = True
        model_error = None
        last_inference_error = RuntimeError("OSNet inference failed")

        def __init__(self) -> None:
            self.fallback_calls = 0

        def extract_deep_features(
            self,
            image: np.ndarray,
        ) -> None:
            return None

        def extract_features(self, image: np.ndarray) -> np.ndarray:
            self.fallback_calls += 1
            return np.ones(4, dtype=np.float32)

    extractor = RuntimeFailingFeatureExtractor()
    tracker = DeepSORTTracker(
        min_hits=1,
        allow_reid_fallback=False,
        feature_extractor=extractor,
        feature_gallery=FakeFeatureGallery(),
        enable_diagnostics=False,
    )

    with pytest.raises(
        DeepSORTUnavailableError,
        match="will not switch feature dimensions",
    ) as caught:
        tracker.process(
            {
                "frame": np.zeros((60, 80, 3), dtype=np.uint8),
                "detections": [[5.0, 5.0, 25.0, 45.0, 0.9]],
                "detection_metadata": {"skipped": False},
            }
        )

    assert caught.value.__cause__ is extractor.last_inference_error
    assert extractor.fallback_calls == 0


def test_deepsort_treats_qwen_cadence_skips_as_predictions() -> None:
    gallery = FakeFeatureGallery()
    tracker = DeepSORTTracker(
        min_hits=1,
        allow_reid_fallback=False,
        feature_extractor=FakeFeatureExtractor(),
        feature_gallery=gallery,
        enable_diagnostics=False,
    )
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    observed = tracker.process(
        {
            "frame": frame,
            "detections": [[5.0, 5.0, 25.0, 45.0, None]],
            "detection_metadata": {"skipped": False},
            "detection_provenance": {"model_id": DEFAULT_QWEN_MODEL_ID},
        }
    )
    track_id = observed["tracks"][0]["id"]

    skipped = tracker.process(
        {
            "frame": frame,
            "detections": [],
            "detection_metadata": {
                "skipped": True,
                "skip_reason": "cadence",
            },
            "detection_provenance": None,
        }
    )

    assert skipped["tracks"][0]["id"] == track_id
    assert skipped["tracks"][0]["is_prediction"] is True
    assert skipped["tracks"][0]["time_since_update"] == 0
    assert skipped["detection_metadata"]["skip_reason"] == "cadence"
    assert skipped["detection_provenance"] is None
    assert gallery.frame_count == 1


def test_deepsort_accepts_new_qwen_detection_with_nullable_confidence() -> None:
    tracker = DeepSORTTracker(
        min_hits=1,
        allow_reid_fallback=False,
        feature_extractor=FakeFeatureExtractor(),
        feature_gallery=FakeFeatureGallery(),
        enable_diagnostics=False,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    tracker.process(
        {
            "frame": frame,
            "detections": [[5.0, 5.0, 25.0, 45.0, None]],
            "detection_metadata": {"skipped": False},
        }
    )

    output = tracker.process(
        {
            "frame": frame,
            "detections": [
                [5.0, 5.0, 25.0, 45.0, None],
                [60.0, 5.0, 90.0, 50.0, None],
            ],
            "detection_metadata": {"skipped": False},
        }
    )

    assert len(output["tracks"]) == 2
    assert {track["score"] for track in output["tracks"]} == {None}


def test_deepsort_moves_a_track_to_lost_state_exactly_once() -> None:
    tracker = DeepSORTTracker(
        min_hits=1,
        allow_reid_fallback=False,
        feature_extractor=FakeFeatureExtractor(),
        feature_gallery=FakeFeatureGallery(),
        enable_diagnostics=False,
    )
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    observed = tracker.process(
        {
            "frame": frame,
            "detections": [[5.0, 5.0, 25.0, 45.0, 0.9]],
            "detection_metadata": {"skipped": False},
        }
    )
    track_id = observed["tracks"][0]["id"]

    for _ in range(3):
        tracker.process(
            {
                "frame": frame,
                "detections": [],
                "detection_metadata": {"skipped": False},
            }
        )

    assert tracker.tracks == []
    assert len(tracker.lost_tracks) == 1
    lost_track = tracker.lost_tracks[0]
    assert lost_track["id"] == track_id
    assert lost_track["time_since_update"] == 3

    tracker.process(
        {
            "frame": frame,
            "detections": [],
            "detection_metadata": {"skipped": False},
        }
    )

    assert tracker.tracks == []
    assert len(tracker.lost_tracks) == 1
    assert tracker.lost_tracks[0] is lost_track
    assert lost_track["time_since_update"] == 4
