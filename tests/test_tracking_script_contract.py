from pathlib import Path

import numpy as np
import pytest
import yaml

from naturallab.spatial_tracking.calibration import (
    FloorPlaneCalibrationArtifact,
    ImageSize,
    IntrinsicCalibrationArtifact,
)
from scripts.track_people_in_video import (
    add_distance_statistics,
    build_argument_parser,
    discover_video_files,
    load_floor_tracker,
    select_track_evidence_rows,
    summarize_track_records,
)


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_floor_tracker_accepts_canonical_calibration_keys(tmp_path: Path) -> None:
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(
        camera_path,
        {
            "camera_matrix": np.eye(3).tolist(),
            "dist_coeff": [0, 0, 0, 0, 0],
        },
    )
    write_yaml(floor_path, {"floor_plane": [0, 1, 0, -1000]})

    tracker = load_floor_tracker(camera_path, floor_path)

    np.testing.assert_allclose(tracker.camera_matrix, np.eye(3))
    np.testing.assert_allclose(tracker.floor_plane, [0, 1, 0, -1000])
    assert tracker.correction_factor == 1.0


def test_floor_tracker_migrates_consolidated_script_keys(tmp_path: Path) -> None:
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(
        camera_path,
        {
            "camera_matrix": np.eye(3).tolist(),
            "dist_coeffs": [0, 0, 0, 0, 0],
        },
    )
    write_yaml(
        floor_path,
        {"plane_normal": [0, 1, 0], "plane_d": -1000},
    )

    tracker = load_floor_tracker(camera_path, floor_path, correction_factor=1.25)

    np.testing.assert_allclose(tracker.dist_coeffs, [0, 0, 0, 0, 0])
    np.testing.assert_allclose(tracker.floor_plane, [0, 1, 0, -1000])
    assert tracker.correction_factor == 1.25


def test_floor_tracker_scales_outlier_thresholds_to_calibration_units(
    tmp_path: Path,
) -> None:
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(
        camera_path,
        {
            "camera_matrix": np.eye(3).tolist(),
            "dist_coeffs": [0, 0, 0, 0, 0],
        },
    )
    write_yaml(
        floor_path,
        {
            "plane_normal": [0, 1, 0],
            "plane_d": -1,
            "units": "m",
        },
    )

    tracker = load_floor_tracker(camera_path, floor_path)

    assert tracker.min_movement == pytest.approx(0.005)
    assert tracker.max_movement == pytest.approx(0.2)


def test_floor_tracker_validates_versioned_artifact_binding(tmp_path: Path) -> None:
    intrinsics = IntrinsicCalibrationArtifact(
        camera_id="ceiling-01",
        image_size=ImageSize(1920, 1080),
        camera_matrix=np.eye(3).tolist(),
        dist_coeff=[0, 0, 0, 0, 0],
    )
    floor = FloorPlaneCalibrationArtifact(
        camera_id=intrinsics.camera_id,
        image_size=intrinsics.image_size,
        floor_plane=[0, 1, 0, -1000],
        units="mm",
        coordinate_frame=intrinsics.coordinate_frame,
        intrinsic_sha256=intrinsics.sha256,
    )
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(camera_path, intrinsics.to_dict())
    write_yaml(floor_path, floor.to_dict())

    tracker = load_floor_tracker(camera_path, floor_path)

    assert tracker.calibration_bundle is not None
    assert tracker.units == "mm"

    tampered = floor.to_dict()
    tampered["intrinsic_sha256"] = "0" * 64
    write_yaml(floor_path, tampered)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_floor_tracker(camera_path, floor_path)


def test_versioned_calibration_rejects_empirical_correction_factor(
    tmp_path: Path,
) -> None:
    intrinsics = IntrinsicCalibrationArtifact(
        camera_id="ceiling-01",
        image_size=ImageSize(640, 480),
        camera_matrix=np.eye(3).tolist(),
        dist_coeff=[0, 0, 0, 0, 0],
    )
    floor = FloorPlaneCalibrationArtifact(
        camera_id=intrinsics.camera_id,
        image_size=intrinsics.image_size,
        floor_plane=[0, 1, 0, -1000],
        units="mm",
        coordinate_frame=intrinsics.coordinate_frame,
        intrinsic_sha256=intrinsics.sha256,
    )
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(camera_path, intrinsics.to_dict())
    write_yaml(floor_path, floor.to_dict())

    with pytest.raises(ValueError, match="legacy option"):
        load_floor_tracker(
            camera_path,
            floor_path,
            correction_factor=1.2,
        )


def test_legacy_correction_factor_is_hidden_from_tracking_help() -> None:
    parser = build_argument_parser()

    assert "--correction-factor" not in parser.format_help()
    args = parser.parse_args(
        [
            "--input",
            "video.mp4",
            "--output",
            "results",
            "--correction-factor",
            "1.25",
        ]
    )
    assert args.correction_factor == pytest.approx(1.25)


def test_distance_statistics_use_declared_units() -> None:
    millimetres = {}
    add_distance_statistics(millimetres, 1250.0, "mm")
    assert millimetres == {
        "total_distance": 1250.0,
        "distance_units": "mm",
        "total_distance_mm": 1250.0,
        "total_distance_m": 1.25,
    }

    centimetres = {}
    add_distance_statistics(centimetres, 125.0, "cm")
    assert centimetres == {
        "total_distance": 125.0,
        "distance_units": "cm",
        "total_distance_cm": 125.0,
        "total_distance_m": 1.25,
    }

    micrometres = {}
    add_distance_statistics(micrometres, 1_250_000.0, "µm")
    assert micrometres == {
        "total_distance": 1_250_000.0,
        "distance_units": "um",
        "total_distance_um": 1_250_000.0,
        "total_distance_m": 1.25,
    }


@pytest.mark.parametrize(
    "correction_factor",
    [0, -1, float("nan"), float("inf"), float("-inf"), True],
)
def test_floor_tracker_rejects_invalid_legacy_correction_factor(
    tmp_path: Path,
    correction_factor,
) -> None:
    camera_path = tmp_path / "intrinsics.yaml"
    floor_path = tmp_path / "floor.yaml"
    write_yaml(
        camera_path,
        {
            "camera_matrix": np.eye(3).tolist(),
            "dist_coeff": [0, 0, 0, 0, 0],
        },
    )
    write_yaml(floor_path, {"floor_plane": [0, 1, 0, -1000]})

    with pytest.raises(ValueError, match="finite positive"):
        load_floor_tracker(
            camera_path,
            floor_path,
            correction_factor=correction_factor,
        )


def test_track_statistics_separate_observations_and_predictions() -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [3, 4, 5, 8],
            "is_prediction": [False, True, True, False],
        }
    )

    assert summarize_track_records("track-1", track_df, fps=2.0) == {
        "track_id": "track-1",
        "first_frame": 3,
        "last_frame": 8,
        "track_records": 4,
        "observed_frames": 2,
        "predicted_frames": 2,
        "span_frames": 5,
        "covered_frame_count": 6,
        "duration_seconds": 2.5,
        "timing_basis": "nominal_fps",
    }


def test_track_statistics_prefer_source_timestamps_for_vfr() -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [10, 11, 12],
            "is_prediction": [False, True, False],
            "timestamp_seconds": [1.0, 1.04, 1.11],
        }
    )

    stat = summarize_track_records("track-1", track_df, fps=25.0)

    assert stat["first_timestamp_seconds"] == 1.0
    assert stat["last_timestamp_seconds"] == 1.11
    assert stat["duration_seconds"] == pytest.approx(0.11)
    assert stat["timing_basis"] == "source_timestamps"


@pytest.mark.parametrize(
    "timestamps,fps,expected_duration,expected_basis",
    [
        ([0.0, None, None], 0.0, None, None),
        ([None, 0.04, 0.08], 25.0, 0.08, "nominal_fps"),
    ],
)
def test_track_statistics_require_timed_track_endpoints(
    timestamps,
    fps,
    expected_duration,
    expected_basis,
) -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "is_prediction": [False, True, False],
            "timestamp_seconds": timestamps,
        }
    )

    stat = summarize_track_records("track-1", track_df, fps=fps)

    if expected_duration is None:
        assert stat["duration_seconds"] is None
    else:
        assert stat["duration_seconds"] == pytest.approx(expected_duration)
    assert stat["timing_basis"] == expected_basis
    assert "first_timestamp_seconds" not in stat
    assert "last_timestamp_seconds" not in stat


def test_single_frame_track_has_zero_elapsed_duration() -> None:
    import pandas as pd

    untimed = pd.DataFrame(
        {"frame": [7], "is_prediction": [False]}
    )
    timed = untimed.assign(timestamp_seconds=[2.5])

    untimed_stat = summarize_track_records("track-1", untimed, fps=25.0)
    timed_stat = summarize_track_records("track-1", timed, fps=25.0)

    assert untimed_stat["span_frames"] == 0
    assert untimed_stat["covered_frame_count"] == 1
    assert untimed_stat["duration_seconds"] == 0.0
    assert timed_stat["duration_seconds"] == 0.0


def test_video_directory_discovery_is_case_insensitive_and_natural(
    tmp_path: Path,
) -> None:
    for name in (
        "session10.MP4",
        "session2.mkv",
        "session1.mov",
        "notes.txt",
    ):
        (tmp_path / name).write_bytes(b"placeholder")

    assert [path.name for path in discover_video_files(tmp_path)] == [
        "session1.mov",
        "session2.mkv",
        "session10.MP4",
    ]


def test_identity_evidence_prefers_observed_track_rows() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "is_prediction": [False, True, True, False],
        }
    )

    selected = select_track_evidence_rows(rows)

    assert selected["frame"].tolist() == [0, 3]


def test_identity_evidence_falls_back_when_only_predictions_exist() -> None:
    import pandas as pd

    rows = pd.DataFrame(
        {
            "frame": [1, 2],
            "is_prediction": [True, True],
        }
    )

    selected = select_track_evidence_rows(rows)

    assert selected["frame"].tolist() == [1, 2]


def test_owl_tracker_detector_fails_loudly_when_model_is_unavailable() -> None:
    from naturallab.spatial_tracking.detection.owl_detector import (
        OWLDetectorModule,
    )

    detector = OWLDetectorModule.__new__(OWLDetectorModule)
    detector.has_model = False
    detector.load_error = "test initialization failure"

    with pytest.raises(RuntimeError, match="test initialization failure"):
        detector.process({"frame": np.zeros((2, 2, 3), dtype=np.uint8)})
