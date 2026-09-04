import json
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
    FLOOR_STATISTICS_OUTPUT_COLUMNS,
    STATISTICS_OUTPUT_COLUMNS,
    TRACK_OUTPUT_COLUMNS,
    add_distance_statistics,
    build_argument_parser,
    build_video_output_plan,
    configured_role_evidence_limit,
    discover_video_files,
    expected_decoded_frame_count,
    floor_metric_provenance,
    load_floor_tracker,
    main,
    prepare_video_output,
    probe_video,
    select_track_evidence_rows,
    summarize_track_records,
    validate_cli_args,
    validate_video_output_plan,
    write_track_tables,
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
    assert not hasattr(tracker, "correction_factor")


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

    tracker = load_floor_tracker(camera_path, floor_path)

    np.testing.assert_allclose(tracker.dist_coeffs, [0, 0, 0, 0, 0])
    np.testing.assert_allclose(tracker.floor_plane, [0, 1, 0, -1000])
    assert not hasattr(tracker, "correction_factor")


def test_floor_tracker_has_no_hidden_per_update_distance_filter(
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
    positions = iter(
        (
            np.array([0.0, 0.0, 1.0]),
            np.array([0.001, 0.0, 1.0]),
            np.array([0.501, 0.0, 1.0]),
        )
    )
    tracker.project_to_floor = lambda _point: next(positions)

    for _ in range(3):
        tracker.update_track("participant", [0, 0, 10, 10])

    assert tracker.get_distance("participant") == pytest.approx(0.501)


def test_floor_tracker_accumulates_unscaled_calibration_distance(
    tmp_path: Path,
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
    tracker = load_floor_tracker(camera_path, floor_path)
    positions = iter(
        (
            np.array([0.0, 0.0, 1000.0]),
            np.array([10.0, 0.0, 1000.0]),
        )
    )
    tracker.project_to_floor = lambda _point: next(positions)

    tracker.update_track("participant", [0, 0, 10, 10])
    tracker.update_track("participant", [0, 0, 10, 10])

    assert tracker.get_distance("participant") == pytest.approx(10.0)


def test_floor_tracker_ignores_nonfinite_projected_positions(
    tmp_path: Path,
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
    tracker = load_floor_tracker(camera_path, floor_path)
    positions = iter(
        (
            np.array([0.0, 0.0, 1000.0]),
            np.array([np.nan, 0.0, 1000.0]),
            np.array([10.0, 0.0, 1000.0]),
        )
    )
    tracker.project_to_floor = lambda _point: next(positions)

    assert tracker.update_track("participant", [0, 0, 10, 10]) is not None
    assert tracker.update_track("participant", [0, 0, 10, 10]) is None
    assert tracker.update_track("participant", [0, 0, 10, 10]) is not None

    assert tracker.get_distance("participant") == pytest.approx(10.0)
    assert tracker.get_projection_summary("participant") == {
        "floor_projection_attempts": 3,
        "floor_projection_valid": 2,
        "floor_projection_missed": 1,
        "floor_projection_gap_count": 1,
        "floor_projection_coverage": pytest.approx(2 / 3),
        "distance_complete": False,
        "distance_status": "partial_projection_gaps",
    }


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
    metric_provenance = floor_metric_provenance(tracker)
    assert metric_provenance == {
        "units": "mm",
        "scale_source": "canonical-calibration",
        "distance_accumulation": (
            "all_finite_consecutive_floor_displacements"
        ),
        "projection_gap_policy": (
            "bridge_valid_endpoints_and_mark_distance_partial"
        ),
        "completeness_output": "per-track track_statistics.csv fields",
        "per_update_filter": None,
        "camera_id": "ceiling-01",
        "intrinsic_sha256": intrinsics.sha256,
        "floor_sha256": floor.sha256,
    }

    tampered = floor.to_dict()
    tampered["intrinsic_sha256"] = "0" * 64
    write_yaml(floor_path, tampered)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_floor_tracker(camera_path, floor_path)


def test_tracking_cli_rejects_removed_correction_factor() -> None:
    parser = build_argument_parser()

    assert "--correction-factor" not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--input",
                "video.mp4",
                "--output",
                "results",
                "--correction-factor",
                "1.25",
            ]
        )


def test_identity_evidence_limit_comes_from_quality_preset() -> None:
    parser = build_argument_parser()
    limit = configured_role_evidence_limit()
    defaults = parser.parse_args(
        ["--input", "video.mp4", "--output", "results"]
    )

    assert defaults.identity_evidence_frames == limit == 5

    excessive = parser.parse_args(
        [
            "--input",
            "video.mp4",
            "--output",
            "results",
            "--identity-evidence-frames",
            str(limit + 1),
        ]
    )
    with pytest.raises(SystemExit):
        validate_cli_args(parser, excessive)


def test_tracking_cli_does_not_advertise_reserved_no_op_options() -> None:
    parser = build_argument_parser()
    help_text = parser.format_help()

    assert "--visualize" not in help_text
    assert "--batch-size" not in help_text
    assert parser.parse_args(
        ["--input", "video.mp4", "--output", "results"]
    ).yolo_model == "yolo11n.pt"


def test_tracking_cli_rejects_two_role_description_sources() -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "--input",
                "video.mp4",
                "--output",
                "results",
                "--identities",
                '{"participant":"person"}',
                "--identity-file",
                "roles.json",
            ]
        )


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
        "timestamp_validation_status": "not_available",
        "timestamp_source_kinds": "",
    }


def test_track_statistics_prefer_source_timestamps_for_vfr() -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [10, 11, 12],
            "is_prediction": [False, True, False],
            "timestamp_seconds": [1.0, 1.04, 1.11],
            "timestamp_source": ["container_pts"] * 3,
        }
    )

    stat = summarize_track_records("track-1", track_df, fps=25.0)

    assert stat["first_timestamp_seconds"] == 1.0
    assert stat["last_timestamp_seconds"] == 1.11
    assert stat["duration_seconds"] == pytest.approx(0.11)
    assert stat["timing_basis"] == "source_timestamps:container_pts"
    assert stat["timestamp_validation_status"] == "valid"
    assert stat["timestamp_source_kinds"] == "container_pts"


def test_track_statistics_disclose_mixed_timestamp_sources() -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "timestamp_seconds": [0.0, 0.04, 0.08],
            "timestamp_source": [
                "container_pts",
                "synthesized_fps",
                "synthesized_fps",
            ],
        }
    )

    stat = summarize_track_records("track-1", track_df, fps=25.0)

    assert stat["duration_seconds"] == pytest.approx(0.08)
    assert stat["timing_basis"] == "mixed_source_timestamps"
    assert stat["timestamp_validation_status"] == "valid"
    assert stat["timestamp_source_kinds"] == (
        "container_pts|synthesized_fps"
    )


def test_track_statistics_reject_nonmonotonic_source_timestamps() -> None:
    import pandas as pd

    track_df = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "timestamp_seconds": [1.0, 0.9, 1.1],
            "timestamp_source": ["container_pts"] * 3,
        }
    )

    stat = summarize_track_records("track-1", track_df, fps=25.0)

    assert stat["duration_seconds"] == pytest.approx(0.08)
    assert stat["timing_basis"] == "nominal_fps"
    assert stat["timestamp_validation_status"] == "nonmonotonic"
    assert "first_timestamp_seconds" not in stat
    assert "last_timestamp_seconds" not in stat


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


def test_explicit_non_video_file_is_not_accepted_as_video(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not a video", encoding="utf-8")

    assert discover_video_files(input_path) == []


def test_same_stem_inputs_fail_instead_of_sharing_results(
    tmp_path: Path,
) -> None:
    videos = [tmp_path / "session.mp4", tmp_path / "session.mov"]

    with pytest.raises(ValueError, match="same output directory"):
        build_video_output_plan(videos, tmp_path / "results")


def test_existing_results_fail_closed_and_overwrite_removes_all_stale_files(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "session.mp4"
    destination = tmp_path / "results" / "session"
    frames = destination / "frames"
    frames.mkdir(parents=True)
    (destination / "tracks.csv").write_text("old", encoding="utf-8")
    (destination / "identity_matches.json").write_text(
        "old",
        encoding="utf-8",
    )
    (frames / "frame_000000.jpg").write_bytes(b"old")
    plan = build_video_output_plan([video_path], tmp_path / "results")

    with pytest.raises(FileExistsError, match="--overwrite"):
        validate_video_output_plan(plan, overwrite=False)

    validate_video_output_plan(plan, overwrite=True)
    prepare_video_output(destination, overwrite=True)

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_overwrite_never_replaces_a_non_directory_destination(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "session.mp4"
    destination = tmp_path / "results" / "session"
    destination.parent.mkdir()
    destination.write_text("unrelated file", encoding="utf-8")
    plan = build_video_output_plan([video_path], tmp_path / "results")

    with pytest.raises(FileExistsError, match="not a directory"):
        validate_video_output_plan(plan, overwrite=True)

    assert destination.read_text(encoding="utf-8") == "unrelated file"


def test_zero_track_run_writes_stable_empty_csv_contracts(
    tmp_path: Path,
) -> None:
    import pandas as pd

    video_output = tmp_path / "session"
    video_output.mkdir()

    tracks, statistics = write_track_tables(
        video_output,
        [],
        fps=25.0,
        floor_tracker=None,
    )

    assert tracks.empty
    assert statistics.empty
    assert pd.read_csv(video_output / "tracks.csv").columns.tolist() == list(
        TRACK_OUTPUT_COLUMNS
    )
    assert pd.read_csv(
        video_output / "track_statistics.csv"
    ).columns.tolist() == list(STATISTICS_OUTPUT_COLUMNS)


def test_floor_statistics_export_projection_completeness(
    tmp_path: Path,
) -> None:
    import pandas as pd

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
    positions = iter((np.array([0.0, 0.0, 1000.0]), None))
    tracker.project_to_floor = lambda _point: next(positions)
    first = tracker.update_track("participant", [0, 0, 10, 10])
    assert first is not None
    assert tracker.update_track("participant", [0, 0, 10, 10]) is None

    rows = [
        {
            "frame": 0,
            "track_id": "participant",
            "x1": 0,
            "y1": 0,
            "x2": 10,
            "y2": 10,
            "confidence": 0.9,
            "is_prediction": False,
            "timestamp_seconds": 0.0,
            "timestamp_ns": 0,
            "timestamp_source": "test",
            "floor_x": first[0],
            "floor_y": first[1],
            "floor_z": first[2],
        },
        {
            "frame": 1,
            "track_id": "participant",
            "x1": 0,
            "y1": 0,
            "x2": 10,
            "y2": 10,
            "confidence": 0.9,
            "is_prediction": False,
            "timestamp_seconds": 0.04,
            "timestamp_ns": 40_000_000,
            "timestamp_source": "test",
        },
    ]
    _, statistics = write_track_tables(
        tmp_path,
        rows,
        fps=25.0,
        floor_tracker=tracker,
    )

    row = statistics.iloc[0]
    assert row["floor_projection_attempts"] == 2
    assert row["floor_projection_valid"] == 1
    assert row["floor_projection_missed"] == 1
    assert row["floor_projection_gap_count"] == 1
    assert row["floor_projection_coverage"] == pytest.approx(0.5)
    assert bool(row["distance_complete"]) is False
    assert row["distance_status"] == "partial_projection_gaps"
    persisted = pd.read_csv(tmp_path / "track_statistics.csv")
    assert set(FLOOR_STATISTICS_OUTPUT_COLUMNS).issubset(persisted.columns)


def test_empty_input_directory_returns_nonzero_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input"
    input_path.mkdir()
    output_path = tmp_path / "results"

    assert main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert "No supported video files found" in capsys.readouterr().out
    assert not output_path.exists()


def test_undecodable_video_returns_nonzero_without_changing_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "broken.mp4"
    input_path.write_bytes(b"not a video container")
    output_path = tmp_path / "results"

    assert probe_video(input_path) is None
    assert main(["--input", str(input_path), "--output", str(output_path)]) == 1
    assert "Could not establish a complete, decodable video contract" in (
        capsys.readouterr().out
    )
    assert not output_path.exists()


def test_probe_rejects_unknown_frame_count_even_when_first_frame_decodes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cv2

    class UnknownLengthCapture:
        def __init__(self, _path) -> None:
            self.released = False

        def isOpened(self):
            return True

        def get(self, property_id):
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0
            if property_id == cv2.CAP_PROP_FPS:
                return 25.0
            raise AssertionError(f"unexpected property: {property_id}")

        def read(self):
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(cv2, "VideoCapture", UnknownLengthCapture)
    input_path = tmp_path / "unknown-length.mp4"
    input_path.write_bytes(b"test fixture")

    assert probe_video(input_path) is None


def test_truncated_video_returns_nonzero_and_discards_partial_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    import naturallab.media
    import naturallab.spatial_tracking.detection.yolo_detector as yolo_module
    import scripts.track_people_in_video as tracking_script

    class EmptyDetector:
        def __init__(self, **_kwargs) -> None:
            pass

        def process(self, data):
            return {
                **data,
                "detections": [],
                "detection_metadata": {"skipped": False},
                "detection_provenance": {"backend": "test-empty"},
            }

    class TruncatedSource:
        def __init__(self, path, *, stop_frame=None) -> None:
            self.path = path
            self.stop_frame = stop_frame

        def __iter__(self):
            yield SimpleNamespace(
                frame_index=0,
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                metadata={"timestamp_source": "test"},
                source_timestamp=0.0,
                timestamp_ns=0,
            )

    monkeypatch.setattr(yolo_module, "YOLODetectorModule", EmptyDetector)
    monkeypatch.setattr(naturallab.media, "VideoFileSource", TruncatedSource)
    monkeypatch.setattr(
        tracking_script,
        "probe_video",
        lambda _path: (100, 25.0),
    )
    input_path = tmp_path / "truncated.mp4"
    input_path.write_bytes(b"test fixture")
    output_path = tmp_path / "results"

    result = main(
        ["--input", str(input_path), "--output", str(output_path)]
    )

    captured = capsys.readouterr().out
    assert result == 1
    assert "decoding ended early after 1 of 100 expected frames" in captured
    assert "Processing complete!" not in captured
    assert not (output_path / "truncated").exists()


def test_failed_requested_frame_encoding_discards_partial_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from types import SimpleNamespace

    import cv2

    import naturallab.media
    import naturallab.spatial_tracking.detection.yolo_detector as yolo_module
    import scripts.track_people_in_video as tracking_script

    class EmptyDetector:
        def __init__(self, **_kwargs) -> None:
            pass

        def process(self, data):
            return {
                **data,
                "detections": [],
                "detection_metadata": {"skipped": False},
                "detection_provenance": {"backend": "test-empty"},
            }

    class OneFrameSource:
        def __init__(self, path, *, stop_frame=None) -> None:
            self.path = path
            self.stop_frame = stop_frame

        def __iter__(self):
            yield SimpleNamespace(
                frame_index=0,
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                metadata={"timestamp_source": "test"},
                source_timestamp=0.0,
                timestamp_ns=0,
            )

    monkeypatch.setattr(yolo_module, "YOLODetectorModule", EmptyDetector)
    monkeypatch.setattr(naturallab.media, "VideoFileSource", OneFrameSource)
    monkeypatch.setattr(
        tracking_script,
        "probe_video",
        lambda _path: (1, 25.0),
    )
    monkeypatch.setattr(cv2, "imwrite", lambda *_args, **_kwargs: False)
    input_path = tmp_path / "session.mp4"
    input_path.write_bytes(b"test fixture")
    output_path = tmp_path / "results"

    result = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--save-frames",
            "--frame-interval",
            "1",
        ]
    )

    captured = capsys.readouterr().out
    assert result == 1
    assert "could not encode annotated frame" in captured
    assert "Processing complete!" not in captured
    assert not (output_path / "session").exists()


@pytest.mark.parametrize(
    ("total_frames", "max_frames", "expected"),
    [
        (100, None, 100),
        (100, 20, 20),
        (10, 20, 10),
        (0, None, None),
        (0, 20, None),
    ],
)
def test_expected_decoded_frame_count(
    total_frames: int,
    max_frames: int | None,
    expected: int | None,
) -> None:
    assert expected_decoded_frame_count(total_frames, max_frames) == expected


def test_valid_zero_detection_run_writes_all_requested_empty_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import pandas as pd

    import naturallab.media
    import naturallab.spatial_tracking.detection.yolo_detector as yolo_module
    import scripts.track_people_in_video as tracking_script

    class EmptyDetector:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def process(self, data):
            return {
                **data,
                "detections": [],
                "detection_metadata": {"skipped": False},
                "detection_provenance": {"backend": "test-empty"},
            }

    class TwoFrameSource:
        def __init__(self, path, *, stop_frame=None) -> None:
            self.path = path
            self.stop_frame = stop_frame

        def __iter__(self):
            frame_count = min(2, self.stop_frame or 2)
            for frame_index in range(frame_count):
                yield SimpleNamespace(
                    frame_index=frame_index,
                    image=np.zeros((16, 16, 3), dtype=np.uint8),
                    metadata={"timestamp_source": "test"},
                    source_timestamp=frame_index / 25,
                    timestamp_ns=frame_index * 40_000_000,
                )

    monkeypatch.setattr(yolo_module, "YOLODetectorModule", EmptyDetector)
    monkeypatch.setattr(naturallab.media, "VideoFileSource", TwoFrameSource)
    monkeypatch.setattr(
        tracking_script,
        "probe_video",
        lambda path: (2, 25.0),
    )
    input_path = tmp_path / "session.mp4"
    input_path.write_bytes(b"test fixture")
    output_path = tmp_path / "results"

    result = main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--yolo-model",
            str(tmp_path / "private" / "local-model.pt"),
            "--identities",
            '{"participant":"the study participant"}',
        ]
    )

    video_output = output_path / "session"
    assert result == 0
    assert pd.read_csv(video_output / "tracks.csv").empty
    assert pd.read_csv(video_output / "track_statistics.csv").empty
    identity_output = json.loads(
        (video_output / "identity_matches.json").read_text(encoding="utf-8")
    )
    assert identity_output["assignments"] == {}
    run_metadata = json.loads(
        (video_output / "run_metadata.json").read_text(encoding="utf-8")
    )
    assert run_metadata["input_video"] == "session.mp4"
    assert run_metadata["detector_settings"]["model_path"] == (
        "local-model.pt"
    )
    assert run_metadata["metric_tracking"] is None
    assert str(tmp_path) not in repr(run_metadata)
    assert not (video_output / "frames").exists()


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
