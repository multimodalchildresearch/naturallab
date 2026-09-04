from __future__ import annotations

from pathlib import Path
from argparse import Namespace

import pytest

from naturallab.spatial_tracking.calibration import commands
from naturallab.spatial_tracking.calibration.commands import (
    build_parser,
    run_calibration_command,
)


def test_calibration_cli_defines_inner_corner_semantics() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "intrinsic",
            "--video",
            "board.mp4",
            "--camera-id",
            "camera-1",
            "--square-size-mm",
            "30",
            "--output-dir",
            "calibration/camera-1",
        ]
    )

    assert args.inner_cols == 7
    assert args.inner_rows == 7
    assert args.input_rotation == "none"
    assert args.minimum_views == 16
    assert args.minimum_tilted_views == 4


def test_floor_rotation_is_inherited_not_independently_configurable() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "floor",
                "--video",
                "floor.mp4",
                "--intrinsics",
                "intrinsics.yaml",
                "--square-size-mm",
                "30",
                "--output-dir",
                "calibration/camera-1",
                "--input-rotation",
                "90_cw",
            ]
        )


def test_old_manual_and_hidden_scale_paths_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts/calibrate_camera_system.py",
        root / "naturallab/spatial_tracking/calibration/camera_calib.py",
        root / "naturallab/spatial_tracking/calibration/floor_calib.py",
        root / "naturallab/spatial_tracking/calibration/measure_distance.py",
    )

    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "cv2.waitKey" not in combined
    assert "correction_factor = 1.1" not in combined
    assert "Press SPACE" not in combined


def test_verification_fail_has_a_distinct_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commands,
        "_run_verify",
        lambda args: {"status": "fail"},
    )

    assert (
        run_calibration_command(Namespace(calibration_command="verify"))
        == 3
    )


def test_verification_warning_is_a_completed_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commands,
        "_run_verify",
        lambda args: {"status": "warning"},
    )

    assert (
        run_calibration_command(Namespace(calibration_command="verify"))
        == 0
    )


def test_verification_accepts_one_bundle_path() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "verify",
            "--video",
            "verification.mp4",
            "--bundle",
            "calibration-bundle.yaml",
            "--square-size-mm",
            "30",
            "--output-dir",
            "verification",
        ]
    )

    assert args.bundle == "calibration-bundle.yaml"
    assert args.intrinsics is None
    assert args.floor is None
    assert args.minimum_placements == 3
    assert args.maximum_placements == 20
    assert args.minimum_center_span_fraction == 0.10


def test_extrinsics_is_manifest_driven() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "extrinsics",
            "--manifest",
            "shared-board.yaml",
            "--output-dir",
            "shared-room",
            "--save-frames",
        ]
    )

    assert args.manifest == "shared-board.yaml"
    assert args.output_dir == "shared-room"
    assert args.save_frames is True
    assert not hasattr(args, "input_rotation")


def test_extrinsics_fail_has_a_distinct_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        commands,
        "_run_extrinsics",
        lambda args: {"status": "fail"},
    )

    assert (
        run_calibration_command(Namespace(calibration_command="extrinsics"))
        == 3
    )
