from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    ("status", "expected_name", "obsolete_name"),
    [
        ("pass", "room-registration.yaml", "candidate-room-registration.yaml"),
        ("fail", "candidate-room-registration.yaml", "room-registration.yaml"),
    ],
)
def test_extrinsics_overwrite_publishes_only_the_current_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    expected_name: str,
    obsolete_name: str,
) -> None:
    output = tmp_path / "shared-room"
    stale_frames = output / "annotated-placements" / "removed-camera"
    stale_frames.mkdir(parents=True)
    (stale_frames / "frame_00000001.jpg").write_bytes(b"old")
    for name in (
        "room-registration.yaml",
        "candidate-room-registration.yaml",
        "extrinsics-report.json",
        "shared-observations.csv",
    ):
        (output / name).write_text("old", encoding="utf-8")

    registration = SimpleNamespace(
        to_dict=lambda: {"schema": "naturallab.room-registration/v1"}
    )
    monkeypatch.setattr(
        commands,
        "calibrate_extrinsics_from_manifest",
        lambda _manifest: SimpleNamespace(
            report={"status": status},
            room_registration=registration,
            observations=({"placement_id": 1},),
        ),
    )
    args = Namespace(
        calibration_command="extrinsics",
        manifest="shared-board.yaml",
        output_dir=str(output),
        save_frames=False,
        overwrite=True,
        json_output=True,
    )

    report = commands._run_extrinsics(args)

    assert report["status"] == status
    assert (output / expected_name).is_file()
    assert not (output / obsolete_name).exists()
    assert not (output / "annotated-placements").exists()


def test_extrinsics_overwrite_leaves_prior_outputs_untouched_if_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "shared-room"
    output.mkdir()
    prior = {
        "room-registration.yaml": "old registration",
        "extrinsics-report.json": "old report",
        "shared-observations.csv": "old observations",
    }
    for name, content in prior.items():
        (output / name).write_text(content, encoding="utf-8")

    registration = SimpleNamespace(
        to_dict=lambda: {"schema": "naturallab.room-registration/v1"}
    )
    monkeypatch.setattr(
        commands,
        "calibrate_extrinsics_from_manifest",
        lambda _manifest: SimpleNamespace(
            report={"status": "fail"},
            room_registration=registration,
            observations=({"placement_id": 1},),
        ),
    )

    def fail_csv(*_args, **_kwargs):
        raise OSError("simulated full disk")

    monkeypatch.setattr(commands, "write_measurements_csv", fail_csv)
    args = Namespace(
        calibration_command="extrinsics",
        manifest="shared-board.yaml",
        output_dir=str(output),
        save_frames=False,
        overwrite=True,
        json_output=True,
    )

    with pytest.raises(OSError, match="simulated full disk"):
        commands._run_extrinsics(args)

    for name, content in prior.items():
        assert (output / name).read_text(encoding="utf-8") == content
    assert not (output / "candidate-room-registration.yaml").exists()
    assert not list(tmp_path.glob(".naturallab-extrinsics-*"))


def test_extrinsics_report_uses_relative_annotated_frame_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "shared-room"
    view = SimpleNamespace(
        view_id="camera-01",
        video_path=tmp_path / "private" / "shared-board.mp4",
    )
    run = SimpleNamespace(
        report={"status": "pass"},
        room_registration=SimpleNamespace(
            to_dict=lambda: {"schema": "naturallab.room-registration/v1"}
        ),
        observations=({"placement_id": 1},),
        manifest=SimpleNamespace(views=(view,), board=object()),
        bundles_by_view={
            "camera-01": SimpleNamespace(input_rotation="none")
        },
        selected_detections_by_view={"camera-01": (object(),)},
    )
    monkeypatch.setattr(
        commands,
        "calibrate_extrinsics_from_manifest",
        lambda _manifest: run,
    )

    def save_frames(_video, **kwargs):
        frame = Path(kwargs["output_directory"]) / "frame_00000001.jpg"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
        return (frame,)

    monkeypatch.setattr(commands, "save_annotated_detections", save_frames)
    args = Namespace(
        calibration_command="extrinsics",
        manifest=str(tmp_path / "private" / "shared-board.yaml"),
        output_dir=str(output),
        save_frames=True,
        overwrite=False,
        json_output=True,
    )

    report = commands._run_extrinsics(args)

    assert report["outputs"]["annotated_frames"] == {
        "camera-01": ["annotated-placements/camera-01/frame_00000001.jpg"]
    }
    assert str(tmp_path) not in (output / "extrinsics-report.json").read_text(
        encoding="utf-8"
    )


def test_calibration_output_commit_restores_prior_set_on_install_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    staging = tmp_path / "staging"
    output.mkdir()
    staging.mkdir()
    (output / "first.txt").write_text("old first", encoding="utf-8")
    (output / "second.txt").write_text("old second", encoding="utf-8")
    (staging / "first.txt").write_text("new first", encoding="utf-8")
    (staging / "second.txt").write_text("new second", encoding="utf-8")
    real_replace = commands.os.replace

    def fail_second_install(source, destination):
        if Path(source) == staging / "second.txt":
            raise OSError("simulated install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(commands.os, "replace", fail_second_install)

    with pytest.raises(OSError, match="simulated install failure"):
        commands._commit_staged_outputs(
            output=output,
            staging=staging,
            managed=(Path("first.txt"), Path("second.txt")),
            produced=(Path("first.txt"), Path("second.txt")),
        )

    assert (output / "first.txt").read_text(encoding="utf-8") == "old first"
    assert (output / "second.txt").read_text(encoding="utf-8") == "old second"
    assert not list(tmp_path.glob(".naturallab-calibration-backup-*"))
