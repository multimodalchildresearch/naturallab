"""Researcher-facing command line for automatic camera calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Optional, Sequence

from naturallab.provenance import runtime_provenance

from .artifacts import CalibrationBundle, InputRotation
from .automatic import (
    AutomaticCalibrationError,
    BoardSpec,
    calibrate_floor_from_video,
    calibrate_intrinsics_from_video,
    load_calibration_bundle,
    load_calibration_bundle_file,
    load_intrinsic_artifact,
    save_annotated_detections,
    source_identity,
    verify_floor_from_video,
    write_json_report,
    write_measurements_csv,
    write_yaml_artifact,
    write_yaml_document,
)
from .extrinsics import calibrate_extrinsics_from_manifest


def _add_board_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_square_size: bool = True,
) -> None:
    parser.add_argument(
        "--inner-cols",
        "--pattern-cols",
        dest="inner_cols",
        type=int,
        default=7,
        help=(
            "Number of internal chessboard corners across a row (default: 7). "
            "A board with 8 squares across has 7 internal corners."
        ),
    )
    parser.add_argument(
        "--inner-rows",
        "--pattern-rows",
        dest="inner_rows",
        type=int,
        default=7,
        help=(
            "Number of internal chessboard corners down a column (default: 7). "
            "A board with 8 squares down has 7 internal corners."
        ),
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        required=require_square_size,
        help="Measured side length of one chessboard square in millimetres.",
    )


def _add_sampling_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_placement_limit: bool,
    maximum_placements_default: int = 12,
) -> None:
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Approximate interval between screened frames (default: 1.0).",
    )
    parser.add_argument(
        "--stationary-motion-px",
        type=float,
        default=20.0,
        help=(
            "Maximum median corner motion from the start of one stationary "
            "run (default: 20 pixels)."
        ),
    )
    parser.add_argument(
        "--minimum-stationary-samples",
        type=int,
        default=3,
        help=(
            "Minimum detected samples while the board remains still "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--minimum-separation-px",
        type=float,
        default=80.0,
        help=(
            "Minimum image-plane separation between retained placements "
            "(default: 80 pixels)."
        ),
    )
    if include_placement_limit:
        parser.add_argument(
            "--maximum-placements",
            type=int,
            default=maximum_placements_default,
            help=(
                "Maximum distinct placements to retain "
                f"(default: {maximum_placements_default})."
            ),
        )


def _add_common_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for canonical artifacts and diagnostic reports.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated selected frames for visual quality control.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace known output files from an earlier run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Print the result summary as JSON.",
    )


def add_calibration_commands(
    parser: argparse.ArgumentParser,
) -> None:
    """Configure automatic intrinsic, floor, extrinsic, and QC commands."""

    subparsers = parser.add_subparsers(dest="calibration_command")

    intrinsic = subparsers.add_parser(
        "intrinsic",
        help="Automatically calibrate camera intrinsics from one video.",
        description=(
            "Detect a chessboard automatically, select diverse sharp views, "
            "reject high-error views, and write a canonical intrinsic artifact."
        ),
    )
    intrinsic.add_argument("--video", required=True)
    intrinsic.add_argument("--camera-id", required=True)
    intrinsic.add_argument(
        "--input-rotation",
        choices=tuple(rotation.value for rotation in InputRotation),
        default=InputRotation.NONE.value,
        help=(
            "Rotation applied after decoding and later reused by tracking "
            "(default: none). Do not pre-rotate and also set this flag."
        ),
    )
    intrinsic.add_argument(
        "--coordinate-frame",
        help=(
            "Camera coordinate-frame name. Defaults to "
            "'camera/<camera-id>/opencv'."
        ),
    )
    _add_board_arguments(intrinsic)
    intrinsic.add_argument(
        "--sample-seconds",
        type=float,
        default=1.0,
        help="Approximate interval between screened frames (default: 1.0).",
    )
    intrinsic.add_argument(
        "--target-views",
        type=int,
        default=28,
        help="Maximum diverse candidate views selected initially (default: 28).",
    )
    intrinsic.add_argument(
        "--minimum-views",
        type=int,
        default=16,
        help="Minimum accepted calibration views (default: 16).",
    )
    intrinsic.add_argument(
        "--maximum-view-rms-pixels",
        type=float,
        default=3.0,
        help=(
            "Reject a selected view above this Euclidean reprojection RMS; "
            "fail if it remains above the limit at minimum views (default: 3)."
        ),
    )
    intrinsic.add_argument(
        "--minimum-center-span-fraction",
        type=float,
        default=0.20,
        help=(
            "Require selected board centres to span this fraction of both "
            "image axes (default: 0.20)."
        ),
    )
    intrinsic.add_argument(
        "--minimum-scale-ratio",
        type=float,
        default=1.20,
        help=(
            "Require this near/far ratio in detected board linear scale "
            "(default: 1.20)."
        ),
    )
    intrinsic.add_argument(
        "--minimum-perspective-change",
        type=float,
        default=0.02,
        help=(
            "Require at least this edge-scale change from out-of-plane tilt "
            "around each board axis (default: 0.02)."
        ),
    )
    intrinsic.add_argument(
        "--minimum-tilted-views",
        type=int,
        default=4,
        help=(
            "Minimum selected views with measurable out-of-plane perspective "
            "(default: 4)."
        ),
    )
    _add_common_output_arguments(intrinsic)

    floor = subparsers.add_parser(
        "floor",
        help="Automatically fit a metric floor plane from one video.",
        description=(
            "Identify stationary board placements automatically, recover full "
            "PnP poses, fit the floor plane, and write a hash-bound artifact."
        ),
    )
    floor.add_argument(
        "--intrinsics",
        "--camera-calib",
        dest="intrinsics",
        required=True,
        help="Canonical intrinsic YAML produced by the intrinsic step.",
    )
    floor.add_argument("--video", required=True)
    _add_board_arguments(floor)
    _add_sampling_arguments(
        floor,
        include_placement_limit=True,
        maximum_placements_default=12,
    )
    floor.add_argument(
        "--minimum-placements",
        type=int,
        default=3,
        help=(
            "Minimum spatially distinct stationary placements (default: 3; "
            "recording at least 5 is recommended)."
        ),
    )
    floor.add_argument(
        "--maximum-normal-deviation-degrees",
        type=float,
        default=5.0,
        help=(
            "Fail when individual board normals disagree by more than this "
            "amount (default: 5 degrees)."
        ),
    )
    floor.add_argument(
        "--maximum-centroid-offset-mm",
        type=float,
        default=50.0,
        help=(
            "Fail when one board placement sits this far from the common plane "
            "(default: 50 mm)."
        ),
    )
    _add_common_output_arguments(floor)

    extrinsics = subparsers.add_parser(
        "extrinsics",
        help="Recover fixed multi-camera geometry from shared-board footage.",
        description=(
            "Find stationary chessboard placements visible in every declared "
            "view, resolve symmetric corner ordering, recover fixed-intrinsic "
            "stereo transforms, and write a validated room registration."
        ),
    )
    extrinsics.add_argument(
        "--manifest",
        required=True,
        help=(
            "Shared-board YAML declaring each video, calibration bundle, "
            "board, timing offsets, and quality limits."
        ),
    )
    _add_common_output_arguments(extrinsics)

    verify = subparsers.add_parser(
        "verify",
        help="Automatically verify a fixed calibration on a separate video.",
        description=(
            "Detect stationary chessboards in an independent recording, project "
            "their known boundary spans through the fixed floor plane, and "
            "report measured distances without refitting or manual clicks."
        ),
    )
    verify.add_argument(
        "--intrinsics",
        "--camera-calib",
        dest="intrinsics",
        help="Canonical intrinsic YAML produced by the intrinsic step.",
    )
    verify.add_argument(
        "--floor",
        "--floor-calib",
        dest="floor",
        help="Canonical floor YAML produced by the floor step.",
    )
    verify.add_argument(
        "--bundle",
        help=(
            "Canonical calibration-bundle YAML produced by the floor step. "
            "Use this instead of --intrinsics and --floor."
        ),
    )
    verify.add_argument("--video", required=True)
    _add_board_arguments(verify)
    _add_sampling_arguments(
        verify,
        include_placement_limit=True,
        maximum_placements_default=20,
    )
    verify.add_argument(
        "--minimum-placements",
        type=int,
        default=3,
        help=(
            "Minimum spatially distinct stationary verification placements "
            "(default: 3)."
        ),
    )
    verify.add_argument(
        "--minimum-center-span-fraction",
        type=float,
        default=0.10,
        help=(
            "Require verification placement centres to span this fraction of "
            "both image axes (default: 0.10)."
        ),
    )
    verify.add_argument(
        "--pass-threshold-percent",
        type=float,
        default=3.0,
        help=(
            "Operational pass threshold applied to mean, P90, and worst-"
            "placement mean absolute distance error (default: 3%%)."
        ),
    )
    verify.add_argument(
        "--warning-threshold-percent",
        type=float,
        default=5.0,
        help=(
            "Operational warning/fail boundary applied to the same three "
            "distance-error metrics (default: 5%%)."
        ),
    )
    _add_common_output_arguments(verify)


def _output_directory(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).expanduser().resolve()


def _board(args: argparse.Namespace) -> BoardSpec:
    return BoardSpec(
        internal_columns=args.inner_cols,
        internal_rows=args.inner_rows,
        square_size_mm=args.square_size_mm,
    )


def _ensure_available(
    paths: Sequence[Path],
    *,
    overwrite: bool,
) -> None:
    if overwrite:
        return
    existing = [
        path for path in paths if path.exists() or path.is_symlink()
    ]
    if existing:
        rendered = ", ".join(str(path) for path in existing)
        raise AutomaticCalibrationError(
            f"output already exists: {rendered}; pass --overwrite to replace it"
        )


def _remove_managed_path(path: Path) -> None:
    """Remove one known output without following a directory symlink."""

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _create_staging_directory(output: Path, operation: str) -> Path:
    """Create a same-filesystem staging directory for one command run."""

    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".naturallab-{operation}-",
            dir=str(output.parent),
        )
    )


def _commit_staged_outputs(
    *,
    output: Path,
    staging: Path,
    managed: Sequence[Path],
    produced: Sequence[Path],
) -> None:
    """Install a complete output set and restore the old set on failure.

    ``managed`` includes outputs that this run intentionally omits.  Moving
    every prior managed path to a same-filesystem backup before installation
    both removes stale diagnostics and enforces mutual exclusion between the
    operational and candidate room-registration artifacts.
    """

    managed_paths = tuple(Path(path) for path in managed)
    produced_paths = tuple(Path(path) for path in produced)
    for relative in (*managed_paths, *produced_paths):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("managed calibration outputs must be relative")
    if not set(produced_paths).issubset(set(managed_paths)):
        raise ValueError("produced calibration outputs must be managed")
    missing = [
        relative
        for relative in produced_paths
        if not (staging / relative).exists()
        and not (staging / relative).is_symlink()
    ]
    if missing:
        raise AutomaticCalibrationError(
            "staged calibration output is missing: "
            + ", ".join(str(path) for path in missing)
        )

    output.mkdir(parents=True, exist_ok=True)
    backup = Path(
        tempfile.mkdtemp(
            prefix=".naturallab-calibration-backup-",
            dir=str(output.parent),
        )
    )
    moved_old: list[tuple[Path, Path]] = []
    installed: list[Path] = []
    try:
        for relative in managed_paths:
            destination = output / relative
            if not destination.exists() and not destination.is_symlink():
                continue
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, saved)
            moved_old.append((destination, saved))

        for relative in produced_paths:
            source = staging / relative
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            installed.append(destination)
    except Exception as commit_error:
        try:
            for destination in reversed(installed):
                _remove_managed_path(destination)
            for destination, saved in reversed(moved_old):
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(saved, destination)
        except Exception as rollback_error:
            raise AutomaticCalibrationError(
                "could not commit calibration outputs or restore the prior "
                f"outputs; recovery backup retained at {backup}: "
                f"{rollback_error}"
            ) from commit_error
        shutil.rmtree(backup)
        raise
    else:
        shutil.rmtree(backup)


def _attach_runtime_provenance(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """Attach reproducibility metadata without retaining secret parameters."""

    report["runtime_provenance"] = runtime_provenance(
        operation=f"calibrate.{args.calibration_command}",
        parameters=vars(args),
    )


def _print_intrinsic(
    report: dict[str, Any],
    output_directory: Path,
) -> None:
    holdout = report["holdout_corner_error_pixels"]
    print("Automatic intrinsic calibration complete")
    print(f"Camera: {report['camera_id']}")
    print(
        "Views: "
        f"{report['detected_view_count']} detected, "
        f"{report['accepted_view_count']} accepted"
    )
    print(f"OpenCV RMS: {report['opencv_rms_pixels']:.3f} px")
    geometry = report["view_geometry"]
    print(
        "Centre coverage x/y: "
        f"{geometry['center_span_fraction_x']:.3f} / "
        f"{geometry['center_span_fraction_y']:.3f}"
    )
    print(
        "Scale ratio / tilted views: "
        f"{geometry['board_linear_scale_ratio']:.2f}x / "
        f"{geometry['tilted_view_count']}"
    )
    if holdout["mean"] is not None:
        print(f"Internal holdout mean: {holdout['mean']:.3f} px")
    print(f"Artifact SHA-256: {report['artifact_sha256']}")
    print(f"Outputs: {output_directory}")


def _print_floor(
    report: dict[str, Any],
    output_directory: Path,
) -> None:
    internal = report["internal_leave_one_placement_out"]
    print("Automatic floor-plane calibration complete")
    print(f"Camera: {report['camera_id']}")
    print(f"Placements: {report['selected_placement_count']}")
    if report["rejected_placements"]:
        print(
            "Rejected invalid poses: "
            f"{len(report['rejected_placements'])}"
        )
    print(
        "Plane [a, b, c, d]: "
        + ", ".join(f"{value:.8g}" for value in report["floor_plane"])
    )
    print(
        "Plane point RMS: "
        f"{report['plane_fit_residuals']['rms_mm']:.2f} mm"
    )
    if internal is not None:
        print(
            "Internal LOPO mean absolute error: "
            f"{internal['mean_absolute_error_mm']:.2f} mm "
            f"({internal['mean_absolute_error_percent']:.2f}%)"
        )
    print(f"Artifact SHA-256: {report['artifact_sha256']}")
    print("Next: run verify on a separately recorded board video.")
    print(f"Outputs: {output_directory}")


def _print_verification(
    report: dict[str, Any],
    output_directory: Path,
) -> None:
    measurements = report["measurements"]
    print(f"Calibration verification: {report['status'].upper()}")
    print(f"Camera: {report['camera_id']}")
    print(f"Placements: {report['selected_placement_count']}")
    if report["rejected_placements"]:
        print(
            "Rejected invalid poses: "
            f"{len(report['rejected_placements'])}"
        )
    print(
        "Mean known/measured span: "
        f"{measurements['mean_known_distance_mm']:.1f} / "
        f"{measurements['mean_measured_distance_mm']:.1f} mm"
    )
    print(
        "Mean absolute error: "
        f"{measurements['mean_absolute_error_mm']:.1f} mm "
        f"({measurements['mean_absolute_error_percent']:.2f}%)"
    )
    print(
        "P90 / maximum error: "
        f"{measurements['p90_absolute_error_percent']:.2f}% / "
        f"{measurements['maximum_absolute_error_percent']:.2f}%"
    )
    worst_placement = report["decision_errors_percent"][
        "worst_placement_mean_absolute_error_percent"
    ]
    print(
        "Worst placement mean: "
        f"{worst_placement:.2f}%"
    )
    print(f"Outputs: {output_directory}")


def _print_extrinsics(
    report: dict[str, Any],
    output_directory: Path,
) -> None:
    print(f"Shared-room calibration: {report['status'].upper()}")
    print(f"Rig: {report['rig_id']}")
    print(f"Anchor view: {report['anchor_view_id']}")
    print(f"Shared placements: {report['shared_placement_count']}")
    for view_id, pair in report["pair_recoveries"].items():
        holdout = pair[
            "leave_one_placement_out_corner_transfer_pixels"
        ]
        print(
            f"{view_id}: stereo RMS "
            f"{pair['fixed_intrinsic_stereo_rms_pixels']:.3f} px, "
            f"held-out P90 {holdout['p90']:.3f} px"
        )
    print(
        "Validation scope: planar floor only; volumetric accuracy is not "
        "yet certified."
    )
    print(f"Outputs: {output_directory}")


def _run_intrinsic(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_directory(args)
    artifact_path = output / "intrinsics.yaml"
    report_path = output / "intrinsic-report.json"
    selected_path = output / "selected-views.csv"
    frame_directory = output / "intrinsic-selected-views"
    managed = (
        Path("intrinsics.yaml"),
        Path("intrinsic-report.json"),
        Path("selected-views.csv"),
        Path("intrinsic-selected-views"),
    )
    targets = [output / relative for relative in managed]
    _ensure_available(targets, overwrite=args.overwrite)

    run = calibrate_intrinsics_from_video(
        args.video,
        camera_id=args.camera_id,
        board=_board(args),
        input_rotation=InputRotation(args.input_rotation),
        coordinate_frame=args.coordinate_frame,
        sample_seconds=args.sample_seconds,
        target_views=args.target_views,
        minimum_views=args.minimum_views,
        maximum_view_rms_pixels=args.maximum_view_rms_pixels,
        minimum_center_span_fraction=args.minimum_center_span_fraction,
        minimum_scale_ratio=args.minimum_scale_ratio,
        minimum_perspective_change=args.minimum_perspective_change,
        minimum_tilted_views=args.minimum_tilted_views,
    )
    report = dict(run.report)
    report["opencv_version"] = __import__("cv2").__version__
    report["outputs"] = {
        "intrinsics": artifact_path.name,
        "report": report_path.name,
        "selected_views": selected_path.name,
    }
    if args.save_frames:
        report["outputs"]["annotated_frames"] = frame_directory.name
    _attach_runtime_provenance(report, args)
    produced = list(managed[:3])
    staging = _create_staging_directory(output, "intrinsic")
    try:
        write_yaml_artifact(
            staging / "intrinsics.yaml",
            run.artifact,
        )
        write_json_report(staging / "intrinsic-report.json", report)
        write_measurements_csv(
            staging / "selected-views.csv",
            report["selected_views"],
        )
        if args.save_frames:
            save_annotated_detections(
                args.video,
                detections=run.selected_detections,
                board=_board(args),
                input_rotation=run.artifact.input_rotation,
                output_directory=(
                    staging / "intrinsic-selected-views"
                ),
            )
            produced.append(Path("intrinsic-selected-views"))
        _commit_staged_outputs(
            output=output,
            staging=staging,
            managed=managed,
            produced=produced,
        )
    finally:
        _remove_managed_path(staging)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_intrinsic(report, output)
    return report


def _run_floor(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_directory(args)
    artifact_path = output / "floor.yaml"
    bundle_path = output / "calibration-bundle.yaml"
    report_path = output / "floor-report.json"
    measurements_path = output / "floor-internal-measurements.csv"
    frame_directory = output / "floor-selected-placements"
    managed = (
        Path("floor.yaml"),
        Path("calibration-bundle.yaml"),
        Path("floor-report.json"),
        Path("floor-internal-measurements.csv"),
        Path("floor-selected-placements"),
    )
    targets = [output / relative for relative in managed]
    _ensure_available(targets, overwrite=args.overwrite)

    intrinsics = load_intrinsic_artifact(args.intrinsics)
    run = calibrate_floor_from_video(
        args.video,
        intrinsics=intrinsics,
        board=_board(args),
        sample_seconds=args.sample_seconds,
        stationary_distance_pixels=args.stationary_motion_px,
        minimum_stationary_samples=args.minimum_stationary_samples,
        minimum_separation_pixels=args.minimum_separation_px,
        minimum_placements=args.minimum_placements,
        maximum_placements=args.maximum_placements,
        maximum_normal_deviation_degrees=(
            args.maximum_normal_deviation_degrees
        ),
        maximum_centroid_offset_mm=args.maximum_centroid_offset_mm,
    )
    bundle = CalibrationBundle(
        intrinsics=intrinsics,
        floor_plane=run.artifact,
    )
    report = dict(run.report)
    report["opencv_version"] = __import__("cv2").__version__
    report["intrinsics_file"] = source_identity(args.intrinsics)
    report["outputs"] = {
        "floor": artifact_path.name,
        "bundle": bundle_path.name,
        "report": report_path.name,
        "internal_measurements": measurements_path.name,
    }
    if args.save_frames:
        report["outputs"]["annotated_frames"] = frame_directory.name
    _attach_runtime_provenance(report, args)
    produced = list(managed[:4])
    staging = _create_staging_directory(output, "floor")
    try:
        write_yaml_artifact(
            staging / "floor.yaml",
            run.artifact,
        )
        write_yaml_document(
            staging / "calibration-bundle.yaml",
            bundle.to_dict(),
        )
        write_json_report(staging / "floor-report.json", report)
        write_measurements_csv(
            staging / "floor-internal-measurements.csv",
            run.internal_measurements,
        )
        if args.save_frames:
            save_annotated_detections(
                args.video,
                detections=tuple(
                    pose.detection for pose in run.selected_poses
                ),
                board=_board(args),
                input_rotation=intrinsics.input_rotation,
                output_directory=(
                    staging / "floor-selected-placements"
                ),
            )
            produced.append(Path("floor-selected-placements"))
        _commit_staged_outputs(
            output=output,
            staging=staging,
            managed=managed,
            produced=produced,
        )
    finally:
        _remove_managed_path(staging)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_floor(report, output)
    return report


def _run_extrinsics(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_directory(args)
    registration_path = output / "room-registration.yaml"
    candidate_registration_path = (
        output / "candidate-room-registration.yaml"
    )
    report_path = output / "extrinsics-report.json"
    observations_path = output / "shared-observations.csv"
    managed = (
        Path("room-registration.yaml"),
        Path("candidate-room-registration.yaml"),
        Path("extrinsics-report.json"),
        Path("shared-observations.csv"),
        Path("annotated-placements"),
    )
    targets = [output / relative for relative in managed]
    _ensure_available(targets, overwrite=args.overwrite)

    run = calibrate_extrinsics_from_manifest(args.manifest)
    report = dict(run.report)
    if report["status"] == "fail":
        registration_path = candidate_registration_path
    report["opencv_version"] = __import__("cv2").__version__
    report["outputs"] = {
        "room_registration": registration_path.name,
        "report": report_path.name,
        "shared_observations": observations_path.name,
    }
    _attach_runtime_provenance(report, args)
    registration_name = Path(registration_path.name)
    produced = [
        registration_name,
        Path("extrinsics-report.json"),
        Path("shared-observations.csv"),
    ]
    staging = _create_staging_directory(output, "extrinsics")
    try:
        write_yaml_document(
            staging / registration_name,
            run.room_registration.to_dict(),
        )
        write_measurements_csv(
            staging / "shared-observations.csv",
            run.observations,
        )
        if args.save_frames:
            frame_outputs = {}
            views = {
                view.view_id: view for view in run.manifest.views
            }
            for view_id, detections in (
                run.selected_detections_by_view.items()
            ):
                staged_view_directory = (
                    staging / "annotated-placements" / view_id
                )
                written = save_annotated_detections(
                    views[view_id].video_path,
                    detections=detections,
                    board=run.manifest.board,
                    input_rotation=run.bundles_by_view[
                        view_id
                    ].input_rotation,
                    output_directory=staged_view_directory,
                )
                final_view_directory = Path("annotated-placements") / view_id
                frame_outputs[view_id] = [
                    str(final_view_directory / path.name)
                    for path in written
                ]
            report["outputs"]["annotated_frames"] = frame_outputs
            produced.append(Path("annotated-placements"))
        write_json_report(
            staging / "extrinsics-report.json",
            report,
        )
        _commit_staged_outputs(
            output=output,
            staging=staging,
            managed=managed,
            produced=produced,
        )
    finally:
        _remove_managed_path(staging)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_extrinsics(report, output)
    return report


def _run_verify(args: argparse.Namespace) -> dict[str, Any]:
    output = _output_directory(args)
    report_path = output / "verification.json"
    measurements_path = output / "measurements.csv"
    frame_directory = output / "annotated-placements"
    managed = (
        Path("verification.json"),
        Path("measurements.csv"),
        Path("annotated-placements"),
    )
    targets = [output / relative for relative in managed]
    _ensure_available(targets, overwrite=args.overwrite)

    if args.bundle:
        if args.intrinsics or args.floor:
            raise AutomaticCalibrationError(
                "--bundle cannot be combined with --intrinsics or --floor"
            )
        bundle = load_calibration_bundle_file(args.bundle)
        calibration_inputs = {
            "bundle_file": source_identity(args.bundle)
        }
    else:
        if not args.intrinsics or not args.floor:
            raise AutomaticCalibrationError(
                "provide either --bundle or both --intrinsics and --floor"
            )
        bundle = load_calibration_bundle(args.intrinsics, args.floor)
        calibration_inputs = {
            "intrinsics_file": source_identity(args.intrinsics),
            "floor_file": source_identity(args.floor),
        }
    run = verify_floor_from_video(
        args.video,
        bundle=bundle,
        board=_board(args),
        sample_seconds=args.sample_seconds,
        stationary_distance_pixels=args.stationary_motion_px,
        minimum_stationary_samples=args.minimum_stationary_samples,
        minimum_separation_pixels=args.minimum_separation_px,
        minimum_placements=args.minimum_placements,
        maximum_placements=args.maximum_placements,
        minimum_center_span_fraction=args.minimum_center_span_fraction,
        pass_threshold_percent=args.pass_threshold_percent,
        warning_threshold_percent=args.warning_threshold_percent,
    )
    report = dict(run.report)
    report["opencv_version"] = __import__("cv2").__version__
    report.update(calibration_inputs)
    report["outputs"] = {
        "report": report_path.name,
        "measurements": measurements_path.name,
    }
    if args.save_frames:
        report["outputs"]["annotated_frames"] = frame_directory.name
    _attach_runtime_provenance(report, args)
    produced = list(managed[:2])
    staging = _create_staging_directory(output, "verify")
    try:
        write_measurements_csv(
            staging / "measurements.csv",
            run.measurements,
        )
        if args.save_frames:
            save_annotated_detections(
                args.video,
                detections=run.selected_detections,
                board=_board(args),
                input_rotation=bundle.input_rotation,
                output_directory=staging / "annotated-placements",
                measurements=run.measurements,
            )
            produced.append(Path("annotated-placements"))
        write_json_report(staging / "verification.json", report)
        _commit_staged_outputs(
            output=output,
            staging=staging,
            managed=managed,
            produced=produced,
        )
    finally:
        _remove_managed_path(staging)
    if args.json_output:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_verification(report, output)
    return report


def run_calibration_command(args: argparse.Namespace) -> int:
    """Execute one parsed calibration command and return a process status."""

    if args.calibration_command is None:
        return 0
    try:
        if args.calibration_command == "intrinsic":
            _run_intrinsic(args)
        elif args.calibration_command == "floor":
            _run_floor(args)
        elif args.calibration_command == "extrinsics":
            report = _run_extrinsics(args)
            if report["status"] == "fail":
                return 3
        elif args.calibration_command == "verify":
            report = _run_verify(args)
            if report["status"] == "fail":
                return 3
        else:
            raise AutomaticCalibrationError(
                f"unknown calibration command: {args.calibration_command}"
            )
    except (AutomaticCalibrationError, ValueError, OSError) as exc:
        print(f"naturallab calibrate: error: {exc}", file=sys.stderr)
        return 2
    return 0


def build_parser(
    *,
    prog: str = "naturallab calibrate",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Click-free, schema-versioned camera, floor, and shared-room "
            "calibration."
        ),
    )
    add_calibration_commands(parser)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.calibration_command is None:
        parser.print_help()
        return 0
    return run_calibration_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
