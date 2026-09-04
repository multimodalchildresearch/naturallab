#!/usr/bin/env python3
"""
Track People in Video
=====================

A general-purpose script for tracking people in any video file.
Outputs movement trajectories, distances traveled, and interaction metrics.

Use Cases:
- Behavioral research (any population)
- Retail analytics (customer movement)
- Sports analysis (player tracking)
- Security/surveillance analysis
- Occupancy monitoring

Example Usage:
    # Basic tracking
    python track_people_in_video.py --input video.mp4 --output results/
    
    # With floor calibration for real-world distances
    python track_people_in_video.py --input video.mp4 --output results/ \\
        --camera-calib calibration/camera-01/intrinsic/intrinsics.yaml \\
        --floor-calib calibration/camera-01/floor/floor.yaml
    
    # With identity labels
    python track_people_in_video.py --input video.mp4 --output results/ \\
        --identities '{"Person A": "person wearing red", "Person B": "person in blue"}'
"""

import argparse
from dataclasses import replace
import json
import math
import re
import shutil
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

VIDEO_SUFFIXES = frozenset(
    {
        ".avi",
        ".m2ts",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp4",
        ".mpeg",
        ".mpg",
        ".mts",
        ".webm",
    }
)

TRACK_OUTPUT_COLUMNS = (
    "frame",
    "track_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "is_prediction",
    "timestamp_seconds",
    "timestamp_ns",
    "timestamp_source",
)
FLOOR_OUTPUT_COLUMNS = ("floor_x", "floor_y", "floor_z")
STATISTICS_OUTPUT_COLUMNS = (
    "track_id",
    "first_frame",
    "last_frame",
    "track_records",
    "observed_frames",
    "predicted_frames",
    "span_frames",
    "covered_frame_count",
    "duration_seconds",
    "timing_basis",
)


def discover_video_files(input_path):
    """Return one input file or naturally ordered supported directory videos."""
    from naturallab.media import natural_path_sort_key

    if input_path.is_file():
        return (
            [input_path]
            if input_path.suffix.lower() in VIDEO_SUFFIXES
            else []
        )
    return sorted(
        (
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        ),
        key=natural_path_sort_key,
    )


def build_video_output_plan(videos, output_path):
    """Map videos to unambiguous per-video output directories."""

    plan = []
    destinations = {}
    for video_path in videos:
        destination = output_path / video_path.stem
        destination_key = destination.name.casefold()
        conflicting_video = destinations.get(destination_key)
        if conflicting_video is not None:
            raise ValueError(
                "input videos map to the same output directory: "
                f"{conflicting_video.name} and {video_path.name} -> "
                f"{destination.name}"
            )
        destinations[destination_key] = video_path
        plan.append((video_path, destination))
    return plan


def validate_video_output_plan(plan, *, overwrite):
    """Fail before processing when a destination could retain old results."""

    for _, destination in plan:
        if destination.is_symlink():
            raise FileExistsError(
                f"refusing symlink output directory: {destination}"
            )
        if not destination.exists():
            continue
        if not destination.is_dir():
            raise FileExistsError(
                f"output destination is not a directory; refusing to replace: "
                f"{destination}"
            )
        is_nonempty_directory = (
            destination.is_dir() and next(destination.iterdir(), None) is not None
        )
        if is_nonempty_directory and not overwrite:
            raise FileExistsError(
                f"output already contains results: {destination}; "
                "pass --overwrite to replace that video's complete output"
            )


def prepare_video_output(destination, *, overwrite):
    """Create one clean output directory after preflight has succeeded."""

    if destination.is_symlink():
        raise FileExistsError(
            f"refusing symlink output directory: {destination}"
        )
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(
                f"output destination is not a directory; refusing to replace: "
                f"{destination}"
            )
        is_nonempty_directory = (
            destination.is_dir() and next(destination.iterdir(), None) is not None
        )
        if is_nonempty_directory:
            if not overwrite:
                raise FileExistsError(
                    f"output already contains results: {destination}; "
                    "pass --overwrite to replace it"
                )
            shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)


def probe_video(video_path):
    """Return nominal metadata only when at least one frame can be decoded."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return None
        reported_total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = (
            int(reported_total_frames)
            if math.isfinite(reported_total_frames)
            and reported_total_frames > 0
            else 0
        )
        if total_frames == 0:
            return None
        reported_fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = (
            reported_fps
            if math.isfinite(reported_fps) and reported_fps > 0
            else 0.0
        )
        decoded, frame = capture.read()
        if not decoded or frame is None or frame.size == 0:
            return None
        return total_frames, fps
    finally:
        capture.release()


def expected_decoded_frame_count(total_frames, max_frames):
    """Return the minimum decoded-frame count promised by container metadata."""

    if total_frames <= 0:
        return None
    if max_frames is None:
        return total_frames
    return min(total_frames, max_frames)


def write_track_tables(video_output, all_tracks, *, fps, floor_tracker):
    """Write stable CSV schemas, including for a valid zero-track run."""

    import pandas as pd

    track_columns = list(TRACK_OUTPUT_COLUMNS)
    if floor_tracker is not None:
        track_columns.extend(FLOOR_OUTPUT_COLUMNS)
    dataframe = pd.DataFrame(all_tracks)
    if dataframe.empty:
        dataframe = pd.DataFrame(columns=track_columns)
    else:
        ordered_columns = [
            column for column in track_columns if column in dataframe.columns
        ]
        ordered_columns.extend(
            column
            for column in dataframe.columns
            if column not in ordered_columns
        )
        dataframe = dataframe.loc[:, ordered_columns]
    dataframe.to_csv(video_output / "tracks.csv", index=False)

    statistics = []
    if not dataframe.empty:
        for track_id in dataframe["track_id"].unique():
            track_dataframe = dataframe[dataframe["track_id"] == track_id]
            statistic = summarize_track_records(
                track_id,
                track_dataframe,
                fps,
            )

            # Use the floor tracker's filtered accumulator so an explicit
            # legacy correction factor is reflected in the export.
            if (
                floor_tracker is not None
                and "floor_x" in track_dataframe.columns
                and track_dataframe["floor_x"].notna().any()
            ):
                distance = floor_tracker.get_distance(track_id)
                add_distance_statistics(
                    statistic,
                    distance,
                    floor_tracker.units,
                )
            statistics.append(statistic)

    statistics_dataframe = pd.DataFrame(statistics)
    if statistics_dataframe.empty:
        statistics_dataframe = pd.DataFrame(
            columns=STATISTICS_OUTPUT_COLUMNS
        )
    statistics_dataframe.to_csv(
        video_output / "track_statistics.csv",
        index=False,
    )
    return dataframe, statistics_dataframe


def load_floor_tracker(camera_calib_path, floor_calib_path, correction_factor=1.0):
    """Load a validated canonical pair or explicitly migrate legacy YAML."""
    import numpy as np
    import yaml

    from naturallab.spatial_tracking.calibration import (
        CalibrationBundle,
        FloorPlaneCalibrationArtifact,
        IntrinsicCalibrationArtifact,
    )
    from naturallab.spatial_tracking.movement.floor_tracker import SimpleFloorTracker

    if (
        isinstance(correction_factor, bool)
        or not isinstance(correction_factor, (int, float))
        or not math.isfinite(correction_factor)
        or correction_factor <= 0
    ):
        raise ValueError("correction_factor must be a finite positive number")

    with open(camera_calib_path, encoding="utf-8") as handle:
        camera_data = yaml.safe_load(handle)
    with open(floor_calib_path, encoding="utf-8") as handle:
        floor_data = yaml.safe_load(handle)

    if not isinstance(camera_data, dict) or not isinstance(floor_data, dict):
        raise ValueError("calibration files must each contain a YAML mapping")

    uses_artifact_contract = any(
        marker in camera_data or marker in floor_data
        for marker in ("schema_version", "kind", "intrinsic_sha256")
    )
    calibration_bundle = None
    if uses_artifact_contract:
        if correction_factor != 1.0:
            raise ValueError(
                "--correction-factor is a legacy option and cannot be applied "
                "to versioned calibration artifacts"
            )
        intrinsics = IntrinsicCalibrationArtifact.from_dict(camera_data)
        floor = FloorPlaneCalibrationArtifact.from_dict(
            floor_data,
            intrinsic=intrinsics,
        )
        calibration_bundle = CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=floor,
        )
        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeff
        floor_plane = floor.floor_plane
        units = floor.units
    else:
        dist_coeffs = camera_data.get(
            "dist_coeff",
            camera_data.get("dist_coeffs"),
        )
        if dist_coeffs is None:
            raise ValueError("camera calibration must contain dist_coeff")

        floor_plane = floor_data.get("floor_plane")
        if floor_plane is None:
            normal = floor_data.get("plane_normal")
            plane_d = floor_data.get("plane_d")
            if normal is not None and plane_d is not None:
                floor_plane = [*normal, plane_d]
        if floor_plane is None:
            raise ValueError("floor calibration must contain floor_plane")
        if "camera_matrix" not in camera_data:
            raise ValueError("camera calibration must contain camera_matrix")
        camera_matrix = camera_data["camera_matrix"]
        units = str(floor_data.get("units", "mm"))

    tracker = SimpleFloorTracker(
        camera_matrix=np.asarray(camera_matrix, dtype=float),
        dist_coeffs=np.asarray(dist_coeffs, dtype=float),
        floor_plane=np.asarray(floor_plane, dtype=float),
        correction_factor=correction_factor,
        units=units,
    )
    tracker.calibration_bundle = calibration_bundle
    return tracker


def prepare_frame_for_calibration(frame, floor_tracker):
    """Apply the rotation encoded by a canonical calibration and check size."""
    if floor_tracker is None or floor_tracker.calibration_bundle is None:
        return frame

    import cv2

    bundle = floor_tracker.calibration_bundle
    rotation = bundle.input_rotation.value
    if rotation == "90_cw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "180":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == "90_ccw":
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    actual_height, actual_width = frame.shape[:2]
    expected = bundle.image_size
    if (actual_width, actual_height) != (expected.width, expected.height):
        raise ValueError(
            "video frame geometry does not match calibration after input "
            f"rotation: got {actual_width}x{actual_height}, expected "
            f"{expected.width}x{expected.height}"
        )
    return frame


def extract_track_evidence(
    video_path,
    tracks,
    floor_tracker,
    frames_per_track=5,
):
    """Extract representative JPEG crops without loading a legacy CLIP model."""
    import cv2
    import numpy as np

    from naturallab.spatial_tracking.vlm import EvidenceImage

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"could not reopen video for role evidence: {video_path}")

    evidence_by_track = {}
    try:
        for track_id, track_rows in tracks.groupby("track_id", sort=False):
            track_rows = select_track_evidence_rows(track_rows)
            sample_count = min(frames_per_track, len(track_rows))
            row_indices = np.linspace(
                0,
                len(track_rows) - 1,
                sample_count,
                dtype=int,
            )
            evidence = []
            for row_index in sorted(set(row_indices.tolist())):
                row = track_rows.iloc[row_index]
                frame_index = int(row["frame"])
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = prepare_frame_for_calibration(frame, floor_tracker)
                height, width = frame.shape[:2]
                x1 = max(0, min(width, int(round(row["x1"]))))
                y1 = max(0, min(height, int(round(row["y1"]))))
                x2 = max(0, min(width, int(round(row["x2"]))))
                y2 = max(0, min(height, int(round(row["y2"]))))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                encoded_ok, encoded = cv2.imencode(
                    ".jpg",
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                if not encoded_ok:
                    continue
                evidence.append(
                    EvidenceImage(
                        encoded.tobytes(),
                        label=f"frame-{frame_index}",
                    )
                )
            evidence_by_track[str(track_id)] = evidence
    finally:
        capture.release()
    return evidence_by_track


def select_track_evidence_rows(track_rows):
    """Prefer detector-observed boxes, falling back to predictions if needed."""
    if "is_prediction" not in track_rows.columns:
        return track_rows
    observed_rows = track_rows[
        ~track_rows["is_prediction"].fillna(False).astype(bool)
    ]
    return observed_rows if not observed_rows.empty else track_rows


def serialize_role_assignment(assignment):
    """Return a JSON-safe role-assignment record."""
    return {
        "track_id": assignment.track_id,
        "role": assignment.role,
        "abstained": assignment.abstained,
        "confidence": assignment.confidence,
        "reason": assignment.reason,
        "provenance": assignment.provenance.as_dict(),
    }


def add_distance_statistics(stat, distance, units):
    """Add a unit-aware distance value and metres when conversion is known."""
    normalized_units = (
        units.strip().lower().replace("µ", "u").replace("μ", "u")
    )
    safe_units = re.sub(r"[^a-z0-9]+", "_", normalized_units).strip("_")
    safe_units = safe_units or "calibration_units"
    canonical_units = safe_units
    if normalized_units in {
        "mm",
        "millimeter",
        "millimeters",
        "millimetre",
        "millimetres",
    }:
        canonical_units = "mm"
    elif normalized_units in {
        "cm",
        "centimeter",
        "centimeters",
        "centimetre",
        "centimetres",
    }:
        canonical_units = "cm"
    elif normalized_units in {"m", "meter", "meters", "metre", "metres"}:
        canonical_units = "m"
    elif normalized_units in {
        "um",
        "micrometer",
        "micrometers",
        "micrometre",
        "micrometres",
    }:
        canonical_units = "um"

    stat["total_distance"] = float(distance)
    stat["distance_units"] = canonical_units
    stat[f"total_distance_{canonical_units}"] = float(distance)

    metres = None
    if normalized_units in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        metres = distance / 1000.0
    elif normalized_units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        metres = distance / 100.0
    elif normalized_units in {"m", "meter", "meters", "metre", "metres"}:
        metres = distance
    elif normalized_units in {
        "um",
        "micrometer",
        "micrometers",
        "micrometre",
        "micrometres",
    }:
        metres = distance / 1_000_000.0
    if metres is not None:
        stat["total_distance_m"] = float(metres)


def summarize_track_records(track_id, track_df, fps):
    """Summarize observed detections separately from temporal predictions."""
    if "is_prediction" in track_df.columns:
        prediction_flags = (
            track_df["is_prediction"].fillna(False).astype(bool)
        )
        predicted_frames = int(prediction_flags.sum())
    else:
        predicted_frames = 0
    track_records = len(track_df)
    first_frame = track_df["frame"].min()
    last_frame = track_df["frame"].max()
    span_frames = last_frame - first_frame
    covered_frame_count = span_frames + 1
    stat = {
        "track_id": track_id,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "track_records": track_records,
        "observed_frames": track_records - predicted_frames,
        "predicted_frames": predicted_frames,
        "span_frames": span_frames,
        "covered_frame_count": covered_frame_count,
        "duration_seconds": span_frames / fps if fps > 0 else None,
        "timing_basis": "nominal_fps" if fps > 0 else None,
    }
    if "timestamp_seconds" in track_df.columns:
        ordered_rows = track_df.sort_values("frame")
        endpoint_timestamps = ordered_rows["timestamp_seconds"].iloc[
            [0, -1]
        ]
        if endpoint_timestamps.notna().all():
            first_timestamp = float(
                endpoint_timestamps.iloc[0]
            )
            last_timestamp = float(
                endpoint_timestamps.iloc[-1]
            )
            if last_timestamp >= first_timestamp:
                stat["first_timestamp_seconds"] = first_timestamp
                stat["last_timestamp_seconds"] = last_timestamp
                stat["duration_seconds"] = last_timestamp - first_timestamp
                stat["timing_basis"] = "source_timestamps"
    return stat


def build_argument_parser():
    """Build the compatibility tracking CLI parser."""

    parser = argparse.ArgumentParser(
        description="Track people in video and extract movement metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input video file or directory of videos")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for results")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace complete per-video result directories. Without this "
            "flag, existing non-empty results make the command fail."
        ),
    )
    
    # Detection settings
    parser.add_argument(
        "--detector",
        choices=["yolo", "owl", "qwen"],
        default="yolo",
        help="Person detector (qwen is the preferred quality path)",
    )
    parser.add_argument("--yolo-model", default="yolo11x.pt",
                       help="Path to YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument(
        "--qwen-cadence",
        type=int,
        default=10,
        help="Run Qwen detection every N frames (default: 10)",
    )
    
    # Calibration (optional, for real-world measurements)
    parser.add_argument("--camera-calib", 
                       help="Camera calibration file (YAML)")
    parser.add_argument("--floor-calib",
                       help="Floor calibration file (YAML)")
    # Retained only so historical command lines using unversioned calibration
    # files continue to run. Canonical artifacts reject any value other than
    # 1.0, and this option is intentionally absent from researcher-facing help.
    parser.add_argument(
        "--correction-factor",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    
    # Identity matching (optional)
    parser.add_argument("--identities", type=str,
                       help='JSON dict of identity descriptions, e.g., \'{"Coach": "person in red", "Player": "person in white"}\'')
    parser.add_argument("--identity-file",
                       help="JSON file with identity descriptions")
    parser.add_argument(
        "--identity-evidence-frames",
        type=int,
        default=5,
        help="Representative crops supplied to Qwen per track (default: 5)",
    )
    
    # Tracking settings
    parser.add_argument(
        "--tracker",
        choices=["kalman", "deepsort"],
        default="kalman",
        help=(
            "Temporal tracker (default: kalman). DeepSORT uses the validated "
            "quality preset and its OSNet-AIN ReID model."
        ),
    )
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help=(
            "Max detector updates to keep an unmatched track alive "
            "(default: 30)"
        ),
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=None,
        help=(
            "Detections needed to confirm a track "
            "(default: 1 for qwen, 3 otherwise)"
        ),
    )
    parser.add_argument(
        "--reid-model",
        type=Path,
        metavar="PATH",
        help=(
            "Existing OSNet-AIN checkpoint override for DeepSORT. When "
            "omitted, the quality factory uses its verified cache/download "
            "policy."
        ),
    )
    parser.add_argument(
        "--allow-reid-fallback",
        action="store_true",
        help=(
            "Explicitly allow histogram features if the verified ReID model "
            "cannot be acquired or loaded. Valid only with --tracker "
            "deepsort."
        ),
    )
    
    # Output options
    parser.add_argument("--visualize", action="store_true",
                       help="Reserved; use --save-frames in this compatibility script")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save sample frames with annotations")
    parser.add_argument("--frame-interval", type=int, default=100,
                       help="Interval for saving sample frames (default: 100)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process (for testing)")
    
    # Processing options
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Local model device (default: auto)",
    )
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Reserved for detector adapters (default: 1)")

    return parser


def validate_cli_args(parser, args):
    """Validate argument combinations before any files or models are touched."""

    if bool(args.camera_calib) != bool(args.floor_calib):
        parser.error("--camera-calib and --floor-calib must be provided together")
    if args.allow_reid_fallback and args.tracker != "deepsort":
        parser.error(
            "--allow-reid-fallback is valid only with --tracker deepsort"
        )
    if args.reid_model is not None and args.tracker != "deepsort":
        parser.error("--reid-model is valid only with --tracker deepsort")
    if args.visualize:
        print("Warning: --visualize is not implemented yet; use --save-frames")
    if args.batch_size != 1:
        print("Warning: --batch-size is reserved for a future detector adapter")
    if args.qwen_cadence < 1:
        parser.error("--qwen-cadence must be positive")
    if (
        not math.isfinite(args.confidence)
        or not 0 <= args.confidence <= 1
    ):
        parser.error("--confidence must be between 0 and 1")
    if args.max_age < 0:
        parser.error("--max-age must be non-negative")
    if args.tracker == "deepsort" and args.max_age == 0:
        parser.error("--max-age must be positive with --tracker deepsort")
    if args.min_hits is None:
        args.min_hits = 1 if args.detector == "qwen" else 3
    elif args.min_hits < 1:
        parser.error("--min-hits must be positive")
    if (
        not math.isfinite(args.correction_factor)
        or args.correction_factor <= 0
    ):
        parser.error("--correction-factor must be a finite positive number")
    if args.identity_evidence_frames < 1:
        parser.error("--identity-evidence-frames must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive when provided")

    return args


def resolve_reid_device(requested_device):
    """Resolve ``auto`` to an available PyTorch device for OSNet inference."""

    if requested_device != "auto":
        return requested_device

    try:
        import torch
    except ImportError:
        # The validated factory will provide the actionable dependency error.
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


def build_deepsort_components(
    args,
    *,
    pipeline_builder=None,
    preset_loader=None,
):
    """Construct the selected DeepSORT tracker through the quality factory."""

    if pipeline_builder is None or preset_loader is None:
        from naturallab.spatial_tracking.pipeline import (
            build_spatial_pipeline,
            load_spatial_pipeline_preset,
        )

        pipeline_builder = pipeline_builder or build_spatial_pipeline
        preset_loader = preset_loader or load_spatial_pipeline_preset

    preset = preset_loader()
    tracker_config = replace(
        preset.tracker,
        max_age=args.max_age,
        min_hits=args.min_hits,
        reid_device=resolve_reid_device(args.device),
    )
    effective_preset = replace(preset, tracker=tracker_config)
    return pipeline_builder(
        preset=effective_preset,
        reid_model_path=args.reid_model,
        allow_reid_fallback=args.allow_reid_fallback,
    )


def tracker_run_provenance(args, components=None):
    """Return secret-free tracker and ReID metadata for ``run_metadata.json``."""

    if args.tracker == "kalman":
        return {
            "backend": "kalman",
            "parameters": {
                "max_age": args.max_age,
                "min_hits": args.min_hits,
            },
            "reid_model": None,
        }

    if components is None:
        raise ValueError("DeepSORT provenance requires pipeline components")
    pipeline_provenance = components.provenance()
    return {
        "backend": pipeline_provenance["tracker_backend"],
        "parameters": pipeline_provenance["tracker_parameters"],
        "reid_model": pipeline_provenance["reid_model"],
    }


def main(argv=None):
    parser = build_argument_parser()
    args = validate_cli_args(parser, parser.parse_args(argv))
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1

    output_path = Path(args.output)
    if output_path.exists() and not output_path.is_dir():
        print(f"Error: Output path is not a directory: {output_path}")
        return 1

    # Load identity descriptions if provided
    identities = None
    if args.identities:
        identities = json.loads(args.identities)
    elif args.identity_file:
        with open(args.identity_file, encoding="utf-8") as f:
            identities = json.load(f)
    if identities is not None and (
        not isinstance(identities, dict)
        or not identities
        or not all(
            isinstance(role, str)
            and role.strip()
            and isinstance(description, str)
            and description.strip()
            for role, description in identities.items()
        )
    ):
        parser.error(
            "identities must be a non-empty JSON object mapping role names "
            "to non-empty descriptions"
        )
    
    print("=" * 60)
    print("NaturalLab - People Tracking Pipeline")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Detector: {args.detector}")
    print(f"Tracker: {args.tracker}")
    print(f"Confidence threshold: {args.confidence}")
    if args.camera_calib:
        print(f"Camera calibration: {args.camera_calib}")
    if args.floor_calib:
        print(f"Floor calibration: {args.floor_calib}")
    if identities:
        print(f"Identities: {list(identities.keys())}")
    print()
    
    # Resolve and validate the complete batch before creating output or
    # loading models. A directory with no supported videos is an error, not a
    # successful no-op.
    videos = discover_video_files(input_path)
    if not videos:
        print(
            "Error: No supported video files found. Supported extensions: "
            + ", ".join(sorted(VIDEO_SUFFIXES))
        )
        return 1
    try:
        output_plan = build_video_output_plan(videos, output_path)
        validate_video_output_plan(
            output_plan,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError) as error:
        print(f"Error: {error}")
        return 1

    video_metadata = {}
    unusable_videos = []
    for video_path in videos:
        metadata = probe_video(video_path)
        if metadata is None:
            unusable_videos.append(video_path)
        else:
            video_metadata[video_path] = metadata
    if unusable_videos:
        print(
            "Error: Could not establish a complete, decodable video "
            "contract for:"
        )
        for video_path in unusable_videos:
            print(f"  - {video_path}")
        print(
            "Each file must open, report a positive frame count, and decode "
            "its first frame."
        )
        print("No videos were processed and no result directories were changed.")
        return 1

    output_path.mkdir(parents=True, exist_ok=True)

    # Import tracking modules only after input and output preflight succeeds.
    try:
        from naturallab.spatial_tracking.tracking.kalman_tracker import (
            KalmanPersonTracker,
        )
    except ImportError as error:
        print(f"Error importing modules: {error}")
        print("Make sure naturallab is installed: pip install -e .")
        return 1

    print(f"Found {len(videos)} video(s) to process")

    qwen_config = None
    if args.detector == "qwen" or identities:
        from naturallab.spatial_tracking.vlm import QwenBackendConfig

        effective_detection_cadence = (
            args.qwen_cadence if args.detector == "qwen" else 1
        )
        qwen_config = QwenBackendConfig(
            detection_cadence_frames=effective_detection_cadence
        )
    
    for video_path, video_output in output_plan:
        print(f"\nProcessing: {video_path.name}")
        print("-" * 40)

        # Initialize detector
        detector_device = None if args.device == "auto" else args.device
        if args.detector == "yolo":
            from naturallab.spatial_tracking.detection.yolo_detector import (
                YOLODetectorModule,
            )

            detector = YOLODetectorModule(
                model_path=args.yolo_model,
                confidence=args.confidence,
                device=detector_device,
            )
        elif args.detector == "owl":
            from naturallab.spatial_tracking.detection.owl_detector import (
                OWLDetectorModule,
            )

            detector = OWLDetectorModule(
                confidence=args.confidence,
                device=detector_device,
            )
            if not detector.has_model:
                print(
                    "Error: OWLv2 detector initialization failed: "
                    f"{detector.load_error}"
                )
                return 1
        else:
            from naturallab.spatial_tracking.detection.qwen_detector import (
                QwenDetectorModule,
            )
            from naturallab.spatial_tracking.vlm import QwenPersonGrounder

            detector = QwenDetectorModule(
                grounder=QwenPersonGrounder(config=qwen_config),
                cadence_frames=args.qwen_cadence,
                confidence_threshold=args.confidence,
            )
        
        # Initialize the requested temporal tracker. The compatibility default
        # remains Kalman; DeepSORT construction always goes through the
        # validated quality preset so checkpoint and fallback policy cannot be
        # bypassed accidentally.
        deepsort_components = None
        if args.tracker == "kalman":
            tracker = KalmanPersonTracker(
                max_age=args.max_age,
                min_hits=args.min_hits,
            )
        else:
            try:
                deepsort_components = build_deepsort_components(args)
            except Exception as error:
                print(
                    "Error: DeepSORT/ReID setup failed: "
                    f"{error}",
                    file=sys.stderr,
                )
                if not args.allow_reid_fallback:
                    print(
                        "No ReID fallback was used. After reviewing the "
                        "warning, rerun with --allow-reid-fallback only if "
                        "histogram features are acceptable for this run.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        "Histogram fallback was explicitly allowed, but the "
                        "tracker still could not be constructed.",
                        file=sys.stderr,
                    )
                return 1
            tracker = deepsort_components.tracker
        tracker_provenance = tracker_run_provenance(
            args,
            deepsort_components,
        )
        
        # Initialize floor tracker if calibration available
        floor_tracker = None
        if args.camera_calib and args.floor_calib:
            floor_tracker = load_floor_tracker(
                args.camera_calib,
                args.floor_calib,
                correction_factor=args.correction_factor
            )
        
        # Actual decoding goes through the common FrameSource contract so
        # container timestamps are retained. Preflight already proved that at
        # least one frame is decodable.
        import cv2
        from naturallab.media import VideoFileSource

        total_frames, fps = video_metadata[video_path]
        print(
            f"  Total frames: {total_frames if total_frames else 'unknown'}"
        )
        print(f"  Nominal FPS: {fps if fps else 'unknown'}")

        # Start from a clean directory only after all input/model/calibration
        # setup for this video has succeeded. This removes prior frames and
        # identity results as well as CSVs when --overwrite is explicit.
        try:
            prepare_video_output(
                video_output,
                overwrite=args.overwrite,
            )
        except FileExistsError as error:
            print(f"Error: {error}")
            return 1
        
        # Storage for results
        all_tracks = []
        detection_provenance = None
        frames_processed = 0
        timestamp_sources = set()
        
        # Progress tracking
        from tqdm import tqdm
        expected_frames = total_frames or None
        if args.max_frames is not None:
            expected_frames = (
                min(args.max_frames, total_frames)
                if total_frames
                else args.max_frames
            )
        frame_source = VideoFileSource(
            video_path,
            stop_frame=args.max_frames,
        )
        
        for packet in tqdm(
            frame_source,
            total=expected_frames,
            desc="  Processing",
        ):
            frame_idx = packet.frame_index
            frame = packet.image
            if packet.metadata.get("timestamp_source"):
                timestamp_sources.add(
                    packet.metadata["timestamp_source"]
                )
            frame = prepare_frame_for_calibration(frame, floor_tracker)
            
            # Detector and tracker modules share one dictionary contract.
            detection_data = detector.process(
                {"frame": frame, "frame_idx": frame_idx}
            )
            frame_provenance = detection_data.get("detection_provenance")
            if frame_provenance is not None:
                if (
                    detection_provenance is not None
                    and detection_provenance != frame_provenance
                ):
                    raise RuntimeError(
                        "detector provenance changed during one video"
                    )
                detection_provenance = frame_provenance
            tracks = tracker.process(detection_data)["tracks"]
            
            # Store track data
            for track in tracks:
                track_data = {
                    "frame": frame_idx,
                    "track_id": track["id"],
                    "x1": track["bbox"][0],
                    "y1": track["bbox"][1],
                    "x2": track["bbox"][2],
                    "y2": track["bbox"][3],
                    "confidence": track["score"],
                    "is_prediction": track.get("is_prediction", False),
                    "timestamp_seconds": packet.source_timestamp,
                    "timestamp_ns": packet.timestamp_ns,
                    "timestamp_source": packet.metadata.get(
                        "timestamp_source"
                    ),
                }
                
                # Add floor position if available
                if floor_tracker:
                    floor_pos = floor_tracker.update_track(
                        track["id"], track["bbox"]
                    )
                    if floor_pos is not None:
                        track_data["floor_x"] = floor_pos[0]
                        track_data["floor_y"] = floor_pos[1]
                        track_data["floor_z"] = floor_pos[2]
                
                all_tracks.append(track_data)
            
            # Save sample frames if requested
            if args.save_frames and frame_idx % args.frame_interval == 0:
                frame_path = video_output / "frames" / f"frame_{frame_idx:06d}.jpg"
                frame_path.parent.mkdir(exist_ok=True)
                # Draw tracks on frame
                annotated = frame.copy()
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track["bbox"])
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID: {track['id']}",
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(frame_path), annotated)
            
            frames_processed += 1

        required_frame_count = expected_decoded_frame_count(
            total_frames,
            args.max_frames,
        )
        if frames_processed == 0:
            shutil.rmtree(video_output)
            print(
                "  Error: Video produced no decodable frames; no result "
                "directory was kept."
            )
            return 1
        if (
            required_frame_count is not None
            and frames_processed < required_frame_count
        ):
            shutil.rmtree(video_output)
            print(
                "  Error: Video decoding ended early after "
                f"{frames_processed} of {required_frame_count} expected "
                "frames; no result directory was kept."
            )
            return 1

        detector_settings = {
            "confidence_threshold": args.confidence,
            "device": args.device,
        }
        if args.detector == "yolo":
            detector_settings["model_path"] = args.yolo_model
        if args.detector == "qwen":
            detector_settings["detection_cadence_frames"] = (
                args.qwen_cadence
            )
        with open(
            video_output / "run_metadata.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                {
                    "input_video": str(video_path),
                    "detector": args.detector,
                    "detector_settings": detector_settings,
                    "detection_provenance": detection_provenance,
                    "tracker": args.tracker,
                    "tracker_provenance": tracker_provenance,
                    "reid_provenance": tracker_provenance["reid_model"],
                    "frames_processed": frames_processed,
                    "timestamp_sources": sorted(timestamp_sources),
                },
                handle,
                indent=2,
                sort_keys=True,
            )
        
        # Always write the two table contracts, including headers for a valid
        # run that found no publishable tracks.
        df, stats_df = write_track_tables(
            video_output,
            all_tracks,
            fps=fps,
            floor_tracker=floor_tracker,
        )
        print(f"  Saved {len(df)} track records to tracks.csv")
        print(f"  Saved statistics for {len(stats_df)} tracks")

        # Identity matching if requested. An empty valid run still receives a
        # deterministic identity file with an empty assignments mapping.
        if identities:
            assignments = {}
            if not df.empty:
                print("  Performing Qwen role assignment...")
                from naturallab.spatial_tracking.vlm import (
                    QwenTrackRoleAssigner,
                )

                evidence_by_track = extract_track_evidence(
                    video_path,
                    df,
                    floor_tracker,
                    frames_per_track=args.identity_evidence_frames,
                )
                assigner = QwenTrackRoleAssigner(
                    roles=tuple(identities),
                    role_descriptions=identities,
                    evidence_images_per_track=(
                        args.identity_evidence_frames
                    ),
                    config=qwen_config,
                )
                for track_id in map(str, df["track_id"].unique()):
                    evidence = evidence_by_track.get(track_id, [])
                    if not evidence:
                        assignments[track_id] = {
                            "track_id": track_id,
                            "role": None,
                            "abstained": True,
                            "confidence": None,
                            "reason": "No valid track crops were available.",
                            "provenance": None,
                        }
                        continue
                    assignment = assigner.assign_role(track_id, evidence)
                    assignments[track_id] = serialize_role_assignment(
                        assignment
                    )

            with open(
                video_output / "identity_matches.json",
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                        "role_descriptions": identities,
                        "evidence_images_per_track": (
                            args.identity_evidence_frames
                        ),
                        "assignments": assignments,
                    },
                    handle,
                    indent=2,
                    sort_keys=True,
                )
            print("  Identity matches saved")
        
        print(f"  Results saved to: {video_output}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
