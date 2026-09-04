"""Automatic shared-board recovery of fixed multi-camera extrinsics.

The workflow in this module is deliberately manifest-driven.  Every view is
bound to one explicit video and one canonical intrinsic/floor bundle.  A
stationary chessboard visible in all views supplies metric correspondences.

OpenCV does not promise a physical corner origin for a symmetric chessboard.
Consequently, raw corner order is never used as a cross-camera identity.  The
implementation resolves the valid grid symmetries against one shared floor
homography before estimating fixed-intrinsic stereo transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from ..multiview.registration import RoomRegistration, ViewRegistration
from .artifacts import CalibrationBundle
from .automatic import (
    AutomaticCalibrationError,
    BoardDetection,
    BoardSpec,
    VideoMetadata,
    group_stationary_detections,
    load_calibration_bundle_file,
    scan_calibration_video,
    source_identity,
)


EXTRINSICS_INPUT_KIND = "shared_board_extrinsics_input"
EXTRINSICS_INPUT_SCHEMA_VERSION = "1.0"
EXTRINSICS_REPORT_KIND = "shared_board_extrinsics_report"

_MANIFEST_FIELDS = {
    "schema_version",
    "kind",
    "rig_id",
    "anchor_view_id",
    "room_coordinate_frame",
    "room_frame_mode",
    "board",
    "sampling",
    "quality_limits",
    "views",
}
_VIEW_FIELDS = {
    "view_id",
    "video",
    "calibration_bundle",
    "time_offset_seconds",
}
_BOARD_FIELDS = {
    "internal_columns",
    "internal_rows",
    "square_size_mm",
}
_SAMPLING_FIELDS = {
    "sample_seconds",
    "stationary_motion_pixels",
    "minimum_stationary_samples",
    "minimum_placement_separation_pixels",
    "minimum_shared_placements",
    "maximum_shared_placements",
    "time_tolerance_seconds",
}
_QUALITY_FIELDS = {
    "maximum_stereo_rms_pixels",
    "maximum_holdout_p90_pixels",
    "maximum_transform_p90_rotation_degrees",
    "maximum_transform_p90_translation_mm",
    "maximum_floor_normal_angle_degrees",
    "maximum_floor_offset_mm",
    "maximum_triangulated_board_p90_error_percent",
    "minimum_orientation_margin_pixels",
}
_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AutomaticCalibrationError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise AutomaticCalibrationError(f"{field_name} keys must be strings")
    return value


def _strict_fields(
    value: Mapping[str, Any],
    *,
    field_name: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AutomaticCalibrationError(
            f"{field_name} contains unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(required - set(value))
    if missing:
        raise AutomaticCalibrationError(
            f"{field_name} is missing field(s): {', '.join(missing)}"
        )


def _string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutomaticCalibrationError(
            f"{field_name} must be a non-empty string"
        )
    return value.strip()


def _identifier(value: Any, field_name: str) -> str:
    result = _string(value, field_name)
    if not _SAFE_IDENTIFIER.fullmatch(result):
        raise AutomaticCalibrationError(
            f"{field_name} must contain only letters, digits, '.', '_', or "
            "'-' and must start with a letter or digit"
        )
    return result


def _float(
    value: Any,
    field_name: str,
    *,
    minimum: Optional[float] = None,
    strictly_greater: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise AutomaticCalibrationError(
            f"{field_name} must be a finite number"
        )
    result = float(value)
    if minimum is not None:
        invalid = result <= minimum if strictly_greater else result < minimum
        if invalid:
            comparison = "greater than" if strictly_greater else "at least"
            raise AutomaticCalibrationError(
                f"{field_name} must be {comparison} {minimum}"
            )
    return result


def _integer(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AutomaticCalibrationError(f"{field_name} must be an integer")
    if value < minimum:
        raise AutomaticCalibrationError(
            f"{field_name} must be at least {minimum}"
        )
    return value


def _resolve_path(value: Any, *, base: Path, field_name: str) -> Path:
    rendered = _string(value, field_name)
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    result = candidate.resolve()
    if not result.is_file():
        raise AutomaticCalibrationError(
            f"{field_name} does not exist: {result}"
        )
    return result


@dataclass(frozen=True)
class ExtrinsicViewInput:
    """One fixed view and the calibration/video evidence bound to it."""

    view_id: str
    video_path: Path
    bundle_path: Path
    time_offset_seconds: float


@dataclass(frozen=True)
class ExtrinsicQualityLimits:
    maximum_stereo_rms_pixels: float = 5.0
    maximum_holdout_p90_pixels: float = 20.0
    maximum_transform_p90_rotation_degrees: float = 1.0
    maximum_transform_p90_translation_mm: float = 60.0
    maximum_floor_normal_angle_degrees: float = 2.0
    maximum_floor_offset_mm: float = 60.0
    maximum_triangulated_board_p90_error_percent: float = 5.0
    minimum_orientation_margin_pixels: float = 25.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ExtrinsicQualityLimits":
        _strict_fields(
            value,
            field_name="quality_limits",
            allowed=_QUALITY_FIELDS,
            required=set(),
        )
        defaults = cls()
        values: Dict[str, float] = {}
        for name in _QUALITY_FIELDS:
            values[name] = _float(
                value.get(name, getattr(defaults, name)),
                f"quality_limits.{name}",
                minimum=0.0,
                strictly_greater=True,
            )
        return cls(**values)

    def to_dict(self) -> Dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in sorted(_QUALITY_FIELDS)
        }


@dataclass(frozen=True)
class SharedBoardExtrinsicsManifest:
    """Strict, resolved input contract for one fixed camera rig."""

    source_path: Path
    rig_id: str
    anchor_view_id: str
    room_coordinate_frame: str
    room_frame_mode: str
    board: BoardSpec
    sample_seconds: float
    stationary_motion_pixels: float
    minimum_stationary_samples: int
    minimum_placement_separation_pixels: float
    minimum_shared_placements: int
    maximum_shared_placements: int
    time_tolerance_seconds: float
    quality_limits: ExtrinsicQualityLimits
    views: Tuple[ExtrinsicViewInput, ...]

    @property
    def anchor_view(self) -> ExtrinsicViewInput:
        return next(
            view for view in self.views if view.view_id == self.anchor_view_id
        )


@dataclass(frozen=True)
class SharedPlacement:
    """One stationary board placement observed by every configured view."""

    placement_id: int
    detections: Mapping[str, BoardDetection]
    maximum_time_delta_seconds: float


@dataclass(frozen=True)
class GridSymmetry:
    name: str
    indices: Tuple[int, ...]


@dataclass(frozen=True)
class PairRecovery:
    target_view_id: str
    symmetry_names: Tuple[str, ...]
    symmetry_indices: Tuple[Tuple[int, ...], ...]
    orientation_diagnostics: Mapping[str, Any]
    target_from_anchor: np.ndarray
    stereo_rms_pixels: float
    transfer_errors_pixels: Tuple[float, ...]
    holdout_errors_pixels: Tuple[float, ...]
    holdout_triangulation_errors_percent: Tuple[float, ...]
    holdout_ray_angles_degrees: Tuple[float, ...]
    stability_rotation_degrees: Tuple[float, ...]
    stability_translation_mm: Tuple[float, ...]


@dataclass(frozen=True)
class ExtrinsicCalibrationRun:
    room_registration: RoomRegistration
    report: Mapping[str, Any]
    observations: Tuple[Mapping[str, Any], ...]
    selected_detections_by_view: Mapping[str, Tuple[BoardDetection, ...]]
    manifest: SharedBoardExtrinsicsManifest
    bundles_by_view: Mapping[str, CalibrationBundle]


def load_extrinsics_manifest(
    path: Path | str,
) -> SharedBoardExtrinsicsManifest:
    """Load and strictly validate a shared-board calibration manifest."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise AutomaticCalibrationError(
            f"extrinsics manifest does not exist: {source}"
        )
    try:
        with source.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise AutomaticCalibrationError(
            f"could not read extrinsics manifest {source}: {exc}"
        ) from exc
    values = _mapping(document, "extrinsics manifest")
    _strict_fields(
        values,
        field_name="extrinsics manifest",
        allowed=_MANIFEST_FIELDS,
        required=_MANIFEST_FIELDS - {"quality_limits"},
    )
    if str(values["schema_version"]) != EXTRINSICS_INPUT_SCHEMA_VERSION:
        raise AutomaticCalibrationError(
            "extrinsics manifest.schema_version must be "
            f"{EXTRINSICS_INPUT_SCHEMA_VERSION!r}"
        )
    if values["kind"] != EXTRINSICS_INPUT_KIND:
        raise AutomaticCalibrationError(
            f"extrinsics manifest.kind must be {EXTRINSICS_INPUT_KIND!r}"
        )

    board_values = _mapping(values["board"], "board")
    _strict_fields(
        board_values,
        field_name="board",
        allowed=_BOARD_FIELDS,
        required=_BOARD_FIELDS,
    )
    board = BoardSpec(
        internal_columns=_integer(
            board_values["internal_columns"],
            "board.internal_columns",
            minimum=2,
        ),
        internal_rows=_integer(
            board_values["internal_rows"],
            "board.internal_rows",
            minimum=2,
        ),
        square_size_mm=_float(
            board_values["square_size_mm"],
            "board.square_size_mm",
            minimum=0.0,
            strictly_greater=True,
        ),
    )

    sampling = _mapping(values["sampling"], "sampling")
    _strict_fields(
        sampling,
        field_name="sampling",
        allowed=_SAMPLING_FIELDS,
        required=_SAMPLING_FIELDS,
    )
    minimum_shared = _integer(
        sampling["minimum_shared_placements"],
        "sampling.minimum_shared_placements",
        minimum=3,
    )
    maximum_shared = _integer(
        sampling["maximum_shared_placements"],
        "sampling.maximum_shared_placements",
        minimum=minimum_shared,
    )

    raw_views = values["views"]
    if (
        isinstance(raw_views, (str, bytes))
        or not isinstance(raw_views, Sequence)
        or len(raw_views) < 2
    ):
        raise AutomaticCalibrationError(
            "views must contain at least two explicit view mappings"
        )
    views = []
    for index, raw_view in enumerate(raw_views):
        view_values = _mapping(raw_view, f"views[{index}]")
        _strict_fields(
            view_values,
            field_name=f"views[{index}]",
            allowed=_VIEW_FIELDS,
            required=_VIEW_FIELDS - {"time_offset_seconds"},
        )
        views.append(
            ExtrinsicViewInput(
                view_id=_identifier(
                    view_values["view_id"],
                    f"views[{index}].view_id",
                ),
                video_path=_resolve_path(
                    view_values["video"],
                    base=source.parent,
                    field_name=f"views[{index}].video",
                ),
                bundle_path=_resolve_path(
                    view_values["calibration_bundle"],
                    base=source.parent,
                    field_name=f"views[{index}].calibration_bundle",
                ),
                time_offset_seconds=_float(
                    view_values.get("time_offset_seconds", 0.0),
                    f"views[{index}].time_offset_seconds",
                ),
            )
        )
    view_ids = [view.view_id for view in views]
    if len(set(view_ids)) != len(view_ids):
        raise AutomaticCalibrationError("view_id values must be unique")
    anchor_view_id = _identifier(
        values["anchor_view_id"],
        "anchor_view_id",
    )
    if anchor_view_id not in view_ids:
        raise AutomaticCalibrationError(
            f"anchor_view_id {anchor_view_id!r} is not present in views"
        )

    room_frame_mode = _string(values["room_frame_mode"], "room_frame_mode")
    if room_frame_mode not in {"anchor_opencv", "floor_aligned_anchor"}:
        raise AutomaticCalibrationError(
            "room_frame_mode must be 'anchor_opencv' or "
            "'floor_aligned_anchor'"
        )
    quality = ExtrinsicQualityLimits.from_mapping(
        _mapping(values.get("quality_limits", {}), "quality_limits")
    )
    return SharedBoardExtrinsicsManifest(
        source_path=source,
        rig_id=_identifier(values["rig_id"], "rig_id"),
        anchor_view_id=anchor_view_id,
        room_coordinate_frame=_string(
            values["room_coordinate_frame"],
            "room_coordinate_frame",
        ),
        room_frame_mode=room_frame_mode,
        board=board,
        sample_seconds=_float(
            sampling["sample_seconds"],
            "sampling.sample_seconds",
            minimum=0.0,
            strictly_greater=True,
        ),
        stationary_motion_pixels=_float(
            sampling["stationary_motion_pixels"],
            "sampling.stationary_motion_pixels",
            minimum=0.0,
        ),
        minimum_stationary_samples=_integer(
            sampling["minimum_stationary_samples"],
            "sampling.minimum_stationary_samples",
            minimum=2,
        ),
        minimum_placement_separation_pixels=_float(
            sampling["minimum_placement_separation_pixels"],
            "sampling.minimum_placement_separation_pixels",
            minimum=0.0,
        ),
        minimum_shared_placements=minimum_shared,
        maximum_shared_placements=maximum_shared,
        time_tolerance_seconds=_float(
            sampling["time_tolerance_seconds"],
            "sampling.time_tolerance_seconds",
            minimum=0.0,
            strictly_greater=True,
        ),
        quality_limits=quality,
        views=tuple(views),
    )


def grid_symmetries(board: BoardSpec) -> Tuple[GridSymmetry, ...]:
    """Return every shape-preserving corner permutation for the grid."""

    grid = np.arange(board.corner_count, dtype=int).reshape(
        board.internal_rows,
        board.internal_columns,
    )
    candidates = [
        ("identity", grid),
        ("flip_columns", np.fliplr(grid)),
        ("flip_rows", np.flipud(grid)),
        ("rotate_180", np.flipud(np.fliplr(grid))),
    ]
    if board.internal_rows == board.internal_columns:
        transposed = grid.T
        candidates.extend(
            [
                ("transpose", transposed),
                ("rotate_90", np.fliplr(transposed)),
                ("rotate_270", np.flipud(transposed)),
                (
                    "anti_transpose",
                    np.flipud(np.fliplr(transposed)),
                ),
            ]
        )
    result = []
    seen = set()
    for name, candidate in candidates:
        indices = tuple(int(item) for item in candidate.reshape(-1))
        if indices in seen:
            continue
        seen.add(indices)
        result.append(GridSymmetry(name=name, indices=indices))
    return tuple(result)


def _bundle_geometry(
    manifest: SharedBoardExtrinsicsManifest,
) -> Dict[str, CalibrationBundle]:
    bundles: Dict[str, CalibrationBundle] = {}
    camera_ids = []
    units = set()
    for view in manifest.views:
        bundle = load_calibration_bundle_file(view.bundle_path)
        if bundle.camera_id in camera_ids:
            raise AutomaticCalibrationError(
                f"camera_id {bundle.camera_id!r} is reused by multiple views"
            )
        camera_ids.append(bundle.camera_id)
        if bundle.intrinsics.coordinate_frame != (
            bundle.floor_plane.coordinate_frame
        ):
            raise AutomaticCalibrationError(
                f"view {view.view_id!r} intrinsic and floor coordinate "
                "frames disagree"
            )
        units.add(bundle.floor_plane.units)
        bundles[view.view_id] = bundle
    if units != {"mm"}:
        raise AutomaticCalibrationError(
            "all floor calibration bundles must use millimetres"
        )
    return bundles


def _scan_views(
    manifest: SharedBoardExtrinsicsManifest,
    bundles: Mapping[str, CalibrationBundle],
) -> Tuple[
    Dict[str, Tuple[BoardDetection, ...]],
    Dict[str, Tuple[Tuple[BoardDetection, ...], ...]],
    Dict[str, VideoMetadata],
]:
    detections_by_view: Dict[str, Tuple[BoardDetection, ...]] = {}
    groups_by_view: Dict[str, Tuple[Tuple[BoardDetection, ...], ...]] = {}
    metadata_by_view: Dict[str, VideoMetadata] = {}
    for view in manifest.views:
        bundle = bundles[view.view_id]
        detections, metadata = scan_calibration_video(
            view.video_path,
            board=manifest.board,
            input_rotation=bundle.input_rotation,
            sample_seconds=manifest.sample_seconds,
        )
        expected_size = (
            bundle.intrinsics.image_size.width,
            bundle.intrinsics.image_size.height,
        )
        if metadata.image_size != expected_size:
            raise AutomaticCalibrationError(
                f"view {view.view_id!r} video geometry "
                f"{metadata.image_size[0]}x{metadata.image_size[1]} does not "
                f"match its bundle geometry {expected_size[0]}x"
                f"{expected_size[1]} after input rotation"
            )
        groups = group_stationary_detections(
            detections,
            sample_step_frames=metadata.sample_step_frames,
            maximum_center_motion_pixels=(
                manifest.stationary_motion_pixels
            ),
            minimum_samples=manifest.minimum_stationary_samples,
        )
        detections_by_view[view.view_id] = detections
        groups_by_view[view.view_id] = groups
        metadata_by_view[view.view_id] = metadata
    return detections_by_view, groups_by_view, metadata_by_view


def _common_time(
    detection: BoardDetection,
    view: ExtrinsicViewInput,
) -> float:
    return detection.timestamp_seconds + view.time_offset_seconds


def _shared_placements(
    manifest: SharedBoardExtrinsicsManifest,
    groups_by_view: Mapping[
        str,
        Sequence[Sequence[BoardDetection]],
    ],
) -> Tuple[SharedPlacement, ...]:
    views = {view.view_id: view for view in manifest.views}
    anchor_id = manifest.anchor_view_id
    eligible = {
        view_id: tuple(
            detection
            for group in groups
            for detection in group
        )
        for view_id, groups in groups_by_view.items()
    }
    candidates = []
    for anchor_group in groups_by_view[anchor_id]:
        alternatives = []
        for anchor_detection in anchor_group:
            common_time = _common_time(
                anchor_detection,
                views[anchor_id],
            )
            matched = {anchor_id: anchor_detection}
            deltas = []
            sharpness = anchor_detection.sharpness
            for view in manifest.views:
                if view.view_id == anchor_id:
                    continue
                if not eligible[view.view_id]:
                    matched = {}
                    break
                nearest = min(
                    eligible[view.view_id],
                    key=lambda item: abs(
                        _common_time(item, view) - common_time
                    ),
                )
                delta = abs(_common_time(nearest, view) - common_time)
                if delta > manifest.time_tolerance_seconds:
                    matched = {}
                    break
                matched[view.view_id] = nearest
                deltas.append(delta)
                sharpness += nearest.sharpness
            if matched:
                alternatives.append(
                    (
                        sum(deltas),
                        -sharpness,
                        max(deltas, default=0.0),
                        matched,
                    )
                )
        if alternatives:
            _, _, maximum_delta, matched = min(
                alternatives,
                key=lambda item: (item[0], item[1]),
            )
            candidates.append((matched, maximum_delta))

    if len(candidates) < manifest.minimum_shared_placements:
        raise AutomaticCalibrationError(
            f"only {len(candidates)} stationary board placements were "
            "simultaneously visible in every view; at least "
            f"{manifest.minimum_shared_placements} are required"
        )

    first = int(
        np.argmax(
            [
                sum(
                    detection.sharpness
                    for detection in matched.values()
                )
                for matched, _ in candidates
            ]
        )
    )
    chosen = [first]
    while len(chosen) < min(
        manifest.maximum_shared_placements,
        len(candidates),
    ):
        remaining = [
            index
            for index in range(len(candidates))
            if index not in chosen
        ]
        scored = []
        for index in remaining:
            center = candidates[index][0][anchor_id].center
            separation = min(
                float(
                    np.linalg.norm(
                        center
                        - candidates[chosen_index][0][anchor_id].center
                    )
                )
                for chosen_index in chosen
            )
            scored.append((separation, index))
        best_separation, best_index = max(scored)
        if (
            best_separation
            < manifest.minimum_placement_separation_pixels
        ):
            break
        chosen.append(best_index)
    if len(chosen) < manifest.minimum_shared_placements:
        raise AutomaticCalibrationError(
            f"only {len(chosen)} spatially distinct shared placements remain; "
            f"at least {manifest.minimum_shared_placements} are required"
        )
    selected = [candidates[index] for index in chosen]
    selected.sort(
        key=lambda item: item[0][anchor_id].timestamp_seconds
    )
    return tuple(
        SharedPlacement(
            placement_id=index + 1,
            detections=dict(matched),
            maximum_time_delta_seconds=float(maximum_delta),
        )
        for index, (matched, maximum_delta) in enumerate(selected)
    )


def _camera_arrays(
    bundle: CalibrationBundle,
) -> Tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(bundle.intrinsics.camera_matrix, dtype=np.float64),
        np.asarray(bundle.intrinsics.dist_coeff, dtype=np.float64),
    )


def _undistorted_pixels(
    detection: BoardDetection,
    bundle: CalibrationBundle,
) -> np.ndarray:
    camera_matrix, distortion = _camera_arrays(bundle)
    return cv2.undistortPoints(
        detection.corners,
        camera_matrix,
        distortion,
        P=camera_matrix,
    ).reshape(-1, 2)


def _rms_vectors(residuals: np.ndarray) -> float:
    values = np.asarray(residuals, dtype=np.float64).reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(values * values, axis=1))))


def _apply_homography(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(points, dtype=np.float64), np.ones(len(points))]
    )
    transformed = (matrix @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2, None]


def _fit_homography(
    anchor_points: Sequence[np.ndarray],
    target_points: Sequence[np.ndarray],
    assignments: Sequence[GridSymmetry],
) -> np.ndarray:
    source = np.vstack(anchor_points).astype(np.float64)
    destination = np.vstack(
        [
            points[np.asarray(symmetry.indices, dtype=int)]
            for points, symmetry in zip(target_points, assignments)
        ]
    ).astype(np.float64)
    matrix, _ = cv2.findHomography(source, destination, method=0)
    if matrix is None or not np.all(np.isfinite(matrix)):
        raise AutomaticCalibrationError(
            "shared-board homography estimation failed"
        )
    return matrix


def _resolve_orientations(
    anchor_points: Sequence[np.ndarray],
    target_points: Sequence[np.ndarray],
    symmetries: Sequence[GridSymmetry],
) -> Tuple[Tuple[GridSymmetry, ...], Dict[str, Any]]:
    if len(anchor_points) != len(target_points) or not anchor_points:
        raise ValueError("paired non-empty point sets are required")
    solutions = {}
    for seed_index in range(len(anchor_points)):
        for seed_symmetry in symmetries:
            matrix, _ = cv2.findHomography(
                np.asarray(anchor_points[seed_index], dtype=np.float64),
                np.asarray(
                    target_points[seed_index][
                        np.asarray(seed_symmetry.indices, dtype=int)
                    ],
                    dtype=np.float64,
                ),
                method=0,
            )
            if matrix is None:
                continue
            assignments: Tuple[GridSymmetry, ...] = tuple(
                seed_symmetry for _ in anchor_points
            )
            for _ in range(10):
                updated = []
                for anchor, target in zip(anchor_points, target_points):
                    predicted = _apply_homography(anchor, matrix)
                    updated.append(
                        min(
                            symmetries,
                            key=lambda symmetry: _rms_vectors(
                                predicted
                                - target[
                                    np.asarray(
                                        symmetry.indices,
                                        dtype=int,
                                    )
                                ]
                            ),
                        )
                    )
                updated_tuple = tuple(updated)
                matrix = _fit_homography(
                    anchor_points,
                    target_points,
                    updated_tuple,
                )
                if tuple(item.name for item in updated_tuple) == tuple(
                    item.name for item in assignments
                ):
                    assignments = updated_tuple
                    break
                assignments = updated_tuple
            residuals = []
            per_placement = []
            margins = []
            ratios = []
            for anchor, target, selected in zip(
                anchor_points,
                target_points,
                assignments,
            ):
                predicted = _apply_homography(anchor, matrix)
                scores = sorted(
                    (
                        _rms_vectors(
                            predicted
                            - target[
                                np.asarray(symmetry.indices, dtype=int)
                            ]
                        ),
                        symmetry,
                    )
                    for symmetry in symmetries
                )
                selected_points = target[
                    np.asarray(selected.indices, dtype=int)
                ]
                residuals.append(predicted - selected_points)
                best = scores[0][0]
                second = scores[1][0]
                per_placement.append(best)
                margins.append(second - best)
                ratios.append(second / max(best, 1e-9))
            total_rms = _rms_vectors(np.vstack(residuals))
            signature = tuple(item.name for item in assignments)
            previous = solutions.get(signature)
            candidate = (
                total_rms,
                assignments,
                matrix,
                per_placement,
                margins,
                ratios,
            )
            if previous is None or total_rms < previous[0]:
                solutions[signature] = candidate
    if not solutions:
        raise AutomaticCalibrationError(
            "could not resolve cross-view chessboard orientation"
        )
    ranked = sorted(solutions.values(), key=lambda item: item[0])
    (
        total_rms,
        assignments,
        matrix,
        per_placement,
        margins,
        ratios,
    ) = ranked[0]
    diagnostics: Dict[str, Any] = {
        "method": "joint_undistorted_floor_homography_d4",
        "homography_anchor_to_target": matrix.tolist(),
        "global_rms_pixels": float(total_rms),
        "next_global_rms_pixels": (
            float(ranked[1][0]) if len(ranked) > 1 else None
        ),
        "per_placement_selected_rms_pixels": [
            float(value) for value in per_placement
        ],
        "per_placement_next_margin_pixels": [
            float(value) for value in margins
        ],
        "per_placement_next_ratio": [
            float(value) for value in ratios
        ],
        "minimum_next_margin_pixels": float(min(margins)),
        "minimum_next_ratio": float(min(ratios)),
        "candidate_assignment_count": len(ranked),
    }
    return assignments, diagnostics


def _stereo_calibrate(
    board: BoardSpec,
    anchor_detections: Sequence[BoardDetection],
    target_detections: Sequence[BoardDetection],
    assignments: Sequence[GridSymmetry],
    anchor_bundle: CalibrationBundle,
    target_bundle: CalibrationBundle,
) -> Tuple[float, np.ndarray]:
    if not (
        len(anchor_detections)
        == len(target_detections)
        == len(assignments)
    ):
        raise ValueError("stereo observations must have equal lengths")
    if len(anchor_detections) < 2:
        raise AutomaticCalibrationError(
            "fixed-intrinsic stereo calibration requires two placements"
        )
    object_sets = [
        board.object_points().astype(np.float32)
        for _ in anchor_detections
    ]
    anchor_sets = [
        detection.corners.astype(np.float32)
        for detection in anchor_detections
    ]
    target_sets = [
        detection.corners[
            np.asarray(symmetry.indices, dtype=int)
        ].astype(np.float32)
        for detection, symmetry in zip(target_detections, assignments)
    ]
    anchor_matrix, anchor_distortion = _camera_arrays(anchor_bundle)
    target_matrix, target_distortion = _camera_arrays(target_bundle)
    image_size = (
        anchor_bundle.intrinsics.image_size.width,
        anchor_bundle.intrinsics.image_size.height,
    )
    try:
        result = cv2.stereoCalibrate(
            object_sets,
            anchor_sets,
            target_sets,
            anchor_matrix.copy(),
            anchor_distortion.copy(),
            target_matrix.copy(),
            target_distortion.copy(),
            image_size,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-9,
            ),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    except cv2.error as exc:
        raise AutomaticCalibrationError(
            f"fixed-intrinsic stereo calibration failed: {exc}"
        ) from exc
    rms = float(result[0])
    rotation = np.asarray(result[5], dtype=np.float64)
    translation = np.asarray(result[6], dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    if not math.isfinite(rms) or not np.all(np.isfinite(transform)):
        raise AutomaticCalibrationError(
            "fixed-intrinsic stereo calibration produced non-finite values"
        )
    return rms, transform


def _pose_transform(
    detection: BoardDetection,
    *,
    board: BoardSpec,
    bundle: CalibrationBundle,
    symmetry: Optional[GridSymmetry] = None,
) -> np.ndarray:
    camera_matrix, distortion = _camera_arrays(bundle)
    corners = detection.corners
    if symmetry is not None:
        corners = corners[np.asarray(symmetry.indices, dtype=int)]
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        board.object_points(),
        corners,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise AutomaticCalibrationError(
            f"solvePnP failed for frame {detection.frame_index}"
        )
    if hasattr(cv2, "solvePnPRefineLM"):
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            board.object_points(),
            corners,
            camera_matrix,
            distortion,
            rotation_vector,
            translation_vector,
        )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(
        translation_vector,
        dtype=np.float64,
    ).reshape(3)
    camera_points = (
        rotation @ board.object_points().T
        + transform[:3, 3:4]
    ).T
    if np.any(camera_points[:, 2] <= 0.0):
        raise AutomaticCalibrationError(
            f"frame {detection.frame_index} produced a pose behind the camera"
        )
    return transform


def _project_board_to_target(
    *,
    board: BoardSpec,
    anchor_detection: BoardDetection,
    anchor_bundle: CalibrationBundle,
    target_bundle: CalibrationBundle,
    target_from_anchor: np.ndarray,
) -> np.ndarray:
    anchor_from_board = _pose_transform(
        anchor_detection,
        board=board,
        bundle=anchor_bundle,
    )
    target_from_board = target_from_anchor @ anchor_from_board
    rotation_vector, _ = cv2.Rodrigues(target_from_board[:3, :3])
    camera_matrix, distortion = _camera_arrays(target_bundle)
    projected, _ = cv2.projectPoints(
        board.object_points(),
        rotation_vector,
        target_from_board[:3, 3],
        camera_matrix,
        distortion,
    )
    return projected.reshape(-1, 2)


def _rotation_difference_degrees(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    relative = first @ second.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _triangulation_metrics(
    *,
    board: BoardSpec,
    anchor_detection: BoardDetection,
    target_detection: BoardDetection,
    symmetry: GridSymmetry,
    anchor_bundle: CalibrationBundle,
    target_bundle: CalibrationBundle,
    target_from_anchor: np.ndarray,
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    anchor_matrix, anchor_distortion = _camera_arrays(anchor_bundle)
    target_matrix, target_distortion = _camera_arrays(target_bundle)
    anchor_normalized = cv2.undistortPoints(
        anchor_detection.corners,
        anchor_matrix,
        anchor_distortion,
    ).reshape(-1, 2)
    target_normalized = cv2.undistortPoints(
        target_detection.corners[
            np.asarray(symmetry.indices, dtype=int)
        ],
        target_matrix,
        target_distortion,
    ).reshape(-1, 2)
    projection_anchor = np.column_stack(
        [np.eye(3), np.zeros(3)]
    ).astype(np.float64)
    projection_target = target_from_anchor[:3].astype(np.float64)
    homogeneous = cv2.triangulatePoints(
        projection_anchor,
        projection_target,
        anchor_normalized.T,
        target_normalized.T,
    )
    points = (homogeneous[:3] / homogeneous[3]).T
    target_points = (
        target_from_anchor[:3, :3] @ points.T
        + target_from_anchor[:3, 3:4]
    ).T
    if (
        not np.all(np.isfinite(points))
        or np.any(points[:, 2] <= 0.0)
        or np.any(target_points[:, 2] <= 0.0)
    ):
        raise AutomaticCalibrationError(
            "held-out triangulation produced points behind a camera"
        )

    errors = []
    columns = board.internal_columns
    rows = board.internal_rows
    for row in range(rows):
        for column in range(columns - 1):
            first = row * columns + column
            second = first + 1
            measured = float(np.linalg.norm(points[first] - points[second]))
            errors.append(
                abs(measured - board.square_size_mm)
                / board.square_size_mm
                * 100.0
            )
    for row in range(rows - 1):
        for column in range(columns):
            first = row * columns + column
            second = first + columns
            measured = float(np.linalg.norm(points[first] - points[second]))
            errors.append(
                abs(measured - board.square_size_mm)
                / board.square_size_mm
                * 100.0
            )

    anchor_rays = np.column_stack(
        [anchor_normalized, np.ones(len(anchor_normalized))]
    )
    anchor_rays /= np.linalg.norm(anchor_rays, axis=1, keepdims=True)
    target_rays = np.column_stack(
        [target_normalized, np.ones(len(target_normalized))]
    )
    target_rays = (
        target_from_anchor[:3, :3].T @ target_rays.T
    ).T
    target_rays /= np.linalg.norm(target_rays, axis=1, keepdims=True)
    cosines = np.clip(
        np.sum(anchor_rays * target_rays, axis=1),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(cosines))
    # Conditioning depends on the acute intersection angle between the two
    # unoriented rays.  Cameras facing one another can otherwise report an
    # obtuse angle despite strong triangulation geometry.
    angles = np.minimum(angles, 180.0 - angles)
    return (
        tuple(float(value) for value in errors),
        tuple(float(value) for value in angles),
    )


def _recover_pair(
    *,
    manifest: SharedBoardExtrinsicsManifest,
    placements: Sequence[SharedPlacement],
    bundles: Mapping[str, CalibrationBundle],
    target_view_id: str,
) -> PairRecovery:
    anchor_id = manifest.anchor_view_id
    anchor_bundle = bundles[anchor_id]
    target_bundle = bundles[target_view_id]
    anchor_detections = [
        placement.detections[anchor_id] for placement in placements
    ]
    target_detections = [
        placement.detections[target_view_id] for placement in placements
    ]
    anchor_points = [
        _undistorted_pixels(detection, anchor_bundle)
        for detection in anchor_detections
    ]
    target_points = [
        _undistorted_pixels(detection, target_bundle)
        for detection in target_detections
    ]
    assignments, orientation = _resolve_orientations(
        anchor_points,
        target_points,
        grid_symmetries(manifest.board),
    )
    stereo_rms, target_from_anchor = _stereo_calibrate(
        manifest.board,
        anchor_detections,
        target_detections,
        assignments,
        anchor_bundle,
        target_bundle,
    )

    transfer_errors = []
    stability_rotation = []
    stability_translation = []
    for anchor_detection, target_detection, symmetry in zip(
        anchor_detections,
        target_detections,
        assignments,
    ):
        projected = _project_board_to_target(
            board=manifest.board,
            anchor_detection=anchor_detection,
            anchor_bundle=anchor_bundle,
            target_bundle=target_bundle,
            target_from_anchor=target_from_anchor,
        )
        observed = target_detection.corners[
            np.asarray(symmetry.indices, dtype=int)
        ].reshape(-1, 2)
        transfer_errors.extend(
            np.linalg.norm(projected - observed, axis=1).tolist()
        )
        anchor_from_board = _pose_transform(
            anchor_detection,
            board=manifest.board,
            bundle=anchor_bundle,
        )
        target_from_board = _pose_transform(
            target_detection,
            board=manifest.board,
            bundle=target_bundle,
            symmetry=symmetry,
        )
        placement_target_from_anchor = (
            target_from_board @ np.linalg.inv(anchor_from_board)
        )
        stability_rotation.append(
            _rotation_difference_degrees(
                placement_target_from_anchor[:3, :3],
                target_from_anchor[:3, :3],
            )
        )
        stability_translation.append(
            float(
                np.linalg.norm(
                    placement_target_from_anchor[:3, 3]
                    - target_from_anchor[:3, 3]
                )
            )
        )

    holdout_errors = []
    triangulation_errors = []
    ray_angles = []
    for heldout_index in range(len(placements)):
        training = [
            index
            for index in range(len(placements))
            if index != heldout_index
        ]
        _, heldout_target_from_anchor = _stereo_calibrate(
            manifest.board,
            [anchor_detections[index] for index in training],
            [target_detections[index] for index in training],
            [assignments[index] for index in training],
            anchor_bundle,
            target_bundle,
        )
        projected = _project_board_to_target(
            board=manifest.board,
            anchor_detection=anchor_detections[heldout_index],
            anchor_bundle=anchor_bundle,
            target_bundle=target_bundle,
            target_from_anchor=heldout_target_from_anchor,
        )
        heldout_detection = target_detections[heldout_index]
        heldout_symmetry = min(
            grid_symmetries(manifest.board),
            key=lambda item: _rms_vectors(
                projected
                - heldout_detection.corners[
                    np.asarray(item.indices, dtype=int)
                ].reshape(-1, 2)
            ),
        )
        observed = heldout_detection.corners[
            np.asarray(heldout_symmetry.indices, dtype=int)
        ].reshape(-1, 2)
        holdout_errors.extend(
            np.linalg.norm(projected - observed, axis=1).tolist()
        )
        local_errors, local_angles = _triangulation_metrics(
            board=manifest.board,
            anchor_detection=anchor_detections[heldout_index],
            target_detection=heldout_detection,
            symmetry=heldout_symmetry,
            anchor_bundle=anchor_bundle,
            target_bundle=target_bundle,
            target_from_anchor=heldout_target_from_anchor,
        )
        triangulation_errors.extend(local_errors)
        ray_angles.extend(local_angles)

    return PairRecovery(
        target_view_id=target_view_id,
        symmetry_names=tuple(item.name for item in assignments),
        symmetry_indices=tuple(item.indices for item in assignments),
        orientation_diagnostics=orientation,
        target_from_anchor=target_from_anchor,
        stereo_rms_pixels=stereo_rms,
        transfer_errors_pixels=tuple(
            float(value) for value in transfer_errors
        ),
        holdout_errors_pixels=tuple(
            float(value) for value in holdout_errors
        ),
        holdout_triangulation_errors_percent=tuple(
            float(value) for value in triangulation_errors
        ),
        holdout_ray_angles_degrees=tuple(
            float(value) for value in ray_angles
        ),
        stability_rotation_degrees=tuple(stability_rotation),
        stability_translation_mm=tuple(stability_translation),
    )


def _floor_aligned_anchor_transform(
    bundle: CalibrationBundle,
) -> np.ndarray:
    plane = np.asarray(
        bundle.floor_plane.floor_plane,
        dtype=np.float64,
    )
    normal = plane[:3] / np.linalg.norm(plane[:3])
    z_axis = -normal
    x_hint = np.asarray([1.0, 0.0, 0.0])
    x_axis = x_hint - normal * float(np.dot(normal, x_hint))
    if np.linalg.norm(x_axis) < 1e-8:
        x_hint = np.asarray([0.0, 1.0, 0.0])
        x_axis = x_hint - normal * float(np.dot(normal, x_hint))
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.vstack([x_axis, y_axis, z_axis])
    closest_floor_point = -plane[3] * normal
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ closest_floor_point
    return transform


def transform_plane_to_room(
    floor_plane: Sequence[float],
    room_from_camera: np.ndarray,
) -> np.ndarray:
    """Transform camera-frame plane coefficients into a room frame."""

    plane = np.asarray(floor_plane, dtype=np.float64).reshape(4)
    normal = plane[:3]
    distance = float(plane[3])
    rotation = room_from_camera[:3, :3]
    translation = room_from_camera[:3, 3]
    room_normal = rotation @ normal
    room_distance = distance - float(np.dot(room_normal, translation))
    length = float(np.linalg.norm(room_normal))
    return np.asarray(
        [*(room_normal / length), room_distance / length],
        dtype=np.float64,
    )


def _summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(np.max(array)),
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _quality_report(
    *,
    manifest: SharedBoardExtrinsicsManifest,
    recoveries: Sequence[PairRecovery],
    floor_checks: Mapping[str, Mapping[str, float]],
) -> Tuple[str, Tuple[Mapping[str, Any], ...]]:
    limits = manifest.quality_limits
    checks = []

    def add(
        name: str,
        actual: float,
        limit: float,
        *,
        view_id: str,
        direction: str = "maximum",
    ) -> None:
        passed = actual <= limit if direction == "maximum" else actual >= limit
        checks.append(
            {
                "view_id": view_id,
                "name": name,
                "actual": float(actual),
                "limit": float(limit),
                "direction": direction,
                "passed": bool(passed),
            }
        )

    for recovery in recoveries:
        target = recovery.target_view_id
        add(
            "fixed_intrinsic_stereo_rms_pixels",
            recovery.stereo_rms_pixels,
            limits.maximum_stereo_rms_pixels,
            view_id=target,
        )
        add(
            "heldout_corner_transfer_p90_pixels",
            float(np.percentile(recovery.holdout_errors_pixels, 90)),
            limits.maximum_holdout_p90_pixels,
            view_id=target,
        )
        add(
            "placement_transform_p90_rotation_degrees",
            float(
                np.percentile(
                    recovery.stability_rotation_degrees,
                    90,
                )
            ),
            limits.maximum_transform_p90_rotation_degrees,
            view_id=target,
        )
        add(
            "placement_transform_p90_translation_mm",
            float(
                np.percentile(
                    recovery.stability_translation_mm,
                    90,
                )
            ),
            limits.maximum_transform_p90_translation_mm,
            view_id=target,
        )
        add(
            "triangulated_board_p90_error_percent",
            float(
                np.percentile(
                    recovery.holdout_triangulation_errors_percent,
                    90,
                )
            ),
            limits.maximum_triangulated_board_p90_error_percent,
            view_id=target,
        )
        add(
            "orientation_minimum_next_margin_pixels",
            float(
                recovery.orientation_diagnostics[
                    "minimum_next_margin_pixels"
                ]
            ),
            limits.minimum_orientation_margin_pixels,
            view_id=target,
            direction="minimum",
        )
    for view_id, floor in floor_checks.items():
        add(
            "floor_normal_angle_degrees",
            floor["normal_angle_degrees"],
            limits.maximum_floor_normal_angle_degrees,
            view_id=view_id,
        )
        add(
            "floor_offset_mm",
            floor["offset_mm"],
            limits.maximum_floor_offset_mm,
            view_id=view_id,
        )
    status = "pass" if all(check["passed"] for check in checks) else "fail"
    return status, tuple(checks)


def calibrate_extrinsics_from_manifest(
    manifest_path: Path | str,
) -> ExtrinsicCalibrationRun:
    """Recover and validate fixed view-to-room transforms from shared footage."""

    manifest = load_extrinsics_manifest(manifest_path)
    bundles = _bundle_geometry(manifest)
    _, groups, metadata = _scan_views(manifest, bundles)
    placements = _shared_placements(manifest, groups)

    recoveries = []
    for view in manifest.views:
        if view.view_id == manifest.anchor_view_id:
            continue
        recoveries.append(
            _recover_pair(
                manifest=manifest,
                placements=placements,
                bundles=bundles,
                target_view_id=view.view_id,
            )
        )

    if manifest.room_frame_mode == "floor_aligned_anchor":
        room_from_anchor = _floor_aligned_anchor_transform(
            bundles[manifest.anchor_view_id]
        )
    else:
        room_from_anchor = np.eye(4, dtype=np.float64)
    room_from_camera = {
        manifest.anchor_view_id: room_from_anchor,
    }
    for recovery in recoveries:
        room_from_camera[recovery.target_view_id] = (
            room_from_anchor @ np.linalg.inv(recovery.target_from_anchor)
        )

    anchor_plane = transform_plane_to_room(
        bundles[manifest.anchor_view_id].floor_plane.floor_plane,
        room_from_camera[manifest.anchor_view_id],
    )
    floor_checks: Dict[str, Dict[str, float]] = {}
    transformed_planes = {}
    for view in manifest.views:
        transformed = transform_plane_to_room(
            bundles[view.view_id].floor_plane.floor_plane,
            room_from_camera[view.view_id],
        )
        if float(np.dot(transformed[:3], anchor_plane[:3])) < 0.0:
            transformed = -transformed
        angle = math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(transformed[:3], anchor_plane[:3]),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        transformed_planes[view.view_id] = transformed.tolist()
        floor_checks[view.view_id] = {
            "normal_angle_degrees": float(angle),
            "offset_mm": abs(float(transformed[3] - anchor_plane[3])),
        }

    status, quality_checks = _quality_report(
        manifest=manifest,
        recoveries=recoveries,
        floor_checks=floor_checks,
    )
    manifest_identity = source_identity(manifest.source_path)
    video_identities = {
        view.view_id: source_identity(view.video_path)
        for view in manifest.views
    }
    views_by_id = {view.view_id: view for view in manifest.views}
    recovery_by_id = {
        recovery.target_view_id: recovery for recovery in recoveries
    }
    registrations = []
    for view in manifest.views:
        bundle = bundles[view.view_id]
        recovery = recovery_by_id.get(view.view_id)
        registrations.append(
            ViewRegistration(
                view_id=view.view_id,
                camera_id=bundle.camera_id,
                source_coordinate_frame=bundle.floor_plane.coordinate_frame,
                source_floor_calibration_sha256=bundle.floor_plane.sha256,
                room_coordinate_frame=manifest.room_coordinate_frame,
                units=bundle.floor_plane.units,
                transform_to_room=tuple(
                    tuple(float(item) for item in row)
                    for row in room_from_camera[view.view_id]
                ),
                provenance={
                    "method": (
                        "shared_stationary_chessboard_fixed_intrinsics_v1"
                    ),
                    "rig_id": manifest.rig_id,
                    "anchor_view_id": manifest.anchor_view_id,
                    "room_frame_mode": manifest.room_frame_mode,
                    "manifest_sha256": manifest_identity["sha256"],
                    "video_sha256": video_identities[
                        view.view_id
                    ]["sha256"],
                    "intrinsic_sha256": bundle.intrinsics.sha256,
                    "shared_placement_count": len(placements),
                    "pair_stereo_rms_pixels": (
                        recovery.stereo_rms_pixels
                        if recovery is not None
                        else 0.0
                    ),
                    "validation_status": status,
                    "validation_scope": "planar_floor_only",
                    "volumetric_validated": False,
                },
            )
        )
    registration = RoomRegistration(
        room_coordinate_frame=manifest.room_coordinate_frame,
        units="mm",
        views=tuple(registrations),
    )

    pair_reports = {}
    for recovery in recoveries:
        pair_reports[recovery.target_view_id] = {
            "target_view_id": recovery.target_view_id,
            "transform_direction": "target_from_anchor",
            "target_from_anchor": recovery.target_from_anchor.tolist(),
            "anchor_from_target": np.linalg.inv(
                recovery.target_from_anchor
            ).tolist(),
            "fixed_intrinsic_stereo_rms_pixels": (
                recovery.stereo_rms_pixels
            ),
            "orientation": dict(recovery.orientation_diagnostics),
            "selected_symmetry_by_placement": [
                {
                    "placement_id": placement.placement_id,
                    "name": name,
                    "indices": list(indices),
                }
                for placement, name, indices in zip(
                    placements,
                    recovery.symmetry_names,
                    recovery.symmetry_indices,
                )
            ],
            "in_sample_corner_transfer_pixels": _summary(
                recovery.transfer_errors_pixels
            ),
            "leave_one_placement_out_corner_transfer_pixels": _summary(
                recovery.holdout_errors_pixels
            ),
            "leave_one_placement_out_triangulated_board_error_percent": (
                _summary(
                    recovery.holdout_triangulation_errors_percent
                )
            ),
            "leave_one_placement_out_ray_angle_degrees": _summary(
                recovery.holdout_ray_angles_degrees
            ),
            "per_placement_transform_rotation_degrees": _summary(
                recovery.stability_rotation_degrees
            ),
            "per_placement_transform_translation_mm": _summary(
                recovery.stability_translation_mm
            ),
        }

    observation_rows = []
    selected_by_view: Dict[str, Tuple[BoardDetection, ...]] = {}
    for view in manifest.views:
        selected = tuple(
            placement.detections[view.view_id]
            for placement in placements
        )
        selected_by_view[view.view_id] = selected
        for placement, detection in zip(placements, selected):
            recovery = recovery_by_id.get(view.view_id)
            if recovery is None:
                symmetry_name = "anchor_raw_order"
            else:
                symmetry_name = recovery.symmetry_names[
                    placement.placement_id - 1
                ]
            observation_rows.append(
                {
                    "placement_id": placement.placement_id,
                    "view_id": view.view_id,
                    "camera_id": bundles[view.view_id].camera_id,
                    "frame_index": detection.frame_index,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "common_timestamp_seconds": _common_time(
                        detection,
                        views_by_id[view.view_id],
                    ),
                    "maximum_shared_time_delta_seconds": (
                        placement.maximum_time_delta_seconds
                    ),
                    "center_x_pixels": float(detection.center[0]),
                    "center_y_pixels": float(detection.center[1]),
                    "sharpness": detection.sharpness,
                    "corner_symmetry_to_anchor": symmetry_name,
                }
            )

    registration_document = registration.to_dict()
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "kind": EXTRINSICS_REPORT_KIND,
        "status": status,
        "rig_id": manifest.rig_id,
        "anchor_view_id": manifest.anchor_view_id,
        "room_coordinate_frame": manifest.room_coordinate_frame,
        "room_frame_mode": manifest.room_frame_mode,
        "transform_convention": (
            "transform_to_room maps homogeneous camera coordinates into the "
            "declared room coordinate frame"
        ),
        "manifest": {
            **manifest_identity,
            "kind": EXTRINSICS_INPUT_KIND,
        },
        "board": manifest.board.to_dict(),
        "shared_placement_count": len(placements),
        "minimum_required_shared_placements": (
            manifest.minimum_shared_placements
        ),
        "synchronization": {
            "method": (
                "reported_frame_index_divided_by_reported_fps_plus_explicit_"
                "per_view_offset"
            ),
            "dynamic_synchronization_validated": False,
            "maximum_allowed_time_delta_seconds": (
                manifest.time_tolerance_seconds
            ),
            "maximum_selected_time_delta_seconds": max(
                placement.maximum_time_delta_seconds
                for placement in placements
            ),
            "interpretation": (
                "The board is stationary at each selected placement, so small "
                "timestamp differences do not bias this fixed-rig recovery. "
                "This does not validate synchronization for moving subjects."
            ),
        },
        "views": {
            view.view_id: {
                "camera_id": bundles[view.view_id].camera_id,
                "video": video_identities[view.view_id],
                "calibration_bundle": str(view.bundle_path),
                "intrinsic_sha256": bundles[
                    view.view_id
                ].intrinsics.sha256,
                "floor_sha256": bundles[
                    view.view_id
                ].floor_plane.sha256,
                "input_rotation": bundles[
                    view.view_id
                ].input_rotation.value,
                "time_offset_seconds": view.time_offset_seconds,
                "sampling": metadata[view.view_id].to_dict(),
                "transform_to_room": room_from_camera[
                    view.view_id
                ].tolist(),
                "transformed_floor_plane": transformed_planes[
                    view.view_id
                ],
                "floor_agreement_with_anchor": floor_checks[
                    view.view_id
                ],
            }
            for view in manifest.views
        },
        "pair_recoveries": pair_reports,
        "quality_limits": manifest.quality_limits.to_dict(),
        "quality_checks": list(quality_checks),
        "room_registration_document_sha256": _canonical_sha256(
            registration_document
        ),
        "validation_scope": "planar_floor_only",
        "volumetric_validated": False,
        "supported_now": [
            "shared-floor cross-view registration and fusion",
            "provisional multi-view triangulation with explicit uncertainty",
        ],
        "not_yet_demonstrated": [
            "volumetric 3D accuracy away from the floor plane",
            "dynamic inter-camera synchronization",
        ],
        "interpretation": (
            "A pass establishes metric multi-view geometry on the observed "
            "floor plane. Because every shared target placement is planar, it "
            "does not by itself certify volumetric skeleton accuracy."
        ),
    }
    return ExtrinsicCalibrationRun(
        room_registration=registration,
        report=report,
        observations=tuple(observation_rows),
        selected_detections_by_view=selected_by_view,
        manifest=manifest,
        bundles_by_view=bundles,
    )


__all__ = [
    "EXTRINSICS_INPUT_KIND",
    "EXTRINSICS_INPUT_SCHEMA_VERSION",
    "EXTRINSICS_REPORT_KIND",
    "ExtrinsicCalibrationRun",
    "ExtrinsicQualityLimits",
    "ExtrinsicViewInput",
    "GridSymmetry",
    "SharedBoardExtrinsicsManifest",
    "SharedPlacement",
    "calibrate_extrinsics_from_manifest",
    "grid_symmetries",
    "load_extrinsics_manifest",
    "transform_plane_to_room",
]
