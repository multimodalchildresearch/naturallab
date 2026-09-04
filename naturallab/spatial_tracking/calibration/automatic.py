"""Click-free camera, floor-plane, and calibration-verification workflows.

The public functions in this module operate on one explicitly supplied video.
They never select a file from a directory, open an interactive window, or
apply an empirical distance correction.  Chessboard dimensions are expressed
as *internal corners*, matching OpenCV's calibration APIs.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .artifacts import (
    CalibrationArtifactError,
    CalibrationBundle,
    FloorPlaneCalibrationArtifact,
    ImageSize,
    InputRotation,
    IntrinsicCalibrationArtifact,
)


class AutomaticCalibrationError(RuntimeError):
    """Raised when automatic calibration cannot produce a defensible result."""


def _finite_number(
    value: Any,
    name: str,
    *,
    minimum: Optional[float] = None,
    minimum_inclusive: bool = True,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None:
        invalid = (
            number < minimum
            if minimum_inclusive
            else number <= minimum
        )
        if invalid:
            comparator = "at least" if minimum_inclusive else "greater than"
            raise ValueError(f"{name} must be {comparator} {minimum}")
    return number


def _integer(
    value: Any,
    name: str,
    *,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class BoardSpec:
    """Metric chessboard definition using OpenCV internal-corner counts."""

    internal_columns: int
    internal_rows: int
    square_size_mm: float

    def __post_init__(self) -> None:
        for name, value in (
            ("internal_columns", self.internal_columns),
            ("internal_rows", self.internal_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer of at least 2")
        if (
            isinstance(self.square_size_mm, bool)
            or not isinstance(self.square_size_mm, (int, float))
            or not math.isfinite(float(self.square_size_mm))
            or float(self.square_size_mm) <= 0.0
        ):
            raise ValueError("square_size_mm must be a finite positive number")
        object.__setattr__(self, "square_size_mm", float(self.square_size_mm))

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.internal_columns, self.internal_rows)

    @property
    def corner_count(self) -> int:
        return self.internal_columns * self.internal_rows

    def object_points(self) -> np.ndarray:
        points = np.zeros((self.corner_count, 3), dtype=np.float32)
        points[:, :2] = (
            np.mgrid[
                0 : self.internal_columns,
                0 : self.internal_rows,
            ]
            .T.reshape(-1, 2)
            .astype(np.float32)
            * self.square_size_mm
        )
        return points

    def to_dict(self) -> Dict[str, Any]:
        return {
            "internal_columns": self.internal_columns,
            "internal_rows": self.internal_rows,
            "square_size_mm": self.square_size_mm,
        }


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    reported_frame_count: int
    decoded_frame_count: int
    sampled_frame_count: int
    sample_step_frames: int
    image_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fps": self.fps,
            "reported_frame_count": self.reported_frame_count,
            "decoded_frame_count": self.decoded_frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "sample_step_frames": self.sample_step_frames,
            "image_size": {
                "width": self.image_size[0],
                "height": self.image_size[1],
            },
        }


@dataclass(frozen=True)
class BoardDetection:
    frame_index: int
    timestamp_seconds: float
    corners: np.ndarray
    center: np.ndarray
    sharpness: float
    feature: np.ndarray

    def metadata(self) -> Dict[str, Any]:
        xy = self.corners.reshape(-1, 2)
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "center_x": float(self.center[0]),
            "center_y": float(self.center[1]),
            "sharpness": self.sharpness,
            "minimum_x": float(np.min(xy[:, 0])),
            "maximum_x": float(np.max(xy[:, 0])),
            "minimum_y": float(np.min(xy[:, 1])),
            "maximum_y": float(np.max(xy[:, 1])),
        }


@dataclass(frozen=True)
class BoardPose:
    detection: BoardDetection
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    rotation_matrix: np.ndarray
    plateau_start_frame: int
    plateau_end_frame: int
    plateau_sample_count: int
    reprojection_rms_pixels: float

    def metadata(self) -> Dict[str, Any]:
        return {
            **self.detection.metadata(),
            "plateau_start_frame": self.plateau_start_frame,
            "plateau_end_frame": self.plateau_end_frame,
            "plateau_sample_count": self.plateau_sample_count,
            "reprojection_rms_pixels": self.reprojection_rms_pixels,
            "rotation_vector": self.rotation_vector.reshape(-1).tolist(),
            "translation_vector_mm": (
                self.translation_vector.reshape(-1).tolist()
            ),
        }


@dataclass(frozen=True)
class IntrinsicCalibrationRun:
    artifact: IntrinsicCalibrationArtifact
    report: Mapping[str, Any]
    selected_detections: Tuple[BoardDetection, ...]


@dataclass(frozen=True)
class FloorCalibrationRun:
    artifact: FloorPlaneCalibrationArtifact
    report: Mapping[str, Any]
    selected_poses: Tuple[BoardPose, ...]
    internal_measurements: Tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class VerificationRun:
    report: Mapping[str, Any]
    measurements: Tuple[Mapping[str, Any], ...]
    selected_detections: Tuple[BoardDetection, ...]


def rotate_frame(
    frame: np.ndarray,
    rotation: InputRotation | str,
) -> np.ndarray:
    """Apply one artifact-compatible right-angle input rotation."""

    rotation_value = (
        rotation if isinstance(rotation, InputRotation) else InputRotation(rotation)
    )
    if rotation_value is InputRotation.CLOCKWISE_90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation_value is InputRotation.ROTATE_180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation_value is InputRotation.COUNTERCLOCKWISE_90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def detect_chessboard_corners(
    gray: np.ndarray,
    board: BoardSpec,
) -> Optional[np.ndarray]:
    """Detect at reduced resolution and refine on the original image."""

    if gray.ndim != 2:
        raise ValueError("gray must be a two-dimensional grayscale image")
    scale = 0.5 if min(gray.shape[:2]) >= 720 else 1.0
    if scale == 1.0:
        screened = gray
    else:
        screened = cv2.resize(
            gray,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(
        screened,
        board.pattern_size,
        classic_flags,
    )
    if not found and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = (
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
        found, corners = cv2.findChessboardCornersSB(
            screened,
            board.pattern_size,
            sb_flags,
        )
    if not found or corners is None:
        return None

    corners = np.asarray(corners, dtype=np.float32)
    scale_x = screened.shape[1] / gray.shape[1]
    scale_y = screened.shape[0] / gray.shape[0]
    corners[..., 0] /= scale_x
    corners[..., 1] /= scale_y
    return cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        ),
    )


def _sharpness(gray: np.ndarray, corners: np.ndarray) -> float:
    xy = corners.reshape(-1, 2)
    x, y, width, height = cv2.boundingRect(xy.astype(np.float32))
    margin = 20
    crop = gray[
        max(0, y - margin) : min(gray.shape[0], y + height + margin),
        max(0, x - margin) : min(gray.shape[1], x + width + margin),
    ]
    if not crop.size:
        return 0.0
    return float(cv2.Laplacian(crop, cv2.CV_64F).var())


def _detection_feature(
    corners: np.ndarray,
    image_size: Tuple[int, int],
    board: BoardSpec,
) -> np.ndarray:
    width, height = image_size
    xy = corners.reshape(-1, 2)
    center = xy.mean(axis=0)
    hull_area = float(
        cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))
    )
    row_vector = xy[board.internal_columns - 1] - xy[0]
    angle = math.atan2(float(row_vector[1]), float(row_vector[0]))
    x_span = max(float(np.ptp(xy[:, 0])), 1e-6)
    y_span = max(float(np.ptp(xy[:, 1])), 1e-6)
    return np.asarray(
        [
            center[0] / width,
            center[1] / height,
            math.log(max(hull_area / (width * height), 1e-9)),
            math.cos(angle),
            math.sin(angle),
            math.log(x_span / y_span),
        ],
        dtype=np.float64,
    )


def intrinsic_view_geometry(
    detections: Sequence[BoardDetection],
    *,
    image_size: Tuple[int, int],
    board: BoardSpec,
    perspective_threshold: float,
) -> Dict[str, Any]:
    """Summarize calibration-view coverage and out-of-plane perspective.

    Reprojection RMS alone cannot identify a degenerate planar calibration:
    translated, fronto-parallel views can fit almost perfectly while producing
    incorrect focal lengths and distortion.  Edge-scale changes are measured
    directly from the detected grid and therefore provide a calibration-model
    independent signal that the board was tilted around both image axes.
    """

    if not detections:
        raise ValueError("at least one detection is required")
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError("image_size must contain positive dimensions")
    if not math.isfinite(float(perspective_threshold)) or (
        perspective_threshold < 0.0
    ):
        raise ValueError("perspective_threshold must be finite and nonnegative")

    centers = np.vstack([detection.center for detection in detections])
    areas = []
    row_edge_changes = []
    column_edge_changes = []
    per_view = []
    for detection in detections:
        grid = detection.corners.reshape(
            board.internal_rows,
            board.internal_columns,
            2,
        )
        top = float(np.linalg.norm(grid[0, -1] - grid[0, 0]))
        bottom = float(np.linalg.norm(grid[-1, -1] - grid[-1, 0]))
        left = float(np.linalg.norm(grid[-1, 0] - grid[0, 0]))
        right = float(np.linalg.norm(grid[-1, -1] - grid[0, -1]))
        if min(top, bottom, left, right) <= 0.0:
            raise AutomaticCalibrationError(
                f"frame {detection.frame_index} contains a collapsed board edge"
            )
        row_change = abs(math.log(top / bottom))
        column_change = abs(math.log(left / right))
        xy = detection.corners.reshape(-1, 2)
        area_fraction = float(
            cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))
            / (width * height)
        )
        areas.append(area_fraction)
        row_edge_changes.append(row_change)
        column_edge_changes.append(column_change)
        per_view.append(
            {
                "frame_index": detection.frame_index,
                "board_area_fraction": area_fraction,
                "row_edge_scale_change": row_change,
                "column_edge_scale_change": column_change,
                "has_out_of_plane_perspective": (
                    max(row_change, column_change) >= perspective_threshold
                ),
            }
        )

    minimum_area = float(np.min(areas))
    maximum_area = float(np.max(areas))
    return {
        "center_span_fraction_x": float(np.ptp(centers[:, 0]) / width),
        "center_span_fraction_y": float(np.ptp(centers[:, 1]) / height),
        "minimum_board_area_fraction": minimum_area,
        "maximum_board_area_fraction": maximum_area,
        "board_linear_scale_ratio": float(
            math.sqrt(maximum_area / minimum_area)
            if minimum_area > 0.0
            else math.inf
        ),
        "maximum_row_edge_scale_change": float(
            np.max(row_edge_changes)
        ),
        "maximum_column_edge_scale_change": float(
            np.max(column_edge_changes)
        ),
        "perspective_threshold": float(perspective_threshold),
        "tilted_view_count": sum(
            bool(item["has_out_of_plane_perspective"]) for item in per_view
        ),
        "views": per_view,
    }


def _validate_intrinsic_view_geometry(
    geometry: Mapping[str, Any],
    *,
    minimum_center_span_fraction: float,
    minimum_scale_ratio: float,
    minimum_perspective_change: float,
    minimum_tilted_views: int,
) -> None:
    failures = []
    for axis in ("x", "y"):
        actual = float(geometry[f"center_span_fraction_{axis}"])
        if actual < minimum_center_span_fraction:
            failures.append(
                f"board-centre {axis}-coverage {actual:.3f} is below "
                f"{minimum_center_span_fraction:.3f}"
            )
    scale_ratio = float(geometry["board_linear_scale_ratio"])
    if scale_ratio < minimum_scale_ratio:
        failures.append(
            f"near/far linear scale ratio {scale_ratio:.3f} is below "
            f"{minimum_scale_ratio:.3f}"
        )
    row_change = float(geometry["maximum_row_edge_scale_change"])
    if row_change < minimum_perspective_change:
        failures.append(
            f"maximum row-edge perspective change {row_change:.3f} is below "
            f"{minimum_perspective_change:.3f}"
        )
    column_change = float(geometry["maximum_column_edge_scale_change"])
    if column_change < minimum_perspective_change:
        failures.append(
            "maximum column-edge perspective change "
            f"{column_change:.3f} is below {minimum_perspective_change:.3f}"
        )
    tilted_count = int(geometry["tilted_view_count"])
    if tilted_count < minimum_tilted_views:
        failures.append(
            f"only {tilted_count} views contain measurable out-of-plane "
            f"perspective; at least {minimum_tilted_views} are required"
        )
    if failures:
        raise AutomaticCalibrationError(
            "intrinsic recording is geometrically under-diverse: "
            + "; ".join(failures)
            + ". Re-record with the board near and far, across both image "
            "axes, and visibly tilted around both horizontal and vertical axes."
        )


def _validate_intrinsic_parameters(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Reject non-finite or plainly implausible pinhole calibration values."""

    matrix = np.asarray(camera_matrix, dtype=float)
    coefficients = np.asarray(distortion, dtype=float).reshape(-1)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise AutomaticCalibrationError(
            "intrinsic calibration produced a non-finite 3x3 camera matrix"
        )
    if not np.all(np.isfinite(coefficients)):
        raise AutomaticCalibrationError(
            "intrinsic calibration produced non-finite distortion coefficients"
        )
    width, height = image_size
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    cx = float(matrix[0, 2])
    cy = float(matrix[1, 2])
    maximum_dimension = float(max(width, height))
    failures = []
    if fx <= 0.1 * maximum_dimension or fx >= 20.0 * maximum_dimension:
        failures.append(f"fx={fx:.3f}px is outside a broad plausible range")
    if fy <= 0.1 * maximum_dimension or fy >= 20.0 * maximum_dimension:
        failures.append(f"fy={fy:.3f}px is outside a broad plausible range")
    aspect_ratio = fx / fy if fy else math.inf
    if not 0.5 <= aspect_ratio <= 2.0:
        failures.append(
            f"focal aspect ratio fx/fy={aspect_ratio:.3f} is implausible"
        )
    if not -0.25 * width <= cx <= 1.25 * width:
        failures.append(f"principal point cx={cx:.3f}px is far outside the image")
    if not -0.25 * height <= cy <= 1.25 * height:
        failures.append(f"principal point cy={cy:.3f}px is far outside the image")
    if failures:
        raise AutomaticCalibrationError(
            "intrinsic parameters failed plausibility checks: "
            + "; ".join(failures)
        )
    return {
        "fx_pixels": fx,
        "fy_pixels": fy,
        "principal_point_x_pixels": cx,
        "principal_point_y_pixels": cy,
        "focal_aspect_ratio": aspect_ratio,
        "distortion_coefficient_count": int(coefficients.size),
        "maximum_absolute_distortion_coefficient": (
            float(np.max(np.abs(coefficients))) if coefficients.size else 0.0
        ),
    }


def scan_calibration_video(
    video: Path | str,
    *,
    board: BoardSpec,
    input_rotation: InputRotation = InputRotation.NONE,
    sample_seconds: float = 1.0,
) -> Tuple[Tuple[BoardDetection, ...], VideoMetadata]:
    """Screen one explicit video for automatic chessboard detections."""

    source = Path(video).expanduser().resolve()
    if not source.is_file():
        raise AutomaticCalibrationError(f"video does not exist: {source}")
    if (
        isinstance(sample_seconds, bool)
        or not isinstance(sample_seconds, (int, float))
        or not math.isfinite(float(sample_seconds))
        or float(sample_seconds) <= 0.0
    ):
        raise ValueError("sample_seconds must be a finite positive number")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise AutomaticCalibrationError(f"could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0.0:
        capture.release()
        raise AutomaticCalibrationError(
            f"video reports an invalid frame rate: {fps}"
        )
    reported_frame_count_value = float(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    if (
        not math.isfinite(reported_frame_count_value)
        or reported_frame_count_value < 1.0
    ):
        capture.release()
        raise AutomaticCalibrationError(
            "video does not report a trustworthy positive frame count; "
            "use a complete, finalized recording"
        )
    reported_frame_count = int(reported_frame_count_value)
    sample_step = max(1, round(fps * float(sample_seconds)))

    detections = []
    frame_index = 0
    sampled_count = 0
    image_size: Optional[Tuple[int, int]] = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % sample_step:
            frame_index += 1
            continue
        sampled_count += 1
        frame = rotate_frame(frame, input_rotation)
        current_size = (int(frame.shape[1]), int(frame.shape[0]))
        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            capture.release()
            raise AutomaticCalibrationError(
                "video frame dimensions changed during decoding"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = detect_chessboard_corners(gray, board)
        if corners is not None:
            xy = corners.reshape(-1, 2)
            detections.append(
                BoardDetection(
                    frame_index=frame_index,
                    timestamp_seconds=frame_index / fps,
                    corners=corners,
                    center=xy.mean(axis=0),
                    sharpness=_sharpness(gray, corners),
                    feature=_detection_feature(corners, current_size, board),
                )
            )
        frame_index += 1
    capture.release()

    if image_size is None:
        raise AutomaticCalibrationError(
            f"no frames could be decoded from video: {source}"
        )
    # Some OpenCV/container combinations differ by one terminal frame.  A
    # larger shortfall means calibration would silently use only a prefix of a
    # damaged or incompletely copied recording, which is not an acceptable
    # scientific input contract.
    if frame_index + 1 < reported_frame_count:
        raise AutomaticCalibrationError(
            "video decoding ended early after "
            f"{frame_index} of {reported_frame_count} reported frames; "
            "use a complete, finalized recording"
        )
    return (
        tuple(detections),
        VideoMetadata(
            fps=fps,
            reported_frame_count=reported_frame_count,
            decoded_frame_count=frame_index,
            sampled_frame_count=sampled_count,
            sample_step_frames=sample_step,
            image_size=image_size,
        ),
    )


def select_diverse_detections(
    detections: Sequence[BoardDetection],
    target_count: int,
) -> Tuple[int, ...]:
    """Select deterministic, spatially and perspectivally diverse views."""

    if isinstance(target_count, bool) or not isinstance(target_count, int):
        raise ValueError("target_count must be an integer")
    if target_count < 1:
        raise ValueError("target_count must be at least 1")
    if not detections:
        return ()

    features = np.vstack([detection.feature for detection in detections])
    median = np.median(features, axis=0)
    scale = (
        np.percentile(features, 90, axis=0)
        - np.percentile(features, 10, axis=0)
    )
    scale[scale < 1e-6] = 1.0
    standardized = (features - median) / scale
    if standardized.shape[1] >= 5:
        standardized[:, 3:5] *= 0.5

    sharpness = np.asarray(
        [detection.sharpness for detection in detections],
        dtype=float,
    )
    if len(sharpness) == 1:
        sharpness_rank = np.ones(1)
    else:
        sharpness_rank = (
            np.argsort(np.argsort(sharpness)).astype(float)
            / (len(sharpness) - 1)
        )
    first = int(
        np.argmax(
            np.linalg.norm(standardized, axis=1)
            + 0.15 * sharpness_rank
        )
    )
    chosen = [first]
    minimum_distance = np.linalg.norm(
        standardized - standardized[first],
        axis=1,
    )
    while len(chosen) < min(target_count, len(detections)):
        score = minimum_distance * (0.8 + 0.2 * sharpness_rank)
        score[chosen] = -1.0
        next_index = int(np.argmax(score))
        chosen.append(next_index)
        distance = np.linalg.norm(
            standardized - standardized[next_index],
            axis=1,
        )
        minimum_distance = np.minimum(minimum_distance, distance)
    return tuple(chosen)


def _calibrate_selected_views(
    detections: Sequence[BoardDetection],
    indices: Sequence[int],
    board: BoardSpec,
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    object_template = board.object_points()
    object_sets = [object_template.copy() for _ in indices]
    image_sets = [detections[index].corners for index in indices]
    rms, matrix, distortion, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            object_sets,
            image_sets,
            image_size,
            None,
            None,
        )
    )
    per_view_rms = []
    for object_points, image_points, rotation_vector, translation_vector in zip(
        object_sets,
        image_sets,
        rotation_vectors,
        translation_vectors,
    ):
        predicted, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            matrix,
            distortion,
        )
        residual = (
            image_points.reshape(-1, 2) - predicted.reshape(-1, 2)
        )
        per_view_rms.append(
            float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        )
    return {
        "opencv_rms_pixels": float(rms),
        "camera_matrix": matrix,
        "dist_coeff": distortion,
        "per_view_rms_pixels": per_view_rms,
    }


def _error_summary(values: Iterable[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
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


def _evaluate_intrinsics(
    detections: Sequence[BoardDetection],
    indices: Iterable[int],
    board: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeff: np.ndarray,
) -> Dict[str, Any]:
    errors = []
    object_points = board.object_points()
    for index in indices:
        detection = detections[index]
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            detection.corners,
            camera_matrix,
            dist_coeff,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        predicted, _ = cv2.projectPoints(
            object_points,
            rotation_vector,
            translation_vector,
            camera_matrix,
            dist_coeff,
        )
        residual = (
            detection.corners.reshape(-1, 2)
            - predicted.reshape(-1, 2)
        )
        errors.append(
            float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        )
    return _error_summary(errors)


def calibrate_intrinsics_from_video(
    video: Path | str,
    *,
    camera_id: str,
    board: BoardSpec,
    input_rotation: InputRotation = InputRotation.NONE,
    coordinate_frame: Optional[str] = None,
    sample_seconds: float = 1.0,
    target_views: int = 28,
    minimum_views: int = 16,
    maximum_view_rms_pixels: float = 3.0,
    minimum_center_span_fraction: float = 0.20,
    minimum_scale_ratio: float = 1.20,
    minimum_perspective_change: float = 0.02,
    minimum_tilted_views: int = 4,
) -> IntrinsicCalibrationRun:
    """Automatically estimate one camera's canonical intrinsic artifact."""

    _integer(minimum_views, "minimum_views", minimum=5)
    _integer(target_views, "target_views", minimum=minimum_views)
    if target_views < minimum_views:
        raise ValueError("target_views must be at least minimum_views")
    maximum_view_rms_pixels = _finite_number(
        maximum_view_rms_pixels,
        "maximum_view_rms_pixels",
        minimum=0.0,
        minimum_inclusive=False,
    )
    minimum_center_span_fraction = _finite_number(
        minimum_center_span_fraction,
        "minimum_center_span_fraction",
        minimum=0.0,
    )
    if minimum_center_span_fraction > 1.0:
        raise ValueError("minimum_center_span_fraction cannot exceed 1")
    minimum_scale_ratio = _finite_number(
        minimum_scale_ratio,
        "minimum_scale_ratio",
        minimum=1.0,
    )
    minimum_perspective_change = _finite_number(
        minimum_perspective_change,
        "minimum_perspective_change",
        minimum=0.0,
    )
    _integer(
        minimum_tilted_views,
        "minimum_tilted_views",
        minimum=1,
    )
    if minimum_tilted_views > minimum_views:
        raise ValueError(
            "minimum_tilted_views cannot exceed minimum_views"
        )

    source = Path(video).expanduser().resolve()
    detections, metadata = scan_calibration_video(
        source,
        board=board,
        input_rotation=input_rotation,
        sample_seconds=sample_seconds,
    )
    if len(detections) < minimum_views:
        raise AutomaticCalibrationError(
            f"only {len(detections)} chessboard detections were found; "
            f"at least {minimum_views} are required"
        )

    retained = list(
        select_diverse_detections(detections, target_views)
    )
    removals = []
    while True:
        result = _calibrate_selected_views(
            detections,
            retained,
            board,
            metadata.image_size,
        )
        errors = np.asarray(result["per_view_rms_pixels"], dtype=float)
        worst_local_index = int(np.argmax(errors))
        worst_error = float(errors[worst_local_index])
        if worst_error <= maximum_view_rms_pixels:
            break
        if len(retained) <= minimum_views:
            raise AutomaticCalibrationError(
                "intrinsic calibration still contains a "
                f"{worst_error:.3f}-pixel view at the minimum retained-view "
                f"count ({minimum_views}); record a sharper, more diverse video "
                "or explicitly review a different RMS limit"
            )
        removed = retained.pop(worst_local_index)
        removals.append(
            {
                "frame_index": detections[removed].frame_index,
                "per_view_rms_pixels": worst_error,
                "limit_pixels": float(maximum_view_rms_pixels),
            }
        )

    selected = tuple(detections[index] for index in retained)
    view_geometry = intrinsic_view_geometry(
        selected,
        image_size=metadata.image_size,
        board=board,
        perspective_threshold=minimum_perspective_change,
    )
    _validate_intrinsic_view_geometry(
        view_geometry,
        minimum_center_span_fraction=minimum_center_span_fraction,
        minimum_scale_ratio=minimum_scale_ratio,
        minimum_perspective_change=minimum_perspective_change,
        minimum_tilted_views=minimum_tilted_views,
    )
    parameter_summary = _validate_intrinsic_parameters(
        result["camera_matrix"],
        result["dist_coeff"],
        metadata.image_size,
    )

    coordinate_frame_value = (
        coordinate_frame
        if coordinate_frame is not None
        else f"camera/{camera_id}/opencv"
    )
    artifact = IntrinsicCalibrationArtifact(
        camera_id=camera_id,
        image_size=ImageSize(
            width=metadata.image_size[0],
            height=metadata.image_size[1],
        ),
        camera_matrix=result["camera_matrix"],
        dist_coeff=result["dist_coeff"],
        units="pixels",
        coordinate_frame=coordinate_frame_value,
        input_rotation=input_rotation,
    )
    retained_set = set(retained)
    holdout = (
        index for index in range(len(detections)) if index not in retained_set
    )
    report = {
        "schema_version": "1.0",
        "kind": "automatic_intrinsic_calibration_report",
        "camera_id": artifact.camera_id,
        "source_video_identity": source_identity(source),
        "board": board.to_dict(),
        "input_rotation": input_rotation.value,
        "sampling": {
            "sample_seconds": float(sample_seconds),
            **metadata.to_dict(),
        },
        "detected_view_count": len(detections),
        "initial_selected_view_count": min(target_views, len(detections)),
        "accepted_view_count": len(selected),
        "selected_views": [
            {
                **detection.metadata(),
                "per_view_rms_pixels": float(
                    result["per_view_rms_pixels"][local_index]
                ),
            }
            for local_index, detection in enumerate(selected)
        ],
        "rejected_views": removals,
        "opencv_rms_pixels": result["opencv_rms_pixels"],
        "view_geometry": view_geometry,
        "intrinsic_parameters": parameter_summary,
        "quality_limits": {
            "maximum_view_rms_pixels": maximum_view_rms_pixels,
            "minimum_center_span_fraction_per_axis": (
                minimum_center_span_fraction
            ),
            "minimum_linear_scale_ratio": minimum_scale_ratio,
            "minimum_perspective_change_per_axis": (
                minimum_perspective_change
            ),
            "minimum_tilted_views": minimum_tilted_views,
        },
        "holdout_corner_error_pixels": _evaluate_intrinsics(
            detections,
            holdout,
            board,
            result["camera_matrix"],
            result["dist_coeff"],
        ),
        "artifact_sha256": artifact.sha256,
    }
    return IntrinsicCalibrationRun(
        artifact=artifact,
        report=report,
        selected_detections=selected,
    )


def group_stationary_detections(
    detections: Sequence[BoardDetection],
    *,
    sample_step_frames: int,
    maximum_center_motion_pixels: float,
    minimum_samples: int,
) -> Tuple[Tuple[BoardDetection, ...], ...]:
    """Group consecutive detections where the board remains stationary."""

    _integer(sample_step_frames, "sample_step_frames", minimum=1)
    maximum_center_motion_pixels = _finite_number(
        maximum_center_motion_pixels,
        "maximum_center_motion_pixels",
        minimum=0.0,
    )
    _integer(minimum_samples, "minimum_samples", minimum=1)

    groups = []
    current = []
    # Tolerate one missed sampled detection without joining genuinely separate
    # placements.  Stationarity is still checked against the first detection
    # in the run, preventing slow chained drift.
    maximum_gap = max(1, int(math.ceil(sample_step_frames * 2.5)))
    for detection in detections:
        if not current:
            current = [detection]
            continue
        previous = current[-1]
        anchor = current[0]
        contiguous = (
            detection.frame_index - previous.frame_index <= maximum_gap
        )
        corner_displacement = np.linalg.norm(
            detection.corners.reshape(-1, 2)
            - anchor.corners.reshape(-1, 2),
            axis=1,
        )
        stationary = float(np.median(corner_displacement)) <= (
            maximum_center_motion_pixels
        )
        if contiguous and stationary:
            current.append(detection)
        else:
            if len(current) >= minimum_samples:
                groups.append(tuple(current))
            current = [detection]
    if len(current) >= minimum_samples:
        groups.append(tuple(current))
    return tuple(groups)


def choose_stationary_placements(
    groups: Sequence[Sequence[BoardDetection]],
    *,
    minimum_separation_pixels: float,
    maximum_placements: int,
) -> Tuple[Tuple[BoardDetection, Tuple[int, int, int]], ...]:
    """Choose sharp representatives with broad image-plane coverage."""

    minimum_separation_pixels = _finite_number(
        minimum_separation_pixels,
        "minimum_separation_pixels",
        minimum=0.0,
    )
    _integer(maximum_placements, "maximum_placements", minimum=1)
    representatives = [
        (
            max(group, key=lambda item: item.sharpness),
            (
                group[0].frame_index,
                group[-1].frame_index,
                len(group),
            ),
        )
        for group in groups
        if group
    ]
    if not representatives:
        return ()

    first_index = int(
        np.argmax(
            [representative[0].sharpness for representative in representatives]
        )
    )
    chosen_indices = [first_index]
    while len(chosen_indices) < min(
        maximum_placements,
        len(representatives),
    ):
        remaining = [
            index
            for index in range(len(representatives))
            if index not in chosen_indices
        ]
        scored = []
        for index in remaining:
            center = representatives[index][0].center
            minimum_distance = min(
                float(
                    np.linalg.norm(
                        center - representatives[chosen][0].center
                    )
                )
                for chosen in chosen_indices
            )
            scored.append((minimum_distance, index))
        best_distance, best_index = max(scored)
        if best_distance < minimum_separation_pixels:
            break
        chosen_indices.append(best_index)
    chosen = [representatives[index] for index in chosen_indices]
    chosen.sort(key=lambda item: item[0].frame_index)
    return tuple(chosen)


def estimate_board_pose(
    detection: BoardDetection,
    *,
    board: BoardSpec,
    intrinsics: IntrinsicCalibrationArtifact,
    plateau: Tuple[int, int, int],
    maximum_reprojection_rms_pixels: float = 3.0,
) -> BoardPose:
    """Estimate the complete board pose using iterative planar PnP."""

    maximum_reprojection_rms_pixels = _finite_number(
        maximum_reprojection_rms_pixels,
        "maximum_reprojection_rms_pixels",
        minimum=0.0,
        minimum_inclusive=False,
    )
    camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    dist_coeff = np.asarray(intrinsics.dist_coeff, dtype=np.float64)
    ok, rotation_vector, translation_vector = cv2.solvePnP(
        board.object_points(),
        detection.corners,
        camera_matrix,
        dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise AutomaticCalibrationError(
            f"solvePnP failed for frame {detection.frame_index}"
        )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    camera_points = (
        rotation_matrix @ board.object_points().T + translation_vector
    ).T
    if (
        not np.all(np.isfinite(camera_points))
        or np.any(camera_points[:, 2] <= 0.0)
    ):
        raise AutomaticCalibrationError(
            f"frame {detection.frame_index} produced a board pose behind the camera"
        )
    predicted, _ = cv2.projectPoints(
        board.object_points(),
        rotation_vector,
        translation_vector,
        np.asarray(intrinsics.camera_matrix, dtype=np.float64),
        np.asarray(intrinsics.dist_coeff, dtype=np.float64),
    )
    residual = (
        detection.corners.reshape(-1, 2) - predicted.reshape(-1, 2)
    )
    reprojection_rms = float(
        np.sqrt(np.mean(np.sum(residual * residual, axis=1)))
    )
    if reprojection_rms > maximum_reprojection_rms_pixels:
        raise AutomaticCalibrationError(
            f"frame {detection.frame_index} board-pose reprojection RMS "
            f"{reprojection_rms:.3f}px exceeds "
            f"{maximum_reprojection_rms_pixels:.3f}px"
        )
    return BoardPose(
        detection=detection,
        rotation_vector=rotation_vector,
        translation_vector=translation_vector,
        rotation_matrix=rotation_matrix,
        plateau_start_frame=plateau[0],
        plateau_end_frame=plateau[1],
        plateau_sample_count=plateau[2],
        reprojection_rms_pixels=reprojection_rms,
    )


def transformed_board_points(
    pose: BoardPose,
    board: BoardSpec,
) -> np.ndarray:
    """Transform the known board corners into the camera coordinate frame."""

    return (
        pose.rotation_matrix @ board.object_points().T
        + pose.translation_vector
    ).T


def fit_floor_plane(
    poses: Sequence[BoardPose],
    board: BoardSpec,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fit a metric plane to complete PnP-transformed board points."""

    if not poses:
        raise ValueError("at least one board pose is required")
    points = np.vstack(
        [transformed_board_points(pose, board) for pose in poses]
    ).astype(np.float64)
    centroid = np.mean(points, axis=0)
    _, _, right_vectors = np.linalg.svd(points - centroid)
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    distance = -float(np.dot(normal, centroid))
    # Canonicalize the otherwise arbitrary sign: the camera origin should lie
    # on the negative side of the floor plane.
    if distance > 0.0:
        normal = -normal
        distance = -distance
    residuals = points @ normal + distance
    return (
        np.asarray([*normal, distance], dtype=np.float64),
        {
            "point_count": int(points.shape[0]),
            "rms_mm": float(np.sqrt(np.mean(residuals * residuals))),
            "mean_absolute_mm": float(np.mean(np.abs(residuals))),
            "maximum_absolute_mm": float(np.max(np.abs(residuals))),
        },
    )


def floor_pose_consistency(
    poses: Sequence[BoardPose],
    board: BoardSpec,
    floor_plane: Sequence[float],
) -> Dict[str, Any]:
    """Describe how consistently individual board poses support one plane."""

    plane = np.asarray(floor_plane, dtype=np.float64)
    normal = plane[:3] / np.linalg.norm(plane[:3])
    angular_deviations = []
    centroid_offsets = []
    for pose in poses:
        pose_normal = np.asarray(pose.rotation_matrix[:, 2], dtype=float)
        pose_normal /= np.linalg.norm(pose_normal)
        cosine = float(np.clip(abs(np.dot(pose_normal, normal)), 0.0, 1.0))
        angular_deviations.append(math.degrees(math.acos(cosine)))
        centroid = np.mean(transformed_board_points(pose, board), axis=0)
        centroid_offsets.append(
            abs(float(np.dot(normal, centroid) + plane[3]))
        )
    return {
        "per_pose_normal_deviation_degrees": angular_deviations,
        "maximum_normal_deviation_degrees": float(
            np.max(angular_deviations)
        ),
        "per_pose_centroid_offset_mm": centroid_offsets,
        "maximum_centroid_offset_mm": float(np.max(centroid_offsets)),
        "centroid_offset_standard_deviation_mm": (
            float(np.std(centroid_offsets, ddof=1))
            if len(centroid_offsets) > 1
            else 0.0
        ),
    }


def project_image_points_to_plane(
    image_points: np.ndarray,
    *,
    intrinsics: IntrinsicCalibrationArtifact,
    floor_plane: Sequence[float],
) -> np.ndarray:
    """Intersect undistorted camera rays with a camera-frame floor plane."""

    plane = np.asarray(floor_plane, dtype=np.float64).reshape(-1)
    if plane.shape != (4,):
        raise ValueError("floor_plane must contain [a, b, c, d]")
    camera_matrix = np.asarray(intrinsics.camera_matrix, dtype=np.float64)
    dist_coeff = np.asarray(intrinsics.dist_coeff, dtype=np.float64)
    normalized = cv2.undistortPoints(
        np.asarray(image_points, dtype=np.float32),
        camera_matrix,
        dist_coeff,
    ).reshape(-1, 2)
    rays = np.column_stack(
        [normalized, np.ones(len(normalized), dtype=np.float64)]
    )
    denominator = rays @ plane[:3]
    if np.any(np.abs(denominator) < 1e-9):
        raise AutomaticCalibrationError(
            "a viewing ray is parallel to the fitted floor plane"
        )
    scale = -plane[3] / denominator
    if np.any(scale <= 0.0):
        raise AutomaticCalibrationError(
            "the floor plane intersects one or more rays behind the camera; "
            "check the input rotation and calibration pairing"
        )
    return rays * scale[:, None]


def boundary_edges(
    board: BoardSpec,
) -> Tuple[Tuple[str, int, int, float], ...]:
    columns = board.internal_columns
    rows = board.internal_rows
    return (
        ("top", 0, columns - 1, (columns - 1) * board.square_size_mm),
        (
            "bottom",
            (rows - 1) * columns,
            rows * columns - 1,
            (columns - 1) * board.square_size_mm,
        ),
        ("left", 0, (rows - 1) * columns, (rows - 1) * board.square_size_mm),
        (
            "right",
            columns - 1,
            rows * columns - 1,
            (rows - 1) * board.square_size_mm,
        ),
    )


def measure_board_on_plane(
    detection: BoardDetection,
    *,
    placement_id: int,
    board: BoardSpec,
    intrinsics: IntrinsicCalibrationArtifact,
    floor_plane: Sequence[float],
) -> Tuple[Mapping[str, Any], ...]:
    """Measure automatically detected board spans after floor projection."""

    floor_points = project_image_points_to_plane(
        detection.corners,
        intrinsics=intrinsics,
        floor_plane=floor_plane,
    )
    rows = []
    for edge_name, first, second, true_mm in boundary_edges(board):
        measured_mm = float(
            np.linalg.norm(floor_points[first] - floor_points[second])
        )
        absolute_error_mm = abs(measured_mm - true_mm)
        rows.append(
            {
                "placement_id": placement_id,
                "frame_index": detection.frame_index,
                "timestamp_seconds": detection.timestamp_seconds,
                "edge_name": edge_name,
                "corner_index_1": first,
                "corner_index_2": second,
                "known_distance_mm": float(true_mm),
                "measured_distance_mm": measured_mm,
                "absolute_error_mm": absolute_error_mm,
                "absolute_error_cm": absolute_error_mm / 10.0,
                "absolute_error_percent": absolute_error_mm / true_mm * 100.0,
            }
        )
    return tuple(rows)


def summarize_measurements(
    measurements: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not measurements:
        raise ValueError("at least one measurement is required")
    measured = np.asarray(
        [row["measured_distance_mm"] for row in measurements],
        dtype=float,
    )
    known = np.asarray(
        [row["known_distance_mm"] for row in measurements],
        dtype=float,
    )
    absolute_mm = np.asarray(
        [row["absolute_error_mm"] for row in measurements],
        dtype=float,
    )
    percentage = np.asarray(
        [row["absolute_error_percent"] for row in measurements],
        dtype=float,
    )
    placement_means = []
    for placement_id in sorted(
        {int(row["placement_id"]) for row in measurements}
    ):
        local = [
            float(row["absolute_error_percent"])
            for row in measurements
            if int(row["placement_id"]) == placement_id
        ]
        placement_means.append(float(np.mean(local)))
    return {
        "measurement_count": len(measurements),
        "placement_count": len(placement_means),
        "mean_known_distance_mm": float(np.mean(known)),
        "mean_measured_distance_mm": float(np.mean(measured)),
        "mean_absolute_error_mm": float(np.mean(absolute_mm)),
        "mean_absolute_error_cm": float(np.mean(absolute_mm)) / 10.0,
        "mean_absolute_error_percent": float(np.mean(percentage)),
        "median_absolute_error_percent": float(np.median(percentage)),
        "p90_absolute_error_percent": float(np.percentile(percentage, 90)),
        "maximum_absolute_error_percent": float(np.max(percentage)),
        "placement_mean_absolute_error_percent": {
            "mean": float(np.mean(placement_means)),
            "standard_deviation": (
                float(np.std(placement_means, ddof=1))
                if len(placement_means) > 1
                else 0.0
            ),
            "maximum": float(np.max(placement_means)),
            "values": placement_means,
        },
    }


def _load_yaml_mapping(path: Path | str) -> Tuple[Path, Mapping[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise AutomaticCalibrationError(
            f"calibration file does not exist: {source}"
        )
    try:
        with source.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:
        raise AutomaticCalibrationError(
            f"could not read calibration file {source}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise AutomaticCalibrationError(
            f"calibration file must contain a YAML mapping: {source}"
        )
    return source, data


def load_intrinsic_artifact(
    path: Path | str,
) -> IntrinsicCalibrationArtifact:
    """Load a canonical intrinsic artifact for automatic downstream stages."""

    source, data = _load_yaml_mapping(path)
    try:
        artifact = IntrinsicCalibrationArtifact.from_dict(
            data,
            warn_legacy=False,
        )
    except CalibrationArtifactError as exc:
        raise AutomaticCalibrationError(
            f"{source} is not a complete canonical intrinsic artifact: {exc}. "
            "Run the automatic intrinsic step first."
        ) from exc
    if artifact.legacy:
        raise AutomaticCalibrationError(
            f"{source} is a legacy intrinsic file without a complete geometry "
            "binding; run the automatic intrinsic step first"
        )
    return artifact


def load_calibration_bundle(
    intrinsic_path: Path | str,
    floor_path: Path | str,
) -> CalibrationBundle:
    intrinsics = load_intrinsic_artifact(intrinsic_path)
    source, data = _load_yaml_mapping(floor_path)
    try:
        floor = FloorPlaneCalibrationArtifact.from_dict(
            data,
            intrinsic=intrinsics,
            warn_legacy=False,
        )
        bundle = CalibrationBundle(intrinsics=intrinsics, floor_plane=floor)
    except CalibrationArtifactError as exc:
        raise AutomaticCalibrationError(
            f"{source} is not a compatible canonical floor artifact: {exc}"
        ) from exc
    if floor.legacy:
        raise AutomaticCalibrationError(
            f"{source} is a legacy floor file; run the automatic floor step first"
        )
    return bundle


def load_calibration_bundle_file(path: Path | str) -> CalibrationBundle:
    """Load one canonical intrinsic/floor bundle YAML."""

    source, data = _load_yaml_mapping(path)
    try:
        bundle = CalibrationBundle.from_dict(data, warn_legacy=False)
    except CalibrationArtifactError as exc:
        raise AutomaticCalibrationError(
            f"{source} is not a compatible canonical calibration bundle: {exc}"
        ) from exc
    if bundle.legacy:
        raise AutomaticCalibrationError(
            f"{source} contains legacy calibration data; rerun the automatic "
            "intrinsic and floor steps"
        )
    return bundle


def _selected_floor_poses(
    video: Path | str,
    *,
    intrinsics: IntrinsicCalibrationArtifact,
    board: BoardSpec,
    sample_seconds: float,
    stationary_distance_pixels: float,
    minimum_stationary_samples: int,
    minimum_separation_pixels: float,
    maximum_placements: int,
) -> Tuple[
    Tuple[BoardPose, ...],
    VideoMetadata,
    int,
    int,
    Tuple[Mapping[str, Any], ...],
]:
    stationary_distance_pixels = _finite_number(
        stationary_distance_pixels,
        "stationary_distance_pixels",
        minimum=0.0,
    )
    minimum_separation_pixels = _finite_number(
        minimum_separation_pixels,
        "minimum_separation_pixels",
        minimum=0.0,
    )
    _integer(
        minimum_stationary_samples,
        "minimum_stationary_samples",
        minimum=1,
    )
    _integer(maximum_placements, "maximum_placements", minimum=1)
    detections, metadata = scan_calibration_video(
        video,
        board=board,
        input_rotation=intrinsics.input_rotation,
        sample_seconds=sample_seconds,
    )
    expected_size = (
        intrinsics.image_size.width,
        intrinsics.image_size.height,
    )
    if metadata.image_size != expected_size:
        raise AutomaticCalibrationError(
            "floor video geometry does not match the intrinsic artifact after "
            f"its declared input rotation: got {metadata.image_size[0]}x"
            f"{metadata.image_size[1]}, expected {expected_size[0]}x"
            f"{expected_size[1]}"
        )
    groups = group_stationary_detections(
        detections,
        sample_step_frames=metadata.sample_step_frames,
        maximum_center_motion_pixels=stationary_distance_pixels,
        minimum_samples=minimum_stationary_samples,
    )
    placements = choose_stationary_placements(
        groups,
        minimum_separation_pixels=minimum_separation_pixels,
        maximum_placements=maximum_placements,
    )
    poses = []
    rejected = []
    for detection, plateau in placements:
        try:
            poses.append(
                estimate_board_pose(
                    detection,
                    board=board,
                    intrinsics=intrinsics,
                    plateau=plateau,
                )
            )
        except AutomaticCalibrationError as exc:
            rejected.append(
                {
                    "frame_index": detection.frame_index,
                    "timestamp_seconds": detection.timestamp_seconds,
                    "plateau_start_frame": plateau[0],
                    "plateau_end_frame": plateau[1],
                    "plateau_sample_count": plateau[2],
                    "reason": str(exc),
                }
            )
    return (
        tuple(poses),
        metadata,
        len(detections),
        len(groups),
        tuple(rejected),
    )


def calibrate_floor_from_video(
    video: Path | str,
    *,
    intrinsics: IntrinsicCalibrationArtifact,
    board: BoardSpec,
    sample_seconds: float = 1.0,
    stationary_distance_pixels: float = 20.0,
    minimum_stationary_samples: int = 3,
    minimum_separation_pixels: float = 80.0,
    minimum_placements: int = 3,
    maximum_placements: int = 12,
    maximum_normal_deviation_degrees: float = 5.0,
    maximum_centroid_offset_mm: float = 50.0,
) -> FloorCalibrationRun:
    """Fit a canonical floor plane from automatically selected placements."""

    _integer(minimum_placements, "minimum_placements", minimum=2)
    _integer(
        maximum_placements,
        "maximum_placements",
        minimum=minimum_placements,
    )
    if maximum_placements < minimum_placements:
        raise ValueError(
            "maximum_placements must be at least minimum_placements"
        )
    maximum_normal_deviation_degrees = _finite_number(
        maximum_normal_deviation_degrees,
        "maximum_normal_deviation_degrees",
        minimum=0.0,
        minimum_inclusive=False,
    )
    maximum_centroid_offset_mm = _finite_number(
        maximum_centroid_offset_mm,
        "maximum_centroid_offset_mm",
        minimum=0.0,
        minimum_inclusive=False,
    )
    source = Path(video).expanduser().resolve()
    (
        poses,
        metadata,
        detection_count,
        stationary_count,
        rejected_placements,
    ) = (
        _selected_floor_poses(
            source,
            intrinsics=intrinsics,
            board=board,
            sample_seconds=sample_seconds,
            stationary_distance_pixels=stationary_distance_pixels,
            minimum_stationary_samples=minimum_stationary_samples,
            minimum_separation_pixels=minimum_separation_pixels,
            maximum_placements=maximum_placements,
        )
    )
    if len(poses) < minimum_placements:
        raise AutomaticCalibrationError(
            f"only {len(poses)} spatially distinct stationary board placements "
            f"were retained after rejecting {len(rejected_placements)} invalid "
            f"pose(s); at least {minimum_placements} are required. Hold the "
            "board flat and motionless at more floor locations."
        )

    plane, residuals = fit_floor_plane(poses, board)
    pose_consistency = floor_pose_consistency(poses, board, plane)
    if (
        pose_consistency["maximum_normal_deviation_degrees"]
        > maximum_normal_deviation_degrees
    ):
        raise AutomaticCalibrationError(
            "selected placements do not describe one flat floor: maximum "
            "board-normal deviation is "
            f"{pose_consistency['maximum_normal_deviation_degrees']:.3f}°, "
            f"above {maximum_normal_deviation_degrees:.3f}°"
        )
    if (
        pose_consistency["maximum_centroid_offset_mm"]
        > maximum_centroid_offset_mm
    ):
        raise AutomaticCalibrationError(
            "selected placements do not describe one flat floor: maximum "
            "board-centroid offset is "
            f"{pose_consistency['maximum_centroid_offset_mm']:.1f} mm, "
            f"above {maximum_centroid_offset_mm:.1f} mm"
        )
    artifact = FloorPlaneCalibrationArtifact(
        camera_id=intrinsics.camera_id,
        image_size=intrinsics.image_size,
        floor_plane=plane,
        units="mm",
        coordinate_frame=intrinsics.coordinate_frame,
        intrinsic_sha256=intrinsics.sha256,
        input_rotation=intrinsics.input_rotation,
    )
    CalibrationBundle(intrinsics=intrinsics, floor_plane=artifact)

    internal_measurements = []
    if len(poses) >= 2:
        for placement_id, holdout in enumerate(poses, start=1):
            training = [pose for pose in poses if pose is not holdout]
            loo_plane, _ = fit_floor_plane(training, board)
            internal_measurements.extend(
                measure_board_on_plane(
                    holdout.detection,
                    placement_id=placement_id,
                    board=board,
                    intrinsics=intrinsics,
                    floor_plane=loo_plane,
                )
            )
    warnings = []
    if rejected_placements:
        warnings.append(
            f"{len(rejected_placements)} automatically selected placement(s) "
            "failed pose quality checks and were excluded."
        )
    if len(poses) < 3:
        warnings.append(
            "Only two placements were retained. Record at least three "
            "well-separated placements for a stronger floor-plane estimate."
        )
    report = {
        "schema_version": "1.0",
        "kind": "automatic_floor_calibration_report",
        "camera_id": intrinsics.camera_id,
        "source_video_identity": source_identity(source),
        "intrinsic_sha256": intrinsics.sha256,
        "board": board.to_dict(),
        "input_rotation": intrinsics.input_rotation.value,
        "sampling": {
            "sample_seconds": float(sample_seconds),
            "stationary_distance_pixels": float(
                stationary_distance_pixels
            ),
            "minimum_stationary_samples": minimum_stationary_samples,
            "minimum_separation_pixels": float(
                minimum_separation_pixels
            ),
            **metadata.to_dict(),
        },
        "sampled_detection_count": detection_count,
        "stationary_run_count": stationary_count,
        "selected_placement_count": len(poses),
        "selected_placements": [pose.metadata() for pose in poses],
        "rejected_placements": list(rejected_placements),
        "floor_plane": artifact.to_dict()["floor_plane"],
        "plane_fit_residuals": residuals,
        "pose_consistency": pose_consistency,
        "quality_limits": {
            "maximum_normal_deviation_degrees": float(
                maximum_normal_deviation_degrees
            ),
            "maximum_centroid_offset_mm": float(
                maximum_centroid_offset_mm
            ),
        },
        "internal_leave_one_placement_out": (
            summarize_measurements(internal_measurements)
            if internal_measurements
            else None
        ),
        "warnings": warnings,
        "artifact_sha256": artifact.sha256,
        "interpretation": (
            "The internal leave-one-placement-out result measures consistency "
            "within this recording. Run the separate verify command on a new "
            "recording for independent operational validation."
        ),
    }
    return FloorCalibrationRun(
        artifact=artifact,
        report=report,
        selected_poses=poses,
        internal_measurements=tuple(internal_measurements),
    )


def verify_floor_from_video(
    video: Path | str,
    *,
    bundle: CalibrationBundle,
    board: BoardSpec,
    sample_seconds: float = 1.0,
    stationary_distance_pixels: float = 20.0,
    minimum_stationary_samples: int = 3,
    minimum_separation_pixels: float = 80.0,
    minimum_placements: int = 3,
    maximum_placements: int = 20,
    minimum_center_span_fraction: float = 0.10,
    pass_threshold_percent: float = 3.0,
    warning_threshold_percent: float = 5.0,
) -> VerificationRun:
    """Automatically measure a known board on a fixed calibrated floor."""

    _integer(minimum_placements, "minimum_placements", minimum=3)
    _integer(
        maximum_placements,
        "maximum_placements",
        minimum=minimum_placements,
    )
    minimum_center_span_fraction = _finite_number(
        minimum_center_span_fraction,
        "minimum_center_span_fraction",
        minimum=0.0,
    )
    if minimum_center_span_fraction > 1.0:
        raise ValueError("minimum_center_span_fraction cannot exceed 1")
    pass_threshold_percent = _finite_number(
        pass_threshold_percent,
        "pass_threshold_percent",
        minimum=0.0,
        minimum_inclusive=False,
    )
    warning_threshold_percent = _finite_number(
        warning_threshold_percent,
        "warning_threshold_percent",
        minimum=pass_threshold_percent,
    )
    if warning_threshold_percent < pass_threshold_percent:
        raise ValueError(
            "warning_threshold_percent must be at least the pass threshold"
        )
    source = Path(video).expanduser().resolve()
    (
        poses,
        metadata,
        detection_count,
        stationary_count,
        rejected_placements,
    ) = (
        _selected_floor_poses(
            source,
            intrinsics=bundle.intrinsics,
            board=board,
            sample_seconds=sample_seconds,
            stationary_distance_pixels=stationary_distance_pixels,
            minimum_stationary_samples=minimum_stationary_samples,
            minimum_separation_pixels=minimum_separation_pixels,
            maximum_placements=maximum_placements,
        )
    )
    if len(poses) < minimum_placements:
        raise AutomaticCalibrationError(
            f"only {len(poses)} spatially distinct stationary verification "
            f"placements were retained after rejecting "
            f"{len(rejected_placements)} invalid pose(s); at least "
            f"{minimum_placements} are required"
        )
    centers = np.vstack([pose.detection.center for pose in poses])
    placement_coverage = {
        "center_span_fraction_x": float(
            np.ptp(centers[:, 0]) / metadata.image_size[0]
        ),
        "center_span_fraction_y": float(
            np.ptp(centers[:, 1]) / metadata.image_size[1]
        ),
        "minimum_required_per_axis": minimum_center_span_fraction,
    }
    coverage_failures = [
        axis
        for axis in ("x", "y")
        if placement_coverage[f"center_span_fraction_{axis}"]
        < minimum_center_span_fraction
    ]
    if coverage_failures:
        details = ", ".join(
            f"{axis}={placement_coverage[f'center_span_fraction_{axis}']:.3f}"
            for axis in coverage_failures
        )
        raise AutomaticCalibrationError(
            "verification placements do not cover enough of the image: "
            f"{details}; each axis must span at least "
            f"{minimum_center_span_fraction:.3f}"
        )
    measurements = []
    for placement_id, pose in enumerate(poses, start=1):
        measurements.extend(
            measure_board_on_plane(
                pose.detection,
                placement_id=placement_id,
                board=board,
                intrinsics=bundle.intrinsics,
                floor_plane=bundle.floor_plane.floor_plane,
            )
        )
    summary = summarize_measurements(measurements)
    placement_summary = summary["placement_mean_absolute_error_percent"]
    decision_errors = {
        "mean_absolute_error_percent": float(
            summary["mean_absolute_error_percent"]
        ),
        "p90_absolute_error_percent": float(
            summary["p90_absolute_error_percent"]
        ),
        "worst_placement_mean_absolute_error_percent": float(
            placement_summary["maximum"]
        ),
    }
    if all(
        value <= pass_threshold_percent
        for value in decision_errors.values()
    ):
        status = "pass"
    elif all(
        value <= warning_threshold_percent
        for value in decision_errors.values()
    ):
        status = "warning"
    else:
        status = "fail"
    report = {
        "schema_version": "1.0",
        "kind": "automatic_floor_verification_report",
        "status": status,
        "camera_id": bundle.camera_id,
        "source_video_identity": source_identity(source),
        "intrinsic_sha256": bundle.intrinsics.sha256,
        "floor_calibration_sha256": bundle.floor_plane.sha256,
        "board": board.to_dict(),
        "input_rotation": bundle.input_rotation.value,
        "thresholds": {
            "pass_error_percent": float(pass_threshold_percent),
            "warning_error_percent": float(warning_threshold_percent),
            "decision_metrics": list(decision_errors),
        },
        "sampling": {
            "sample_seconds": float(sample_seconds),
            "stationary_distance_pixels": float(
                stationary_distance_pixels
            ),
            "minimum_stationary_samples": minimum_stationary_samples,
            "minimum_separation_pixels": float(
                minimum_separation_pixels
            ),
            **metadata.to_dict(),
        },
        "sampled_detection_count": detection_count,
        "stationary_run_count": stationary_count,
        "selected_placement_count": len(poses),
        "minimum_required_placements": minimum_placements,
        "placement_coverage": placement_coverage,
        "selected_placements": [
            pose.detection.metadata() for pose in poses
        ],
        "rejected_placements": list(rejected_placements),
        "measurements": summary,
        "decision_errors_percent": decision_errors,
        "interpretation": (
            "This result measures automatically detected board spans in the "
            "supplied recording. It is independent only when the verification "
            "video was recorded separately from the floor-calibration video."
        ),
    }
    return VerificationRun(
        report=report,
        measurements=tuple(measurements),
        selected_detections=tuple(pose.detection for pose in poses),
    )


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise AutomaticCalibrationError(
            f"output already exists: {path}; pass --overwrite to replace it"
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(destination: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_yaml_artifact(
    path: Path | str,
    artifact: (
        IntrinsicCalibrationArtifact | FloorPlaneCalibrationArtifact
    ),
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    _prepare_output(destination, overwrite=overwrite)
    _atomic_write_text(
        destination,
        yaml.safe_dump(artifact.to_dict(), sort_keys=False),
    )
    return destination


def write_yaml_document(
    path: Path | str,
    document: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    _prepare_output(destination, overwrite=overwrite)
    _atomic_write_text(
        destination,
        yaml.safe_dump(dict(document), sort_keys=False),
    )
    return destination


def write_json_report(
    path: Path | str,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    destination = Path(path).expanduser().resolve()
    _prepare_output(destination, overwrite=overwrite)
    _atomic_write_text(
        destination,
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
    )
    return destination


def write_measurements_csv(
    path: Path | str,
    measurements: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    if not measurements:
        raise ValueError("measurements must not be empty")
    destination = Path(path).expanduser().resolve()
    _prepare_output(destination, overwrite=overwrite)
    fieldnames = list(measurements[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(measurements)
    _atomic_write_text(destination, buffer.getvalue())
    return destination


def default_report_path(artifact_path: Path | str) -> Path:
    path = Path(artifact_path).expanduser().resolve()
    return path.with_suffix(".report.json")


def save_annotated_detections(
    video: Path | str,
    *,
    detections: Sequence[BoardDetection],
    board: BoardSpec,
    input_rotation: InputRotation,
    output_directory: Path | str,
    measurements: Optional[Sequence[Mapping[str, Any]]] = None,
    overwrite: bool = False,
) -> Tuple[Path, ...]:
    """Save selected frames with detected corners and optional measurements."""

    source = Path(video).expanduser().resolve()
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for previous in destination.glob("frame_*.jpg"):
            if previous.is_file():
                previous.unlink()
    measurement_lookup: Dict[Tuple[int, str], Mapping[str, Any]] = {}
    for row in measurements or ():
        measurement_lookup[(int(row["frame_index"]), str(row["edge_name"]))] = row

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise AutomaticCalibrationError(f"could not reopen video: {source}")
    written = []
    for detection in detections:
        capture.set(cv2.CAP_PROP_POS_FRAMES, detection.frame_index)
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise AutomaticCalibrationError(
                f"could not read selected frame {detection.frame_index}"
            )
        frame = rotate_frame(frame, input_rotation)
        annotated = frame.copy()
        cv2.drawChessboardCorners(
            annotated,
            board.pattern_size,
            detection.corners,
            True,
        )
        xy = detection.corners.reshape(-1, 2)
        for edge_name, first, second, _ in boundary_edges(board):
            row = measurement_lookup.get(
                (detection.frame_index, edge_name)
            )
            if row is None:
                continue
            point_1 = tuple(np.rint(xy[first]).astype(int))
            point_2 = tuple(np.rint(xy[second]).astype(int))
            cv2.line(annotated, point_1, point_2, (0, 255, 0), 3)
            midpoint = (
                (point_1[0] + point_2[0]) // 2,
                (point_1[1] + point_2[1]) // 2,
            )
            cv2.putText(
                annotated,
                f"{row['measured_distance_mm']:.1f} mm",
                midpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        output = destination / (
            f"frame_{detection.frame_index:08d}.jpg"
        )
        if output.exists() and not overwrite:
            capture.release()
            raise AutomaticCalibrationError(
                f"annotated frame already exists: {output}"
            )
        if not cv2.imwrite(str(output), annotated):
            capture.release()
            raise AutomaticCalibrationError(
                f"could not write annotated frame: {output}"
            )
        written.append(output)
    capture.release()
    return tuple(written)


def source_identity(path: Path | str) -> Dict[str, Any]:
    """Return path-free source provenance with a content fingerprint."""

    source = Path(path).expanduser().resolve()
    stat = os.stat(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }
