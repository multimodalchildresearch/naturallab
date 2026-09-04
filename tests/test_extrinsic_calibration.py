from __future__ import annotations

import math

import cv2
import numpy as np

from naturallab.spatial_tracking.calibration.artifacts import (
    CalibrationBundle,
    FloorPlaneCalibrationArtifact,
    ImageSize,
    IntrinsicCalibrationArtifact,
)
from naturallab.spatial_tracking.calibration.automatic import (
    BoardDetection,
    BoardSpec,
)
from naturallab.spatial_tracking.calibration.extrinsics import (
    _resolve_orientations,
    _stereo_calibrate,
    grid_symmetries,
    transform_plane_to_room,
)


def _bundle(camera_id: str) -> CalibrationBundle:
    intrinsics = IntrinsicCalibrationArtifact(
        camera_id=camera_id,
        image_size=ImageSize(1280, 960),
        camera_matrix=(
            (900.0, 0.0, 640.0),
            (0.0, 900.0, 480.0),
            (0.0, 0.0, 1.0),
        ),
        dist_coeff=(0.0, 0.0, 0.0, 0.0, 0.0),
        coordinate_frame=f"camera/{camera_id}/opencv",
    )
    floor = FloorPlaneCalibrationArtifact(
        camera_id=camera_id,
        image_size=intrinsics.image_size,
        floor_plane=(0.0, 1.0, 0.0, -2000.0),
        units="mm",
        coordinate_frame=intrinsics.coordinate_frame,
        intrinsic_sha256=intrinsics.sha256,
    )
    return CalibrationBundle(intrinsics=intrinsics, floor_plane=floor)


def _detection(
    corners: np.ndarray,
    *,
    frame_index: int,
) -> BoardDetection:
    points = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
    return BoardDetection(
        frame_index=frame_index,
        timestamp_seconds=float(frame_index),
        corners=points,
        center=points.reshape(-1, 2).mean(axis=0),
        sharpness=100.0,
        feature=np.zeros(7, dtype=np.float64),
    )


def _transform(rotation_vector: tuple[float, float, float], translation) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rotation_vector, dtype=np.float64))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _project(
    board: BoardSpec,
    camera_from_board: np.ndarray,
    bundle: CalibrationBundle,
) -> np.ndarray:
    rotation_vector, _ = cv2.Rodrigues(camera_from_board[:3, :3])
    projected, _ = cv2.projectPoints(
        board.object_points(),
        rotation_vector,
        camera_from_board[:3, 3],
        np.asarray(bundle.intrinsics.camera_matrix, dtype=np.float64),
        np.asarray(bundle.intrinsics.dist_coeff, dtype=np.float64),
    )
    return projected


def test_square_grid_has_eight_unique_symmetries() -> None:
    board = BoardSpec(7, 7, 20.0)
    symmetries = grid_symmetries(board)

    assert len(symmetries) == 8
    assert len({symmetry.indices for symmetry in symmetries}) == 8
    assert all(
        sorted(symmetry.indices) == list(range(board.corner_count))
        for symmetry in symmetries
    )


def test_joint_homography_recovers_per_placement_corner_flips() -> None:
    board = BoardSpec(7, 7, 20.0)
    symmetries = grid_symmetries(board)
    by_name = {symmetry.name: symmetry for symmetry in symmetries}
    expected = (
        by_name["rotate_90"],
        by_name["rotate_180"],
        by_name["flip_columns"],
    )
    local = board.object_points()[:, :2].astype(np.float64)
    anchor_sets = []
    target_sets = []
    homography = np.asarray(
        [
            [1.1, 0.08, 120.0],
            [-0.03, 0.9, 80.0],
            [0.0002, -0.0001, 1.0],
        ]
    )
    for center, angle, symmetry in zip(
        ((250.0, 250.0), (650.0, 320.0), (420.0, 700.0)),
        (0.1, -0.35, 0.6),
        expected,
    ):
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)],
            ]
        )
        anchor = local @ rotation.T + np.asarray(center)
        homogeneous = np.column_stack([anchor, np.ones(len(anchor))])
        target_physical = (homography @ homogeneous.T).T
        target_physical = (
            target_physical[:, :2] / target_physical[:, 2, None]
        )
        target_raw = np.empty_like(target_physical)
        target_raw[np.asarray(symmetry.indices)] = target_physical
        anchor_sets.append(anchor)
        target_sets.append(target_raw)

    recovered, diagnostics = _resolve_orientations(
        anchor_sets,
        target_sets,
        symmetries,
    )

    assert tuple(item.name for item in recovered) == tuple(
        item.name for item in expected
    )
    assert diagnostics["global_rms_pixels"] < 1e-5
    assert diagnostics["minimum_next_margin_pixels"] > 10.0


def test_fixed_intrinsic_stereo_returns_target_from_anchor() -> None:
    board = BoardSpec(7, 7, 30.0)
    anchor_bundle = _bundle("anchor")
    target_bundle = _bundle("target")
    target_from_anchor = _transform(
        (0.03, -0.12, 0.02),
        (850.0, 20.0, 120.0),
    )
    board_poses = (
        _transform((0.02, 0.1, 0.0), (-250.0, -150.0, 2800.0)),
        _transform((-0.08, 0.04, 0.1), (180.0, -50.0, 3200.0)),
        _transform((0.1, -0.05, -0.08), (-50.0, 220.0, 2500.0)),
        _transform((-0.04, -0.12, 0.03), (300.0, 180.0, 3600.0)),
    )
    anchor = [
        _detection(
            _project(board, pose, anchor_bundle),
            frame_index=index,
        )
        for index, pose in enumerate(board_poses)
    ]
    target = [
        _detection(
            _project(board, target_from_anchor @ pose, target_bundle),
            frame_index=index,
        )
        for index, pose in enumerate(board_poses)
    ]
    identity = next(
        symmetry
        for symmetry in grid_symmetries(board)
        if symmetry.name == "identity"
    )

    rms, recovered = _stereo_calibrate(
        board,
        anchor,
        target,
        [identity] * len(anchor),
        anchor_bundle,
        target_bundle,
    )

    assert rms < 1e-3
    assert np.allclose(recovered, target_from_anchor, atol=1e-3)
    point_anchor = np.asarray([100.0, 50.0, 2500.0, 1.0])
    point_target = target_from_anchor @ point_anchor
    assert np.allclose(
        np.linalg.inv(recovered) @ point_target,
        point_anchor,
        atol=1e-3,
    )


def test_plane_transform_matches_floor_aligned_room() -> None:
    normal = np.asarray([0.2, 0.8, 0.565685424949238])
    normal /= np.linalg.norm(normal)
    plane = np.asarray([*normal, -2100.0])
    z_axis = -normal
    x_axis = np.asarray([1.0, 0.0, 0.0])
    x_axis -= normal * np.dot(normal, x_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    room_from_camera = np.eye(4)
    room_from_camera[:3, :3] = np.vstack([x_axis, y_axis, z_axis])
    floor_point = -plane[3] * normal
    room_from_camera[:3, 3] = (
        -room_from_camera[:3, :3] @ floor_point
    )

    transformed = transform_plane_to_room(plane, room_from_camera)

    assert np.allclose(transformed, (0.0, 0.0, -1.0, 0.0), atol=1e-9)
