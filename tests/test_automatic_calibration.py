from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from naturallab.spatial_tracking.calibration import (
    CalibrationBundle,
    FloorPlaneCalibrationArtifact,
    ImageSize,
    InputRotation,
    IntrinsicCalibrationArtifact,
)
from naturallab.spatial_tracking.calibration import automatic
from naturallab.spatial_tracking.calibration.automatic import (
    AutomaticCalibrationError,
    BoardDetection,
    BoardPose,
    BoardSpec,
    VideoMetadata,
    calibrate_floor_from_video,
    calibrate_intrinsics_from_video,
    fit_floor_plane,
    group_stationary_detections,
    measure_board_on_plane,
    select_diverse_detections,
    source_identity,
    verify_floor_from_video,
)


def _intrinsics(
    *,
    camera_id: str = "camera-test",
    image_size: tuple[int, int] = (1280, 720),
) -> IntrinsicCalibrationArtifact:
    width, height = image_size
    return IntrinsicCalibrationArtifact(
        camera_id=camera_id,
        image_size=ImageSize(width, height),
        camera_matrix=(
            (1000.0, 0.0, width / 2),
            (0.0, 980.0, height / 2),
            (0.0, 0.0, 1.0),
        ),
        dist_coeff=(0.0, 0.0, 0.0, 0.0, 0.0),
        coordinate_frame=f"camera/{camera_id}/opencv",
        input_rotation=InputRotation.NONE,
    )


def _detection(
    frame_index: int,
    corners: np.ndarray,
    *,
    sharpness: float = 100.0,
) -> BoardDetection:
    xy = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return BoardDetection(
        frame_index=frame_index,
        timestamp_seconds=frame_index / 30.0,
        corners=xy.reshape(-1, 1, 2),
        center=xy.mean(axis=0),
        sharpness=sharpness,
        feature=np.asarray(
            [
                xy[:, 0].mean() / 1280,
                xy[:, 1].mean() / 720,
                frame_index / 1000,
                np.cos(frame_index),
                np.sin(frame_index),
                sharpness / 1000,
            ],
            dtype=float,
        ),
    )


def _projected_detection(
    board: BoardSpec,
    intrinsics: IntrinsicCalibrationArtifact,
    *,
    frame_index: int,
    rotation_vector: np.ndarray,
    translation_vector: np.ndarray,
) -> tuple[BoardDetection, BoardPose]:
    corners, _ = cv2.projectPoints(
        board.object_points(),
        rotation_vector,
        translation_vector,
        np.asarray(intrinsics.camera_matrix, dtype=float),
        np.asarray(intrinsics.dist_coeff, dtype=float),
    )
    detection = _detection(
        frame_index,
        corners,
        sharpness=100.0 + frame_index,
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose = BoardPose(
        detection=detection,
        rotation_vector=np.asarray(rotation_vector, dtype=float),
        translation_vector=np.asarray(translation_vector, dtype=float).reshape(
            3, 1
        ),
        rotation_matrix=rotation_matrix,
        plateau_start_frame=frame_index - 30,
        plateau_end_frame=frame_index + 30,
        plateau_sample_count=3,
        reprojection_rms_pixels=0.0,
    )
    return detection, pose


def test_board_dimensions_are_internal_corners_without_subtraction() -> None:
    board = BoardSpec(7, 7, 30.0)

    assert board.pattern_size == (7, 7)
    assert board.corner_count == 49
    assert board.object_points()[-1].tolist() == [180.0, 180.0, 0.0]


def test_source_identity_is_stable_and_does_not_expose_local_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "participant-bearing-name.mp4"
    source.write_bytes(b"same calibration bytes")

    identity = source_identity(source)
    rendered = repr(identity)

    assert identity["size_bytes"] == len(b"same calibration bytes")
    assert len(identity["sha256"]) == 64
    assert str(tmp_path) not in rendered
    assert source.name not in rendered
    assert set(identity) == {"size_bytes", "sha256"}


def test_video_scan_rejects_materially_incomplete_decode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class PartialCapture:
        def __init__(self, _path: str) -> None:
            self.decoded = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 25.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 100.0
            raise AssertionError(f"unexpected property: {property_id}")

        def read(self):
            if self.decoded == 3:
                return False, None
            self.decoded += 1
            return True, np.zeros((16, 16, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    video = tmp_path / "truncated.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(cv2, "VideoCapture", PartialCapture)
    monkeypatch.setattr(
        automatic,
        "detect_chessboard_corners",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(
        AutomaticCalibrationError,
        match="ended early after 3 of 100 reported frames",
    ):
        automatic.scan_calibration_video(
            video,
            board=BoardSpec(7, 7, 30.0),
        )


def test_video_scan_rejects_unknown_reported_frame_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class UnknownLengthCapture:
        def __init__(self, _path: str) -> None:
            pass

        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            if property_id == cv2.CAP_PROP_FPS:
                return 25.0
            if property_id == cv2.CAP_PROP_FRAME_COUNT:
                return 0.0
            raise AssertionError(f"unexpected property: {property_id}")

        def release(self) -> None:
            pass

    video = tmp_path / "unknown-length.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(cv2, "VideoCapture", UnknownLengthCapture)

    with pytest.raises(
        AutomaticCalibrationError,
        match="trustworthy positive frame count",
    ):
        automatic.scan_calibration_video(
            video,
            board=BoardSpec(7, 7, 30.0),
        )


def test_stationary_grouping_uses_whole_run_and_tolerates_one_miss() -> None:
    base = np.mgrid[0:7, 0:7].T.reshape(-1, 2).astype(np.float32)
    detections = (
        _detection(0, base),
        _detection(30, base + 1),
        _detection(90, base + 2),
        # A large rotate/translation from the original anchor starts a new run.
        _detection(120, base + np.asarray([80, 0], dtype=np.float32)),
    )

    groups = group_stationary_detections(
        detections,
        sample_step_frames=30,
        maximum_center_motion_pixels=5.0,
        minimum_samples=3,
    )

    assert len(groups) == 1
    assert [item.frame_index for item in groups[0]] == [0, 30, 90]


def test_diverse_selection_is_deterministic() -> None:
    base = np.mgrid[0:7, 0:7].T.reshape(-1, 2).astype(np.float32)
    detections = tuple(
        _detection(
            index * 30,
            base * (5 + index / 10) + np.asarray([index * 20, index * 4]),
            sharpness=50 + index,
        )
        for index in range(12)
    )

    first = select_diverse_detections(detections, 7)
    second = select_diverse_detections(detections, 7)

    assert first == second
    assert len(first) == 7
    assert len(set(first)) == 7


def test_full_pose_floor_fit_and_edge_measurement_are_metric() -> None:
    board = BoardSpec(7, 7, 30.0)
    intrinsics = _intrinsics()
    poses = []
    detections = []
    for index, translation in enumerate(
        (
            (-300.0, -200.0, 2400.0),
            (100.0, -100.0, 2400.0),
            (-100.0, 200.0, 2400.0),
        ),
        start=1,
    ):
        detection, pose = _projected_detection(
            board,
            intrinsics,
            frame_index=index * 30,
            rotation_vector=np.zeros((3, 1), dtype=float),
            translation_vector=np.asarray(translation, dtype=float).reshape(
                3, 1
            ),
        )
        detections.append(detection)
        poses.append(pose)

    plane, residuals = fit_floor_plane(poses, board)
    measurements = measure_board_on_plane(
        detections[0],
        placement_id=1,
        board=board,
        intrinsics=intrinsics,
        floor_plane=plane,
    )

    assert plane == pytest.approx([0.0, 0.0, 1.0, -2400.0], abs=1e-5)
    assert residuals["rms_mm"] < 1e-8
    assert {row["known_distance_mm"] for row in measurements} == {180.0}
    assert [row["measured_distance_mm"] for row in measurements] == pytest.approx(
        [180.0] * 4,
        abs=1e-3,
    )


def test_intrinsic_video_workflow_emits_canonical_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = BoardSpec(7, 7, 25.0)
    expected = _intrinsics()
    detections = []
    for index in range(20):
        rotation_vector = np.asarray(
            [
                (-0.35, -0.18, 0.18, 0.35)[index % 4],
                (-0.30, -0.15, 0.0, 0.15, 0.30)[index % 5],
                -0.08 + 0.02 * index,
            ],
            dtype=float,
        ).reshape(3, 1)
        translation_vector = np.asarray(
            [
                -400.0 + 200.0 * (index % 5),
                -220.0 + 145.0 * (index // 5),
                1100.0 + 55.0 * index,
            ],
            dtype=float,
        ).reshape(3, 1)
        detection, _ = _projected_detection(
            board,
            expected,
            frame_index=index * 30,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )
        detections.append(detection)
    metadata = VideoMetadata(
        fps=30.0,
        reported_frame_count=600,
        decoded_frame_count=600,
        sampled_frame_count=20,
        sample_step_frames=30,
        image_size=(1280, 720),
    )
    video = tmp_path / "intrinsic.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        automatic,
        "scan_calibration_video",
        lambda *args, **kwargs: (tuple(detections), metadata),
    )

    run = calibrate_intrinsics_from_video(
        video,
        camera_id="camera-test",
        board=board,
        target_views=16,
        minimum_views=10,
        maximum_view_rms_pixels=1.0,
    )

    assert run.artifact.camera_id == "camera-test"
    assert run.artifact.image_size == ImageSize(1280, 720)
    assert np.asarray(run.artifact.camera_matrix) == pytest.approx(
        np.asarray(expected.camera_matrix),
        rel=2e-3,
        abs=1.0,
    )
    assert run.report["opencv_rms_pixels"] < 0.01
    assert run.report["artifact_sha256"] == run.artifact.sha256
    assert run.report["view_geometry"]["tilted_view_count"] >= 4


def test_frontoparallel_intrinsic_views_are_rejected_despite_low_rms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = BoardSpec(7, 7, 25.0)
    expected = _intrinsics()
    detections = []
    for index in range(20):
        detection, _ = _projected_detection(
            board,
            expected,
            frame_index=index * 30,
            rotation_vector=np.zeros((3, 1), dtype=float),
            translation_vector=np.asarray(
                [
                    -400.0 + 200.0 * (index % 5),
                    -220.0 + 145.0 * (index // 5),
                    1100.0 + 55.0 * index,
                ],
                dtype=float,
            ).reshape(3, 1),
        )
        detections.append(detection)
    metadata = VideoMetadata(
        fps=30.0,
        reported_frame_count=600,
        decoded_frame_count=600,
        sampled_frame_count=20,
        sample_step_frames=30,
        image_size=(1280, 720),
    )
    video = tmp_path / "frontoparallel.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        automatic,
        "scan_calibration_video",
        lambda *args, **kwargs: (tuple(detections), metadata),
    )

    with pytest.raises(
        AutomaticCalibrationError,
        match="geometrically under-diverse",
    ):
        calibrate_intrinsics_from_video(
            video,
            camera_id="camera-test",
            board=board,
            target_views=16,
            minimum_views=10,
            maximum_view_rms_pixels=1.0,
        )


def test_floor_and_independent_verification_bind_exact_intrinsics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = BoardSpec(7, 7, 30.0)
    intrinsics = _intrinsics()
    poses = []
    for index, translation in enumerate(
        (
            (-300.0, -200.0, 2400.0),
            (100.0, -100.0, 2400.0),
            (-100.0, 200.0, 2400.0),
            (300.0, 100.0, 2400.0),
        ),
        start=1,
    ):
        _, pose = _projected_detection(
            board,
            intrinsics,
            frame_index=index * 30,
            rotation_vector=np.zeros((3, 1), dtype=float),
            translation_vector=np.asarray(translation, dtype=float).reshape(
                3, 1
            ),
        )
        poses.append(pose)
    metadata = VideoMetadata(
        fps=30.0,
        reported_frame_count=300,
        decoded_frame_count=300,
        sampled_frame_count=10,
        sample_step_frames=30,
        image_size=(1280, 720),
    )
    video = tmp_path / "floor.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        automatic,
        "_selected_floor_poses",
        lambda *args, **kwargs: (tuple(poses), metadata, 12, 4, ()),
    )

    floor_run = calibrate_floor_from_video(
        video,
        intrinsics=intrinsics,
        board=board,
        minimum_placements=3,
    )
    bundle = CalibrationBundle(
        intrinsics=intrinsics,
        floor_plane=floor_run.artifact,
    )
    verification = verify_floor_from_video(
        video,
        bundle=bundle,
        board=board,
    )

    assert floor_run.artifact.intrinsic_sha256 == intrinsics.sha256
    assert floor_run.artifact.floor_plane == pytest.approx(
        [0.0, 0.0, 1.0, -2400.0],
        abs=1e-5,
    )
    assert verification.report["status"] == "pass"
    assert verification.report["measurements"][
        "mean_absolute_error_mm"
    ] == pytest.approx(0.0, abs=1e-3)
    assert verification.report["floor_calibration_sha256"] == (
        floor_run.artifact.sha256
    )


def test_verification_requires_three_spatially_distinct_placements(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = BoardSpec(7, 7, 30.0)
    intrinsics = _intrinsics()
    poses = []
    for index, translation in enumerate(
        ((-200.0, 0.0, 2400.0), (200.0, 0.0, 2400.0)),
        start=1,
    ):
        _, pose = _projected_detection(
            board,
            intrinsics,
            frame_index=index * 30,
            rotation_vector=np.zeros((3, 1), dtype=float),
            translation_vector=np.asarray(translation, dtype=float).reshape(
                3, 1
            ),
        )
        poses.append(pose)
    floor = FloorPlaneCalibrationArtifact(
        camera_id=intrinsics.camera_id,
        image_size=intrinsics.image_size,
        floor_plane=(0.0, 0.0, 1.0, -2400.0),
        units="mm",
        coordinate_frame=intrinsics.coordinate_frame,
        intrinsic_sha256=intrinsics.sha256,
    )
    metadata = VideoMetadata(
        fps=30.0,
        reported_frame_count=120,
        decoded_frame_count=120,
        sampled_frame_count=4,
        sample_step_frames=30,
        image_size=(1280, 720),
    )
    video = tmp_path / "verify.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        automatic,
        "_selected_floor_poses",
        lambda *args, **kwargs: (tuple(poses), metadata, 6, 2, ()),
    )

    with pytest.raises(AutomaticCalibrationError, match="at least 3"):
        verify_floor_from_video(
            video,
            bundle=CalibrationBundle(
                intrinsics=intrinsics,
                floor_plane=floor,
            ),
            board=board,
        )


def test_oblique_camera_floor_fit_remains_metric() -> None:
    board = BoardSpec(7, 7, 30.0)
    intrinsics = _intrinsics()
    rotation_vector = np.asarray([0.35, -0.20, 0.10], dtype=float).reshape(
        3, 1
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    base = np.asarray([0.0, 0.0, 2600.0], dtype=float)
    poses = []
    detections = []
    for index, (x_offset, y_offset) in enumerate(
        ((-350.0, -250.0), (150.0, -100.0), (-100.0, 300.0)),
        start=1,
    ):
        translation = (
            base
            + rotation_matrix[:, 0] * x_offset
            + rotation_matrix[:, 1] * y_offset
        )
        detection, pose = _projected_detection(
            board,
            intrinsics,
            frame_index=index * 30,
            rotation_vector=rotation_vector,
            translation_vector=translation.reshape(3, 1),
        )
        detections.append(detection)
        poses.append(pose)

    plane, residuals = fit_floor_plane(poses, board)
    expected_normal = rotation_matrix[:, 2]
    expected_distance = -float(np.dot(expected_normal, base))
    measurements = measure_board_on_plane(
        detections[0],
        placement_id=1,
        board=board,
        intrinsics=intrinsics,
        floor_plane=plane,
    )

    assert plane == pytest.approx(
        [*expected_normal, expected_distance],
        abs=1e-5,
    )
    assert residuals["rms_mm"] < 1e-8
    assert [row["measured_distance_mm"] for row in measurements] == (
        pytest.approx([180.0] * 4, abs=1e-3)
    )


def test_one_bad_floor_pose_is_reported_and_remaining_poses_continue(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    board = BoardSpec(7, 7, 30.0)
    intrinsics = _intrinsics()
    poses_by_frame = {}
    detections = []
    for index, translation in enumerate(
        (
            (-350.0, -250.0, 2400.0),
            (200.0, -200.0, 2400.0),
            (-250.0, 250.0, 2400.0),
            (250.0, 200.0, 2400.0),
        ),
        start=1,
    ):
        detection, pose = _projected_detection(
            board,
            intrinsics,
            frame_index=index * 30,
            rotation_vector=np.zeros((3, 1), dtype=float),
            translation_vector=np.asarray(translation, dtype=float).reshape(
                3, 1
            ),
        )
        detections.append(detection)
        poses_by_frame[detection.frame_index] = pose
    metadata = VideoMetadata(
        fps=30.0,
        reported_frame_count=150,
        decoded_frame_count=150,
        sampled_frame_count=5,
        sample_step_frames=30,
        image_size=(1280, 720),
    )
    video = tmp_path / "floor.mp4"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(
        automatic,
        "scan_calibration_video",
        lambda *args, **kwargs: (tuple(detections), metadata),
    )

    def fake_estimate(detection, **kwargs):
        if detection.frame_index == 60:
            raise AutomaticCalibrationError("synthetic pose rejection")
        return poses_by_frame[detection.frame_index]

    monkeypatch.setattr(automatic, "estimate_board_pose", fake_estimate)
    poses, _, _, _, rejected = automatic._selected_floor_poses(
        video,
        intrinsics=intrinsics,
        board=board,
        sample_seconds=1.0,
        stationary_distance_pixels=0.0,
        minimum_stationary_samples=1,
        minimum_separation_pixels=0.0,
        maximum_placements=4,
    )

    assert len(poses) == 3
    assert [item.detection.frame_index for item in poses] == [30, 90, 120]
    assert len(rejected) == 1
    assert rejected[0]["frame_index"] == 60
    assert "synthetic pose rejection" in rejected[0]["reason"]


def test_bundle_rejects_a_floor_artifact_from_another_intrinsic() -> None:
    first = _intrinsics(camera_id="first")
    second = _intrinsics(camera_id="second")
    floor = FloorPlaneCalibrationArtifact(
        camera_id=first.camera_id,
        image_size=first.image_size,
        floor_plane=(0.0, 0.0, 1.0, -2000.0),
        units="mm",
        coordinate_frame=first.coordinate_frame,
        intrinsic_sha256=first.sha256,
    )

    with pytest.raises(ValueError, match="camera_id mismatch"):
        CalibrationBundle(intrinsics=second, floor_plane=floor)
