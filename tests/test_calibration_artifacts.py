"""Tests for the versioned calibration artifact contract."""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError
from typing import List, Optional

import pytest

from naturallab.spatial_tracking.calibration import (
    CalibrationArtifactError,
    CalibrationBundle,
    FloorPlaneCalibrationArtifact,
    ImageSize,
    InputRotation,
    IntrinsicCalibrationArtifact,
    LegacyCalibrationWarning,
)


CAMERA_MATRIX = [
    [1000.0, 0.0, 960.0],
    [0.0, 1001.0, 540.0],
    [0.0, 0.0, 1.0],
]
DIST_COEFF = [0.1, -0.2, 0.001, -0.002, 0.03]


def make_intrinsics(
    *,
    camera_id: str = "ceiling-01",
    image_size: ImageSize = ImageSize(1920, 1080),
    input_rotation: InputRotation = InputRotation.NONE,
) -> IntrinsicCalibrationArtifact:
    return IntrinsicCalibrationArtifact(
        camera_id=camera_id,
        image_size=image_size,
        camera_matrix=CAMERA_MATRIX,
        dist_coeff=DIST_COEFF,
        units="pixels",
        coordinate_frame="opencv_camera",
        input_rotation=input_rotation,
    )


def make_floor(
    intrinsics: IntrinsicCalibrationArtifact,
    *,
    camera_id: Optional[str] = None,
    image_size: Optional[ImageSize] = None,
    input_rotation: Optional[InputRotation] = None,
    intrinsic_sha256: Optional[str] = None,
) -> FloorPlaneCalibrationArtifact:
    return FloorPlaneCalibrationArtifact(
        camera_id=camera_id or intrinsics.camera_id,
        image_size=image_size or intrinsics.image_size,
        floor_plane=[0.0, 2.0, 0.0, -3000.0],
        units="mm",
        coordinate_frame="opencv_camera",
        intrinsic_sha256=intrinsic_sha256 or intrinsics.sha256,
        input_rotation=(
            intrinsics.input_rotation
            if input_rotation is None
            else input_rotation
        ),
    )


def test_intrinsics_are_immutable_and_serialize_canonically() -> None:
    artifact = make_intrinsics()

    assert artifact.schema_version == "1.0"
    assert artifact.kind == "intrinsics"
    assert artifact.camera_matrix == tuple(tuple(row) for row in CAMERA_MATRIX)
    assert artifact.dist_coeff == tuple(DIST_COEFF)
    assert artifact.to_dict()["image_size"] == {"width": 1920, "height": 1080}
    assert artifact.to_dict()["input_rotation"] == "none"
    assert "dist_coeff" in artifact.to_dict()
    assert "dist_coeffs" not in artifact.to_dict()

    with pytest.raises(FrozenInstanceError):
        artifact.camera_id = "different"  # type: ignore[misc]
    with pytest.raises(TypeError):
        artifact.dist_coeff[0] = 99.0  # type: ignore[index]


def test_legacy_dist_coeffs_are_migrated_with_warning_and_flag() -> None:
    canonical = make_intrinsics()
    legacy_data = {
        "camera_id": canonical.camera_id,
        "image_size": canonical.image_size.to_dict(),
        "input_rotation": "none",
        "units": "pixels",
        "coordinate_frame": "opencv_camera",
        "camera_matrix": CAMERA_MATRIX,
        "dist_coeffs": [DIST_COEFF],  # historical OpenCV row-vector shape
    }

    with pytest.warns(LegacyCalibrationWarning, match="legacy intrinsics"):
        migrated = IntrinsicCalibrationArtifact.from_dict(legacy_data)

    assert migrated.legacy
    assert migrated.is_legacy
    assert migrated.to_dict() == canonical.to_dict()
    assert migrated.sha256 == canonical.sha256
    assert "dist_coeffs" not in migrated.to_dict()


def test_legacy_floor_plane_is_normalized_and_bound_to_intrinsics() -> None:
    intrinsics = make_intrinsics(input_rotation=InputRotation.CLOCKWISE_90)
    legacy_data = {
        "plane_normal": [0.0, 2.0, 0.0],
        "plane_d": -3000.0,
        "units": "mm",
        "coordinate_frame": "opencv_camera",
    }

    with pytest.warns(LegacyCalibrationWarning, match="legacy floor_plane"):
        floor = FloorPlaneCalibrationArtifact.from_dict(
            legacy_data,
            intrinsic=intrinsics,
        )

    assert floor.legacy
    assert floor.camera_id == intrinsics.camera_id
    assert floor.image_size == intrinsics.image_size
    assert floor.input_rotation is InputRotation.CLOCKWISE_90
    assert floor.intrinsic_sha256 == intrinsics.sha256
    assert floor.floor_plane == pytest.approx((0.0, 1.0, 0.0, -1500.0))
    assert floor.to_dict()["floor_plane"] == pytest.approx(
        [0.0, 1.0, 0.0, -1500.0]
    )
    assert "plane_normal" not in floor.to_dict()
    assert "plane_d" not in floor.to_dict()


@pytest.mark.parametrize(
    "camera_matrix",
    [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        [[1.0, 0.0, 1.0], [0.0, float("nan"), 1.0], [0.0, 0.0, 1.0]],
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, float("inf"), 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
    ],
)
def test_intrinsics_reject_malformed_or_nonfinite_camera_matrix(
    camera_matrix: List[List[float]],
) -> None:
    with pytest.raises(CalibrationArtifactError, match="camera_matrix"):
        IntrinsicCalibrationArtifact(
            camera_id="ceiling-01",
            image_size=ImageSize(1920, 1080),
            camera_matrix=camera_matrix,
            dist_coeff=DIST_COEFF,
        )


def test_intrinsics_reject_unsupported_distortion_vector_length() -> None:
    with pytest.raises(CalibrationArtifactError, match="dist_coeff"):
        IntrinsicCalibrationArtifact(
            camera_id="ceiling-01",
            image_size=ImageSize(1920, 1080),
            camera_matrix=CAMERA_MATRIX,
            dist_coeff=[0.0, 0.0, 0.0],
        )


@pytest.mark.parametrize(
    "floor_plane",
    [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, float("nan"), 0.0],
    ],
)
def test_floor_plane_rejects_invalid_shape_normal_or_values(
    floor_plane: List[float],
) -> None:
    intrinsics = make_intrinsics()
    with pytest.raises(CalibrationArtifactError, match="floor_plane"):
        FloorPlaneCalibrationArtifact(
            camera_id=intrinsics.camera_id,
            image_size=intrinsics.image_size,
            floor_plane=floor_plane,
            units="mm",
            coordinate_frame="opencv_camera",
            intrinsic_sha256=intrinsics.sha256,
        )


def test_floor_reader_rejects_hidden_correction_factor() -> None:
    intrinsics = make_intrinsics()
    with pytest.raises(CalibrationArtifactError, match="correction_factor"):
        FloorPlaneCalibrationArtifact.from_dict(
            {
                "floor_plane": [0.0, 1.0, 0.0, -1500.0],
                "correction_factor": 1.1,
                "units": "mm",
                "coordinate_frame": "opencv_camera",
            },
            intrinsic=intrinsics,
        )


def test_bundle_accepts_only_a_compatible_intrinsic_floor_pair() -> None:
    intrinsics = make_intrinsics()
    floor = make_floor(intrinsics)

    bundle = CalibrationBundle(intrinsics=intrinsics, floor_plane=floor)

    assert bundle.camera_id == "ceiling-01"
    assert bundle.image_size == ImageSize(1920, 1080)
    assert bundle.floor_plane.floor_plane == pytest.approx(
        (0.0, 1.0, 0.0, -1500.0)
    )
    assert not bundle.legacy


def test_bundle_rejects_camera_hash_size_and_rotation_mismatches() -> None:
    intrinsics = make_intrinsics()

    with pytest.raises(CalibrationArtifactError, match="camera_id mismatch"):
        CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=make_floor(intrinsics, camera_id="ceiling-02"),
        )

    with pytest.raises(CalibrationArtifactError, match="SHA-256 mismatch"):
        CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=make_floor(intrinsics, intrinsic_sha256="0" * 64),
        )

    with pytest.raises(CalibrationArtifactError, match="image_size mismatch"):
        CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=make_floor(
                intrinsics, image_size=ImageSize(1080, 1920)
            ),
        )

    with pytest.raises(CalibrationArtifactError, match="input_rotation mismatch"):
        CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=make_floor(
                intrinsics, input_rotation=InputRotation.CLOCKWISE_90
            ),
        )

    with pytest.raises(CalibrationArtifactError, match="coordinate_frame mismatch"):
        CalibrationBundle(
            intrinsics=intrinsics,
            floor_plane=FloorPlaneCalibrationArtifact(
                camera_id=intrinsics.camera_id,
                image_size=intrinsics.image_size,
                floor_plane=[0.0, 1.0, 0.0, -1500.0],
                units="mm",
                coordinate_frame="different_camera_frame",
                intrinsic_sha256=intrinsics.sha256,
            ),
        )


def test_hashes_are_deterministic_across_round_trip_and_key_order() -> None:
    intrinsics = make_intrinsics()
    floor = make_floor(intrinsics)
    bundle = CalibrationBundle(intrinsics=intrinsics, floor_plane=floor)

    reordered_intrinsics = dict(reversed(list(intrinsics.to_dict().items())))
    round_tripped_intrinsics = IntrinsicCalibrationArtifact.from_dict(
        reordered_intrinsics
    )
    round_tripped_bundle = CalibrationBundle.from_dict(bundle.to_dict())

    assert round_tripped_intrinsics.sha256 == intrinsics.sha256
    assert round_tripped_bundle.sha256 == bundle.sha256
    assert intrinsics.sha256 == hashlib.sha256(
        intrinsics.canonical_json().encode("utf-8")
    ).hexdigest()
    assert bundle.sha256 == hashlib.sha256(
        bundle.canonical_json().encode("utf-8")
    ).hexdigest()
    assert len(bundle.sha256) == 64
