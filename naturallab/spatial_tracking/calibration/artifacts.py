"""Versioned, immutable artifacts for camera and floor calibration.

This module deliberately contains no calibration algorithm.  It defines the
data contract shared by calibration producers and spatial-tracking consumers.
Legacy dictionaries can be migrated explicitly, while every serialization
written by these classes uses one canonical schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union


SCHEMA_VERSION = "1.0"
INTRINSICS_KIND = "intrinsics"
FLOOR_PLANE_KIND = "floor_plane"
BUNDLE_KIND = "calibration_bundle"

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_MISSING = object()


class CalibrationArtifactError(ValueError):
    """Raised when calibration data violates the artifact contract."""


class LegacyCalibrationWarning(UserWarning):
    """Warn that legacy calibration data was migrated in memory."""


class InputRotation(str, Enum):
    """Rotation applied to source frames before using calibration parameters."""

    NONE = "none"
    CLOCKWISE_90 = "90_cw"
    ROTATE_180 = "180"
    COUNTERCLOCKWISE_90 = "90_ccw"


_ROTATION_ALIASES = {
    None: InputRotation.NONE,
    0: InputRotation.NONE,
    90: InputRotation.CLOCKWISE_90,
    180: InputRotation.ROTATE_180,
    270: InputRotation.COUNTERCLOCKWISE_90,
    -90: InputRotation.COUNTERCLOCKWISE_90,
    "": InputRotation.NONE,
    "0": InputRotation.NONE,
    "none": InputRotation.NONE,
    "no_rotation": InputRotation.NONE,
    "90": InputRotation.CLOCKWISE_90,
    "90_cw": InputRotation.CLOCKWISE_90,
    "cw90": InputRotation.CLOCKWISE_90,
    "clockwise_90": InputRotation.CLOCKWISE_90,
    "180": InputRotation.ROTATE_180,
    "180_cw": InputRotation.ROTATE_180,
    "270": InputRotation.COUNTERCLOCKWISE_90,
    "-90": InputRotation.COUNTERCLOCKWISE_90,
    "90_ccw": InputRotation.COUNTERCLOCKWISE_90,
    "ccw90": InputRotation.COUNTERCLOCKWISE_90,
    "counterclockwise_90": InputRotation.COUNTERCLOCKWISE_90,
}


def _normalise_rotation(value: Any) -> InputRotation:
    if isinstance(value, InputRotation):
        return value

    lookup_value = value.strip().lower() if isinstance(value, str) else value
    try:
        return _ROTATION_ALIASES[lookup_value]
    except (KeyError, TypeError):
        allowed = ", ".join(rotation.value for rotation in InputRotation)
        raise CalibrationArtifactError(
            "input_rotation must describe a right-angle image rotation; "
            "canonical values are: {0}".format(allowed)
        )


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationArtifactError(
            "{0} must be a non-empty string".format(field_name)
        )
    return value.strip()


def _as_list(value: Any, field_name: str) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CalibrationArtifactError(
            "{0} must be an array-like value".format(field_name)
        )
    return list(value)


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CalibrationArtifactError(
            "{0} must contain numbers, not booleans".format(field_name)
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise CalibrationArtifactError(
            "{0} must contain only numeric values".format(field_name)
        )
    if not math.isfinite(number):
        raise CalibrationArtifactError(
            "{0} must contain only finite values".format(field_name)
        )
    # Avoid two canonical JSON representations for positive and negative zero.
    return 0.0 if number == 0.0 else number


def _normalise_camera_matrix(value: Any) -> Tuple[Tuple[float, ...], ...]:
    rows = _as_list(value, "camera_matrix")
    if len(rows) != 3:
        raise CalibrationArtifactError("camera_matrix must have shape (3, 3)")

    canonical_rows = []
    for row in rows:
        row_values = _as_list(row, "camera_matrix")
        if len(row_values) != 3:
            raise CalibrationArtifactError("camera_matrix must have shape (3, 3)")
        canonical_rows.append(
            tuple(_finite_float(item, "camera_matrix") for item in row_values)
        )
    matrix = tuple(canonical_rows)
    determinant = (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant) <= 1e-12:
        raise CalibrationArtifactError(
            "camera_matrix must be non-singular"
        )
    if matrix[0][0] <= 0.0 or matrix[1][1] <= 0.0:
        raise CalibrationArtifactError(
            "camera_matrix focal lengths must be positive"
        )
    if any(
        abs(actual - expected) > 1e-9
        for actual, expected in zip(matrix[2], (0.0, 0.0, 1.0))
    ):
        raise CalibrationArtifactError(
            "camera_matrix must use the OpenCV homogeneous bottom row "
            "[0, 0, 1]"
        )
    return matrix


def _normalise_dist_coeff(value: Any) -> Tuple[float, ...]:
    values = _as_list(value, "dist_coeff")
    if not values:
        raise CalibrationArtifactError("dist_coeff must not be empty")

    # OpenCV commonly emits either a flat vector, one row, or one column.
    if all(
        not isinstance(item, Sequence) or isinstance(item, (str, bytes))
        for item in values
    ):
        flattened = values
    elif len(values) == 1:
        flattened = _as_list(values[0], "dist_coeff")
    else:
        columns = [_as_list(item, "dist_coeff") for item in values]
        if not all(len(column) == 1 for column in columns):
            raise CalibrationArtifactError(
                "dist_coeff must be a flat vector, one row, or one column"
            )
        flattened = [column[0] for column in columns]

    if not flattened:
        raise CalibrationArtifactError("dist_coeff must not be empty")
    if len(flattened) not in {4, 5, 8, 12, 14}:
        raise CalibrationArtifactError(
            "dist_coeff must contain 4, 5, 8, 12, or 14 OpenCV "
            "distortion coefficients"
        )
    return tuple(_finite_float(item, "dist_coeff") for item in flattened)


def _normalise_floor_plane(value: Any) -> Tuple[float, float, float, float]:
    values = _as_list(value, "floor_plane")
    if len(values) != 4:
        raise CalibrationArtifactError(
            "floor_plane must contain exactly [a, b, c, d]"
        )
    a, b, c, d = (
        _finite_float(item, "floor_plane") for item in values
    )
    normal_length = math.sqrt(a * a + b * b + c * c)
    if normal_length == 0.0:
        raise CalibrationArtifactError(
            "floor_plane normal [a, b, c] must be non-zero"
        )
    return tuple(
        0.0 if item == 0.0 else item
        for item in (
            a / normal_length,
            b / normal_length,
            c / normal_length,
            d / normal_length,
        )
    )  # type: ignore[return-value]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_schema_and_kind(
    data: Mapping[str, Any],
    expected_kind: str,
    legacy_kind_aliases: Sequence[str],
) -> bool:
    """Validate canonical markers and return whether the input is legacy."""

    legacy = False
    schema_version = data.get("schema_version", _MISSING)
    if schema_version is _MISSING:
        legacy = True
    elif str(schema_version) != SCHEMA_VERSION:
        raise CalibrationArtifactError(
            "Unsupported calibration schema_version {0!r}; expected {1!r}".format(
                schema_version, SCHEMA_VERSION
            )
        )

    kind = data.get("kind", _MISSING)
    if kind is _MISSING:
        legacy = True
    elif kind == expected_kind:
        pass
    elif kind in legacy_kind_aliases:
        legacy = True
    else:
        raise CalibrationArtifactError(
            "Calibration kind {0!r} is not {1!r}".format(kind, expected_kind)
        )
    return legacy


def _value_or_context(
    data: Mapping[str, Any],
    field_name: str,
    context_value: Any,
) -> Any:
    if field_name in data:
        return data[field_name]
    if context_value is not None:
        return context_value
    raise CalibrationArtifactError(
        "Missing {0!r}; legacy calibration files must be supplied this "
        "metadata explicitly".format(field_name)
    )


def _warn_legacy(kind: str) -> None:
    warnings.warn(
        "Loaded legacy {0} calibration data. Metadata was migrated in memory; "
        "serialize the artifact to persist schema version {1}.".format(
            kind, SCHEMA_VERSION
        ),
        LegacyCalibrationWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class ImageSize:
    """Image dimensions in pixels, expressed as width then height."""

    width: int
    height: int

    def __post_init__(self) -> None:
        for name, value in (("width", self.width), ("height", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CalibrationArtifactError(
                    "image_size {0} must be a positive integer".format(name)
                )

    @classmethod
    def from_value(cls, value: Any) -> "ImageSize":
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            try:
                return cls(width=value["width"], height=value["height"])
            except KeyError:
                raise CalibrationArtifactError(
                    "image_size must contain width and height"
                )
        values = _as_list(value, "image_size")
        if len(values) != 2:
            raise CalibrationArtifactError(
                "image_size must contain [width, height]"
            )
        return cls(width=values[0], height=values[1])

    def to_dict(self) -> Dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True)
class IntrinsicCalibrationArtifact:
    """Intrinsic camera calibration bound to a camera and image geometry."""

    camera_id: str
    image_size: ImageSize
    camera_matrix: Tuple[Tuple[float, ...], ...]
    dist_coeff: Tuple[float, ...]
    units: str = "pixels"
    coordinate_frame: str = "opencv_camera"
    input_rotation: InputRotation = InputRotation.NONE
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    kind: str = field(default=INTRINSICS_KIND, init=False)
    _legacy: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _non_empty_string(self.camera_id, "camera_id")
        )
        object.__setattr__(self, "image_size", ImageSize.from_value(self.image_size))
        object.__setattr__(
            self, "camera_matrix", _normalise_camera_matrix(self.camera_matrix)
        )
        object.__setattr__(
            self, "dist_coeff", _normalise_dist_coeff(self.dist_coeff)
        )
        object.__setattr__(self, "units", _non_empty_string(self.units, "units"))
        object.__setattr__(
            self,
            "coordinate_frame",
            _non_empty_string(self.coordinate_frame, "coordinate_frame"),
        )
        object.__setattr__(
            self, "input_rotation", _normalise_rotation(self.input_rotation)
        )

    @property
    def legacy(self) -> bool:
        """Whether this instance was read from a legacy representation."""

        return self._legacy

    @property
    def is_legacy(self) -> bool:
        """Alias for :attr:`legacy` for readability at migration call sites."""

        return self._legacy

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical, JSON/YAML-safe representation."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "camera_id": self.camera_id,
            "image_size": self.image_size.to_dict(),
            "input_rotation": self.input_rotation.value,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "camera_matrix": [list(row) for row in self.camera_matrix],
            "dist_coeff": list(self.dist_coeff),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        """SHA-256 digest of canonical JSON."""

        return _canonical_sha256(self.to_dict())

    @property
    def artifact_hash(self) -> str:
        return self.sha256

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        camera_id: Optional[str] = None,
        image_size: Optional[Union[ImageSize, Sequence[int], Mapping[str, int]]] = None,
        units: Optional[str] = None,
        coordinate_frame: Optional[str] = None,
        input_rotation: Any = None,
        warn_legacy: bool = True,
    ) -> "IntrinsicCalibrationArtifact":
        """Read canonical data or explicitly contextualize a legacy mapping."""

        if not isinstance(data, Mapping):
            raise CalibrationArtifactError(
                "Intrinsic calibration data must be a mapping"
            )
        legacy = _validate_schema_and_kind(
            data,
            INTRINSICS_KIND,
            ("camera_intrinsics", "intrinsic_calibration"),
        )

        has_canonical = "dist_coeff" in data
        has_legacy = "dist_coeffs" in data
        if has_canonical and has_legacy:
            raise CalibrationArtifactError(
                "Calibration data must not contain both dist_coeff and "
                "legacy dist_coeffs"
            )
        if has_legacy:
            dist_coeff = data["dist_coeffs"]
            legacy = True
        elif has_canonical:
            dist_coeff = data["dist_coeff"]
        else:
            raise CalibrationArtifactError(
                "Missing 'dist_coeff' (legacy key 'dist_coeffs' is also accepted)"
            )

        if "camera_matrix" not in data:
            raise CalibrationArtifactError("Missing 'camera_matrix'")

        artifact = cls(
            camera_id=_value_or_context(data, "camera_id", camera_id),
            image_size=_value_or_context(data, "image_size", image_size),
            camera_matrix=data["camera_matrix"],
            dist_coeff=dist_coeff,
            units=_value_or_context(data, "units", units),
            coordinate_frame=_value_or_context(
                data, "coordinate_frame", coordinate_frame
            ),
            input_rotation=_value_or_context(
                data, "input_rotation", input_rotation
            ),
            _legacy=legacy,
        )
        if legacy and warn_legacy:
            _warn_legacy(INTRINSICS_KIND)
        return artifact


@dataclass(frozen=True)
class FloorPlaneCalibrationArtifact:
    """Metric floor plane bound to the exact intrinsic artifact it uses."""

    camera_id: str
    image_size: ImageSize
    floor_plane: Tuple[float, float, float, float]
    units: str
    coordinate_frame: str
    intrinsic_sha256: str
    input_rotation: InputRotation = InputRotation.NONE
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    kind: str = field(default=FLOOR_PLANE_KIND, init=False)
    _legacy: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "camera_id", _non_empty_string(self.camera_id, "camera_id")
        )
        object.__setattr__(self, "image_size", ImageSize.from_value(self.image_size))
        object.__setattr__(
            self, "floor_plane", _normalise_floor_plane(self.floor_plane)
        )
        object.__setattr__(self, "units", _non_empty_string(self.units, "units"))
        object.__setattr__(
            self,
            "coordinate_frame",
            _non_empty_string(self.coordinate_frame, "coordinate_frame"),
        )
        object.__setattr__(
            self, "input_rotation", _normalise_rotation(self.input_rotation)
        )
        intrinsic_sha256 = _non_empty_string(
            self.intrinsic_sha256, "intrinsic_sha256"
        ).lower()
        if not _SHA256_PATTERN.fullmatch(intrinsic_sha256):
            raise CalibrationArtifactError(
                "intrinsic_sha256 must be a 64-character hexadecimal SHA-256"
            )
        object.__setattr__(self, "intrinsic_sha256", intrinsic_sha256)

    @property
    def legacy(self) -> bool:
        return self._legacy

    @property
    def is_legacy(self) -> bool:
        return self._legacy

    def to_dict(self) -> Dict[str, Any]:
        """Return the canonical representation without correction factors."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "camera_id": self.camera_id,
            "image_size": self.image_size.to_dict(),
            "input_rotation": self.input_rotation.value,
            "units": self.units,
            "coordinate_frame": self.coordinate_frame,
            "intrinsic_sha256": self.intrinsic_sha256,
            "floor_plane": list(self.floor_plane),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def artifact_hash(self) -> str:
        return self.sha256

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        intrinsic: Optional[IntrinsicCalibrationArtifact] = None,
        camera_id: Optional[str] = None,
        image_size: Optional[Union[ImageSize, Sequence[int], Mapping[str, int]]] = None,
        units: Optional[str] = None,
        coordinate_frame: Optional[str] = None,
        input_rotation: Any = None,
        intrinsic_sha256: Optional[str] = None,
        warn_legacy: bool = True,
    ) -> "FloorPlaneCalibrationArtifact":
        """Read a floor plane, accepting ``plane_normal`` + ``plane_d``."""

        if not isinstance(data, Mapping):
            raise CalibrationArtifactError(
                "Floor-plane calibration data must be a mapping"
            )
        for forbidden_key in (
            "correction_factor",
            "distance_correction_factor",
            "scale_correction",
        ):
            if forbidden_key in data:
                raise CalibrationArtifactError(
                    "{0!r} is not part of the calibration contract; correct "
                    "the calibration geometry instead".format(forbidden_key)
                )

        legacy = _validate_schema_and_kind(
            data,
            FLOOR_PLANE_KIND,
            ("floor_calibration", "floor_plane_calibration"),
        )
        has_canonical = "floor_plane" in data
        has_normal = "plane_normal" in data
        has_d = "plane_d" in data
        if has_canonical and (has_normal or has_d):
            raise CalibrationArtifactError(
                "Calibration data must not mix floor_plane with legacy "
                "plane_normal/plane_d"
            )
        if has_normal != has_d:
            raise CalibrationArtifactError(
                "Legacy floor calibration requires both plane_normal and plane_d"
            )
        floor_plane: Any
        if has_normal:
            normal = _as_list(data["plane_normal"], "plane_normal")
            if len(normal) != 3:
                raise CalibrationArtifactError(
                    "plane_normal must contain exactly three values"
                )
            floor_plane = normal + [data["plane_d"]]
            legacy = True
        elif has_canonical:
            floor_plane = data["floor_plane"]
        else:
            raise CalibrationArtifactError(
                "Missing 'floor_plane' (legacy plane_normal + plane_d are "
                "also accepted)"
            )

        if intrinsic is not None and not isinstance(
            intrinsic, IntrinsicCalibrationArtifact
        ):
            raise CalibrationArtifactError(
                "intrinsic must be an IntrinsicCalibrationArtifact"
            )

        inferred_camera_id = (
            intrinsic.camera_id if intrinsic is not None else camera_id
        )
        inferred_image_size = (
            intrinsic.image_size if intrinsic is not None else image_size
        )
        inferred_rotation = (
            intrinsic.input_rotation if intrinsic is not None else input_rotation
        )
        inferred_hash = (
            intrinsic.sha256 if intrinsic is not None else intrinsic_sha256
        )

        artifact = cls(
            camera_id=_value_or_context(data, "camera_id", inferred_camera_id),
            image_size=_value_or_context(
                data, "image_size", inferred_image_size
            ),
            floor_plane=floor_plane,
            units=_value_or_context(data, "units", units),
            coordinate_frame=_value_or_context(
                data, "coordinate_frame", coordinate_frame
            ),
            intrinsic_sha256=_value_or_context(
                data, "intrinsic_sha256", inferred_hash
            ),
            input_rotation=_value_or_context(
                data, "input_rotation", inferred_rotation
            ),
            _legacy=legacy,
        )
        if legacy and warn_legacy:
            _warn_legacy(FLOOR_PLANE_KIND)
        return artifact


@dataclass(frozen=True)
class CalibrationBundle:
    """A compatible intrinsic/floor pair safe for spatial projection."""

    intrinsics: IntrinsicCalibrationArtifact
    floor_plane: FloorPlaneCalibrationArtifact
    schema_version: str = field(default=SCHEMA_VERSION, init=False)
    kind: str = field(default=BUNDLE_KIND, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsics, IntrinsicCalibrationArtifact):
            raise CalibrationArtifactError(
                "intrinsics must be an IntrinsicCalibrationArtifact"
            )
        if not isinstance(self.floor_plane, FloorPlaneCalibrationArtifact):
            raise CalibrationArtifactError(
                "floor_plane must be a FloorPlaneCalibrationArtifact"
            )
        if self.intrinsics.camera_id != self.floor_plane.camera_id:
            raise CalibrationArtifactError(
                "Calibration camera_id mismatch: intrinsics use {0!r}, floor "
                "plane uses {1!r}".format(
                    self.intrinsics.camera_id, self.floor_plane.camera_id
                )
            )
        if self.intrinsics.sha256 != self.floor_plane.intrinsic_sha256:
            raise CalibrationArtifactError(
                "Floor calibration is not bound to this intrinsic artifact "
                "(intrinsic SHA-256 mismatch)"
            )
        if self.intrinsics.image_size != self.floor_plane.image_size:
            raise CalibrationArtifactError(
                "Calibration image_size mismatch: intrinsics use {0}x{1}, "
                "floor plane uses {2}x{3}".format(
                    self.intrinsics.image_size.width,
                    self.intrinsics.image_size.height,
                    self.floor_plane.image_size.width,
                    self.floor_plane.image_size.height,
                )
            )
        if self.intrinsics.input_rotation != self.floor_plane.input_rotation:
            raise CalibrationArtifactError(
                "Calibration input_rotation mismatch: intrinsics use {0!r}, "
                "floor plane uses {1!r}".format(
                    self.intrinsics.input_rotation.value,
                    self.floor_plane.input_rotation.value,
                )
            )
        if (
            self.intrinsics.coordinate_frame
            != self.floor_plane.coordinate_frame
        ):
            raise CalibrationArtifactError(
                "Calibration coordinate_frame mismatch: intrinsics use {0!r}, "
                "floor plane uses {1!r}".format(
                    self.intrinsics.coordinate_frame,
                    self.floor_plane.coordinate_frame,
                )
            )

    @property
    def camera_id(self) -> str:
        return self.intrinsics.camera_id

    @property
    def image_size(self) -> ImageSize:
        return self.intrinsics.image_size

    @property
    def input_rotation(self) -> InputRotation:
        return self.intrinsics.input_rotation

    @property
    def legacy(self) -> bool:
        return self.intrinsics.legacy or self.floor_plane.legacy

    @property
    def is_legacy(self) -> bool:
        return self.legacy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "intrinsics": self.intrinsics.to_dict(),
            "floor_plane": self.floor_plane.to_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def artifact_hash(self) -> str:
        return self.sha256

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        warn_legacy: bool = True,
    ) -> "CalibrationBundle":
        if not isinstance(data, Mapping):
            raise CalibrationArtifactError(
                "Calibration bundle data must be a mapping"
            )
        legacy = _validate_schema_and_kind(
            data, BUNDLE_KIND, ("camera_calibration_bundle",)
        )
        if "intrinsics" not in data or "floor_plane" not in data:
            raise CalibrationArtifactError(
                "Calibration bundle must contain intrinsics and floor_plane"
            )

        intrinsics = IntrinsicCalibrationArtifact.from_dict(
            data["intrinsics"], warn_legacy=warn_legacy
        )
        floor_plane = FloorPlaneCalibrationArtifact.from_dict(
            data["floor_plane"],
            intrinsic=intrinsics,
            warn_legacy=warn_legacy,
        )
        bundle = cls(intrinsics=intrinsics, floor_plane=floor_plane)
        if legacy and warn_legacy:
            _warn_legacy(BUNDLE_KIND)
        return bundle


# Concise aliases for callers that do not need the explicit "Artifact" suffix.
IntrinsicCalibration = IntrinsicCalibrationArtifact
FloorPlaneCalibration = FloorPlaneCalibrationArtifact


__all__ = [
    "BUNDLE_KIND",
    "FLOOR_PLANE_KIND",
    "INTRINSICS_KIND",
    "SCHEMA_VERSION",
    "CalibrationArtifactError",
    "CalibrationBundle",
    "FloorPlaneCalibration",
    "FloorPlaneCalibrationArtifact",
    "ImageSize",
    "InputRotation",
    "IntrinsicCalibration",
    "IntrinsicCalibrationArtifact",
    "LegacyCalibrationWarning",
]
