"""Explicit registration of camera-floor coordinates into one room frame."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml  # type: ignore[import-untyped]


REGISTRATION_SCHEMA_VERSION = "1.0"
VIEW_REGISTRATION_KIND = "view_room_registration"
ROOM_REGISTRATION_KIND = "room_registration"
_VIEW_REGISTRATION_FIELDS = {
    "schema_version",
    "kind",
    "view_id",
    "camera_id",
    "source_coordinate_frame",
    "source_floor_calibration_sha256",
    "room_coordinate_frame",
    "units",
    "transform_to_room",
    "provenance",
}
_LOWERCASE_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROOM_REGISTRATION_FIELDS = {
    "schema_version",
    "kind",
    "room_coordinate_frame",
    "units",
    "views",
}

Point3D = Tuple[float, float, float]
RigidTransform = Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]


class RoomRegistrationError(ValueError):
    """Raised when a view-to-room registration is incomplete or inconsistent."""


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoomRegistrationError(f"{field_name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise RoomRegistrationError(f"{field_name} keys must be strings")
    return value


def _validate_document_fields(
    value: Mapping[str, Any],
    *,
    field_name: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RoomRegistrationError(
            f"{field_name} contains unknown field(s): {', '.join(unknown)}"
        )
    missing = sorted(required - set(value))
    if missing:
        raise RoomRegistrationError(
            f"{field_name} is missing field(s): {', '.join(missing)}"
        )


def _validate_artifact_header(
    value: Mapping[str, Any],
    *,
    field_name: str,
    expected_kind: str,
) -> None:
    if str(value.get("schema_version")) != REGISTRATION_SCHEMA_VERSION:
        raise RoomRegistrationError(
            f"{field_name}.schema_version must be {REGISTRATION_SCHEMA_VERSION!r}"
        )
    if value.get("kind") != expected_kind:
        raise RoomRegistrationError(f"{field_name}.kind must be {expected_kind!r}")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoomRegistrationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _lowercase_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _LOWERCASE_SHA256_PATTERN.fullmatch(value):
        raise RoomRegistrationError(
            f"{field_name} must be a lowercase 64-character SHA-256 digest"
        )
    return value


def _sequence(value: Any, field_name: str) -> list:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RoomRegistrationError(f"{field_name} must be an array-like value")
    return list(value)


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RoomRegistrationError(f"{field_name} must contain real numbers")
    number = float(value)
    if not math.isfinite(number):
        raise RoomRegistrationError(f"{field_name} must contain finite numbers")
    return 0.0 if number == 0.0 else number


def _normalise_point(value: Any, field_name: str = "floor_point") -> Point3D:
    values = _sequence(value, field_name)
    if len(values) != 3:
        raise RoomRegistrationError(
            f"{field_name} must contain exactly three coordinates"
        )
    return tuple(_finite_float(item, field_name) for item in values)  # type: ignore[return-value]


def _normalise_rigid_transform(value: Any) -> RigidTransform:
    rows = _sequence(value, "transform_to_room")
    if len(rows) != 4:
        raise RoomRegistrationError("transform_to_room must have shape (4, 4)")

    canonical_rows = []
    for row in rows:
        row_values = _sequence(row, "transform_to_room")
        if len(row_values) != 4:
            raise RoomRegistrationError("transform_to_room must have shape (4, 4)")
        canonical_rows.append(
            tuple(_finite_float(item, "transform_to_room") for item in row_values)
        )
    matrix = np.asarray(canonical_rows, dtype=float)

    if not np.allclose(
        matrix[3],
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        rtol=0.0,
        atol=1e-9,
    ):
        raise RoomRegistrationError(
            "transform_to_room must use homogeneous bottom row [0, 0, 0, 1]"
        )

    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if (
        not math.isfinite(determinant)
        or not np.allclose(
            rotation.T @ rotation,
            np.eye(3),
            rtol=0.0,
            atol=1e-6,
        )
        or not math.isclose(
            determinant,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ):
        raise RoomRegistrationError(
            "transform_to_room must be a non-singular rigid transform "
            "with a proper orthonormal rotation"
        )

    return tuple(tuple(float(item) for item in row) for row in matrix)  # type: ignore[return-value]


def _freeze_provenance(value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoomRegistrationError("provenance must be a mapping")
    try:
        detached = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RoomRegistrationError(
            "provenance must contain JSON-compatible values"
        ) from exc
    return MappingProxyType(detached)


@dataclass(frozen=True)
class ViewRegistration:
    """Rigid transform from one explicitly named view into a room frame."""

    view_id: str
    camera_id: str
    source_coordinate_frame: str
    source_floor_calibration_sha256: str
    room_coordinate_frame: str
    units: str
    transform_to_room: RigidTransform
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = field(
        default=REGISTRATION_SCHEMA_VERSION,
        init=False,
    )
    kind: str = field(default=VIEW_REGISTRATION_KIND, init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "view_id",
            "camera_id",
            "source_coordinate_frame",
            "room_coordinate_frame",
            "units",
        ):
            object.__setattr__(
                self,
                field_name,
                _non_empty_string(getattr(self, field_name), field_name),
            )
        object.__setattr__(
            self,
            "source_floor_calibration_sha256",
            _lowercase_sha256(
                self.source_floor_calibration_sha256,
                "source_floor_calibration_sha256",
            ),
        )
        object.__setattr__(
            self,
            "transform_to_room",
            _normalise_rigid_transform(self.transform_to_room),
        )
        object.__setattr__(
            self,
            "provenance",
            _freeze_provenance(self.provenance),
        )

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        field_name: str = "view_registration",
    ) -> "ViewRegistration":
        """Parse one strict, schema-versioned view registration."""

        values = _mapping(data, field_name)
        _validate_document_fields(
            values,
            field_name=field_name,
            allowed=_VIEW_REGISTRATION_FIELDS,
            required=_VIEW_REGISTRATION_FIELDS - {"provenance"},
        )
        _validate_artifact_header(
            values,
            field_name=field_name,
            expected_kind=VIEW_REGISTRATION_KIND,
        )
        return cls(
            view_id=values["view_id"],
            camera_id=values["camera_id"],
            source_coordinate_frame=values["source_coordinate_frame"],
            source_floor_calibration_sha256=values["source_floor_calibration_sha256"],
            room_coordinate_frame=values["room_coordinate_frame"],
            units=values["units"],
            transform_to_room=values["transform_to_room"],
            provenance=_mapping(
                values.get("provenance", {}),
                f"{field_name}.provenance",
            ),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "ViewRegistration":
        """Load one strict view-registration artifact from JSON or YAML."""

        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
            raise RoomRegistrationError(
                "registration file must use .json, .yaml, or .yml"
            )
        try:
            with source.open("r", encoding="utf-8") as handle:
                if source.suffix.lower() == ".json":
                    data = json.load(handle)
                else:
                    data = yaml.safe_load(handle)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise RoomRegistrationError(
                f"could not read view registration {source}: {exc}"
            ) from exc
        return cls.from_dict(_mapping(data, "view_registration"))

    @property
    def is_identity(self) -> bool:
        """Whether this view already uses the room coordinate system."""

        return bool(
            np.allclose(
                np.asarray(self.transform_to_room),
                np.eye(4),
                rtol=0.0,
                atol=1e-9,
            )
        )

    def transform_floor_point(
        self,
        point: Any,
        *,
        source_floor_calibration_sha256: str,
    ) -> Point3D:
        """Transform a point produced by the exact bound floor calibration."""

        supplied_digest = _lowercase_sha256(
            source_floor_calibration_sha256,
            "source_floor_calibration_sha256",
        )
        if supplied_digest != self.source_floor_calibration_sha256:
            raise RoomRegistrationError(
                f"view {self.view_id!r} floor-calibration SHA-256 "
                f"{supplied_digest!r} does not match the registration binding "
                f"{self.source_floor_calibration_sha256!r}"
            )
        x, y, z = _normalise_point(point)
        transformed = np.asarray(self.transform_to_room) @ np.asarray([x, y, z, 1.0])
        return tuple(0.0 if value == 0.0 else float(value) for value in transformed[:3])  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/YAML-safe registration representation."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "view_id": self.view_id,
            "camera_id": self.camera_id,
            "source_coordinate_frame": self.source_coordinate_frame,
            "source_floor_calibration_sha256": (self.source_floor_calibration_sha256),
            "room_coordinate_frame": self.room_coordinate_frame,
            "units": self.units,
            "transform_to_room": [list(row) for row in self.transform_to_room],
            "provenance": dict(self.provenance),
        }

    @property
    def sha256(self) -> str:
        """Stable geometry/configuration identifier used in result provenance."""

        value = self.to_dict()
        value.pop("provenance")
        canonical = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RoomRegistration:
    """Validated registrations for an explicitly enumerated set of views."""

    room_coordinate_frame: str
    units: str
    views: Tuple[ViewRegistration, ...]
    schema_version: str = field(
        default=REGISTRATION_SCHEMA_VERSION,
        init=False,
    )
    kind: str = field(default=ROOM_REGISTRATION_KIND, init=False)

    def __post_init__(self) -> None:
        room_coordinate_frame = _non_empty_string(
            self.room_coordinate_frame,
            "room_coordinate_frame",
        )
        units = _non_empty_string(self.units, "units")
        object.__setattr__(
            self,
            "room_coordinate_frame",
            room_coordinate_frame,
        )
        object.__setattr__(self, "units", units)

        if isinstance(self.views, (str, bytes)) or not isinstance(
            self.views,
            Sequence,
        ):
            raise RoomRegistrationError(
                "views must be an explicit sequence of ViewRegistration values"
            )
        views = tuple(self.views)
        if not views:
            raise RoomRegistrationError(
                "views must explicitly contain at least one registration"
            )
        if not all(isinstance(view, ViewRegistration) for view in views):
            raise RoomRegistrationError(
                "views must contain only ViewRegistration values"
            )

        view_ids = [view.view_id for view in views]
        camera_ids = [view.camera_id for view in views]
        if len(set(view_ids)) != len(view_ids):
            raise RoomRegistrationError("view_id values must be unique")
        if len(set(camera_ids)) != len(camera_ids):
            raise RoomRegistrationError("camera_id values must be unique")

        for view in views:
            if view.room_coordinate_frame != room_coordinate_frame:
                raise RoomRegistrationError(
                    f"view {view.view_id!r} targets room frame "
                    f"{view.room_coordinate_frame!r}, expected "
                    f"{room_coordinate_frame!r}"
                )
            if view.units != units:
                raise RoomRegistrationError(
                    f"view {view.view_id!r} uses units {view.units!r}, "
                    f"expected {units!r}"
                )
        object.__setattr__(
            self,
            "views",
            tuple(sorted(views, key=lambda view: view.view_id)),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RoomRegistration":
        """Parse a strict room-registration artifact from JSON/YAML data."""

        values = _mapping(data, "room_registration")
        _validate_document_fields(
            values,
            field_name="room_registration",
            allowed=_ROOM_REGISTRATION_FIELDS,
            required=_ROOM_REGISTRATION_FIELDS,
        )
        _validate_artifact_header(
            values,
            field_name="room_registration",
            expected_kind=ROOM_REGISTRATION_KIND,
        )
        raw_views = values["views"]
        if isinstance(raw_views, (str, bytes)) or not isinstance(
            raw_views,
            Sequence,
        ):
            raise RoomRegistrationError("room_registration.views must be a list")
        return cls(
            room_coordinate_frame=values["room_coordinate_frame"],
            units=values["units"],
            views=tuple(
                ViewRegistration.from_dict(
                    _mapping(value, f"room_registration.views[{index}]"),
                    field_name=f"room_registration.views[{index}]",
                )
                for index, value in enumerate(raw_views)
            ),
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "RoomRegistration":
        """Load and validate a ``.json``, ``.yaml``, or ``.yml`` artifact."""

        source = Path(path).expanduser().resolve()
        if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
            raise RoomRegistrationError(
                "registration file must use .json, .yaml, or .yml"
            )
        try:
            with source.open("r", encoding="utf-8") as handle:
                if source.suffix.lower() == ".json":
                    data = json.load(handle)
                else:
                    data = yaml.safe_load(handle)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise RoomRegistrationError(
                f"could not read room registration {source}: {exc}"
            ) from exc
        return cls.from_dict(_mapping(data, "room_registration"))

    @property
    def view_count(self) -> int:
        """Number of explicitly registered views; no count is inferred."""

        return len(self.views)

    @property
    def view_ids(self) -> Tuple[str, ...]:
        return tuple(view.view_id for view in self.views)

    def registration_for(
        self,
        view_id: str,
        *,
        camera_id: str | None = None,
    ) -> ViewRegistration:
        """Resolve one explicit registration, optionally checking camera ID."""

        requested_view_id = _non_empty_string(view_id, "view_id")
        for registration in self.views:
            if registration.view_id != requested_view_id:
                continue
            if camera_id is not None and registration.camera_id != _non_empty_string(
                camera_id, "camera_id"
            ):
                raise RoomRegistrationError(
                    f"view {requested_view_id!r} is registered to camera "
                    f"{registration.camera_id!r}, not {camera_id!r}"
                )
            return registration
        raise RoomRegistrationError(
            f"no room registration exists for view {requested_view_id!r}"
        )

    def transform_floor_point(
        self,
        view_id: str,
        point: Any,
        *,
        camera_id: str,
        coordinate_frame: str,
        source_floor_calibration_sha256: str,
        units: str,
    ) -> Point3D:
        """Validate an observation contract and transform its floor point."""

        registration = self.registration_for(
            view_id,
            camera_id=camera_id,
        )
        if (
            _non_empty_string(coordinate_frame, "coordinate_frame")
            != registration.source_coordinate_frame
        ):
            raise RoomRegistrationError(
                f"view {registration.view_id!r} observation frame "
                f"{coordinate_frame!r} does not match registered source "
                f"frame {registration.source_coordinate_frame!r}"
            )
        if _non_empty_string(units, "units") != registration.units:
            raise RoomRegistrationError(
                f"view {registration.view_id!r} observation units "
                f"{units!r} do not match registration units "
                f"{registration.units!r}"
            )
        return registration.transform_floor_point(
            point,
            source_floor_calibration_sha256=(source_floor_calibration_sha256),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "room_coordinate_frame": self.room_coordinate_frame,
            "units": self.units,
            "views": [view.to_dict() for view in self.views],
        }


def load_room_registration(path: Path | str) -> RoomRegistration:
    """Load a validated shared-room registration artifact."""

    return RoomRegistration.from_file(path)


def load_view_registration(path: Path | str) -> ViewRegistration:
    """Load one validated view-to-room registration artifact."""

    return ViewRegistration.from_file(path)
