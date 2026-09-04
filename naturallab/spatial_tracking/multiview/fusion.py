"""Opt-in fusion of explicitly identified, room-registered trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .registration import (
    Point3D,
    RoomRegistration,
    RoomRegistrationError,
    _freeze_provenance,
    _lowercase_sha256,
    _non_empty_string,
    _normalise_point,
)


class TrajectoryFusionError(ValueError):
    """Raised when trajectory observations cannot be fused unambiguously."""


def _non_negative_timestamp(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TrajectoryFusionError("timestamp_ns must be a non-negative integer")
    return int(value)


def _normalise_identity(value: Any) -> Optional[str]:
    if value is None:
        return None
    try:
        return _non_empty_string(value, "shared_identity")
    except RoomRegistrationError as error:
        raise TrajectoryFusionError(str(error)) from error


@dataclass(frozen=True)
class TrajectoryObservation:
    """One per-view floor observation before room registration."""

    view_id: str
    camera_id: str
    track_id: str
    timestamp_ns: int
    floor_point: Point3D
    coordinate_frame: str
    source_floor_calibration_sha256: str
    units: str
    shared_identity: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            for field_name in (
                "view_id",
                "camera_id",
                "track_id",
                "coordinate_frame",
                "units",
            ):
                object.__setattr__(
                    self,
                    field_name,
                    _non_empty_string(
                        getattr(self, field_name),
                        field_name,
                    ),
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
                "floor_point",
                _normalise_point(self.floor_point),
            )
            object.__setattr__(
                self,
                "provenance",
                _freeze_provenance(self.provenance),
            )
        except RoomRegistrationError as error:
            raise TrajectoryFusionError(str(error)) from error
        object.__setattr__(
            self,
            "timestamp_ns",
            _non_negative_timestamp(self.timestamp_ns),
        )
        object.__setattr__(
            self,
            "shared_identity",
            _normalise_identity(self.shared_identity),
        )


@dataclass(frozen=True)
class RegisteredTrajectoryObservation:
    """A per-view observation transformed into the configured room frame."""

    source_view_id: str
    source_camera_id: str
    source_track_id: str
    timestamp_ns: int
    source_floor_point: Point3D
    room_floor_point: Point3D
    source_coordinate_frame: str
    source_floor_calibration_sha256: str
    room_coordinate_frame: str
    units: str
    shared_identity: Optional[str]
    source_provenance: Mapping[str, Any]
    registration_sha256: str
    registration_provenance: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_view_id": self.source_view_id,
            "source_camera_id": self.source_camera_id,
            "source_track_id": self.source_track_id,
            "timestamp_ns": self.timestamp_ns,
            "source_floor_point": list(self.source_floor_point),
            "room_floor_point": list(self.room_floor_point),
            "source_coordinate_frame": self.source_coordinate_frame,
            "source_floor_calibration_sha256": (self.source_floor_calibration_sha256),
            "room_coordinate_frame": self.room_coordinate_frame,
            "units": self.units,
            "shared_identity": self.shared_identity,
            "source_provenance": dict(self.source_provenance),
            "registration_sha256": self.registration_sha256,
            "registration_provenance": dict(self.registration_provenance),
        }


@dataclass(frozen=True)
class ViewTrajectoryMetrics:
    """Metrics for one local track, never merged across source views."""

    source_view_id: str
    source_camera_id: str
    source_track_id: str
    shared_identity: Optional[str]
    observation_count: int
    first_timestamp_ns: int
    last_timestamp_ns: int
    path_length: float
    units: str


@dataclass(frozen=True)
class FusedTrajectoryObservation:
    """One room-frame point formed from explicitly corresponding identities."""

    shared_identity: str
    timestamp_ns: int
    room_floor_point: Point3D
    room_coordinate_frame: str
    units: str
    source_observations: Tuple[RegisteredTrajectoryObservation, ...]
    fusion_method: str = "arithmetic_mean"
    timestamp_matching_method: str = "global_nearest_unambiguous"
    timestamp_tolerance_ns: int = 0

    @property
    def source_view_ids(self) -> Tuple[str, ...]:
        return tuple(
            observation.source_view_id for observation in self.source_observations
        )

    @property
    def source_camera_ids(self) -> Tuple[str, ...]:
        return tuple(
            observation.source_camera_id for observation in self.source_observations
        )

    @property
    def source_floor_calibration_sha256_by_view(self) -> Dict[str, str]:
        """Exact source floor-calibration artifacts used by fused views."""

        return {
            observation.source_view_id: (observation.source_floor_calibration_sha256)
            for observation in self.source_observations
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shared_identity": self.shared_identity,
            "timestamp_ns": self.timestamp_ns,
            "room_floor_point": list(self.room_floor_point),
            "room_coordinate_frame": self.room_coordinate_frame,
            "units": self.units,
            "fusion_method": self.fusion_method,
            "timestamp_matching_method": self.timestamp_matching_method,
            "timestamp_tolerance_ns": self.timestamp_tolerance_ns,
            "source_view_ids": list(self.source_view_ids),
            "source_camera_ids": list(self.source_camera_ids),
            "source_floor_calibration_sha256_by_view": (
                self.source_floor_calibration_sha256_by_view
            ),
            "source_observations": [
                observation.to_dict() for observation in self.source_observations
            ],
        }


@dataclass(frozen=True)
class MultiviewTrajectoryResult:
    """Independent registered tracks plus optional fused room observations."""

    room_registration: RoomRegistration
    registered_observations: Tuple[RegisteredTrajectoryObservation, ...]
    per_view_observations: Mapping[
        str,
        Tuple[RegisteredTrajectoryObservation, ...],
    ]
    per_view_metrics: Tuple[ViewTrajectoryMetrics, ...]
    fused_observations: Tuple[FusedTrajectoryObservation, ...] = ()
    fusion_enabled: bool = False


def _register_observation(
    observation: TrajectoryObservation,
    room_registration: RoomRegistration,
) -> RegisteredTrajectoryObservation:
    try:
        registration = room_registration.registration_for(
            observation.view_id,
            camera_id=observation.camera_id,
        )
        room_point = room_registration.transform_floor_point(
            observation.view_id,
            observation.floor_point,
            camera_id=observation.camera_id,
            coordinate_frame=observation.coordinate_frame,
            source_floor_calibration_sha256=(
                observation.source_floor_calibration_sha256
            ),
            units=observation.units,
        )
    except RoomRegistrationError as error:
        raise TrajectoryFusionError(str(error)) from error

    return RegisteredTrajectoryObservation(
        source_view_id=observation.view_id,
        source_camera_id=observation.camera_id,
        source_track_id=observation.track_id,
        timestamp_ns=observation.timestamp_ns,
        source_floor_point=observation.floor_point,
        room_floor_point=room_point,
        source_coordinate_frame=observation.coordinate_frame,
        source_floor_calibration_sha256=(registration.source_floor_calibration_sha256),
        room_coordinate_frame=registration.room_coordinate_frame,
        units=registration.units,
        shared_identity=observation.shared_identity,
        source_provenance=observation.provenance,
        registration_sha256=registration.sha256,
        registration_provenance=registration.provenance,
    )


def _observation_sort_key(
    observation: RegisteredTrajectoryObservation,
) -> tuple:
    return (
        observation.source_view_id,
        observation.timestamp_ns,
        observation.source_track_id,
        observation.shared_identity or "",
    )


def _compute_per_view_metrics(
    observations: Tuple[RegisteredTrajectoryObservation, ...],
) -> Tuple[ViewTrajectoryMetrics, ...]:
    groups: Dict[
        Tuple[str, str],
        list[RegisteredTrajectoryObservation],
    ] = {}
    for observation in observations:
        groups.setdefault(
            (
                observation.source_view_id,
                observation.source_track_id,
            ),
            [],
        ).append(observation)

    metrics = []
    for key in sorted(groups):
        track_observations = sorted(
            groups[key],
            key=lambda item: (
                item.timestamp_ns,
                item.source_track_id,
            ),
        )
        shared_identities = {
            item.shared_identity
            for item in track_observations
            if item.shared_identity is not None
        }
        shared_identity = (
            next(iter(shared_identities)) if len(shared_identities) == 1 else None
        )
        path_length = math.fsum(
            math.dist(
                previous.room_floor_point,
                current.room_floor_point,
            )
            for previous, current in zip(
                track_observations,
                track_observations[1:],
            )
        )
        first = track_observations[0]
        last = track_observations[-1]
        metrics.append(
            ViewTrajectoryMetrics(
                source_view_id=first.source_view_id,
                source_camera_id=first.source_camera_id,
                source_track_id=first.source_track_id,
                shared_identity=shared_identity,
                observation_count=len(track_observations),
                first_timestamp_ns=first.timestamp_ns,
                last_timestamp_ns=last.timestamp_ns,
                path_length=path_length,
                units=first.units,
            )
        )
    return tuple(metrics)


def _validate_fusion_tolerance(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise TrajectoryFusionError(
            "timestamp_tolerance_ns must be a non-negative integer "
            "when fusion is enabled"
        )
    return int(value)


def _fuse_identity_observations(
    observations: list[RegisteredTrajectoryObservation],
    *,
    tolerance_ns: int,
) -> list[FusedTrajectoryObservation]:
    ordered = sorted(
        observations,
        key=lambda item: (
            item.timestamp_ns,
            item.source_view_id,
            item.source_track_id,
        ),
    )

    simultaneous_keys = set()
    for observation in ordered:
        key = (
            observation.source_view_id,
            observation.timestamp_ns,
        )
        if key in simultaneous_keys:
            raise TrajectoryFusionError(
                "shared identity has multiple observations from view "
                f"{observation.source_view_id!r} at timestamp "
                f"{observation.timestamp_ns}; correspondence is ambiguous"
            )
        simultaneous_keys.add(key)

    edges = []
    for left_index, left in enumerate(ordered):
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            distance = right.timestamp_ns - left.timestamp_ns
            if distance > tolerance_ns:
                break
            if left.source_view_id != right.source_view_id:
                edges.append((distance, left_index, right_index))
    edges.sort(
        key=lambda edge: (
            edge[0],
            edge[1],
            edge[2],
        )
    )

    parents = list(range(len(ordered)))
    members = {index: {index} for index in range(len(ordered))}

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def merge(component: set[int]) -> None:
        roots = sorted(find(root) for root in component)
        target = roots[0]
        combined_members = set()
        for root in roots:
            combined_members.update(members[root])
        for root in roots[1:]:
            parents[root] = target
            del members[root]
        members[target] = combined_members

    edge_index = 0
    while edge_index < len(edges):
        distance = edges[edge_index][0]
        bucket = []
        while edge_index < len(edges) and edges[edge_index][0] == distance:
            bucket.append(edges[edge_index])
            edge_index += 1

        adjacency: Dict[int, set[int]] = {}
        for _, left_index, right_index in bucket:
            left_root = find(left_index)
            right_root = find(right_index)
            if left_root == right_root:
                continue
            left_members = members[left_root]
            right_members = members[right_root]
            left_views = {ordered[index].source_view_id for index in left_members}
            right_views = {ordered[index].source_view_id for index in right_members}
            if left_views & right_views:
                # A strictly closer match already owns that source view.
                continue
            combined_timestamps = [
                ordered[index].timestamp_ns for index in left_members | right_members
            ]
            if max(combined_timestamps) - min(combined_timestamps) > tolerance_ns:
                # A strictly closer group cannot be widened past tolerance.
                continue
            adjacency.setdefault(left_root, set()).add(right_root)
            adjacency.setdefault(right_root, set()).add(left_root)

        unseen = set(adjacency)
        components = []
        while unseen:
            start = min(unseen)
            stack = [start]
            component = set()
            while stack:
                root = stack.pop()
                if root in component:
                    continue
                component.add(root)
                unseen.discard(root)
                stack.extend(sorted(adjacency[root] - component, reverse=True))
            components.append(component)

        for component in sorted(components, key=lambda value: min(value)):
            component_members = set().union(
                *(members[find(root)] for root in component)
            )
            component_views = [
                ordered[index].source_view_id for index in component_members
            ]
            component_timestamps = [
                ordered[index].timestamp_ns for index in component_members
            ]
            if (
                len(set(component_views)) != len(component_views)
                or max(component_timestamps) - min(component_timestamps) > tolerance_ns
            ):
                raise TrajectoryFusionError(
                    "timestamp correspondence is ambiguous at globally "
                    f"nearest distance {distance} ns for shared identity "
                    f"{ordered[next(iter(component_members))].shared_identity!r}"
                )
            merge(component)

    fused = []
    for member_indices in members.values():
        if len(member_indices) < 2:
            continue
        sources = tuple(
            sorted(
                (ordered[index] for index in member_indices),
                key=lambda item: (
                    item.source_view_id,
                    item.timestamp_ns,
                    item.source_track_id,
                ),
            )
        )
        source_count = len(sources)
        coordinates = tuple(
            math.fsum(source.room_floor_point[axis] for source in sources)
            / source_count
            for axis in range(3)
        )
        timestamp_sum = sum(source.timestamp_ns for source in sources)
        timestamp_ns = (timestamp_sum + source_count // 2) // source_count
        fused.append(
            FusedTrajectoryObservation(
                shared_identity=sources[0].shared_identity or "",
                timestamp_ns=timestamp_ns,
                room_floor_point=coordinates,  # type: ignore[arg-type]
                room_coordinate_frame=sources[0].room_coordinate_frame,
                units=sources[0].units,
                source_observations=sources,
                timestamp_tolerance_ns=tolerance_ns,
            )
        )
    return sorted(
        fused,
        key=lambda item: (
            item.timestamp_ns,
            item.source_view_ids,
        ),
    )


def process_multiview_trajectories(
    observations: Iterable[TrajectoryObservation],
    room_registration: RoomRegistration,
    *,
    fuse: bool = False,
    timestamp_tolerance_ns: Optional[int] = None,
) -> MultiviewTrajectoryResult:
    """Register observations and optionally fuse explicit shared identities.

    Per-view observations and metrics are always retained. Fusion is disabled
    by default and never uses local track IDs, geometric proximity, or a
    presumed camera count to invent cross-view correspondences.
    """

    if not isinstance(room_registration, RoomRegistration):
        raise TrajectoryFusionError("room_registration must be a RoomRegistration")
    if not isinstance(fuse, bool):
        raise TrajectoryFusionError("fuse must be a boolean")

    registered: list[RegisteredTrajectoryObservation] = []
    sample_keys = set()
    local_track_identities: Dict[Tuple[str, str], str] = {}
    for raw_observation in observations:
        if not isinstance(raw_observation, TrajectoryObservation):
            raise TrajectoryFusionError(
                "observations must contain only TrajectoryObservation values"
            )
        local_track_key = (
            raw_observation.view_id,
            raw_observation.track_id,
        )
        if raw_observation.shared_identity is not None:
            previous_identity = local_track_identities.get(local_track_key)
            if (
                previous_identity is not None
                and previous_identity != raw_observation.shared_identity
            ):
                raise TrajectoryFusionError(
                    "a local track cannot change shared_identity within one "
                    "trajectory; conflicting identities for view "
                    f"{raw_observation.view_id!r}, track "
                    f"{raw_observation.track_id!r}: "
                    f"{previous_identity!r} and "
                    f"{raw_observation.shared_identity!r}"
                )
            local_track_identities[local_track_key] = (
                raw_observation.shared_identity
            )
        sample_key = (
            raw_observation.view_id,
            raw_observation.track_id,
            raw_observation.timestamp_ns,
        )
        if sample_key in sample_keys:
            raise TrajectoryFusionError(
                "a local track must have at most one floor observation per "
                "timestamp; duplicate sample for view "
                f"{raw_observation.view_id!r}, track "
                f"{raw_observation.track_id!r}, timestamp "
                f"{raw_observation.timestamp_ns}"
            )
        sample_keys.add(sample_key)
        registered.append(_register_observation(raw_observation, room_registration))
    registered_observations = tuple(sorted(registered, key=_observation_sort_key))

    per_view: Dict[
        str,
        Tuple[RegisteredTrajectoryObservation, ...],
    ] = {}
    for view_id in room_registration.view_ids:
        view_observations = tuple(
            item for item in registered_observations if item.source_view_id == view_id
        )
        per_view[view_id] = view_observations

    fused_observations: Tuple[FusedTrajectoryObservation, ...] = ()
    if fuse:
        if timestamp_tolerance_ns is None:
            raise TrajectoryFusionError(
                "timestamp_tolerance_ns is required when fusion is enabled"
            )
        tolerance_ns = _validate_fusion_tolerance(timestamp_tolerance_ns)
        by_identity: Dict[
            str,
            list[RegisteredTrajectoryObservation],
        ] = {}
        for registered_observation in registered_observations:
            if registered_observation.shared_identity is None:
                continue
            by_identity.setdefault(
                registered_observation.shared_identity,
                [],
            ).append(registered_observation)

        fused = []
        for shared_identity in sorted(by_identity):
            fused.extend(
                _fuse_identity_observations(
                    by_identity[shared_identity],
                    tolerance_ns=tolerance_ns,
                )
            )
        fused_observations = tuple(
            sorted(
                fused,
                key=lambda item: (
                    item.shared_identity,
                    item.timestamp_ns,
                    item.source_view_ids,
                ),
            )
        )

    return MultiviewTrajectoryResult(
        room_registration=room_registration,
        registered_observations=registered_observations,
        per_view_observations=MappingProxyType(per_view),
        per_view_metrics=_compute_per_view_metrics(registered_observations),
        fused_observations=fused_observations,
        fusion_enabled=fuse,
    )
