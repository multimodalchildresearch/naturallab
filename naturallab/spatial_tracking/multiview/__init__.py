"""Room registration and explicit, opt-in multiview trajectory fusion."""

from .fusion import (
    FusedTrajectoryObservation,
    MultiviewTrajectoryResult,
    RegisteredTrajectoryObservation,
    TrajectoryFusionError,
    TrajectoryObservation,
    ViewTrajectoryMetrics,
    process_multiview_trajectories,
)
from .registration import (
    REGISTRATION_SCHEMA_VERSION,
    ROOM_REGISTRATION_KIND,
    VIEW_REGISTRATION_KIND,
    RoomRegistration,
    RoomRegistrationError,
    ViewRegistration,
    load_room_registration,
    load_view_registration,
)

__all__ = [
    "FusedTrajectoryObservation",
    "MultiviewTrajectoryResult",
    "REGISTRATION_SCHEMA_VERSION",
    "ROOM_REGISTRATION_KIND",
    "RegisteredTrajectoryObservation",
    "RoomRegistration",
    "RoomRegistrationError",
    "TrajectoryFusionError",
    "TrajectoryObservation",
    "VIEW_REGISTRATION_KIND",
    "ViewRegistration",
    "ViewTrajectoryMetrics",
    "process_multiview_trajectories",
    "load_room_registration",
    "load_view_registration",
]
