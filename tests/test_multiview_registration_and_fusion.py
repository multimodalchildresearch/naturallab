"""Focused tests for explicit multiview room registration and fusion."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from naturallab.spatial_tracking.multiview import (
    RoomRegistration,
    RoomRegistrationError,
    TrajectoryFusionError,
    TrajectoryObservation,
    ViewRegistration,
    load_room_registration,
    load_view_registration,
    process_multiview_trajectories,
)


IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def floor_calibration_sha256(view_id: str) -> str:
    return hashlib.sha256(f"floor-calibration:{view_id}".encode()).hexdigest()


def make_view(
    view_id: str,
    *,
    camera_id: str | None = None,
    source_coordinate_frame: str | None = None,
    source_floor_calibration_sha256: str | None = None,
    room_coordinate_frame: str = "nursery-room",
    units: str = "mm",
    transform=IDENTITY,
    provenance=None,
) -> ViewRegistration:
    return ViewRegistration(
        view_id=view_id,
        camera_id=camera_id or f"camera-{view_id}",
        source_coordinate_frame=(source_coordinate_frame or f"{view_id}-floor"),
        source_floor_calibration_sha256=(
            floor_calibration_sha256(view_id)
            if source_floor_calibration_sha256 is None
            else source_floor_calibration_sha256
        ),
        room_coordinate_frame=room_coordinate_frame,
        units=units,
        transform_to_room=transform,
        provenance=provenance or {},
    )


def make_room(*views: ViewRegistration) -> RoomRegistration:
    return RoomRegistration(
        room_coordinate_frame="nursery-room",
        units="mm",
        views=views,
    )


def make_observation(
    view: ViewRegistration,
    *,
    track_id: str,
    timestamp_ns: int,
    floor_point=(0.0, 0.0, 0.0),
    shared_identity: str | None = None,
    source_floor_calibration_sha256: str | None = None,
    provenance=None,
) -> TrajectoryObservation:
    return TrajectoryObservation(
        view_id=view.view_id,
        camera_id=view.camera_id,
        track_id=track_id,
        timestamp_ns=timestamp_ns,
        floor_point=floor_point,
        coordinate_frame=view.source_coordinate_frame,
        source_floor_calibration_sha256=(
            view.source_floor_calibration_sha256
            if source_floor_calibration_sha256 is None
            else source_floor_calibration_sha256
        ),
        units=view.units,
        shared_identity=shared_identity,
        provenance=provenance or {},
    )


def test_registration_accepts_identity_and_transforms_floor_points() -> None:
    identity = make_view("already-room")
    rotated_and_translated = make_view(
        "corner",
        transform=[
            [0.0, -1.0, 0.0, 100.0],
            [1.0, 0.0, 0.0, 200.0],
            [0.0, 0.0, 1.0, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    room = make_room(identity, rotated_and_translated)

    assert identity.is_identity
    assert not rotated_and_translated.is_identity
    assert identity.transform_floor_point(
        (1, 2, 3),
        source_floor_calibration_sha256=(identity.source_floor_calibration_sha256),
    ) == (1.0, 2.0, 3.0)
    with pytest.raises(RoomRegistrationError, match="registration binding"):
        identity.transform_floor_point(
            (1, 2, 3),
            source_floor_calibration_sha256="f" * 64,
        )
    assert room.transform_floor_point(
        "corner",
        (1.0, 2.0, 3.0),
        camera_id=rotated_and_translated.camera_id,
        coordinate_frame=rotated_and_translated.source_coordinate_frame,
        source_floor_calibration_sha256=(
            rotated_and_translated.source_floor_calibration_sha256
        ),
        units="mm",
    ) == pytest.approx((98.0, 201.0, 13.0))


def test_room_registration_has_only_the_explicit_arbitrary_view_set() -> None:
    room = make_room(
        make_view("view-10"),
        make_view("view-2"),
        make_view("view-1"),
    )

    assert room.view_count == 3
    assert room.view_ids == ("view-1", "view-10", "view-2")
    with pytest.raises(RoomRegistrationError, match="no room registration"):
        room.registration_for("view-4")


def test_registration_artifact_round_trips_and_loads_from_json(tmp_path) -> None:
    room = make_room(
        make_view(
            "left",
            provenance={"method": "surveyed-markers", "revision": 2},
        ),
        make_view("right"),
    )
    artifact_path = tmp_path / "room-registration.json"
    artifact_path.write_text(
        json.dumps(room.to_dict()),
        encoding="utf-8",
    )

    loaded = load_room_registration(artifact_path)

    assert loaded.to_dict() == room.to_dict()
    assert loaded.view_ids == ("left", "right")


def test_individual_view_registration_loads_from_manifest_style_path(
    tmp_path,
) -> None:
    registration = make_view("left")
    artifact_path = tmp_path / "room-left.to-room.json"
    artifact_path.write_text(
        json.dumps(registration.to_dict()),
        encoding="utf-8",
    )

    assert load_view_registration(artifact_path) == registration


def test_registration_artifact_rejects_unknown_fields_and_wrong_kind() -> None:
    document = make_room(make_view("left")).to_dict()
    document["unexpected"] = True
    with pytest.raises(RoomRegistrationError, match="unknown field"):
        RoomRegistration.from_dict(document)

    document.pop("unexpected")
    document["kind"] = "camera_calibration"
    with pytest.raises(RoomRegistrationError, match=r"\.kind"):
        RoomRegistration.from_dict(document)


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "a" * 63,
        "A" * 64,
        "g" * 64,
        " a" * 32,
        f" {'a' * 64}",
        f"{'a' * 64} ",
    ],
)
def test_registration_requires_lowercase_floor_calibration_sha256(
    digest,
) -> None:
    with pytest.raises(RoomRegistrationError, match="lowercase.*SHA-256"):
        make_view(
            "left",
            source_floor_calibration_sha256=digest,
        )


def test_strict_loader_requires_floor_calibration_binding() -> None:
    document = make_room(make_view("left")).to_dict()
    document["views"][0].pop("source_floor_calibration_sha256")

    with pytest.raises(RoomRegistrationError, match="missing field"):
        RoomRegistration.from_dict(document)


def test_floor_calibration_binding_changes_registration_digest() -> None:
    first = make_view(
        "left",
        source_floor_calibration_sha256="a" * 64,
    )
    second = make_view(
        "left",
        source_floor_calibration_sha256="b" * 64,
    )

    assert first.sha256 != second.sha256


def test_registration_provenance_must_be_json_compatible() -> None:
    with pytest.raises(RoomRegistrationError, match="JSON-compatible"):
        make_view("left", provenance={"unsupported": object()})


@pytest.mark.parametrize(
    "transform,match",
    [
        (
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rigid",
        ),
        (
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rigid",
        ),
        (
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "rigid",
        ),
        (
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
            "homogeneous",
        ),
    ],
)
def test_registration_rejects_singular_or_non_rigid_transforms(
    transform,
    match,
) -> None:
    with pytest.raises(RoomRegistrationError, match=match):
        make_view("invalid", transform=transform)


def test_room_rejects_mismatched_frames_units_and_duplicate_ids() -> None:
    left = make_view("left")

    with pytest.raises(RoomRegistrationError, match="room frame"):
        make_room(
            left,
            make_view(
                "right",
                room_coordinate_frame="different-room",
            ),
        )
    with pytest.raises(RoomRegistrationError, match="units"):
        make_room(left, make_view("right", units="cm"))
    with pytest.raises(RoomRegistrationError, match="view_id"):
        make_room(left, make_view("left", camera_id="another-camera"))
    with pytest.raises(RoomRegistrationError, match="camera_id"):
        make_room(
            left,
            make_view("right", camera_id=left.camera_id),
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"view_id": "unknown"}, "no room registration"),
        ({"camera_id": "wrong-camera"}, "registered to camera"),
        ({"coordinate_frame": "wrong-floor"}, "observation frame"),
        (
            {"source_floor_calibration_sha256": "f" * 64},
            "does not match the registration binding",
        ),
        ({"units": "cm"}, "observation units"),
    ],
)
def test_observations_require_exact_registration_contract(
    changes,
    match,
) -> None:
    view = make_view("left")
    observation = make_observation(
        view,
        track_id="local-1",
        timestamp_ns=0,
    )

    with pytest.raises(TrajectoryFusionError, match=match):
        process_multiview_trajectories(
            [replace(observation, **changes)],
            make_room(view),
        )


def test_default_keeps_per_view_observations_and_metrics_independent() -> None:
    left = make_view("left")
    right = make_view(
        "right",
        transform=[
            [1.0, 0.0, 0.0, 100.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    )
    observations = [
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=10,
            floor_point=(3, 4, 0),
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=9,
            floor_point=(1, 0, 0),
            shared_identity="infant",
        ),
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=0,
            floor_point=(0, 0, 0),
            shared_identity="infant",
        ),
    ]

    result = process_multiview_trajectories(
        observations,
        make_room(left, right),
    )

    assert not result.fusion_enabled
    assert result.fused_observations == ()
    assert set(result.per_view_observations) == {"left", "right"}
    assert len(result.per_view_observations["left"]) == 2
    assert len(result.per_view_observations["right"]) == 1
    metrics = {metric.source_view_id: metric for metric in result.per_view_metrics}
    assert metrics["left"].observed_chord_sum == 5.0
    assert metrics["left"].chord_count == 1
    assert metrics["left"].maximum_timestamp_gap_ns == 10
    assert metrics["left"].path_completeness == (
        "unassessed_no_expected_cadence"
    )
    assert metrics["left"].observation_count == 2
    assert metrics["right"].observed_chord_sum == 0.0
    assert metrics["right"].chord_count == 0
    assert metrics["right"].maximum_timestamp_gap_ns is None
    assert metrics["right"].path_completeness == (
        "unassessed_no_expected_cadence"
    )
    assert metrics["right"].observation_count == 1


def test_per_view_timeline_rejects_duplicate_local_track_timestamp() -> None:
    view = make_view("left")
    observations = [
        make_observation(
            view,
            track_id="local-1",
            timestamp_ns=10,
            floor_point=(0, 0, 0),
        ),
        make_observation(
            view,
            track_id="local-1",
            timestamp_ns=10,
            floor_point=(5, 0, 0),
        ),
    ]

    with pytest.raises(TrajectoryFusionError, match="duplicate sample"):
        process_multiview_trajectories(
            observations,
            make_room(view),
        )


def test_local_track_rejects_conflicting_shared_identities() -> None:
    view = make_view("left")
    observations = [
        make_observation(
            view,
            track_id="local-1",
            timestamp_ns=10,
            shared_identity="child",
        ),
        make_observation(
            view,
            track_id="local-1",
            timestamp_ns=20,
            shared_identity="caregiver",
        ),
    ]

    with pytest.raises(TrajectoryFusionError, match="conflicting identities"):
        process_multiview_trajectories(
            observations,
            make_room(view),
        )


def test_opt_in_fusion_is_deterministic_and_preserves_provenance() -> None:
    left = make_view(
        "left",
        provenance={"method": "survey"},
    )
    right = make_view(
        "right",
        transform=[
            [1.0, 0.0, 0.0, 10.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        provenance={"method": "fiducials"},
    )
    observations = [
        make_observation(
            right,
            track_id="right-7",
            timestamp_ns=104,
            floor_point=(-9, 0, 0),
            shared_identity="infant",
            provenance={"detector": "qwen"},
        ),
        make_observation(
            left,
            track_id="left-2",
            timestamp_ns=100,
            floor_point=(1, 0, 0),
            shared_identity="infant",
            provenance={"detector": "yolo"},
        ),
    ]
    room = make_room(right, left)

    forward = process_multiview_trajectories(
        observations,
        room,
        fuse=True,
        timestamp_tolerance_ns=5,
    )
    reverse = process_multiview_trajectories(
        reversed(observations),
        room,
        fuse=True,
        timestamp_tolerance_ns=5,
    )

    assert [item.to_dict() for item in forward.fused_observations] == [
        item.to_dict() for item in reverse.fused_observations
    ]
    assert len(forward.fused_observations) == 1
    fused = forward.fused_observations[0]
    assert fused.shared_identity == "infant"
    assert fused.timestamp_ns == 102
    assert fused.room_floor_point == pytest.approx((1.0, 0.0, 0.0))
    assert fused.source_view_ids == ("left", "right")
    assert fused.source_camera_ids == ("camera-left", "camera-right")
    assert fused.source_floor_calibration_sha256_by_view == {
        "left": left.source_floor_calibration_sha256,
        "right": right.source_floor_calibration_sha256,
    }
    assert fused.source_observations[0].source_provenance["detector"] == "yolo"
    assert fused.source_observations[1].registration_provenance["method"] == "fiducials"
    assert all(
        len(source.registration_sha256) == 64 for source in fused.source_observations
    )
    assert fused.to_dict()["source_floor_calibration_sha256_by_view"] == {
        "left": left.source_floor_calibration_sha256,
        "right": right.source_floor_calibration_sha256,
    }


def test_fusion_uses_globally_nearest_timestamps_not_earliest_anchor() -> None:
    left = make_view("left")
    right = make_view("right")
    observations = [
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=0,
            floor_point=(0, 0, 0),
            shared_identity="infant",
        ),
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=10,
            floor_point=(10, 0, 0),
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=6,
            floor_point=(6, 0, 0),
            shared_identity="infant",
        ),
    ]

    result = process_multiview_trajectories(
        observations,
        make_room(left, right),
        fuse=True,
        timestamp_tolerance_ns=6,
    )

    assert len(result.fused_observations) == 1
    fused = result.fused_observations[0]
    assert {source.timestamp_ns for source in fused.source_observations} == {6, 10}
    assert fused.timestamp_ns == 8
    assert fused.room_floor_point == pytest.approx((8, 0, 0))
    assert fused.timestamp_matching_method == "global_nearest_unambiguous"


def test_global_nearest_fusion_forms_deterministic_multiview_groups() -> None:
    left = make_view("left")
    middle = make_view("middle")
    right = make_view("right")
    observations = [
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=10,
            shared_identity="infant",
        ),
        make_observation(
            middle,
            track_id="middle-local",
            timestamp_ns=6,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=8,
            shared_identity="infant",
        ),
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=100,
            shared_identity="infant",
        ),
        make_observation(
            middle,
            track_id="middle-local",
            timestamp_ns=103,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=101,
            shared_identity="infant",
        ),
    ]
    room = make_room(left, middle, right)

    forward = process_multiview_trajectories(
        observations,
        room,
        fuse=True,
        timestamp_tolerance_ns=6,
    )
    reverse = process_multiview_trajectories(
        reversed(observations),
        room,
        fuse=True,
        timestamp_tolerance_ns=6,
    )

    assert [item.to_dict() for item in forward.fused_observations] == [
        item.to_dict() for item in reverse.fused_observations
    ]
    assert [item.source_view_ids for item in forward.fused_observations] == [
        ("left", "middle", "right"),
        ("left", "middle", "right"),
    ]
    assert [
        {source.timestamp_ns for source in item.source_observations}
        for item in forward.fused_observations
    ] == [{6, 8, 10}, {100, 101, 103}]


def test_global_nearest_fusion_rejects_equal_distance_ambiguity() -> None:
    left = make_view("left")
    right = make_view("right")
    observations = [
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=0,
            shared_identity="infant",
        ),
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=10,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=5,
            shared_identity="infant",
        ),
    ]

    with pytest.raises(
        TrajectoryFusionError,
        match="ambiguous.*globally nearest",
    ):
        process_multiview_trajectories(
            observations,
            make_room(left, right),
            fuse=True,
            timestamp_tolerance_ns=5,
        )


def test_multiview_chain_outside_one_tolerance_window_is_ambiguous() -> None:
    left = make_view("left")
    middle = make_view("middle")
    right = make_view("right")
    observations = [
        make_observation(
            left,
            track_id="left-local",
            timestamp_ns=0,
            shared_identity="infant",
        ),
        make_observation(
            middle,
            track_id="middle-local",
            timestamp_ns=6,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-local",
            timestamp_ns=12,
            shared_identity="infant",
        ),
    ]

    with pytest.raises(TrajectoryFusionError, match="ambiguous"):
        process_multiview_trajectories(
            observations,
            make_room(left, middle, right),
            fuse=True,
            timestamp_tolerance_ns=6,
        )


def test_fusion_never_invents_identity_or_ignores_tolerance() -> None:
    left = make_view("left")
    right = make_view("right")
    room = make_room(left, right)
    observations = [
        make_observation(
            left,
            track_id="same-local-id",
            timestamp_ns=0,
            shared_identity=None,
        ),
        make_observation(
            right,
            track_id="same-local-id",
            timestamp_ns=0,
            shared_identity=None,
        ),
        make_observation(
            left,
            track_id="left-infant",
            timestamp_ns=100,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-infant",
            timestamp_ns=111,
            shared_identity="infant",
        ),
    ]

    result = process_multiview_trajectories(
        observations,
        room,
        fuse=True,
        timestamp_tolerance_ns=10,
    )

    assert result.fused_observations == ()
    assert len(result.registered_observations) == 4


def test_fusion_requires_explicit_tolerance_and_unambiguous_sources() -> None:
    left = make_view("left")
    right = make_view("right")
    room = make_room(left, right)
    observations = [
        make_observation(
            left,
            track_id="left-1",
            timestamp_ns=100,
            shared_identity="infant",
        ),
        make_observation(
            left,
            track_id="left-2",
            timestamp_ns=100,
            shared_identity="infant",
        ),
        make_observation(
            right,
            track_id="right-1",
            timestamp_ns=100,
            shared_identity="infant",
        ),
    ]

    with pytest.raises(TrajectoryFusionError, match="required"):
        process_multiview_trajectories(
            observations,
            room,
            fuse=True,
        )
    with pytest.raises(TrajectoryFusionError, match="ambiguous"):
        process_multiview_trajectories(
            observations,
            room,
            fuse=True,
            timestamp_tolerance_ns=0,
        )
