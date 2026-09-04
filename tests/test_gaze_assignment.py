import pytest

from naturallab.gaze_analysis import (
    GazeSample,
    ObjectObservation,
    TimedRecord,
    align_streams,
    assign_gaze_to_objects,
)


def object_box(
    observation_id: str,
    timestamp: float,
    bbox=(10.0, 10.0, 60.0, 60.0),
    *,
    category="toy",
) -> ObjectObservation:
    return ObjectObservation(
        observation_id=observation_id,
        timestamp_seconds=timestamp,
        view_id="scene-camera",
        bbox_xyxy=bbox,
        category=category,
        track_id=f"track-{observation_id}",
    )


def test_gaze_assignment_uses_nearest_frame_within_tolerance() -> None:
    assignments = assign_gaze_to_objects(
        [
            GazeSample(
                sample_id="gaze-1",
                timestamp_seconds=1.02,
                view_id="scene-camera",
                x=30,
                y=20,
            )
        ],
        [
            object_box("early", 0.9),
            object_box("nearest", 1.0),
            object_box("late", 1.1),
        ],
        timestamp_tolerance_seconds=0.03,
    )

    assignment = assignments[0]
    assert assignment.assigned is True
    assert assignment.object_observation_id == "nearest"
    assert assignment.time_delta_seconds == pytest.approx(-0.02)
    assert assignment.provenance.as_dict() == {
        "algorithm": "naturallab-gaze-object/v1",
        "timestamp_tolerance_seconds": 0.03,
        "overlap_policy": "abstain",
        "image_sizes": {},
    }


def test_normalized_gaze_requires_geometry_and_converts_to_pixels() -> None:
    sample = GazeSample(
        sample_id="gaze-1",
        timestamp_seconds=1.0,
        view_id="scene-camera",
        x=0.25,
        y=0.5,
        coordinate_space="normalized",
    )
    observation = object_box(
        "object-1",
        1.0,
        bbox=(40.0, 40.0, 60.0, 60.0),
    )

    with pytest.raises(ValueError, match="image_sizes"):
        assign_gaze_to_objects([sample], [observation])

    assignment = assign_gaze_to_objects(
        [sample],
        [observation],
        image_sizes={"scene-camera": (200, 100)},
    )[0]
    assert assignment.object_observation_id == "object-1"
    assert assignment.provenance.as_dict()["image_sizes"] == {
        "scene-camera": [200, 100]
    }


def test_overlap_abstains_by_default_and_can_choose_smallest_box() -> None:
    sample = GazeSample(
        sample_id="gaze-1",
        timestamp_seconds=1.0,
        view_id="scene-camera",
        x=30,
        y=30,
    )
    objects = [
        object_box("large", 1.0, bbox=(0, 0, 100, 100)),
        object_box("small", 1.0, bbox=(20, 20, 40, 40)),
    ]

    abstained = assign_gaze_to_objects([sample], objects)[0]
    assert abstained.assigned is False
    assert abstained.reason == "ambiguous_object_overlap"
    assert abstained.candidate_object_ids == ("large", "small")

    selected = assign_gaze_to_objects(
        [sample],
        objects,
        overlap_policy="smallest_box",
    )[0]
    assert selected.object_observation_id == "small"
    assert selected.candidate_object_ids == ("large", "small")


def test_invalid_and_unsynchronized_gaze_are_explicit() -> None:
    samples = [
        GazeSample(
            sample_id="invalid",
            timestamp_seconds=0.0,
            view_id="scene-camera",
            x=0,
            y=0,
            valid=False,
        ),
        GazeSample(
            sample_id="late",
            timestamp_seconds=2.0,
            view_id="scene-camera",
            x=20,
            y=20,
        ),
    ]
    assignments = assign_gaze_to_objects(
        samples,
        [object_box("object-1", 1.0)],
    )

    assert assignments[0].reason == "invalid_gaze_sample"
    assert assignments[1].reason == "no_synchronized_object_frame"
    assert assignments[1].object_timestamp_seconds == 1.0


def test_gaze_assignment_rejects_duplicate_ids() -> None:
    duplicate = GazeSample(
        sample_id="same",
        timestamp_seconds=1.0,
        view_id="scene-camera",
        x=0,
        y=0,
    )
    with pytest.raises(ValueError, match="duplicate gaze"):
        assign_gaze_to_objects([duplicate, duplicate], [])


def record(stream: str, record_id: str, timestamp: float) -> TimedRecord:
    return TimedRecord(
        stream_id=stream,
        record_id=record_id,
        timestamp_seconds=timestamp,
        values={"value": record_id},
    )


def test_multimodal_alignment_is_nearest_deterministic_and_auditable() -> None:
    aligned = align_streams(
        [record("gaze", "gaze-1", 1.0)],
        {
            "imu": [
                record("imu", "later", 1.02),
                record("imu", "earlier", 0.98),
            ],
            "audio": [record("audio", "audio-1", 1.2)],
        },
        tolerance_seconds={"imu": 0.03, "audio": 0.05},
        required_stream_ids=("imu", "audio"),
    )[0]

    assert aligned.matches["imu"].record_id == "earlier"
    assert aligned.time_deltas_seconds["imu"] == pytest.approx(-0.02)
    assert aligned.matches["audio"] is None
    assert aligned.time_deltas_seconds["audio"] == pytest.approx(0.2)
    assert aligned.missing_required_streams == ("audio",)
    assert aligned.complete is False
    assert aligned.as_dict()["algorithm"] == (
        "naturallab-nearest-timestamp/v1"
    )
    assert aligned.as_dict()["tolerance_seconds"] == {
        "audio": 0.05,
        "imu": 0.03,
    }
    assert aligned.as_dict()["required_stream_ids"] == ["audio", "imu"]


def test_multimodal_ties_use_smallest_record_id_at_same_timestamp() -> None:
    aligned = align_streams(
        [record("gaze", "anchor", 1.1)],
        {
            "imu": [
                record("imu", "z-record", 1.0),
                record("imu", "a-record", 1.0),
            ]
        },
        tolerance_seconds=0.2,
    )[0]

    assert aligned.matches["imu"].record_id == "a-record"


def test_multimodal_stream_keys_are_normalized_once() -> None:
    aligned = align_streams(
        [record("gaze", "anchor", 1.0)],
        {" imu ": [record("imu", "sample", 1.0)]},
        tolerance_seconds={"imu": 0.1},
        required_stream_ids=(" imu ",),
    )[0]

    assert aligned.matches["imu"].record_id == "sample"
    assert aligned.required_stream_ids == ("imu",)

    with pytest.raises(ValueError, match="unique after whitespace"):
        align_streams(
            [record("gaze", "anchor", 1.0)],
            {
                "imu": [],
                " imu ": [],
            },
            tolerance_seconds=0.1,
        )


def test_multimodal_alignment_rejects_mislabeled_stream_records() -> None:
    with pytest.raises(ValueError, match="expected"):
        align_streams(
            [record("gaze", "gaze-1", 1.0)],
            {"imu": [record("audio", "wrong", 1.0)]},
            tolerance_seconds=0.1,
        )
