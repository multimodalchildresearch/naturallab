"""Deterministic timestamp alignment for arbitrary modality streams."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


MULTIMODAL_ALIGNMENT_ALGORITHM = "naturallab-nearest-timestamp/v1"


def _identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    return value.strip()


def _timestamp(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    value = float(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return value


@dataclass(frozen=True)
class TimedRecord:
    """One immutable record from a named modality stream."""

    stream_id: str
    record_id: str
    timestamp_seconds: float
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stream_id",
            _identifier(self.stream_id, "stream_id"),
        )
        object.__setattr__(
            self,
            "record_id",
            _identifier(self.record_id, "record_id"),
        )
        object.__setattr__(
            self,
            "timestamp_seconds",
            _timestamp(self.timestamp_seconds, "timestamp_seconds"),
        )
        if not isinstance(self.values, Mapping):
            raise ValueError("values must be a mapping")
        object.__setattr__(self, "values", dict(self.values))


@dataclass(frozen=True)
class AlignedRecordSet:
    """One anchor record and its nearest records from other streams."""

    anchor: TimedRecord
    matches: Mapping[str, Optional[TimedRecord]]
    time_deltas_seconds: Mapping[str, Optional[float]]
    missing_required_streams: Tuple[str, ...]
    tolerance_seconds: Mapping[str, float]
    required_stream_ids: Tuple[str, ...]
    algorithm: str = MULTIMODAL_ALIGNMENT_ALGORITHM

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "matches",
            MappingProxyType(dict(self.matches)),
        )
        object.__setattr__(
            self,
            "time_deltas_seconds",
            MappingProxyType(dict(self.time_deltas_seconds)),
        )
        object.__setattr__(
            self,
            "tolerance_seconds",
            MappingProxyType(dict(self.tolerance_seconds)),
        )

    @property
    def complete(self) -> bool:
        return not self.missing_required_streams

    def as_dict(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "anchor_stream_id": self.anchor.stream_id,
            "anchor_record_id": self.anchor.record_id,
            "anchor_timestamp_seconds": self.anchor.timestamp_seconds,
            "matches": {
                stream_id: (
                    None
                    if record is None
                    else {
                        "record_id": record.record_id,
                        "timestamp_seconds": record.timestamp_seconds,
                        "values": dict(record.values),
                    }
                )
                for stream_id, record in self.matches.items()
            },
            "time_deltas_seconds": dict(self.time_deltas_seconds),
            "missing_required_streams": list(
                self.missing_required_streams
            ),
            "tolerance_seconds": dict(self.tolerance_seconds),
            "required_stream_ids": list(self.required_stream_ids),
            "complete": self.complete,
        }


def _nearest_record(
    records: Sequence[TimedRecord],
    timestamps: Sequence[float],
    target: float,
) -> Optional[TimedRecord]:
    if not records:
        return None
    index = bisect_left(timestamps, target)
    candidates = []
    if index < len(records):
        candidates.append(records[index])
    if index:
        previous_timestamp = timestamps[index - 1]
        previous_index = bisect_left(timestamps, previous_timestamp)
        candidates.append(records[previous_index])
    return min(
        candidates,
        key=lambda record: (
            abs(record.timestamp_seconds - target),
            record.timestamp_seconds,
            record.record_id,
        ),
    )


def align_streams(
    anchor_records: Iterable[TimedRecord],
    streams: Mapping[str, Iterable[TimedRecord]],
    *,
    tolerance_seconds: float | Mapping[str, float],
    required_stream_ids: Iterable[str] = (),
) -> Tuple[AlignedRecordSet, ...]:
    """Align each anchor independently; records may be reused across anchors."""

    if not isinstance(streams, Mapping):
        raise ValueError("streams must be a mapping")
    normalized_streams: Dict[str, Iterable[TimedRecord]] = {}
    for stream_id, records in streams.items():
        normalized_stream_id = _identifier(stream_id, "stream ID")
        if normalized_stream_id in normalized_streams:
            raise ValueError(
                "stream IDs must be unique after whitespace normalization: "
                f"{normalized_stream_id!r}"
            )
        normalized_streams[normalized_stream_id] = records

    required = tuple(
        sorted(
            {
                _identifier(stream_id, "required stream ID")
                for stream_id in required_stream_ids
            }
        )
    )
    unknown_required = set(required) - set(normalized_streams)
    if unknown_required:
        raise ValueError(
            "required streams are absent from streams: "
            + ", ".join(sorted(unknown_required))
        )

    if isinstance(tolerance_seconds, Mapping):
        normalized_tolerances: Dict[str, float] = {}
        for stream_id, tolerance in tolerance_seconds.items():
            normalized_stream_id = _identifier(
                stream_id,
                "tolerance stream ID",
            )
            if normalized_stream_id in normalized_tolerances:
                raise ValueError(
                    "tolerance stream IDs must be unique after whitespace "
                    f"normalization: {normalized_stream_id!r}"
                )
            normalized_tolerances[normalized_stream_id] = _timestamp(
                tolerance,
                "tolerance_seconds",
            )
        unknown_tolerances = (
            set(normalized_tolerances) - set(normalized_streams)
        )
        missing_tolerances = (
            set(normalized_streams) - set(normalized_tolerances)
        )
        if unknown_tolerances or missing_tolerances:
            raise ValueError(
                "tolerance mapping must contain exactly the stream IDs"
            )
        tolerances = normalized_tolerances
    else:
        tolerance = _timestamp(tolerance_seconds, "tolerance_seconds")
        tolerances = {
            stream_id: tolerance for stream_id in normalized_streams
        }

    indexed_streams: Dict[str, Tuple[TimedRecord, ...]] = {}
    timestamps_by_stream: Dict[str, Tuple[float, ...]] = {}
    seen_record_ids = set()
    for normalized_stream_id in sorted(normalized_streams):
        records = []
        for record in normalized_streams[normalized_stream_id]:
            if not isinstance(record, TimedRecord):
                raise TypeError("streams must contain TimedRecord values")
            if record.stream_id != normalized_stream_id:
                raise ValueError(
                    f"record {record.record_id!r} declares stream "
                    f"{record.stream_id!r}, expected {normalized_stream_id!r}"
                )
            identity = (record.stream_id, record.record_id)
            if identity in seen_record_ids:
                raise ValueError(
                    "duplicate record ID within stream: "
                    f"{record.stream_id}/{record.record_id}"
                )
            seen_record_ids.add(identity)
            records.append(record)
        ordered = tuple(
            sorted(
                records,
                key=lambda record: (
                    record.timestamp_seconds,
                    record.record_id,
                ),
            )
        )
        indexed_streams[normalized_stream_id] = ordered
        timestamps_by_stream[normalized_stream_id] = tuple(
            record.timestamp_seconds for record in ordered
        )

    anchors = []
    seen_anchors = set()
    for anchor in anchor_records:
        if not isinstance(anchor, TimedRecord):
            raise TypeError("anchor_records must contain TimedRecord values")
        identity = (anchor.stream_id, anchor.record_id)
        if identity in seen_anchors:
            raise ValueError(
                f"duplicate anchor record: {anchor.stream_id}/{anchor.record_id}"
            )
        seen_anchors.add(identity)
        anchors.append(anchor)

    aligned = []
    for anchor in sorted(
        anchors,
        key=lambda record: (
            record.timestamp_seconds,
            record.record_id,
        ),
    ):
        matches: Dict[str, Optional[TimedRecord]] = {}
        deltas: Dict[str, Optional[float]] = {}
        for stream_id in sorted(indexed_streams):
            match = _nearest_record(
                indexed_streams[stream_id],
                timestamps_by_stream[stream_id],
                anchor.timestamp_seconds,
            )
            delta = (
                None
                if match is None
                else match.timestamp_seconds - anchor.timestamp_seconds
            )
            if (
                match is None
                or delta is None
                or abs(delta) > tolerances[stream_id]
            ):
                matches[stream_id] = None
                deltas[stream_id] = delta
            else:
                matches[stream_id] = match
                deltas[stream_id] = delta
        missing = tuple(
            stream_id
            for stream_id in required
            if matches[stream_id] is None
        )
        aligned.append(
            AlignedRecordSet(
                anchor=anchor,
                matches=matches,
                time_deltas_seconds=deltas,
                missing_required_streams=missing,
                tolerance_seconds=tolerances,
                required_stream_ids=required,
            )
        )
    return tuple(aligned)
