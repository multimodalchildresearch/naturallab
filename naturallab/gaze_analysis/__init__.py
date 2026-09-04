"""Timestamp-aware gaze and multimodal analysis contracts."""

from .assignment import (
    GazeAssignmentProvenance,
    GazeObjectAssignment,
    GazeSample,
    ObjectObservation,
    assign_gaze_to_objects,
)
from .multimodal import AlignedRecordSet, TimedRecord, align_streams

__all__ = [
    "AlignedRecordSet",
    "GazeAssignmentProvenance",
    "GazeObjectAssignment",
    "GazeSample",
    "ObjectObservation",
    "TimedRecord",
    "align_streams",
    "assign_gaze_to_objects",
]
