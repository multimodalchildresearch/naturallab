import numpy as np

from naturallab.spatial_tracking.movement.distance_calculator import (
    DistanceCalculator,
)


def test_metrics_use_the_steps_accepted_during_ingestion() -> None:
    calculator = DistanceCalculator()
    calculator.add_position("track-1", np.array([0.0, 0.0]))
    calculator.add_position("track-1", np.array([10.0, 0.0]))
    calculator.add_position("track-1", np.array([500.0, 0.0]))

    metrics = calculator.get_distance_metrics("track-1")

    assert metrics["total_distance"] == 10.0
    assert metrics["average_step"] == 10.0
    assert metrics["min_step"] == 10.0
    assert metrics["max_step"] == 10.0


def test_reset_removes_accepted_step_history() -> None:
    calculator = DistanceCalculator()
    calculator.add_position("track-1", np.array([0.0, 0.0]))
    calculator.add_position("track-1", np.array([10.0, 0.0]))

    calculator.reset("track-1")

    assert "track-1" not in calculator.accepted_steps
