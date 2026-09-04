"""Projection of tracked image boxes onto a calibrated floor plane."""

from __future__ import annotations

import logging

import cv2
import numpy as np


class SimpleFloorTracker:
    """Project box foot points and accumulate distance in calibration units."""

    def __init__(
        self,
        camera_matrix,
        dist_coeffs,
        floor_plane,
        units="mm",
    ):
        self.logger = logging.getLogger(
            "naturallab.spatial_tracking.SimpleFloorTracker"
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.floor_plane = floor_plane
        self.units = units
        self.positions = {}
        self.distances = {}
        self.projection_attempts = {}
        self.valid_projections = {}
        self.projection_gaps = {}
        self._projection_gap_open = {}

    def project_to_floor(self, image_point):
        """Intersect one undistorted camera ray with the calibrated plane."""

        try:
            normalized_point = cv2.undistortPoints(
                np.array([[image_point]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs,
            )
            ray_direction = np.array(
                [
                    normalized_point[0][0][0],
                    normalized_point[0][0][1],
                    1.0,
                ]
            )
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            normal = self.floor_plane[:3]
            plane_offset = self.floor_plane[3]
            denominator = np.dot(normal, ray_direction)
            if abs(denominator) < 1e-6:
                return None
            distance_along_ray = -plane_offset / denominator
            if distance_along_ray <= 0:
                return None
            return ray_direction * distance_along_ray
        except Exception as error:
            self.logger.error("Error in floor projection: %s", error)
            return None

    def update_track(self, track_id, bbox):
        """Project a new box and add its consecutive path displacement."""

        if track_id not in self.positions:
            self.positions[track_id] = []
            self.distances[track_id] = 0.0
            self.projection_attempts[track_id] = 0
            self.valid_projections[track_id] = 0
            self.projection_gaps[track_id] = 0
            self._projection_gap_open[track_id] = False
        self.projection_attempts[track_id] += 1

        bottom_center_x = (bbox[0] + bbox[2]) / 2
        bottom_center_y = bbox[3]
        floor_position = self.project_to_floor(
            np.array([bottom_center_x, bottom_center_y])
        )
        if floor_position is None:
            self._mark_projection_gap(track_id)
            return None
        floor_position = np.asarray(floor_position, dtype=float)
        if floor_position.shape != (3,) or not np.all(
            np.isfinite(floor_position)
        ):
            self._mark_projection_gap(track_id)
            return None

        self.valid_projections[track_id] += 1
        self._projection_gap_open[track_id] = False
        self.positions[track_id].append(floor_position)
        if len(self.positions[track_id]) > 30:
            self.positions[track_id] = self.positions[track_id][-30:]

        if len(self.positions[track_id]) >= 2:
            current_position = self.positions[track_id][-1]
            previous_position = self.positions[track_id][-2]
            step_distance = float(
                np.linalg.norm(current_position - previous_position)
            )
            if np.isfinite(step_distance) and step_distance >= 0.0:
                self.distances[track_id] += step_distance
        return floor_position

    def _mark_projection_gap(self, track_id):
        """Record one contiguous run of failed projection attempts."""

        if not self._projection_gap_open[track_id]:
            self.projection_gaps[track_id] += 1
            self._projection_gap_open[track_id] = True

    def get_position(self, track_id):
        """Return the latest projected position for a track, if available."""

        if track_id not in self.positions or not self.positions[track_id]:
            return None
        return self.positions[track_id][-1]

    def get_distance(self, track_id):
        """Return path distance in the calibration's metric unit."""

        return self.distances.get(track_id, 0.0)

    def get_projection_summary(self, track_id):
        """Return coverage and completeness for one track's metric path."""

        attempted = self.projection_attempts.get(track_id, 0)
        valid = self.valid_projections.get(track_id, 0)
        missed = attempted - valid
        gaps = self.projection_gaps.get(track_id, 0)
        coverage = valid / attempted if attempted else 0.0
        if attempted == 0:
            status = "not_attempted"
        elif valid == 0:
            status = "unavailable_no_valid_projection"
        elif missed:
            status = "partial_projection_gaps"
        else:
            status = "complete"
        return {
            "floor_projection_attempts": attempted,
            "floor_projection_valid": valid,
            "floor_projection_missed": missed,
            "floor_projection_gap_count": gaps,
            "floor_projection_coverage": coverage,
            "distance_complete": status == "complete",
            "distance_status": status,
        }

    def reset(self):
        """Clear all projected positions and distances."""

        self.positions.clear()
        self.distances.clear()
        self.projection_attempts.clear()
        self.valid_projections.clear()
        self.projection_gaps.clear()
        self._projection_gap_open.clear()
        self.logger.info("Floor tracker reset")


__all__ = ["SimpleFloorTracker"]
