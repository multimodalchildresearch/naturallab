import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

class SmoothingMethod:
    """Enum-like class for supported smoothing methods"""
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    KALMAN = "kalman"

class EnhancedFloorTracker:
    """
    Enhanced tracker that projects the bottom center of bounding boxes to the floor
    with support for multiple smoothing methods
    """
    
    def __init__(self, camera_matrix, dist_coeffs, floor_plane, correction_factor=1.0,
                 smoothing_method=SmoothingMethod.MOVING_AVERAGE, smoothing_window=5,
                 smoothing_alpha=0.3, max_movement_threshold=500):
        """
        Initialize the enhanced tracker
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            floor_plane: Floor plane equation
            correction_factor: Factor to apply to distances (e.g., 0.83 to reduce by 17%)
            smoothing_method: Method used for smoothing (moving_average, exponential, kalman)
            smoothing_window: Window size for moving average smoothing
            smoothing_alpha: Alpha value for exponential smoothing (0-1, higher = less smoothing)
            max_movement_threshold: Maximum reasonable movement between frames (mm)
        """
        self.logger = logging.getLogger("motion_tracking.EnhancedFloorTracker")
        
        # Store configuration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.floor_plane = floor_plane
        self.correction_factor = correction_factor
        self.smoothing_method = smoothing_method
        self.smoothing_window = smoothing_window
        self.smoothing_alpha = smoothing_alpha
        self.max_movement_threshold = max_movement_threshold
        
        # Position tracking by track ID
        self.raw_positions = {}      # track_id -> list of raw positions
        self.smoothed_positions = {} # track_id -> list of smoothed positions
        
        # Distance tracking
        self.raw_positions_distances = {}   # track_id -> uncorrected raw positions distance (internal use only)
        self.raw_distances = {}             # track_id -> corrected raw distance
        self.smoothed_distances = {}        # track_id -> corrected smoothed distance
        
        # Kalman filters (if used)
        self.kalman_filters = {}     # track_id -> KalmanFilter object
        
        # ID switch tracking
        self.suspicious_movements = {} # track_id -> list of suspicious movement events
        
        self.logger.info(f"Initialized with correction factor: {correction_factor}")
        self.logger.info(f"Using smoothing method: {smoothing_method} (window={smoothing_window}, alpha={smoothing_alpha})")
        self.logger.info(f"Maximum movement threshold: {max_movement_threshold}mm")
    
    def project_to_floor(self, image_point):
        """Project a 2D image point to the 3D floor plane"""
        try:
            # Undistort point
            normalized_point = cv2.undistortPoints(
                np.array([[image_point]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # Calculate ray direction from camera
            ray_direction = np.array([
                normalized_point[0][0][0],
                normalized_point[0][0][1],
                1.0
            ])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Calculate intersection with floor plane
            normal = self.floor_plane[:3]
            d = self.floor_plane[3]
            
            denominator = np.dot(normal, ray_direction)
            if abs(denominator) < 1e-6:  # Nearly parallel to floor
                return None
                
            t = -d / denominator
            if t <= 0:  # Intersection behind camera
                return None
                
            # Calculate intersection point
            intersection = ray_direction * t
            
            return intersection
            
        except Exception as e:
            self.logger.error(f"Error in floor projection: {e}")
            return None
    
    def _init_kalman_filter(self):
        """Initialize a Kalman filter for 3D position tracking"""
        # State: [x, y, z, dx, dy, dz]
        # Measurement: [x, y, z]
        kalman = cv2.KalmanFilter(6, 3)
        
        # Transition matrix (state update matrix)
        # Position is updated by velocity, velocity stays constant
        kalman.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + dx
            [0, 1, 0, 0, 1, 0],  # y = y + dy
            [0, 0, 1, 0, 0, 1],  # z = z + dz
            [0, 0, 0, 1, 0, 0],  # dx = dx
            [0, 0, 0, 0, 1, 0],  # dy = dy
            [0, 0, 0, 0, 0, 1]   # dz = dz
        ], np.float32)
        
        # Measurement matrix (maps state to measurement)
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],  # measure x
            [0, 1, 0, 0, 0, 0],  # measure y
            [0, 0, 1, 0, 0, 0]   # measure z
        ], np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 0.1
        
        # Error covariance
        kalman.errorCovPost = np.eye(6, dtype=np.float32)
        
        return kalman
    
    def _apply_moving_average(self, positions, new_position):
        """Apply moving average smoothing to positions"""
        # Create a temporary window with the new position
        window = positions[-self.smoothing_window:] if len(positions) >= self.smoothing_window else positions[:]
        window.append(new_position)
        
        # Calculate moving average
        smoothed = np.mean(window, axis=0)
        return smoothed
    
    def _apply_exponential_smoothing(self, prev_smoothed, new_position):
        """Apply exponential smoothing to positions"""
        if prev_smoothed is None:
            return new_position
        
        # Apply exponential smoothing: smoothed = alpha*new + (1-alpha)*prev
        smoothed = self.smoothing_alpha * new_position + (1 - self.smoothing_alpha) * prev_smoothed
        return smoothed
    
    def _apply_kalman_smoothing(self, track_id, new_position):
        """Apply Kalman filter smoothing to positions"""
        # Initialize Kalman filter if needed
        if track_id not in self.kalman_filters:
            self.kalman_filters[track_id] = self._init_kalman_filter()
            # Set initial state
            self.kalman_filters[track_id].statePost = np.array([
                new_position[0], new_position[1], new_position[2], 0, 0, 0
            ], dtype=np.float32).reshape(6, 1)
            return new_position
        
        # Predict next state
        kalman = self.kalman_filters[track_id]
        predicted = kalman.predict()
        
        # Correct with measurement
        measurement = np.array(new_position, dtype=np.float32).reshape(3, 1)
        corrected = kalman.correct(measurement)
        
        # Return smoothed position [x, y, z]
        return np.array([corrected[0, 0], corrected[1, 0], corrected[2, 0]])
    
    def _smooth_position(self, track_id, new_position):
        """Apply selected smoothing method to a new position"""
        # If this is the first position, no smoothing needed
        if track_id not in self.smoothed_positions or not self.smoothed_positions[track_id]:
            return new_position
        
        if self.smoothing_method == SmoothingMethod.MOVING_AVERAGE:
            raw_positions = self.raw_positions[track_id]
            return self._apply_moving_average(raw_positions, new_position)
            
        elif self.smoothing_method == SmoothingMethod.EXPONENTIAL:
            prev_smoothed = self.smoothed_positions[track_id][-1]
            return self._apply_exponential_smoothing(prev_smoothed, new_position)
            
        elif self.smoothing_method == SmoothingMethod.KALMAN:
            return self._apply_kalman_smoothing(track_id, new_position)
            
        else:
            # No smoothing - return raw position
            return new_position
    
    def update_track(self, track_id, bbox, frame_idx=None):
        """
        Update a track with new bounding box
        
        Args:
            track_id: Track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            frame_idx: Optional frame index for logging
        """
        # Get bottom center point of bounding box
        bottom_center_x = (bbox[0] + bbox[2]) / 2
        bottom_center_y = bbox[3]  # Bottom of the bounding box
        
        # Project to floor plane
        floor_position = self.project_to_floor(np.array([bottom_center_x, bottom_center_y]))
        
        if floor_position is None:
            return None
                
        # Initialize track data if needed
        if track_id not in self.raw_positions:
            self.raw_positions[track_id] = []
            self.smoothed_positions[track_id] = []
            self.raw_positions_distances[track_id] = 0
            self.raw_distances[track_id] = 0
            self.smoothed_distances[track_id] = 0
            self.suspicious_movements[track_id] = []
        
        # Check for unreasonable movement (ID switch or tracking error)
        suspicious_movement = False
        movement_distance = 0
        
        if len(self.raw_positions[track_id]) > 0:
            last_position = self.raw_positions[track_id][-1]
            movement_distance = np.linalg.norm(floor_position - last_position)
            
            # If movement is unreasonably large, likely an ID switch or tracking error
            if movement_distance > self.max_movement_threshold:
                # Log the issue
                frame_info = f" at frame {frame_idx}" if frame_idx is not None else ""
                self.logger.warning(f"Detected potential ID switch for track {track_id[:6]}{frame_info}: "
                                  f"movement of {movement_distance:.2f}mm exceeds threshold")
                
                # Record suspicious movement
                self.suspicious_movements[track_id].append({
                    'frame': frame_idx,
                    'movement_distance': movement_distance,
                    'old_position': last_position.tolist(),
                    'new_position': floor_position.tolist()
                })
                
                # Reset position history but keep accumulated distances
                self.raw_positions[track_id] = []
                self.smoothed_positions[track_id] = []
                
                # Optionally reset Kalman filter if using it
                if self.smoothing_method == SmoothingMethod.KALMAN and track_id in self.kalman_filters:
                    del self.kalman_filters[track_id]
                    
                suspicious_movement = True
        
        # Store the new raw position
        self.raw_positions[track_id].append(floor_position)
        
        # Keep only the last 30 raw positions
        if len(self.raw_positions[track_id]) > 30:
            self.raw_positions[track_id] = self.raw_positions[track_id][-30:]
        
        # Apply smoothing and store smoothed position
        smoothed_position = self._smooth_position(track_id, floor_position)
        self.smoothed_positions[track_id].append(smoothed_position)
        
        # Keep only the last 30 smoothed positions
        if len(self.smoothed_positions[track_id]) > 30:
            self.smoothed_positions[track_id] = self.smoothed_positions[track_id][-30:]
        
        # Calculate raw distance if we have at least 2 positions and not a suspicious movement
        if len(self.raw_positions[track_id]) >= 2 and not suspicious_movement:
            # Get the last two raw positions
            current_raw_pos = self.raw_positions[track_id][-1]
            prev_raw_pos = self.raw_positions[track_id][-2]
            
            # Calculate raw position distance
            raw_pos_distance = np.linalg.norm(current_raw_pos - prev_raw_pos)
            
            # Filter out unrealistic movements
            min_movement = 5   # mm
            max_movement = 200  # mm
            
            if min_movement < raw_pos_distance < max_movement:
                # Store uncorrected distance (internal use only)
                if track_id not in self.raw_positions_distances:
                    self.raw_positions_distances[track_id] = 0
                self.raw_positions_distances[track_id] += raw_pos_distance
                
                # Apply correction and add to raw total
                corrected_raw_distance = raw_pos_distance * self.correction_factor
                self.raw_distances[track_id] += corrected_raw_distance
        
        # Calculate smoothed distance if we have at least 2 smoothed positions and not a suspicious movement
        if len(self.smoothed_positions[track_id]) >= 2 and not suspicious_movement:
            # Get the last two smoothed positions
            current_smoothed_pos = self.smoothed_positions[track_id][-1]
            prev_smoothed_pos = self.smoothed_positions[track_id][-2]
            
            # Calculate smoothed distance
            smoothed_pos_distance = np.linalg.norm(current_smoothed_pos - prev_smoothed_pos)
            
            # Filter out unrealistic movements
            min_movement = 5   # mm
            max_movement = 200  # mm
            
            if min_movement < smoothed_pos_distance < max_movement:
                # Apply correction and add to smoothed total
                corrected_smoothed_distance = smoothed_pos_distance * self.correction_factor
                self.smoothed_distances[track_id] += corrected_smoothed_distance
        
        return floor_position, smoothed_position, suspicious_movement, movement_distance
    
    def export_suspicious_movements(self, output_file):
        """
        Export all suspicious movement events to a CSV file
        
        Args:
            output_file: Path to the output CSV file
        """
        import csv
        import os
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'track_id', 
                'frame', 
                'movement_distance_mm',
                'old_x', 'old_y', 'old_z',
                'new_x', 'new_y', 'new_z'
            ])
            
            # For each track, write suspicious movements
            for track_id, movements in self.suspicious_movements.items():
                for movement in movements:
                    writer.writerow([
                        track_id,
                        movement.get('frame', 'unknown'),
                        f"{movement['movement_distance']:.2f}",
                        *movement['old_position'],
                        *movement['new_position']
                    ])
            
            self.logger.info(f"Suspicious movements exported to {output_file}")
            return True
    
    def get_raw_position(self, track_id):
        """Get the latest raw floor position for a track"""
        if track_id not in self.raw_positions or not self.raw_positions[track_id]:
            return None
        
        return self.raw_positions[track_id][-1]
    
    def get_smoothed_position(self, track_id):
        """Get the latest smoothed floor position for a track"""
        if track_id not in self.smoothed_positions or not self.smoothed_positions[track_id]:
            return None
        
        return self.smoothed_positions[track_id][-1]
    
    def get_raw_positions(self, track_id, max_points=None):
        """Get all raw floor positions for a track"""
        if track_id not in self.raw_positions:
            return []
        
        positions = self.raw_positions[track_id]
        if max_points is not None:
            return positions[-max_points:]
        return positions
    
    def get_smoothed_positions(self, track_id, max_points=None):
        """Get all smoothed floor positions for a track"""
        if track_id not in self.smoothed_positions:
            return []
        
        positions = self.smoothed_positions[track_id]
        if max_points is not None:
            return positions[-max_points:]
        return positions
    
    def get_raw_position_distance(self, track_id):
        """Get the total uncorrected raw position distance for a track (internal use only)"""
        return self.raw_positions_distances.get(track_id, 0)
    
    def get_raw_distance(self, track_id):
        """Get the total corrected raw distance for a track"""
        return self.raw_distances.get(track_id, 0)
    
    def get_smoothed_distance(self, track_id):
        """Get the total corrected smoothed distance for a track"""
        return self.smoothed_distances.get(track_id, 0)
    
    def export_distance_statistics(self, output_file):
        """
        Export comprehensive distance statistics for all tracks to a CSV file
        
        Args:
            output_file: Path to the output CSV file
        """
        import csv
        import os
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow([
                'track_id', 
                'total_raw_distance_mm', 
                'total_smoothed_distance_mm',
                'avg_velocity_mm_per_frame',
                'tracking_duration_frames',
                'distance_difference_percent'
            ])
            
            # For each track, write statistics
            for track_id in self.raw_distances.keys():
                raw_distance = self.get_raw_distance(track_id)
                smoothed_distance = self.get_smoothed_distance(track_id)
                
                # Calculate frames tracked (based on raw positions)
                frames_tracked = len(self.raw_positions.get(track_id, []))
                
                # Calculate average velocity (mm per frame)
                avg_velocity = 0
                if frames_tracked > 1:
                    avg_velocity = raw_distance / frames_tracked
                
                # Calculate distance difference as percentage
                diff_percentage = 0
                if raw_distance > 0:
                    diff_percentage = ((smoothed_distance - raw_distance) / raw_distance) * 100
                
                # Write row
                writer.writerow([
                    track_id,
                    f"{raw_distance:.2f}",
                    f"{smoothed_distance:.2f}",
                    f"{avg_velocity:.2f}",
                    frames_tracked,
                    f"{diff_percentage:.2f}"
                ])
            
            self.logger.info(f"Distance statistics exported to {output_file}")
            return True
        
    def reset(self):
        """Reset the tracker"""
        self.raw_positions.clear()
        self.smoothed_positions.clear()
        self.raw_positions_distances.clear()
        self.raw_distances.clear()
        self.smoothed_distances.clear()
        self.kalman_filters.clear()
        self.logger.info("Floor tracker reset")


# Keep the original SimpleFloorTracker for backward compatibility
class SimpleFloorTracker:
    """
    Simple tracker that projects the bottom center of bounding boxes to the floor
    """
    
    def __init__(self, camera_matrix, dist_coeffs, floor_plane, correction_factor=1.0):
        """
        Initialize the simple tracker
        
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            floor_plane: Floor plane equation
            correction_factor: Factor to apply to distances (e.g., 0.83 to reduce by 17%)
        """
        self.logger = logging.getLogger("motion_tracking.SimpleFloorTracker")
        
        # Store configuration
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.floor_plane = floor_plane
        self.correction_factor = correction_factor
        
        # Position tracking by track ID
        self.positions = {}  # track_id -> list of positions
        self.distances = {}  # track_id -> total distance
        self.raw_distances = {}  # track_id -> uncorrected total distance
        
        self.logger.info(f"Initialized with correction factor: {correction_factor}")
    
    def project_to_floor(self, image_point):
        """Project a 2D image point to the 3D floor plane"""
        try:
            # Undistort point
            normalized_point = cv2.undistortPoints(
                np.array([[image_point]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            
            # Calculate ray direction from camera
            ray_direction = np.array([
                normalized_point[0][0][0],
                normalized_point[0][0][1],
                1.0
            ])
            ray_direction = ray_direction / np.linalg.norm(ray_direction)
            
            # Calculate intersection with floor plane
            normal = self.floor_plane[:3]
            d = self.floor_plane[3]
            
            denominator = np.dot(normal, ray_direction)
            if abs(denominator) < 1e-6:  # Nearly parallel to floor
                return None
                
            t = -d / denominator
            if t <= 0:  # Intersection behind camera
                return None
                
            # Calculate intersection point
            intersection = ray_direction * t
            
            return intersection
            
        except Exception as e:
            self.logger.error(f"Error in floor projection: {e}")
            return None
    
    def update_track(self, track_id, bbox):
        """Update a track with new bounding box"""
        # Get bottom center point of bounding box
        bottom_center_x = (bbox[0] + bbox[2]) / 2
        bottom_center_y = bbox[3]  # Bottom of the bounding box
        
        # Project to floor plane
        floor_position = self.project_to_floor(np.array([bottom_center_x, bottom_center_y]))
        
        if floor_position is None:
            return None
            
        # Initialize track data if needed
        if track_id not in self.positions:
            self.positions[track_id] = []
            self.distances[track_id] = 0
            self.raw_distances[track_id] = 0
        
        # Store the new position
        self.positions[track_id].append(floor_position)
        
        # Keep only the last 30 positions
        if len(self.positions[track_id]) > 30:
            self.positions[track_id] = self.positions[track_id][-30:]
        
        # Calculate distance if we have at least 2 positions
        if len(self.positions[track_id]) >= 2:
            # Get the last two positions
            current_pos = self.positions[track_id][-1]
            prev_pos = self.positions[track_id][-2]
            
            # Calculate raw distance
            raw_distance = np.linalg.norm(current_pos - prev_pos)
            
            # Filter out unrealistic movements
            min_movement = 5   # mm
            max_movement = 200  # mm
            
            if min_movement < raw_distance < max_movement:
                # Add to raw total
                self.raw_distances[track_id] += raw_distance
                
                # Apply correction and add to corrected total
                corrected_distance = raw_distance * self.correction_factor
                self.distances[track_id] += corrected_distance
        
        return floor_position
    
    def get_position(self, track_id):
        """Get the latest floor position for a track"""
        if track_id not in self.positions or not self.positions[track_id]:
            return None
        
        return self.positions[track_id][-1]
    
    def get_distance(self, track_id, raw=False):
        """Get the total distance for a track"""
        if raw:
            return self.raw_distances.get(track_id, 0)
        else:
            return self.distances.get(track_id, 0)
    
    def reset(self):
        """Reset the tracker"""
        self.positions.clear()
        self.distances.clear()
        self.raw_distances.clear()
        self.logger.info("Floor tracker reset")


def add_correction_to_movement_data(movement_data, correction_factor):
    """Add corrected distances to the original movement data"""
    for track_id, data in movement_data.items():
        # Store original distance
        if 'original_distance' not in data:
            data['original_distance'] = data.get('total_distance', 0)
            
        # Apply correction factor
        data['total_distance'] = data['original_distance'] * correction_factor
    
    return movement_data