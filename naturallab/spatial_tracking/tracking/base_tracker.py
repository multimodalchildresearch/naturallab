import numpy as np
import uuid
from typing import Dict, List, Tuple, Any, Optional
from filterpy.kalman import KalmanFilter

from motion_tracking.base import TrackerModule
from motion_tracking.utils.data_structures import generate_track_id

class BaseTracker(TrackerModule):
    """Base class for Kalman filter-based trackers"""
    
    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize the base tracker
        
        Args:
            max_age: Maximum frames to keep a track alive without matching detections
            min_hits: Minimum hits to consider a track confirmed
            iou_threshold: IOU threshold for detection matching
        """
        super().__init__(name="BaseTracker")
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        # Tracking state
        self.tracks = []
        self.frame_count = 0
        
        # Color assignment
        self.available_colors = [
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        
        self.log_info(f"Initialized with max_age={max_age}, min_hits={min_hits}, iou_threshold={iou_threshold}")
    
    def _iou(self, bb1: List[float], bb2: List[float]) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        
        Args:
            bb1, bb2: Bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            IOU score (0-1)
        """
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
        
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return iou
    
    def _create_kalman_filter(self, bbox: List[float]) -> KalmanFilter:
        """
        Initialize a Kalman filter for a new track
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2, score]
            
        Returns:
            Initialized KalmanFilter
        """
        # State: [x, y, a, h, vx, vy, va, vh] center position, area, height, and their velocities
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # Initial state
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        height = bbox[3] - bbox[1]
        
        kf.x = np.array([x_center, y_center, area, height, 0, 0, 0, 0]).T
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # a = a + va
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # va = va
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        
        # Measurement matrix (we can only observe position and size)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],  # x
            [0, 1, 0, 0, 0, 0, 0, 0],  # y
            [0, 0, 1, 0, 0, 0, 0, 0],  # a 
            [0, 0, 0, 1, 0, 0, 0, 0],  # h
        ])
        
        # Process uncertainty
        kf.P = np.eye(8) * 10
        kf.P[4:, 4:] *= 100  # Higher uncertainty for velocity
        
        # Process noise
        kf.Q = np.eye(8)
        kf.Q[0:4, 0:4] *= 0.01  # Process noise for position and size
        kf.Q[4:8, 4:8] *= 0.1   # Process noise for velocity
        
        # Measurement noise
        kf.R = np.eye(4) * 1
        
        return kf
    
    def _bbox_from_kalman(self, kf: KalmanFilter) -> List[float]:
        """
        Extract bounding box from Kalman filter state
        
        Args:
            kf: Kalman filter object
            
        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        x_center, y_center, area, height = kf.x[:4]
        width = area / height
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        # Handle both scalar and array cases safely
        try:
            return [float(x1[0]), float(y1[0]), float(x2[0]), float(y2[0])]
        except (IndexError, TypeError):
            # If they're already scalars, this will catch the error
            return [float(x1), float(y1), float(x2), float(y2)]
    
    def _update_kalman(self, kf: KalmanFilter, bbox: List[float]) -> None:
        """
        Update Kalman filter with new measurement
        
        Args:
            kf: Kalman filter to update
            bbox: New bounding box measurement [x1, y1, x2, y2, score]
        """
        x_center = (bbox[0] + bbox[2]) / 2
        y_center = (bbox[1] + bbox[3]) / 2
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        height = bbox[3] - bbox[1]
        
        # Measurement
        z = np.array([x_center, y_center, area, height])
        
        # Update
        kf.update(z)
    
    def _create_new_track(self, detection: List[float]) -> Dict[str, Any]:
        """
        Create a new track from a detection
        
        Args:
            detection: Detection in format [x1, y1, x2, y2, score]
            
        Returns:
            New track dictionary
        """
        track_id = generate_track_id()
        color = self.available_colors[len(self.tracks) % len(self.available_colors)]
        
        return {
            'id': track_id,
            'kalman': self._create_kalman_filter(detection),
            'bbox': detection[:4],
            'score': detection[4],
            'time_since_update': 0,
            'hits': 1,
            'color': color
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update tracks with new detections
        
        This is a basic implementation. Subclasses should override this method.
        
        Args:
            data: Dictionary containing 'frame' and 'detections'
            
        Returns:
            Dictionary with 'tracks' containing current track information
        """
        # Implement the basic tracking logic in subclasses
        raise NotImplementedError("Subclasses must implement process method")
    
    def reset(self) -> None:
        """Reset the tracker state"""
        self.tracks = []
        self.frame_count = 0
        self.log_info("Tracker reset")