import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable

class DistanceCalculator:
    """Calculate, track, and correct distances between positions"""
    
    def __init__(self, correction_factor: float = 1.0):
        """
        Initialize the distance calculator
        
        Args:
            correction_factor: Factor to apply to raw distances (e.g., 0.83 reduces by 17%)
        """
        self.logger = logging.getLogger("motion_tracking.DistanceCalculator")
        self.correction_factor = correction_factor
        self.total_distances = {}  # track_id -> total distance
        self.raw_distances = {}    # track_id -> uncorrected distance
        self.distance_history = {} # track_id -> list of distances
        
        self.logger.info(f"Initialized with correction factor: {correction_factor}")
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray, 
                         method: str = 'euclidean') -> float:
        """
        Calculate distance between two positions using specified method
        
        Args:
            pos1: First position (numpy array)
            pos2: Second position (numpy array)
            method: Distance calculation method ('euclidean', 'manhattan', 'cosine')
            
        Returns:
            Distance value (float)
        """
        if pos1 is None or pos2 is None:
            return 0.0
        
        if method == 'euclidean':
            return np.linalg.norm(pos2 - pos1)
        elif method == 'manhattan':
            return np.sum(np.abs(pos2 - pos1))
        elif method == 'cosine':
            # Cosine similarity converted to distance
            dot_product = np.dot(pos1, pos2)
            norm_product = np.linalg.norm(pos1) * np.linalg.norm(pos2)
            if norm_product == 0:
                return 0.0
            similarity = dot_product / norm_product
            # Convert similarity [-1,1] to distance [0,2]
            return 1.0 - similarity
        else:
            self.logger.warning(f"Unknown distance method: {method}, using euclidean")
            return np.linalg.norm(pos2 - pos1)
    
    def calculate_path_distance(self, positions: List[np.ndarray], 
                              method: str = 'euclidean',
                              apply_correction: bool = True) -> float:
        """
        Calculate the total distance along a path of positions
        
        Args:
            positions: List of positions
            method: Distance calculation method
            apply_correction: Whether to apply the correction factor
            
        Returns:
            Total path distance
        """
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        
        for i in range(1, len(positions)):
            segment_distance = self.calculate_distance(
                positions[i-1], positions[i], method)
            total_distance += segment_distance
        
        # Apply correction if requested
        if apply_correction:
            total_distance *= self.correction_factor
            
        return total_distance
    
    def add_position(self, track_id: str, position: np.ndarray, 
                   min_movement: float = 5.0,
                   max_movement: float = 200.0) -> Tuple[float, float]:
        """
        Add a new position for a track and update distance metrics
        
        Args:
            track_id: Unique track identifier
            position: New position
            min_movement: Minimum distance to consider movement (mm)
            max_movement: Maximum allowed distance per frame (mm)
            
        Returns:
            Tuple of (raw_distance_added, corrected_distance_added)
        """
        if position is None:
            return 0.0, 0.0
            
        # Initialize track data if needed
        if track_id not in self.distance_history:
            self.distance_history[track_id] = []
            self.total_distances[track_id] = 0.0
            self.raw_distances[track_id] = 0.0
        
        # Store position
        self.distance_history[track_id].append(position.copy())
        
        # If we don't have at least 2 positions, no distance to calculate
        if len(self.distance_history[track_id]) < 2:
            return 0.0, 0.0
        
        # Get the last two positions
        current_pos = self.distance_history[track_id][-1]
        prev_pos = self.distance_history[track_id][-2]
        
        # Calculate raw distance
        raw_distance = self.calculate_distance(prev_pos, current_pos)
        
        # Only count distances within reasonable range
        if min_movement <= raw_distance <= max_movement:
            # Update raw distance
            self.raw_distances[track_id] += raw_distance
            
            # Apply correction and update total
            corrected_distance = raw_distance * self.correction_factor
            self.total_distances[track_id] += corrected_distance
            
            return raw_distance, corrected_distance
        
        return 0.0, 0.0
    
    def get_total_distance(self, track_id: str, raw: bool = False) -> float:
        """
        Get the total distance for a track
        
        Args:
            track_id: Track identifier
            raw: Whether to return raw (uncorrected) distance
            
        Returns:
            Total distance value
        """
        if raw:
            return self.raw_distances.get(track_id, 0.0)
        else:
            return self.total_distances.get(track_id, 0.0)
    
    def get_distance_metrics(self, track_id: str) -> Dict[str, float]:
        """
        Get comprehensive distance metrics for a track
        
        Args:
            track_id: Track identifier
            
        Returns:
            Dictionary with various distance metrics
        """
        if track_id not in self.distance_history:
            return {
                'total_distance': 0.0,
                'raw_distance': 0.0,
                'average_step': 0.0,
                'max_step': 0.0,
                'min_step': 0.0,
                'direct_distance': 0.0,
                'path_efficiency': 0.0
            }
        
        positions = self.distance_history.get(track_id, [])
        total_distance = self.total_distances.get(track_id, 0.0)
        raw_distance = self.raw_distances.get(track_id, 0.0)
        
        # Calculate per-step statistics if we have at least 2 positions
        avg_step = 0.0
        max_step = 0.0
        min_step = float('inf')
        
        if len(positions) >= 2:
            steps = []
            for i in range(1, len(positions)):
                step_distance = self.calculate_distance(positions[i-1], positions[i])
                if min_movement <= step_distance <= max_movement:
                    steps.append(step_distance)
            
            if steps:
                avg_step = sum(steps) / len(steps)
                max_step = max(steps)
                min_step = min(steps)
            else:
                min_step = 0.0
        
        # Calculate direct distance (start to end) and path efficiency
        direct_distance = 0.0
        path_efficiency = 0.0
        
        if len(positions) >= 2:
            direct_distance = self.calculate_distance(positions[0], positions[-1])
            
            if total_distance > 0:
                path_efficiency = direct_distance / total_distance
        
        return {
            'total_distance': total_distance,
            'raw_distance': raw_distance,
            'average_step': avg_step,
            'max_step': max_step,
            'min_step': min_step,
            'direct_distance': direct_distance,
            'path_efficiency': path_efficiency
        }
    
    def apply_custom_correction(self, correction_function: Callable[[float], float]) -> None:
        """
        Apply a custom correction function to all distance data
        
        Args:
            correction_function: Function that takes a raw distance and returns a corrected one
        """
        # Update the correction factor for future calculations
        # (Example function might return value * 0.83)
        
        # Recalculate all stored distances
        for track_id in self.raw_distances:
            raw_distance = self.raw_distances[track_id]
            corrected_distance = correction_function(raw_distance)
            self.total_distances[track_id] = corrected_distance
        
        self.logger.info(f"Applied custom correction function to all distances")
    
    def set_correction_factor(self, correction_factor: float) -> None:
        """
        Update the correction factor and recalculate all distances
        
        Args:
            correction_factor: New correction factor
        """
        old_factor = self.correction_factor
        self.correction_factor = correction_factor
        
        # Recalculate all stored distances
        for track_id in self.raw_distances:
            self.total_distances[track_id] = self.raw_distances[track_id] * correction_factor
        
        self.logger.info(f"Updated correction factor from {old_factor} to {correction_factor}")
    
    def reset(self, track_id: Optional[str] = None) -> None:
        """
        Reset distance data
        
        Args:
            track_id: If specified, reset only this track. Otherwise reset all.
        """
        if track_id is not None:
            # Reset specific track
            if track_id in self.distance_history:
                del self.distance_history[track_id]
            if track_id in self.total_distances:
                del self.total_distances[track_id]
            if track_id in self.raw_distances:
                del self.raw_distances[track_id]
                
            self.logger.debug(f"Reset distance data for track {track_id}")
        else:
            # Reset all tracks
            self.distance_history.clear()
            self.total_distances.clear()
            self.raw_distances.clear()
            
            self.logger.info("Reset all distance data")