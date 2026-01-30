import numpy as np
from typing import Dict, List, Any, Tuple

from motion_tracking.tracking.base_tracker import BaseTracker

class KalmanPersonTracker(BaseTracker):
    """Kalman filter-based person tracker"""
    
    def __init__(self, max_age: int = 10, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize the Kalman filter tracker
        
        Args:
            max_age: Maximum frames to keep a track alive without matching detections
            min_hits: Minimum hits to consider a track confirmed
            iou_threshold: IOU threshold for detection matching
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.name = "KalmanTracker"  # Override name from BaseTracker
    
    def _match_detections_to_tracks(self, 
                                   tracks: List[Dict[str, Any]], 
                                   detections: List[List[float]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU
        
        Args:
            tracks: List of track dictionaries
            detections: List of detections [x1, y1, x2, y2, score]
            
        Returns:
            Tuple of (matches, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        track_indices = list(range(len(tracks)))
        detection_indices = list(range(len(detections)))
        
        # Calculate IOU matrix
        iou_matrix = np.zeros((len(track_indices), len(detection_indices)))
        
        for i, track_idx in enumerate(track_indices):
            track_bbox = self._bbox_from_kalman(tracks[track_idx]['kalman'])
            
            for j, det_idx in enumerate(detection_indices):
                det_bbox = detections[det_idx][:4]  # Exclude score
                iou_matrix[i, j] = self._iou(track_bbox, det_bbox)
        
        # Simple greedy assignment
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)
        
        # Sort by IOU (highest first)
        for _ in range(min(len(track_indices), len(detection_indices))):
            if np.max(iou_matrix) < self.iou_threshold:
                break
            
            # Find max IOU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i_track, j_det = track_indices[i], detection_indices[j]
            
            # Add match
            matches.append((i_track, j_det))
            
            # Remove from unmatched
            unmatched_tracks.remove(i_track)
            unmatched_detections.remove(j_det)
            
            # Set row and column to -1 to avoid re-matching
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1
            
        return matches, unmatched_tracks, unmatched_detections
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update tracks with new detections
        
        Args:
            data: Dictionary containing 'frame' and 'detections'
            
        Returns:
            Dictionary with 'tracks' containing current track information
        """
        frame = data['frame']
        detections = data['detections']
        
        # Predict new locations of all tracks
        for track in self.tracks:
            track['kalman'].predict()
        
        # Match detections to existing tracks
        matches, unmatched_tracks, unmatched_detections = self._match_detections_to_tracks(
            self.tracks, detections)
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            # Update Kalman filter with new detection
            self._update_kalman(self.tracks[track_idx]['kalman'], detections[det_idx])
            self.tracks[track_idx]['bbox'] = detections[det_idx][:4]
            self.tracks[track_idx]['score'] = detections[det_idx][4]
            self.tracks[track_idx]['time_since_update'] = 0
            self.tracks[track_idx]['hits'] += 1
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            # Initialize a new track
            new_track = self._create_new_track(detections[det_idx])
            self.tracks.append(new_track)
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['time_since_update'] += 1
        
        # Remove dead tracks
        self.tracks = [track for track in self.tracks 
                      if track['time_since_update'] <= self.max_age]
        
        # Get active tracks (those with enough hits and recent updates)
        active_tracks = [track for track in self.tracks 
                        if track['hits'] >= self.min_hits and track['time_since_update'] <= 1]
        
        # Log tracking stats occasionally
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.log_info(f"Frame {self.frame_count}: {len(active_tracks)} active tracks, {len(self.tracks)} total")
        
        return {
            'tracks': active_tracks,
            **data
        }