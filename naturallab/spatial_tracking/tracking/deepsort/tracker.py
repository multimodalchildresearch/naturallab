import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import time
import uuid

from motion_tracking.base import TrackerModule
from motion_tracking.tracking.base_tracker import BaseTracker
from motion_tracking.tracking.deepsort.feature_extractor import AppearanceFeatureExtractor
from motion_tracking.tracking.deepsort.feature_gallery import PersonFeatureGallery
from motion_tracking.diagnostics.deepsort_diagnostics import DeepSORTDiagnostics
from motion_tracking.utils.data_structures import generate_track_id

class DeepSORTTracker(BaseTracker):
    """Enhanced Kalman tracker with appearance-based re-identification for handling occlusions"""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3,
                 reid_model_path: str = 'osnet_ain_x1_0_msmt17.pt',
                 reid_threshold: float = 0.4, 
                 keep_lost_timeout: int = 60,  # Keep lost tracks longer
                 min_features_for_reid: int = 5,  # Minimum features required for reliable ReID
                 confidence_growth_rate: float = 0.05,  # Track confidence growth rate
                 enable_diagnostics: bool = True):
        """
        Initialize the DeepSORT-inspired tracker
        
        Args:
            max_age: Maximum frames to keep a track alive without matching detections
            min_hits: Minimum hits to consider a track confirmed
            iou_threshold: IOU threshold for detection matching
            reid_model_path: Path to ReID model for appearance feature extraction
            reid_threshold: Threshold for appearance similarity matching
            keep_lost_timeout: How long to keep lost tracks for ReID purposes
            min_features_for_reid: Minimum number of features needed for reliable ReID
            confidence_growth_rate: Rate at which track confidence grows with continuous tracking
            enable_diagnostics: Whether to enable diagnostic logging
        """
        super().__init__(max_age, min_hits, iou_threshold)
        self.name = "DeepSORTTracker"  # Override name from BaseTracker
        
        # Add this line to track all used IDs
        self.used_track_ids = set()
        # Initialize the appearance feature extractor - use CPU for most reliable operation
        self.feature_extractor = AppearanceFeatureExtractor(
            model_path=reid_model_path,
            device='cpu'  # Changed from 'mps' for better compatibility
        )
        
        # Initialize the feature gallery
        self.feature_gallery = PersonFeatureGallery(
            max_features_per_id=30,  # Store more features per ID
            max_inactive_time=keep_lost_timeout,  # Keep lost tracks longer
            similarity_threshold=reid_threshold
        )
        
        # Initialize diagnostics
        self.enable_diagnostics = enable_diagnostics
        if enable_diagnostics:
            self.diagnostics = DeepSORTDiagnostics(
                feature_extractor=self.feature_extractor,
                feature_gallery=self.feature_gallery,
                output_dir="deepsort_diagnostics"
            )
            self.diagnostics.verify_model()
        else:
            self.diagnostics = None
        
        # New parameters for improved tracking
        self.keep_lost_timeout = keep_lost_timeout
        self.min_features_for_reid = min_features_for_reid
        self.confidence_growth_rate = confidence_growth_rate
        self.reid_threshold = reid_threshold
        
        # For storing lost (temporarily invisible) tracks
        self.lost_tracks = []
        
        # Track history for better occlusion handling
        self.track_history = {}  # track_id -> historical data
        
        # Log initialization
        self.log_info(f"Initialized with reid_threshold={reid_threshold}, " 
                     f"keep_lost_timeout={keep_lost_timeout}, min_features_for_reid={min_features_for_reid}")
   
    def _create_new_track(self, detection: np.ndarray) -> Dict[str, Any]:
        """Create a new track with guaranteed unique ID"""
        # Get the original implementation's result (call the parent method)
        new_track = super()._create_new_track(detection)
        
        # Check if this ID has been used before
        original_id = new_track['id']
        while new_track['id'] in self.used_track_ids:
            self.log_warning(f"Duplicate ID {new_track['id'][:8]} detected during creation. Regenerating ID.")
            new_track['id'] = generate_track_id()  # From motion_tracking.utils.data_structures
        
        if original_id != new_track['id']:
            self.log_info(f"Changed track ID from {original_id[:8]} to {new_track['id'][:8]} to ensure uniqueness")
        
        # Add to set of used IDs
        self.used_track_ids.add(new_track['id'])
        
        return new_track

    def _extract_person_crop(self, frame: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """Extract a crop of the person from the frame"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        # Ensure coordinates are within frame
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Extract crop
        if x1 >= x2 or y1 >= y2:
            return None
        
        return frame[y1:y2, x1:x2]
    
    def _initialize_track_history(self, track_id: str) -> None:
        """Initialize history for a new track"""
        self.track_history[track_id] = {
            'feature_count': 0,
            'confidence_score': 0.1,  # Start with low confidence
            'occlusion_count': 0,
            'velocity_history': [],
            'position_history': [],
            'association_history': [],  # Track what this track has been associated with
        }
    
    def _update_track_history(self, track_id: str, detection: np.ndarray, is_occluded: bool = False) -> None:
        """Update track history with new detection information"""
        if track_id not in self.track_history:
            self._initialize_track_history(track_id)
        
        history = self.track_history[track_id]
        
        # Update position history (last 10 positions)
        center_x = (detection[0] + detection[2]) / 2
        center_y = (detection[1] + detection[3]) / 2
        
        if len(history['position_history']) > 0:
            # Calculate velocity
            last_pos = history['position_history'][-1]
            velocity = (center_x - last_pos[0], center_y - last_pos[1])
            history['velocity_history'].append(velocity)
            
            # Keep only last 5 velocities
            if len(history['velocity_history']) > 5:
                history['velocity_history'] = history['velocity_history'][-5:]
        
        history['position_history'].append((center_x, center_y))
        
        # Keep only last 10 positions
        if len(history['position_history']) > 10:
            history['position_history'] = history['position_history'][-10:]
        
        # Update occlusion count
        if is_occluded:
            history['occlusion_count'] += 1
        else:
            # Gradually reduce occlusion count when visible
            history['occlusion_count'] = max(0, history['occlusion_count'] - 0.5)
        
        # Increase confidence with continuous tracking
        if not is_occluded:
            history['confidence_score'] = min(0.9, history['confidence_score'] + self.confidence_growth_rate)
    
    def _predict_next_position(self, track_id: str) -> Optional[Tuple[float, float]]:
        """Predict next position based on velocity history"""
        if track_id not in self.track_history:
            return None
        
        history = self.track_history[track_id]
        
        if len(history['position_history']) == 0 or len(history['velocity_history']) == 0:
            return None
        
        # Get last known position
        last_pos = history['position_history'][-1]
        
        # Calculate average velocity from recent history
        velocities = history['velocity_history']
        if not velocities:
            return last_pos
        
        # Weight recent velocities more heavily
        weights = np.linspace(0.5, 1.0, len(velocities))
        weights = weights / np.sum(weights)
        
        avg_vx = sum(v[0] * w for v, w in zip(velocities, weights))
        avg_vy = sum(v[1] * w for v, w in zip(velocities, weights))
        
        # Predict next position
        next_x = last_pos[0] + avg_vx
        next_y = last_pos[1] + avg_vy
        
        return (next_x, next_y)
    
    def _get_track_confidence(self, track_id: str) -> float:
        """Get the current confidence score for a track"""
        if track_id not in self.track_history:
            return 0.1  # Default low confidence
        
        return self.track_history[track_id]['confidence_score']
    
    def _adaptive_iou_threshold(self, track_id: str) -> float:
        """Get an adaptive IOU threshold based on track history"""
        base_iou = self.iou_threshold
        
        if track_id not in self.track_history:
            return base_iou
        
        history = self.track_history[track_id]
        
        # Reduce threshold for tracks that have been occluded
        if history['occlusion_count'] > 0:
            return max(0.1, base_iou - 0.1)
        
        # Increase threshold for highly confident tracks
        if history['confidence_score'] > 0.7:
            return min(0.8, base_iou + 0.1)
        
        return base_iou
    
    def _has_reliable_features(self, track_id: str) -> bool:
        """Check if a track has enough features for reliable ReID"""
        # Check gallery
        feature_count = self.feature_gallery.get_feature_count(track_id)
        return feature_count >= self.min_features_for_reid
    
    def _match_detections_to_tracks(self, 
                                   tracks: List[Dict[str, Any]], 
                                   detections: List[List[float]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU with adaptive thresholds
        
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
            track = tracks[track_idx]
            track_id = track['id']
            
            # Use adaptive IOU threshold based on track history
            adaptive_threshold = self._adaptive_iou_threshold(track_id)
            
            # Get track bbox from Kalman
            track_bbox = self._bbox_from_kalman(track['kalman'])
            
            # Adjust prediction based on velocity history
            predicted_pos = self._predict_next_position(track_id)
            if predicted_pos is not None:
                # Adjust Kalman prediction with velocity history
                # This helps with rapid movements and occlusions
                width = track_bbox[2] - track_bbox[0]
                height = track_bbox[3] - track_bbox[1]
                
                # Blend Kalman with velocity prediction (70% Kalman, 30% velocity)
                center_x = ((track_bbox[0] + track_bbox[2]) / 2) * 0.7 + predicted_pos[0] * 0.3
                center_y = ((track_bbox[1] + track_bbox[3]) / 2) * 0.7 + predicted_pos[1] * 0.3
                
                # Create adjusted bbox
                track_bbox = [
                    center_x - width/2,
                    center_y - height/2,
                    center_x + width/2,
                    center_y + height/2
                ]
            
            for j, det_idx in enumerate(detection_indices):
                det_bbox = detections[det_idx][:4]  # Exclude score
                iou_matrix[i, j] = self._iou(track_bbox, det_bbox)
        
        # Assignment strategy - greedy assignment but with higher
        # quality matches first (process high IOU matches first)
        matches = []
        unmatched_tracks = list(track_indices)
        unmatched_detections = list(detection_indices)
        
        # Process matches in descending order of IOU
        while True:
            # If no more pairs to consider, break
            if len(unmatched_tracks) == 0 or len(unmatched_detections) == 0:
                break
                
            # Find max IOU
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            i_track, j_det = track_indices[i], detection_indices[j]
            
            # If max IOU is too low, we're done with matching
            if iou_matrix[i, j] < self.iou_threshold:
                break
            
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
        Update tracks with new detections, using both spatial and appearance features
        
        Args:
            data: Dictionary containing 'frame' and 'detections'
            
        Returns:
            Dictionary with 'tracks' containing current track information
        """
        frame = data['frame']
        detections = data['detections']
        
        # Update frame counter
        self.frame_count += 1
        frame_idx = self.frame_count
        self.feature_gallery.update_frame_count()
        
        # Generate diagnostic snapshot of the gallery state
        if self.diagnostics is not None:
            self.diagnostics.snapshot_gallery(frame_idx)
        
        # Log tracking statistics
        if frame_idx % 30 == 0:
            self.log_info(f"Frame {frame_idx}: {len(self.tracks)} active tracks, {len(self.lost_tracks)} lost tracks")
        
        # Predict new locations of all tracks
        for track in self.tracks:
            track['kalman'].predict()
        
        # Also predict for lost tracks
        for track in self.lost_tracks:
            track['kalman'].predict()
            track['time_since_update'] += 1
        
        # Match detections to existing tracks using IOU
        matches, unmatched_tracks, unmatched_detections = self._match_detections_to_tracks(
            self.tracks, detections)
        
        # Process lost tracks - attempt to match with unmatched detections
        # using both appearance and motion prediction
        if unmatched_detections and self.lost_tracks:
            # For each lost track, get the predicted position
            lost_track_predictions = {}
            for i, lost_track in enumerate(self.lost_tracks):
                lost_id = lost_track['id']
                
                # Get position from Kalman
                lost_bbox = self._bbox_from_kalman(lost_track['kalman'])
                
                # Blend with velocity-based prediction if available
                predicted_pos = self._predict_next_position(lost_id)
                if predicted_pos is not None:
                    width = lost_bbox[2] - lost_bbox[0]
                    height = lost_bbox[3] - lost_bbox[1]
                    
                    # Create a blended prediction (60% Kalman, 40% velocity for lost tracks)
                    center_x = ((lost_bbox[0] + lost_bbox[2]) / 2) * 0.6 + predicted_pos[0] * 0.4
                    center_y = ((lost_bbox[1] + lost_bbox[3]) / 2) * 0.6 + predicted_pos[1] * 0.4
                    
                    lost_bbox = [
                        center_x - width/2,
                        center_y - height/2,
                        center_x + width/2,
                        center_y + height/2
                    ]
                
                lost_track_predictions[i] = lost_bbox
            
            # For each unmatched detection, calculate:
            # 1. IOU with predicted positions of lost tracks
            # 2. Appearance similarity with lost tracks
            reid_matched_detections = []
            reid_matched_lost_tracks = []
            
            for det_idx in unmatched_detections:
                det_bbox = detections[det_idx][:4]
                det_crop = self._extract_person_crop(frame, det_bbox)
                
                if det_crop is None:
                    continue
                
                # Get detection features
                det_features = self.feature_extractor.extract_features(det_crop)
                if det_features is None:
                    continue
                
                # Calculate combined score for each lost track
                # Combine IOU and appearance similarity
                best_score = -1
                best_lost_idx = -1
                all_similarities = []
                
                for lost_idx, lost_track in enumerate(self.lost_tracks):
                    if lost_idx in reid_matched_lost_tracks:
                        continue  # Skip if already matched
                    
                    lost_id = lost_track['id']
                    
                    # Skip tracks that don't have enough features for reliable ReID
                    if not self._has_reliable_features(lost_id):
                        continue
                    
                    # Calculate IOU score with predicted position
                    predicted_bbox = lost_track_predictions.get(lost_idx)
                    if predicted_bbox is None:
                        iou_score = 0
                    else:
                        iou_score = self._iou(predicted_bbox, det_bbox)
                    
                    # Calculate appearance similarity
                    appear_score = 0
                    gallery_features = self.feature_gallery.gallery.get(lost_id, {}).get('features', [])
                    
                    # Calculate similarity with all features for this ID
                    similarities = []
                    for feat in gallery_features:
                        similarity = 1.0 - cdist(
                            det_features.reshape(1, -1),
                            feat.reshape(1, -1),
                            metric='cosine'
                        )[0][0]
                        similarities.append(similarity)
                        all_similarities.append((lost_id, similarity))
                    
                    # Use max similarity for this ID
                    appear_score = max(similarities) if similarities else 0
                    
                    # Calculate time penalty (reduce score for tracks that have been lost longer)
                    time_factor = max(0.5, 1.0 - (lost_track['time_since_update'] / self.max_age) * 0.5)
                    
                    # Adaptive weighting based on reliability of each cue
                    # If appearance is more reliable, weight it higher
                    feature_reliability = min(1.0, len(gallery_features) / 10)
                    
                    # If track has been lost for a while, rely more on appearance
                    appearance_weight = 0.7 * feature_reliability + 0.1 * (lost_track['time_since_update'] / 10)
                    appearance_weight = min(0.8, max(0.3, appearance_weight))
                    
                    iou_weight = 1.0 - appearance_weight
                    
                    # Calculate combined score
                    combined_score = (
                        appear_score * appearance_weight + 
                        iou_score * iou_weight
                    ) * time_factor
                    
                    # Check if this is the best match so far
                    if combined_score > best_score:
                        best_score = combined_score
                        best_lost_idx = lost_idx
                
                # Log the ReID matching attempt
                if self.diagnostics is not None:
                    if best_lost_idx >= 0:
                        matched_id = self.lost_tracks[best_lost_idx]['id']
                    else:
                        matched_id = None
                    
                    self.diagnostics.log_reid_match(det_features, matched_id, 
                                                  best_score if best_score > 0 else 0, 
                                                  [sim for _, sim in all_similarities])
                
                # Dynamic threshold based on track confidence
                confidence_threshold = self.reid_threshold
                if best_lost_idx >= 0:
                    matched_id = self.lost_tracks[best_lost_idx]['id']
                    # Lower threshold for tracks we're more confident in
                    track_confidence = self._get_track_confidence(matched_id)
                    adaptive_threshold = confidence_threshold * (1.0 - track_confidence * 0.3)
                    adaptive_threshold = max(0.25, adaptive_threshold)  # Don't go below 0.25
                else:
                    adaptive_threshold = confidence_threshold
                
                # If we found a match above the threshold
                if best_lost_idx >= 0 and best_score > adaptive_threshold:
                    # Include this match
                    reid_matched_detections.append(det_idx)
                    reid_matched_lost_tracks.append(best_lost_idx)
                    
                    # Recover the lost track
                    lost_track = self.lost_tracks[best_lost_idx]
                    
                    # Update Kalman filter with new detection
                    self._update_kalman(lost_track['kalman'], detections[det_idx])
                    
                    # Update track info
                    lost_track['bbox'] = detections[det_idx][:4]
                    lost_track['score'] = detections[det_idx][4]
                    lost_track['time_since_update'] = 0
                    lost_track['hits'] += 1
                    
                    # Update track history
                    self._update_track_history(lost_track['id'], detections[det_idx])
                    
                    # Log track recovery
                    if self.diagnostics is not None:
                        self.diagnostics.log_track_recovered(lost_track['id'], 
                                                           detections[det_idx][:4], 
                                                           frame_idx, best_score)
                    
                    # Move back to active tracks
                    self.tracks.append(lost_track)
                    
                    # Extract and update appearance features
                    crop = self._extract_person_crop(frame, detections[det_idx][:4])
                    if crop is not None:
                        features = self.feature_extractor.extract_features(crop)
                        if self.diagnostics is not None:
                            self.diagnostics.log_feature_extraction(
                                lost_track['id'], crop, features, frame_idx)
                        
                        if features is not None:
                            self.feature_gallery.add_features(lost_track['id'], features)
                            
                            # Update feature count in track history
                            if lost_track['id'] in self.track_history:
                                self.track_history[lost_track['id']]['feature_count'] += 1
            
            # Remove recovered tracks from lost tracks
            self.lost_tracks = [t for i, t in enumerate(self.lost_tracks) 
                               if i not in reid_matched_lost_tracks]
            
            # Remove matched detections from unmatched list
            unmatched_detections = [d for d in unmatched_detections 
                                   if d not in reid_matched_detections]
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            # Update Kalman filter with new detection
            self._update_kalman(self.tracks[track_idx]['kalman'], detections[det_idx])
            self.tracks[track_idx]['bbox'] = detections[det_idx][:4]
            self.tracks[track_idx]['score'] = detections[det_idx][4]
            self.tracks[track_idx]['time_since_update'] = 0
            self.tracks[track_idx]['hits'] += 1
            
            # Update track history
            self._update_track_history(self.tracks[track_idx]['id'], detections[det_idx])
            
            # Log track update
            if self.diagnostics is not None:
                self.diagnostics.log_track_update(self.tracks[track_idx]['id'], 
                                                detections[det_idx][:4], frame_idx)
            
            # Extract and update appearance features
            crop = self._extract_person_crop(frame, detections[det_idx][:4])
            if crop is not None:
                features = self.feature_extractor.extract_features(crop)
                if self.diagnostics is not None:
                    self.diagnostics.log_feature_extraction(
                        self.tracks[track_idx]['id'], crop, features, frame_idx)
                
                if features is not None:
                    self.feature_gallery.add_features(self.tracks[track_idx]['id'], features)
                    
                    # Update feature count in track history
                    if self.tracks[track_idx]['id'] in self.track_history:
                        self.track_history[self.tracks[track_idx]['id']]['feature_count'] += 1
        
        # Mark unmatched tracks as temporarily lost
        # Apply different logic based on track's history and confidence
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            track_id = track['id']
            track['time_since_update'] += 1
            
            # Update track history to indicate potential occlusion
            if track_id in self.track_history:
                self._update_track_history(track_id, track['bbox'], is_occluded=True)
            
            # More selective about when to move to lost tracks
            # If it's a high-confidence track with good feature history, keep it active longer
            confidence = self._get_track_confidence(track_id)
            has_good_features = self._has_reliable_features(track_id)
            
            # Keep high-confidence tracks active longer before marking as lost
            move_to_lost = False
            if confidence > 0.7 and has_good_features:
                # For high-confidence tracks, be more patient (up to 5 frames)
                if track['time_since_update'] > 5:
                    move_to_lost = True
            else:
                # For other tracks, move to lost quickly (after 2 frames)
                if track['time_since_update'] > 2:
                    move_to_lost = True
            
            if move_to_lost:
                if self.diagnostics is not None:
                    self.diagnostics.log_track_lost(track_id, frame_idx)
                self.lost_tracks.append(track)
        
        # Remove lost tracks from active tracks
        self.tracks = [t for t in self.tracks if t['time_since_update'] <= 5 or 
                       (self._get_track_confidence(t['id']) > 0.8 and t['time_since_update'] <= 8)]
        
        # Create new tracks for unmatched detections
        # But be more cautious about creating new tracks when we already have tracks
        for det_idx in unmatched_detections:
            det_bbox = detections[det_idx][:4]
            det_score = detections[det_idx][4]
            
            # If we have active tracks, be more selective about creating new ones
            # This helps prevent duplicate tracks during partial occlusions
            should_create_new = True
            
            # Higher detection score threshold when we already have active tracks
            if len(self.tracks) > 0 and det_score < 0.4:
                should_create_new = False
            
            # Don't create new tracks for detections that are near existing tracks
            # This often happens during occlusions
            for track in self.tracks:
                track_bbox = self._bbox_from_kalman(track['kalman'])
                iou = self._iou(track_bbox, det_bbox)
                
                # If close to an existing track, don't create a new one
                if iou > 0.1:
                    should_create_new = False
                    break
            
            # Check against lost tracks too
            for track in self.lost_tracks:
                # Only consider recently lost tracks
                if track['time_since_update'] <= 10:
                    track_bbox = self._bbox_from_kalman(track['kalman'])
                    iou = self._iou(track_bbox, det_bbox)
                    
                    # If close to a recently lost track, don't create a new one
                    if iou > 0.1:
                        should_create_new = False
                        break
            
            if should_create_new:
                # Initialize a new track
                new_track = self._create_new_track(detections[det_idx])
                
                # Initialize track history
                self._initialize_track_history(new_track['id'])
                
                # Log new track creation
                if self.diagnostics is not None:
                    self.diagnostics.log_track_creation(new_track['id'], det_bbox, frame_idx)
                
                # Check if this is a reappearing person that was not in lost tracks
                crop = self._extract_person_crop(frame, det_bbox)
                if crop is not None:
                    features = self.feature_extractor.extract_features(crop)
                    if self.diagnostics is not None:
                        self.diagnostics.log_feature_extraction(
                            new_track['id'], crop, features, frame_idx)
                    
                    if features is not None:
                        # Look for matching ID in gallery (for tracks that might have been removed)
                        matching_id = self.feature_gallery.find_matching_id(features)
                        if matching_id is not None:
                            # Log the ID reassignment
                            if self.diagnostics is not None:
                                self.diagnostics.log(
                                    f"Re-identified track: {new_track['id'][:6]} → {matching_id[:6]}")
                            
                            # Reuse the existing ID instead of creating a new one
                            new_track['id'] = matching_id
                            
                            # If we had history for this track, recover it
                            if matching_id in self.track_history:
                                # Update history to indicate recovery
                                self.track_history[matching_id]['confidence_score'] = max(
                                    0.6, self.track_history[matching_id]['confidence_score'])
                        
                        # Add features to gallery
                        self.feature_gallery.add_features(new_track['id'], features)
                        
                        # Update track history
                        self._update_track_history(new_track['id'], det_bbox)
                        if new_track['id'] in self.track_history:
                            self.track_history[new_track['id']]['feature_count'] += 1
                
                self.tracks.append(new_track)
        
        # Remove dead tracks from lost_tracks list
        # Keep them longer but apply more selective filtering
        original_lost_count = len(self.lost_tracks)
        
        # Filter based on different criteria
        filtered_lost_tracks = []
        
        for track in self.lost_tracks:
            track_id = track['id']
            
            # If track has good features, keep it longer
            if self._has_reliable_features(track_id):
                # Keep high-quality lost tracks longer
                if track['time_since_update'] <= self.keep_lost_timeout:
                    filtered_lost_tracks.append(track)
            else:
                # For tracks without good features, remove sooner
                if track['time_since_update'] <= self.max_age:
                    filtered_lost_tracks.append(track)
        
        self.lost_tracks = filtered_lost_tracks
        removed_count = original_lost_count - len(self.lost_tracks)
        
        if removed_count > 0 and self.diagnostics is not None:
            self.diagnostics.log(f"Removed {removed_count} dead tracks")
        
        # Return current tracks with more selective filtering
        # A track needs to have enough hits and recent updates to be considered active
        active_tracks = []
        
        for track in self.tracks:
            track_id = track['id']
            
            # Basic criteria - enough hits
            criteria1 = track['hits'] >= self.min_hits
            
            # Time since update criteria - more lenient for high-confidence tracks
            time_threshold = 1  # Default
            if track_id in self.track_history:
                confidence = self.track_history[track_id]['confidence_score']
                if confidence > 0.7:
                    time_threshold = 3  # Allow high-confidence tracks to miss a few frames
            
            criteria2 = track['time_since_update'] <= time_threshold
            
            if criteria1 and criteria2:
                active_tracks.append(track)
        
        # Generate diagnostic visualization
        if self.diagnostics is not None:
            diag_frame = self.diagnostics.generate_diagnostic_frame(
                frame, active_tracks, self.lost_tracks, frame_idx
            )
            data['diagnostic_frame'] = diag_frame
        
        # Fix for duplicate IDs - ensure each active track has a unique ID
        seen_ids = set()
        for i, track in enumerate(active_tracks):
            if track['id'] in seen_ids:
                # Generate a new unique ID for this duplicate
                old_id = track['id']
                new_id = generate_track_id()
                
                # Ensure the new ID is also unique across all time
                while new_id in self.used_track_ids:
                    new_id = generate_track_id()
                    
                self.log_warning(f"Fixing duplicate track ID {old_id[:8]} -> {new_id[:8]}")
                
                # Update the ID
                track['id'] = new_id
                self.used_track_ids.add(new_id) 
                
                # If we have features for this track in the gallery, copy them
                if old_id in self.feature_gallery.gallery:
                    self.feature_gallery.gallery[new_id] = self.feature_gallery.gallery[old_id].copy()
                
                # If we have track history, copy it
                if old_id in self.track_history:
                    self.track_history[new_id] = self.track_history[old_id].copy()
            
            seen_ids.add(track['id'])
        
        return {
            'tracks': active_tracks,
            **data
        }
    
    def reset(self) -> None:
        """Reset the tracker state"""
        super().reset()
        self.lost_tracks = []
        self.track_history = {}
        # Do NOT reset self.used_track_ids to ensure IDs remain unique even after reset
        if self.diagnostics is not None:
            self.diagnostics.log("Tracker reset")

# Import at the bottom to avoid circular imports
from scipy.spatial.distance import cdist