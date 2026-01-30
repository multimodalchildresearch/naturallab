import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
import time
import json
import copy

from motion_tracking.base import TrackerModule

class PoseEstimatorModule(TrackerModule):
    """MediaPipe pose estimation on tracked person regions with aggressive outlier elimination"""
    
    def __init__(self, crop_expansion: float = 0.2, strict_filtering: bool = True):
        """
        Initialize the MediaPipe pose estimator
        
        Args:
            crop_expansion: Fraction to expand bounding boxes for pose estimation
            strict_filtering: Whether to use stricter filtering for outlier landmarks
        """
        super().__init__(name="PoseEstimator")
        
        # Import mediapipe here to allow optional dependency
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=2,
                static_image_mode=False,
                smooth_landmarks=True,
                enable_segmentation=False  # Disable segmentation to avoid ROI issues
            )
            self.mp_draw = mp.solutions.drawing_utils
            self.has_mediapipe = True
            self.log_info("MediaPipe pose initialized successfully")
        except ImportError:
            self.log_error("Failed to import MediaPipe. Install with: pip install mediapipe")
            self.has_mediapipe = False
        
        self.crop_expansion = crop_expansion
        self.strict_filtering = strict_filtering
        self.pose_data = {}  # Store pose data by track ID
        self.frame_count = 0
        self.process_times = []  # For performance monitoring
        
        # Add statistics collection
        self.landmark_stats = {
            'total_landmarks': 0,
            'outside_bbox': 0,
            'filtered_landmarks': 0,
            'by_landmark_id': {},  # Track outliers by landmark ID
            'frames_with_outliers': 0,
            'total_frames': 0
        }
    
    def _expand_bbox(self, bbox: List[float], frame_shape: Tuple[int, int, int]) -> List[int]:
        """
        Expand bounding box with some margin and ensure it's within frame
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (h, w, c)
            
        Returns:
            Expanded bounding box as integers [x1, y1, x2, y2]
        """
        h, w = frame_shape[:2]
        
        # Calculate expansion
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        # Expand by specified fraction
        x1 = max(0, int(bbox[0] - width * self.crop_expansion))
        y1 = max(0, int(bbox[1] - height * self.crop_expansion))
        x2 = min(w - 1, int(bbox[2] + width * self.crop_expansion))
        y2 = min(h - 1, int(bbox[3] + height * self.crop_expansion))
        
        return [x1, y1, x2, y2]
    
    def _adjust_landmarks(self, landmarks, x1, y1, x2, y2):
        """
        Adjust landmarks from cropped coordinates to original frame coordinates
        
        Args:
            landmarks: MediaPipe pose landmarks
            x1, y1, x2, y2: Crop region coordinates
            
        Returns:
            Adjusted landmarks
        """
        # Deep copy the landmarks object
        adjusted_landmarks = copy.deepcopy(landmarks)
        
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # Adjust each landmark's coordinates
        for landmark in adjusted_landmarks.landmark:
            # Adjust x and y coordinates
            landmark.x = (landmark.x * crop_width + x1)
            landmark.y = (landmark.y * crop_height + y1)
            # Z stays the same relative to the crop window
        
        return adjusted_landmarks
    
    def _filter_outlier_landmarks(self, landmarks, bbox, frame_shape):
        """
        Aggressively filter out implausible landmark positions with special handling 
        for known problematic landmarks
        
        Args:
            landmarks: MediaPipe pose landmarks
            bbox: Original tracking bounding box [x1, y1, x2, y2]
            frame_shape: Frame dimensions (h, w, c)
            
        Returns:
            Landmarks with outliers fixed, valid landmarks dict, filtered count, and metadata
        """
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        
        # Custom distance thresholds for different landmark groups
        threshold_map = {
            # Shoulders (very strict - major problem areas)
            11: 0.3,  # left_shoulder 
            12: 0.3,  # right_shoulder
            
            # Head (strict - problem area)
            0: 0.4,   # nose
            7: 0.4,   # left_ear
            8: 0.4,   # right_ear
            
            # Feet (strict - problem area)
            31: 0.4,  # left_foot_index
            32: 0.4,  # right_foot_index
            
            # Default for other landmarks
            'default': 0.8 if not self.strict_filtering else 0.5
        }
        
        # Track which landmarks were filtered
        filtered_count = 0
        outside_bbox_count = 0
        filtered_landmarks = []
        
        # Add landmark metadata for debugging
        landmark_metadata = []
        
        # Create a dictionary of clean valid landmarks
        valid_landmarks = {}
        
        for i, lm in enumerate(landmarks.landmark):
            # Use the appropriate threshold for this landmark
            distance_threshold = threshold_map.get(i, threshold_map['default'])
            
            # Calculate maximum allowed distance for this landmark
            max_x_distance = bbox_width * distance_threshold
            max_y_distance = bbox_height * distance_threshold
            
            # Check if point is inside original bbox
            inside_bbox = (x1 <= lm.x <= x2) and (y1 <= lm.y <= y2)
            if not inside_bbox:
                outside_bbox_count += 1
                
                # Update statistics by landmark ID
                if i not in self.landmark_stats['by_landmark_id']:
                    self.landmark_stats['by_landmark_id'][i] = 0
                self.landmark_stats['by_landmark_id'][i] += 1
            
            # Calculate distances from bbox boundaries
            x_min_distance = x1 - lm.x if lm.x < x1 else 0
            x_max_distance = lm.x - x2 if lm.x > x2 else 0
            y_min_distance = y1 - lm.y if lm.y < y1 else 0
            y_max_distance = lm.y - y2 if lm.y > y2 else 0
            
            # Store metadata
            metadata = {
                'id': i,
                'inside_bbox': inside_bbox,
                'x_dist': max(x_min_distance, x_max_distance),
                'y_dist': max(y_min_distance, y_max_distance),
                'threshold': distance_threshold,
                'x_min_violation': lm.x < (x1 - max_x_distance),
                'x_max_violation': lm.x > (x2 + max_x_distance),
                'y_min_violation': lm.y < (y1 - max_y_distance),
                'y_max_violation': lm.y > (y2 + max_y_distance),
                'outside_frame': lm.x < 0 or lm.x >= w or lm.y < 0 or lm.y >= h,
                'original_visibility': lm.visibility
            }
            
            # Check if point is too far horizontally or vertically from bbox
            x_min_violation = metadata['x_min_violation']
            x_max_violation = metadata['x_max_violation']
            y_min_violation = metadata['y_min_violation']
            y_max_violation = metadata['y_max_violation']
            
            # If ANY dimension is violated, mark as invalid
            should_filter = (x_min_violation or x_max_violation or 
                            y_min_violation or y_max_violation or 
                            metadata['outside_frame'])
            
            # For problematic landmarks, filter if outside bbox regardless of threshold
            if not should_filter and not inside_bbox:
                if i in [11, 12, 0, 7, 8, 31, 32]:  # Problem landmarks
                    should_filter = True
                    
            if should_filter:
                # For debugging, note which dimension caused the filter
                reason = []
                if x_min_violation: reason.append(f"x < min by {x1 - lm.x:.1f}")
                if x_max_violation: reason.append(f"x > max by {lm.x - x2:.1f}")
                if y_min_violation: reason.append(f"y < min by {y1 - lm.y:.1f}")
                if y_max_violation: reason.append(f"y > max by {lm.y - y2:.1f}")
                if metadata['outside_frame']: reason.append("outside frame")
                if not inside_bbox and i in [11, 12, 0, 7, 8, 31, 32]: 
                    reason.append("problem landmark outside bbox")
                
                filtered_landmarks.append(f"lm{i}({', '.join(reason)})")
                lm.visibility = 0.0  # Set to invisible in original structure
                filtered_count += 1
                metadata['filtered'] = True
                metadata['filter_reason'] = reason
                
                # Don't add to valid_landmarks dictionary - this is the key part
                # that removes the landmark completely
            else:
                metadata['filtered'] = False
                metadata['filter_reason'] = []
                
                # Reduce visibility for landmarks outside bbox but not filtered
                if not inside_bbox:
                    # Gradually reduce visibility based on distance from bbox
                    max_dist = max(
                        x_min_distance / max_x_distance if x_min_distance > 0 else 0,
                        x_max_distance / max_x_distance if x_max_distance > 0 else 0,
                        y_min_distance / max_y_distance if y_min_distance > 0 else 0,
                        y_max_distance / max_y_distance if y_max_distance > 0 else 0
                    )
                    # Reduce visibility proportionally to distance
                    reduction_factor = 1.0 - min(max_dist, 0.9)  # Never below 0.1
                    lm.visibility *= reduction_factor
                    metadata['visibility_reduced'] = True
                    metadata['reduction_factor'] = reduction_factor
                else:
                    metadata['visibility_reduced'] = False
                    metadata['reduction_factor'] = 1.0
                
                # Add to valid_landmarks dictionary - only non-filtered landmarks
                valid_landmarks[i] = {
                    'x': lm.x,
                    'y': lm.y,
                    'z': lm.z,
                    'visibility': lm.visibility
                }
            
            landmark_metadata.append(metadata)
        
        # Update statistics
        self.landmark_stats['total_landmarks'] += len(landmarks.landmark)
        self.landmark_stats['outside_bbox'] += outside_bbox_count
        self.landmark_stats['filtered_landmarks'] += filtered_count
        self.landmark_stats['total_frames'] += 1
        if outside_bbox_count > 0:
            self.landmark_stats['frames_with_outliers'] += 1
        
        """
        if filtered_count > 0:
            self.log_warning(f"Filtered {filtered_count} landmarks: {', '.join(filtered_landmarks[:5])}" + 
                            (f" and {len(filtered_landmarks)-5} more" if len(filtered_landmarks) > 5 else ""))
        """
        return landmarks, valid_landmarks, filtered_count, landmark_metadata
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform pose estimation on each tracked person with aggressive outlier removal
        
        Args:
            data: Dictionary containing 'frame' and 'tracks'
            
        Returns:
            Dictionary with 'pose_data' containing pose landmarks for each track
        """
        if not self.has_mediapipe:
            self.log_error("MediaPipe not available. Skipping pose estimation.")
            return {**data, 'pose_data': {}}
        
        start_time = time.time()
        frame = data['frame']
        tracks = data.get('tracks', [])
        frame_idx = data.get('frame_idx', self.frame_count)
        self.frame_count = frame_idx + 1
        
        # Reset pose data for tracks not present
        current_track_ids = [track['id'] for track in tracks]
        self.pose_data = {track_id: self.pose_data.get(track_id) 
                          for track_id in current_track_ids 
                          if track_id in self.pose_data}
        
        # Process each track
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            
            # Expand bbox for better pose estimation
            expanded_bbox = self._expand_bbox(bbox, frame.shape)
            
            # Crop the region
            x1, y1, x2, y2 = expanded_bbox
            person_crop = frame[y1:y2, x1:x2]
            
            if person_crop.size == 0:  # Skip empty crops
                continue
            
            # Convert to RGB for MediaPipe
            rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(rgb_crop)
            
            if results.pose_landmarks:
                # Adjust landmarks to original frame coordinates
                adjusted_landmarks = self._adjust_landmarks(results.pose_landmarks, x1, y1, x2, y2)

                # Filter implausible landmark positions and get metadata
                filtered_landmarks, valid_landmarks, filtered_count, landmark_metadata = self._filter_outlier_landmarks(
                    adjusted_landmarks, 
                    bbox,  # Use original tracking bbox, not expanded
                    frame.shape
                )

                # Count landmarks outside bbox for this track
                outside_bbox_count = sum(1 for meta in landmark_metadata if not meta['inside_bbox'])
                valid_landmarks_count = len(valid_landmarks)
                
                # Then store in pose_data with enhanced metadata
                self.pose_data[track_id] = {
                    'landmarks': filtered_landmarks,  # Original structure with visibility=0 for filtered
                    'valid_landmarks': valid_landmarks,  # NEW: Only valid landmarks in a dict
                    'bbox': bbox,  # Store original bbox
                    'expanded_bbox': expanded_bbox,
                    'original_landmarks': results.pose_landmarks,
                    'crop_origin': (x1, y1),
                    'filtered_landmarks': filtered_count > 0,
                    'landmark_metadata': landmark_metadata,
                    'outside_bbox_count': outside_bbox_count,
                    'total_landmarks': len(adjusted_landmarks.landmark),
                    'valid_landmarks_count': valid_landmarks_count,
                    'outside_bbox_percentage': outside_bbox_count / len(adjusted_landmarks.landmark) * 100 if len(adjusted_landmarks.landmark) > 0 else 0
                }
                
                # If all landmarks were filtered, log a warning
                if valid_landmarks_count == 0 and len(adjusted_landmarks.landmark) > 0:
                    self.log_warning(f"All landmarks filtered for track {track_id[:8]} in frame {frame_idx}")
        
        # Check for tracks without pose data
        missing_pose = [tid for tid in current_track_ids if tid not in self.pose_data]
        if missing_pose and frame_idx % 30 == 0:
            self.log_warning(f"{len(missing_pose)} tracks missing pose data: {[id[:8] for id in missing_pose]}")
        
        # Ensure all tracks have unique IDs for pose data
        unique_pose_data = {}
        for track_id, pose_info in self.pose_data.items():
            # Make sure we don't overwrite existing pose data with None
            if track_id in unique_pose_data and pose_info is None:
                continue
            unique_pose_data[track_id] = pose_info
        
        # Replace the pose data with the deduplicated version
        self.pose_data = unique_pose_data
        
        # Track performance
        elapsed = time.time() - start_time
        self.process_times.append(elapsed)
        if len(self.process_times) > 100:
            self.process_times = self.process_times[-100:]
        
        # Log performance and statistics occasionally
        if frame_idx % 100 == 0:
            avg_time = sum(self.process_times) / len(self.process_times)
            self.log_info(f"Pose estimation average processing time: {avg_time*1000:.1f}ms per frame")
            
            # Log landmark statistics
            if self.landmark_stats['total_landmarks'] > 0:
                outside_pct = self.landmark_stats['outside_bbox'] / self.landmark_stats['total_landmarks'] * 100
                filtered_pct = self.landmark_stats['filtered_landmarks'] / self.landmark_stats['total_landmarks'] * 100
                frames_pct = self.landmark_stats['frames_with_outliers'] / self.landmark_stats['total_frames'] * 100 if self.landmark_stats['total_frames'] > 0 else 0
                
                self.log_info(f"Landmark statistics: {outside_pct:.1f}% outside bbox, {filtered_pct:.1f}% filtered, {frames_pct:.1f}% frames with outliers")
                
                # Find the top 3 most problematic landmarks
                if self.landmark_stats['by_landmark_id']:
                    problem_landmarks = sorted(
                        self.landmark_stats['by_landmark_id'].items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:3]
                    
                    self.log_info(f"Most problematic landmarks: {', '.join([f'lm{lm_id}({count})' for lm_id, count in problem_landmarks])}")
        
        # Return pose data with enhanced debugging info and clean landmarks
        return {
            'pose_data': self.pose_data,
            'landmark_stats': self.landmark_stats,
            **data
        }
    
    def reset(self) -> None:
        """Reset the pose estimator state"""
        self.pose_data = {}
        self.process_times = []
        self.landmark_stats = {
            'total_landmarks': 0,
            'outside_bbox': 0,
            'filtered_landmarks': 0,
            'by_landmark_id': {},
            'frames_with_outliers': 0,
            'total_frames': 0
        }
        self.log_info("Pose estimator reset")
        
    def export_landmark_statistics(self, output_path):
        """
        Export landmark statistics to a JSON file
        
        Args:
            output_path: Path to save the statistics JSON
        """
        with open(output_path, 'w') as f:
            # Convert any NumPy types before serialization
            stats_dict = self._convert_numpy_types(self.landmark_stats)
            json.dump(stats_dict, f, indent=2)
        
        self.log_info(f"Landmark statistics exported to {output_path}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to standard Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {str(key): self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj