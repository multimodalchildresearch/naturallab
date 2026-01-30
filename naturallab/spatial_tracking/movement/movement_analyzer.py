import numpy as np
import cv2
import time
import yaml
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

from motion_tracking.base import TrackerModule
from motion_tracking.diagnostics.movement_diagnostics import MovementDiagnostics

class MovementAnalyzerModule(TrackerModule):
    """Analyze movement, calculate distances, detect leaning and turning"""
    
    def __init__(self, camera_calibration_file: str, floor_calibration_file: str, 
                 max_speed_threshold: float = 5000,  # mm/s, ~5m/s or 18km/h is a reasonable human limit
                 position_smoothing_factor: float = 0.7,
                 distance_correction_factor: float = 1.0,
                 enable_diagnostics: bool = True):
        """
        Initialize the movement analyzer
        
        Args:
            camera_calibration_file: Path to camera calibration file
            floor_calibration_file: Path to floor calibration file
            max_speed_threshold: Maximum allowed speed in mm/s
            position_smoothing_factor: Factor for position smoothing (0-1)
            distance_correction_factor: Factor to apply to distances (e.g., 0.83)
            enable_diagnostics: Whether to enable diagnostic logging
        """
        super().__init__(name="MovementAnalyzer")
        
        # Load calibrations
        try:
            with open(camera_calibration_file) as f:
                camera_data = yaml.safe_load(f)
            self.camera_matrix = np.array(camera_data['camera_matrix'])
            self.dist_coeffs = np.array(camera_data['dist_coeff'])
            
            with open(floor_calibration_file) as f:
                floor_data = yaml.safe_load(f)
            self.floor_plane = np.array(floor_data['floor_plane'])
        except Exception as e:
            self.log_error(f"Error loading calibration files: {e}")
            raise
        
        # Movement analysis data by track ID
        self.movement_data = {}
        
        # Configuration parameters
        self.max_speed_threshold = max_speed_threshold
        self.position_smoothing_factor = position_smoothing_factor
        self.distance_correction_factor = distance_correction_factor
        
        # Track last processed IDs for debugging
        self.last_processed_ids = set()
        
        # Import mediapipe for landmark indices
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.has_mediapipe = True
        except ImportError:
            self.log_warning("MediaPipe not found - some pose-based features will be limited")
            self.has_mediapipe = False
        
        self.log_info(f"Initialized with camera matrix shape: {self.camera_matrix.shape}")
        self.log_info(f"Floor plane: {self.floor_plane}")
        self.log_info(f"Distance correction factor: {distance_correction_factor}")

        # Add diagnostics
        self.enable_diagnostics = enable_diagnostics
        if enable_diagnostics:
            self.diagnostics = MovementDiagnostics(output_dir='movement_diagnostics')
            self.diagnostics.log(f"Movement analyzer initialized with camera matrix shape: {self.camera_matrix.shape}")
            self.diagnostics.log(f"Floor plane: {self.floor_plane}")
        else:
            self.diagnostics = None
        
        # For recording major position changes
        self.last_position_plot_frame = 0
        
        # Debug visualization flag
        self.enable_floor_visualization = True
    
    def _init_person_data(self, track_id: str) -> None:
        """Initialize data for a new person"""
        self.movement_data[track_id] = {
            # Movement tracking
            'position_history': deque(maxlen=100),
            'time_history': deque(maxlen=100),
            'total_distance': 0,
            'current_speed': 0,
            'is_initialized': False,
            'last_valid_speed': 0,  # Store last valid speed
            'speed_history': deque(maxlen=5),  # Store recent speeds for smoothing
            'last_frame_processed': 0,  # Last frame this track was processed
            
            # Position smoothing
            'position_buffer': deque(maxlen=5),
            'last_valid_position': None,
            'last_valid_floor_position': None,  # Store the last valid floor position
            'position_jump_detected': False,  # Flag for detecting position jumps
            
            # Turn detection
            'direction_history': deque(maxlen=6),
            'is_turning': False,
            'turn_cooldown': 0,
            
            # Lean detection
            'is_leaning': False,
            'baseline_torso_vector': None,
            '_lean_counter': 0,
        }
        self.log_debug(f"Initialized movement tracking for new ID: {track_id[:6]}")
    
    def project_point_to_floor(self, image_point: np.ndarray, track_id: str = None) -> Optional[np.ndarray]:
        """Project a 2D image point onto the floor plane with sanity checks"""
        try:
            # Undistort point
            normalized_point = cv2.undistortPoints(
                np.array([[image_point]], dtype=np.float32),
                self.camera_matrix,
                self.dist_coeffs
            )
            
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
                if track_id:
                    self.log_debug(f"Track {track_id[:6]}: Floor projection failed - ray nearly parallel to floor")
                return None
                
            t = -d / denominator
            if t < 0:  # Intersection behind camera
                if track_id:
                    self.log_debug(f"Track {track_id[:6]}: Floor projection failed - intersection behind camera")
                return None
                
            intersection = ray_direction * t
            
            # Sanity check on the calculated position
            if abs(intersection[1]) > 3000:  # More than 3 meters up/down
                if track_id:
                    self.log_debug(f"Track {track_id[:6]}: Floor projection failed - position too high/low: {intersection[1]:.2f}mm")
                return None
                
            # Additional validation for the result
            if intersection is not None and track_id and track_id in self.movement_data:
                history = self.movement_data[track_id].get('position_history', [])
                if len(history) > 0:
                    last_valid = history[-1]
                    distance = np.linalg.norm(intersection - last_valid)
                    
                    # If more than 300mm from previous position, consider it suspect
                    if distance > 300:
                        # Use a weighted average with the previous position
                        intersection = last_valid * 0.7 + intersection * 0.3
                        if self.diagnostics:
                            self.diagnostics.log(f"Track {track_id[:6]}: Excessive projection change - blending result")
            
            return intersection    
            
        except Exception as e:
            if track_id:
                self.log_warning(f"Track {track_id[:6]}: Error in projection: {e}")
            else:
                self.log_warning(f"Error in projection: {e}")
            return None
    
    def get_ankle_position(self, landmarks, valid_landmarks=None):
        """Get average ankle position with confidence check using valid landmarks"""
        if not self.has_mediapipe:
            return None
            
        # Use valid_landmarks if provided
        if valid_landmarks:
            left_ankle_idx = self.mp_pose.PoseLandmark.LEFT_ANKLE.value
            right_ankle_idx = self.mp_pose.PoseLandmark.RIGHT_ANKLE.value
            
            # Convert to string keys if needed
            left_key = str(left_ankle_idx) if str(left_ankle_idx) in valid_landmarks else left_ankle_idx
            right_key = str(right_ankle_idx) if str(right_ankle_idx) in valid_landmarks else right_ankle_idx
            
            # Check if we have at least one valid ankle
            if left_key in valid_landmarks and valid_landmarks[left_key]['visibility'] > 0.3:
                return np.array([valid_landmarks[left_key]['x'], valid_landmarks[left_key]['y']])
                
            if right_key in valid_landmarks and valid_landmarks[right_key]['visibility'] > 0.3:
                return np.array([valid_landmarks[right_key]['x'], valid_landmarks[right_key]['y']])
                
            # If both valid with good visibility, use weighted average
            if (left_key in valid_landmarks and right_key in valid_landmarks and
                valid_landmarks[left_key]['visibility'] > 0.3 and valid_landmarks[right_key]['visibility'] > 0.3):
                
                left_vis = valid_landmarks[left_key]['visibility']
                right_vis = valid_landmarks[right_key]['visibility']
                total_vis = left_vis + right_vis
                
                avg_x = (valid_landmarks[left_key]['x'] * left_vis + valid_landmarks[right_key]['x'] * right_vis) / total_vis
                avg_y = (valid_landmarks[left_key]['y'] * left_vis + valid_landmarks[right_key]['y'] * right_vis) / total_vis
                return np.array([avg_x, avg_y])
                
            return None
    
    
    
    def detect_lean(self, landmarks, track_id: str, valid_landmarks=None):
        """Detect leaning using filtered valid landmarks if available"""
        if not self.has_mediapipe:
            return False
            
        person_data = self.movement_data.get(track_id)
        if not person_data:
            return False
        
        # If no valid landmarks provided, maintain current state
        if not valid_landmarks:
            return person_data.get('is_leaning', False)
        
        try:
            # Get landmark indices
            LEFT_SHOULDER = self.mp_pose.PoseLandmark.LEFT_SHOULDER.value
            RIGHT_SHOULDER = self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            LEFT_HIP = self.mp_pose.PoseLandmark.LEFT_HIP.value
            RIGHT_HIP = self.mp_pose.PoseLandmark.RIGHT_HIP.value
            
            # Convert indices to strings if needed
            ls_key = str(LEFT_SHOULDER) if str(LEFT_SHOULDER) in valid_landmarks else LEFT_SHOULDER
            rs_key = str(RIGHT_SHOULDER) if str(RIGHT_SHOULDER) in valid_landmarks else RIGHT_SHOULDER
            lh_key = str(LEFT_HIP) if str(LEFT_HIP) in valid_landmarks else LEFT_HIP
            rh_key = str(RIGHT_HIP) if str(RIGHT_HIP) in valid_landmarks else RIGHT_HIP
            
            # Check if all required landmarks exist in valid_landmarks
            if not all(k in valid_landmarks for k in [ls_key, rs_key, lh_key, rh_key]):
                return person_data.get('is_leaning', False)
                
            # Get 3D positions from valid landmarks
            ls_3d = np.array([valid_landmarks[ls_key]['x'], valid_landmarks[ls_key]['y'], valid_landmarks[ls_key]['z']])
            rs_3d = np.array([valid_landmarks[rs_key]['x'], valid_landmarks[rs_key]['y'], valid_landmarks[rs_key]['z']])
            lh_3d = np.array([valid_landmarks[lh_key]['x'], valid_landmarks[lh_key]['y'], valid_landmarks[lh_key]['z']])
            rh_3d = np.array([valid_landmarks[rh_key]['x'], valid_landmarks[rh_key]['y'], valid_landmarks[rh_key]['z']])
            
            # Build 3D midpoints
            shoulder_mid = (ls_3d + rs_3d) / 2.0
            hip_mid = (lh_3d + rh_3d) / 2.0
            
            # Torso vector from hips to shoulders
            torso_vector = shoulder_mid - hip_mid
            norm = np.linalg.norm(torso_vector)
            if norm < 1e-6:  # Degenerate case
                return person_data.get('is_leaning', False)
            torso_vector /= norm
            
            # If we haven't established a baseline yet, set it now
            if person_data.get('baseline_torso_vector') is None:
                person_data['baseline_torso_vector'] = torso_vector.copy()
                return person_data.get('is_leaning', False)
            
            # Compute angle between current torso and baseline torso
            baseline = person_data['baseline_torso_vector']
            dot_val = np.clip(np.dot(torso_vector, baseline), -1.0, 1.0)
            angle_rad = np.arccos(dot_val)
            angle_deg = np.degrees(angle_rad)
            
            # Decide on threshold for "leaning"
            lean_threshold_angle = 20.0
            is_leaning_now = angle_deg > lean_threshold_angle
            
            # Simple hysteresis: require 2 consecutive frames to change state
            if is_leaning_now != person_data.get('is_leaning', False):
                if '_lean_counter' not in person_data:
                    person_data['_lean_counter'] = 0
                
                person_data['_lean_counter'] += 1
                
                # If we see the same state 2 frames in a row, commit to it
                if person_data['_lean_counter'] >= 2:
                    person_data['is_leaning'] = is_leaning_now
                    person_data['_lean_counter'] = 0
            else:
                person_data['_lean_counter'] = 0
            
            return person_data.get('is_leaning', False)
        
        except Exception as e:
            self.log_warning(f"Error in lean detection: {e}")
            return person_data.get('is_leaning', False)
    
    def smooth_position(self, position: np.ndarray, track_id: str) -> np.ndarray:
        """Smooth position using weighted moving average"""
        person_data = self.movement_data.get(track_id)
        if not person_data:
            return position
            
        # Store in position buffer
        person_data['position_buffer'].append(position.copy())
        
        # If we have a last valid position, check for position jumps
        if person_data['last_valid_position'] is not None:
            # Calculate distance to last valid position
            dist_to_last = np.linalg.norm(position - person_data['last_valid_position'])
            
            # If distance is too large, flag it as a position jump
            jump_detected = dist_to_last > 500  # 50cm jump threshold
            
            # Log the position and jump information
            if self.diagnostics:
                self.diagnostics.log_position(
                    track_id, 
                    person_data.get('last_frame_processed', 0),
                    None,  # No image position available here
                    position,
                    person_data['last_valid_position'],
                    dist_to_last,
                    jump_detected
                )
            
            if jump_detected:
                person_data['position_jump_detected'] = True
                person_data['jump_cooldown'] = 5  # Add a cooldown period after jumps
                if self.diagnostics:
                    self.diagnostics.log(f"Position jump detected for track {track_id[:6]}: {dist_to_last:.1f}mm")
            else:
                person_data['position_jump_detected'] = False
        
        # Initialize smoothed to the original position in case calculations fail
        smoothed = position.copy()
        
        if len(person_data['position_buffer']) > 0:
            try:
                # Calculate weighted average, giving more weight to recent positions
                weights = np.linspace(0.2, 1.0, len(person_data['position_buffer']))
                weights = weights / np.sum(weights)
                
                smoothed = np.zeros(3)
                for i, pos in enumerate(person_data['position_buffer']):
                    smoothed += pos * weights[i]
                
                # Apply stronger smoothing when recovering from a position jump
                if person_data['position_jump_detected'] and person_data['last_valid_position'] is not None:
                    # Blend with last valid position - use stronger smoothing (0.85 instead of 0.7)
                    alpha = 0.85  # Increased smoothing factor
                    smoothed = smoothed * (1-alpha) + person_data['last_valid_position'] * alpha
            except Exception as e:
                # If anything goes wrong, fall back to the original position
                if self.diagnostics:
                    self.diagnostics.log(f"Error in smoothing for track {track_id[:6]}: {e}")
                smoothed = position.copy()
        
        # Always update the last valid position
        person_data['last_valid_position'] = smoothed.copy()
        
        return smoothed
        
    def _is_position_stable(self, track_id: str, new_position: np.ndarray) -> bool:
        """Check if a position is reasonable based on history"""
        person_data = self.movement_data.get(track_id)
        if not person_data or len(person_data['position_history']) < 3:
            return True  # Not enough history to judge
            
        # Get the last few positions
        recent_positions = list(person_data['position_history'])[-3:]
        
        # Calculate average position
        avg_pos = np.mean(recent_positions, axis=0)
        
        # Check how far the new position is from the average
        distance = np.linalg.norm(new_position - avg_pos)
        
        # If too far from recent average, consider unstable
        return distance < 200  # 20cm threshold
    
    def detect_turn(self, new_position: np.ndarray, track_id: str) -> bool:
        """Detect if the person is currently turning based on movement direction changes"""
        person_data = self.movement_data.get(track_id)
        if not person_data or len(person_data['position_history']) < 2:
            return False
            
        # Get last position and calculate movement
        prev_pos = person_data['position_history'][-1]
        movement_vector = new_position - prev_pos
        movement_distance = np.linalg.norm(movement_vector)
        
        # Minimum movement threshold
        min_movement_for_direction = 30  # mm
        
        # If barely moving, maintain current state but decrease confidence
        if movement_distance < min_movement_for_direction:
            if person_data['turn_cooldown'] > 0:
                person_data['turn_cooldown'] -= 1
            return person_data['turn_cooldown'] > 0
            
        # Calculate movement direction
        movement_direction = np.arctan2(movement_vector[2], movement_vector[0])
        person_data['direction_history'].append(movement_direction)
        
        if len(person_data['direction_history']) < 3:
            return False
            
        # Calculate direction changes over recent history
        direction_changes = np.diff([d for d in person_data['direction_history']])
        # Normalize angles to [-pi, pi]
        direction_changes = np.mod(direction_changes + np.pi, 2 * np.pi) - np.pi
        
        # Look at recent direction changes
        recent_changes = direction_changes[-3:]  # Look at last 3 changes
        
        # Calculate average change magnitude
        avg_change = np.mean(np.abs(recent_changes))
        
        # Turning threshold
        turning_threshold = np.pi/6  # 30 degrees
        
        # Detect start of turn
        if avg_change > turning_threshold:
            person_data['turn_cooldown'] = 3
            return True
            
        # For exit condition, check if movement has been straight
        if person_data.get('is_turning', False):
            # Check if we've been moving straight for a bit
            if avg_change < turning_threshold * 0.3:
                person_data['turn_cooldown'] = 0
                return False
                
        # Handle cooldown
        if person_data['turn_cooldown'] > 0:
            person_data['turn_cooldown'] -= 1
            return True
            
        return False
    
    def calculate_speed(self, current_pos: np.ndarray, last_pos: np.ndarray, 
                        current_time: float, last_time: float, track_id: str) -> float:
        """
        Calculate speed with outlier detection and smoothing
        """
        person_data = self.movement_data.get(track_id)
        if not person_data:
            return 0.0
        
        # Calculate time difference
        dt = current_time - last_time
        
        # Check for valid time difference
        if dt <= 0.001:  # Avoid division by very small numbers
            return person_data.get('last_valid_speed', 0.0)
        
        # Calculate distance
        distance = np.linalg.norm(current_pos - last_pos)
        
        # Calculate raw speed
        raw_speed = distance / dt
        
        # Check if speed is unrealistically high (could be caused by tracking jumps)
        if raw_speed > self.max_speed_threshold:
            # Use the last valid speed instead
            self.log_warning(f"Speed outlier detected for track {track_id[:6]}: {raw_speed:.1f}mm/s")
            return person_data.get('last_valid_speed', 0.0)
        
        # Store in speed history for smoothing
        person_data['speed_history'].append(raw_speed)
        
        # Apply exponential moving average for speed smoothing
        if len(person_data['speed_history']) > 0:
            alpha = 0.3  # Smoothing factor
            if len(person_data['speed_history']) == 1:
                smoothed_speed = raw_speed
            else:
                smoothed_speed = alpha * raw_speed + (1 - alpha) * person_data.get('last_valid_speed', 0.0)
            
            person_data['last_valid_speed'] = smoothed_speed
            return smoothed_speed
        
        return raw_speed
    
    def update_movement_metrics(self, floor_position: np.ndarray, current_time: float, 
                           track_id: str, frame_idx: int) -> None:
        """Update movement metrics with turn and lean detection for a specific person"""
        person_data = self.movement_data.get(track_id)
        if not person_data:
            return
        
        # Update the frame index when this track was last processed
        person_data['last_frame_processed'] = frame_idx
        
        # Initialize if this is the first time
        if not person_data['is_initialized']:
            person_data['is_initialized'] = True
            person_data['position_history'].clear()
            person_data['time_history'].clear()
            person_data['total_distance'] = 0
            person_data['current_speed'] = 0
            person_data['last_valid_floor_position'] = floor_position.copy()
            # Add cooldown timer after jumps
            person_data['jump_cooldown'] = 0
            if self.diagnostics:
                self.diagnostics.log(f"Track {track_id[:6]} initialized at frame {frame_idx}")
        
        # Store the floor position for jump detection in next frame
        prev_floor_position = person_data.get('last_valid_floor_position')
        
        if len(person_data['position_history']) > 0 and prev_floor_position is not None:
            # Check for position jumps (which indicate tracking issues)
            jump_distance = np.linalg.norm(floor_position - prev_floor_position)
            position_jump = jump_distance > 500  # 50cm threshold

            position_stable = self._is_position_stable(track_id, floor_position)

            # Set cooldown when jump detected
            if position_jump:
                person_data['jump_cooldown'] = 5  # Ignore distance for 5 frames after a jump
                if self.diagnostics:
                    self.diagnostics.log(f"Position jump for {track_id[:6]}: {jump_distance:.1f}mm - NOT updating distance")
            elif not position_stable:
                person_data['jump_cooldown'] = 3
                if self.diagnostics:
                    self.diagnostics.log(f"Unstable position for {track_id[:6]} - NOT updating distance")

            # Calculate speed with outlier detection
            if not position_jump and len(person_data['time_history']) > 0:
                speed = self.calculate_speed(
                    floor_position, 
                    person_data['position_history'][-1],
                    current_time, 
                    person_data['time_history'][-1],
                    track_id
                )
                person_data['current_speed'] = speed
            else:
                # Speed can't be reliably calculated during position jumps
                person_data['current_speed'] = 0
            
            # Check if currently turning or leaning
            person_data['is_turning'] = self.detect_turn(floor_position, track_id)
            
            
            # Only calculate distance if position is stable, no jump, and no cooldown
            if not position_jump and position_stable and person_data['jump_cooldown'] <= 0:
                distance = np.linalg.norm(floor_position - prev_floor_position)
                
                # Apply movement thresholds and exclude turns and leaning
                min_movement = 5  # mm
                max_movement = 100  # mm (reduced from 2000mm to avoid large jumps)
                
                if min_movement < distance < max_movement:
                    # Detailed logging of movement
                    excluded_reason = ""
                    if person_data['is_turning']:
                        excluded_reason = " - excluded (turning)"
                    elif person_data['is_leaning']:
                        excluded_reason = " - excluded (leaning)"
                        
                    # Add to total distance only for normal walking
                    if not person_data['is_turning'] and not person_data['is_leaning']:
                        old_distance = person_data['total_distance']
                        
                        # Apply distance correction factor
                        corrected_distance = distance * self.distance_correction_factor
                        person_data['total_distance'] += corrected_distance
                        
                        # Log the distance addition
                        if distance > 20 and self.diagnostics:  # Only log significant movements
                            self.diagnostics.log_distance_added(
                                track_id, frame_idx, corrected_distance, person_data['total_distance'])
                    elif self.diagnostics:
                        # Log excluded movements
                        self.diagnostics.log(
                            f"Track {track_id[:6]}: Movement {distance:.1f}mm{excluded_reason}")
                
                # Generate position plot periodically or after big changes
                if self.diagnostics and ((frame_idx - self.last_position_plot_frame > 300) or (jump_distance > 1000)):
                    self.diagnostics.generate_position_plot(track_id)
                    self.last_position_plot_frame = frame_idx
                    
            elif position_jump and self.diagnostics:
                # Log significant position jumps
                self.diagnostics.log(f"Position jump for {track_id[:6]}: {jump_distance:.1f}mm - NOT updating distance")
            else:
                # Decrement cooldown if active
                if person_data['jump_cooldown'] > 0:
                    person_data['jump_cooldown'] -= 1
        
        # CRITICAL: Ensure unique state by making copies and not sharing references
        person_data['position_history'].append(floor_position.copy())
        person_data['time_history'].append(current_time)
        
        # CRITICAL: Create a new copy for last_valid_floor_position
        person_data['last_valid_floor_position'] = floor_position.copy()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process pose data to calculate movement metrics
        
        Args:
            data: Dictionary containing 'pose_data' and 'tracks'
            
        Returns:
            Dictionary with 'movement_data' containing movement metrics for each person
        """
        pose_data = data.get('pose_data', {})
        tracks = data.get('tracks', [])
        frame = data.get('frame')
        frame_idx = data.get('frame_idx', 0)
        
        if frame is None:
            return {'movement_data': self.movement_data, **data}
        
        # Debug all tracks
        track_ids = [track['id'] for track in tracks]
        if frame_idx % 30 == 0:
            self.log_info(f"Processing frame {frame_idx} with {len(tracks)} tracks: {[id[:6] for id in track_ids]}")
            self.log_debug(f"Pose data available for {len(pose_data)} tracks: {[id[:6] for id in pose_data]}")
        
        # Add camera calibration to data for visualization
        data['camera_matrix'] = self.camera_matrix
        data['dist_coeffs'] = self.dist_coeffs
        
        # Track the IDs we're processing in this frame
        current_track_ids = set()
        
        # Update movement data for each person with pose data
        for track in tracks:
            track_id = track['id']
            current_track_ids.add(track_id)
            
            # Initialize movement data for new persons
            if track_id not in self.movement_data:
                self._init_person_data(track_id)
                
            # Skip if no pose data available
            if track_id not in pose_data:
                if frame_idx % 30 == 0:
                    self.log_debug(f"No pose data for track {track_id[:6]}")
                    
                # Try to use bottom center of bbox as fallback
                if self.has_mediapipe:  # Skip if we don't have pose data
                    continue
                    
                # Fallback to using bbox bottom center
                bbox = track['bbox']
                bottom_center = np.array([(bbox[0] + bbox[2]) / 2, bbox[3]])
                floor_position = self.project_point_to_floor(bottom_center, track_id)
                if floor_position is None:
                    continue
                    
                # Smooth position 
                smoothed_position = self.smooth_position(floor_position, track_id)
                
                # Update metrics
                self.update_movement_metrics(smoothed_position, time.time(), track_id, frame_idx)
                continue
                
            # Get ankle position from pose data
            landmarks = pose_data[track_id].get('landmarks')
            if not landmarks:
                if frame_idx % 30 == 0:
                    self.log_debug(f"No landmarks for track {track_id[:6]}")
                continue
                
            ankle_pos = self.get_ankle_position(landmarks)
            if ankle_pos is None:
                if frame_idx % 30 == 0:
                    self.log_debug(f"No valid ankle position for track {track_id[:6]}")
                continue
            
            # Project to floor
            floor_position = self.project_point_to_floor(ankle_pos, track_id)
            if floor_position is None:
                if frame_idx % 30 == 0:
                    self.log_debug(f"Floor projection failed for track {track_id[:6]}")
                continue
            
            # Smooth position
            smoothed_position = self.smooth_position(floor_position, track_id)
            
            # Update metrics
            self.update_movement_metrics(smoothed_position, time.time(), track_id, frame_idx)
            
            # Detect lean
            self.movement_data[track_id]['is_leaning'] = self.detect_lean(landmarks, track_id)
        
        # Debug missing track IDs
        if current_track_ids != self.last_processed_ids:
            added = current_track_ids - self.last_processed_ids
            removed = self.last_processed_ids - current_track_ids
            
            if added:
                self.log_debug(f"New tracks in frame {frame_idx}: {added}")
            if removed:
                self.log_debug(f"Tracks removed in frame {frame_idx}: {removed}")
            
            self.last_processed_ids = current_track_ids
        
        # Print distance summary every 30 frames
        if frame_idx % 30 == 0:
            self.log_info("\nDISTANCES PER TRACK:")
            for track_id, track_data in self.movement_data.items():
                if track_id in current_track_ids:  # Only show active tracks
                    distance = track_data.get('total_distance', 0)
                    self.log_info(f"  Track {track_id[:6]}: {distance/1000:.2f}m")
        
        # Add floor position visualization if enabled
        if self.enable_floor_visualization and 'frame' in data and self.diagnostics:
            try:
                # Create floor position visualization
                viz_frame = self.diagnostics.visualize_floor_positions(
                    data['frame'],
                    data.get('tracks', []),
                    self.movement_data,
                    self.camera_matrix,
                    self.dist_coeffs
                )
                
                # Add to output data
                data['floor_position_frame'] = viz_frame
            except Exception as e:
                self.log_error(f"Error in floor visualization: {e}")
        
        # Return updated data
        return {
            'movement_data': self.movement_data,
            'frame_idx': frame_idx + 1,
            **data
        }
    
    def reset(self) -> None:
        """Reset the movement analyzer state"""
        self.movement_data = {}
        self.last_processed_ids = set()
        self.last_position_plot_frame = 0
        self.log_info("Movement analyzer reset")