import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from motion_tracking.base import TrackerModule

class VisualizationModule(TrackerModule):
    """Visualization module for multi-person tracking"""
    
    def __init__(self, show_diagnostics: bool = True):
        """
        Initialize the visualization module
        
        Args:
            show_diagnostics: Whether to show diagnostic information
        """
        super().__init__(name="Visualizer")
        
        # Try to import mediapipe for pose visualization
        try:
            import mediapipe as mp
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
            self.has_mediapipe = True
        except ImportError:
            self.log_warning("MediaPipe not available. Pose visualization will be limited.")
            self.has_mediapipe = False
            self.mp_draw = None
            self.mp_pose = None
        
        # Track history of distances to debug
        self.last_distances = {}
        self.show_diagnostics = show_diagnostics
        self.fps_history = []
        self.last_frame_time = time.time()
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Draw tracking visualization on the frame
        
        Args:
            data: Dictionary containing 'frame', 'tracks', 'pose_data', and 'movement_data'
            
        Returns:
            Dictionary with 'output_frame' containing the visualized frame
        """
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        if dt > 0:
            fps = 1.0 / dt
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history = self.fps_history[-30:]
        
        # Use floor position visualization if available, otherwise use the original frame
        if 'floor_position_frame' in data:
            frame = data['floor_position_frame'].copy()
        elif 'diagnostic_frame' in data:
            frame = data['diagnostic_frame'].copy()
        else:
            frame = data['frame'].copy()
        
        tracks = data.get('tracks', [])
        pose_data = data.get('pose_data', {})
        movement_data = data.get('movement_data', {})
        frame_idx = data.get('frame_idx', 0)
        
        # Draw each tracked person
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            color = track['color']
            
            # Track ID above bounding box with different color per person
            id_text = f"ID: {track_id[:6]}"
            cv2.putText(frame, id_text,
                      (int(bbox[0]), int(bbox[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        color, 2)
            
            # Draw pose if available
            if track_id in pose_data and self.has_mediapipe and self.mp_draw:
                person_pose = pose_data[track_id]
                
                if 'landmarks' in person_pose:
                    # Draw MediaPipe pose landmarks with person's color
                    self.mp_draw.draw_landmarks(
                        frame,
                        person_pose['landmarks'],
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=color, thickness=2)
                    )
            
            # Draw movement data if available
            if track_id in movement_data:
                person_movement = movement_data[track_id]
                
                # Get the last two positions if available to show frame-to-frame distances
                positions = list(person_movement.get('position_history', []))
                
                if len(positions) >= 2:
                    # Get the last two positions
                    current_pos = positions[-1]
                    prev_pos = positions[-2]
                    
                    # Calculate distance
                    distance = np.linalg.norm(current_pos - prev_pos)
                    
                    # Position text above the bounding box
                    text_pos = (int(bbox[0]), int(bbox[1] - 25))
                    
                    # Show frame-to-frame distance
                    distance_text = f"Δ={distance:.0f}mm"
                    
                    # Determine color based on distance (red for large movements)
                    text_color = color
                    if distance > 100:
                        text_color = (0, 165, 255)  # Orange for significant
                    if distance > 200:
                        text_color = (0, 0, 255)    # Red for large
                    
                    # Draw with black outline for better visibility
                    cv2.putText(frame, distance_text, 
                              (text_pos[0]+1, text_pos[1]+1),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
                    cv2.putText(frame, distance_text, 
                              text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        
        # Draw overall statistics table with separate columns for each person
        if self.show_diagnostics:
            table_width = 200  # Width per person
            top_margin = 30
            person_count = 0
            
            # Sort tracks by ID to maintain consistent ordering
            sorted_tracks = sorted(tracks, key=lambda t: t['id'])
            
            for track in sorted_tracks:
                track_id = track['id']
                color = track['color']
                
                if track_id in movement_data:
                    person_movement = movement_data[track_id]
                    
                    # Background rectangle for text
                    bg_color = (30, 30, 30)  # Darker background for better contrast
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    
                    # Calculate x position for this person's column
                    x_offset = 10 + person_count * table_width
                    y_pos = top_margin
                    
                    # Draw column header with background
                    text = f"Person {track_id[:6]}"
                    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                (x_offset, y_pos - 20),
                                (x_offset + table_width - 10, y_pos + 5),
                                bg_color, -1)
                    cv2.putText(frame, text, (x_offset + 5, y_pos), 
                              font, 0.7, color, 2)
                    y_pos += 30
                    
                    # Get metrics
                    total_distance = person_movement.get('total_distance', 0)
                    current_speed = person_movement.get('current_speed', 0)
                    
                    # Track distance changes for debugging
                    last_dist = self.last_distances.get(track_id, 0)
                    dist_change = total_distance - last_dist
                    self.last_distances[track_id] = total_distance
                    
                    # Draw distance with background and change indicator
                    text = f"Dist: {total_distance/1000:.2f}m"
                    if dist_change > 0:
                        text += f" (+{dist_change/1000:.2f}m)"
                    
                    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                (x_offset, y_pos - 20),
                                (x_offset + table_width - 10, y_pos + 5),
                                bg_color, -1)
                    cv2.putText(frame, text, (x_offset + 5, y_pos), 
                              font, 0.7, color, 2)
                    y_pos += 30
                    
                    # Draw speed with background
                    text = f"Speed: {current_speed/1000:.2f}m/s"
                    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                (x_offset, y_pos - 20),
                                (x_offset + table_width - 10, y_pos + 5),
                                bg_color, -1)
                    cv2.putText(frame, text, (x_offset + 5, y_pos), 
                              font, 0.7, color, 2)
                    y_pos += 30
                    
                    # Status indicators
                    is_turning = person_movement.get('is_turning', False)
                    turn_status = "Turning" if is_turning else "Walking"
                    turn_color = (0, 0, 255) if is_turning else color
                    
                    text = f"Status: {turn_status}"
                    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                (x_offset, y_pos - 20),
                                (x_offset + table_width - 10, y_pos + 5),
                                bg_color, -1)
                    cv2.putText(frame, text, (x_offset + 5, y_pos), 
                              font, 0.7, turn_color, 2)
                    y_pos += 30
                    
                    # Show leaning status
                    is_leaning = person_movement.get('is_leaning', False)
                    lean_status = "Leaning" if is_leaning else "Upright"
                    lean_color = (0, 0, 255) if is_leaning else color
                    
                    text = f"Posture: {lean_status}"
                    text_size = cv2.getTextSize(text, font, 0.7, 2)[0]
                    cv2.rectangle(frame, 
                                (x_offset, y_pos - 20),
                                (x_offset + table_width - 10, y_pos + 5),
                                bg_color, -1)
                    cv2.putText(frame, text, (x_offset + 5, y_pos), 
                              font, 0.7, lean_color, 2)
                    
                    # Add per-frame distance information
                    y_pos += 30
                    if len(person_movement.get('position_history', [])) >= 2:
                        # Get last two positions
                        positions = list(person_movement.get('position_history', []))
                        current_pos = positions[-1]
                        prev_pos = positions[-2]
                        
                        # Calculate last frame distance
                        last_frame_dist = np.linalg.norm(current_pos - prev_pos)
                        
                        # Show with color coding
                        frame_dist_color = color
                        if last_frame_dist > 100:
                            frame_dist_color = (0, 165, 255)  # Orange for significant
                        if last_frame_dist > 200:
                            frame_dist_color = (0, 0, 255)    # Red for large
                        
                        text = f"Last move: {last_frame_dist:.1f}mm"
                        cv2.rectangle(frame, 
                                    (x_offset, y_pos - 20),
                                    (x_offset + table_width - 10, y_pos + 5),
                                    bg_color, -1)
                        cv2.putText(frame, text, (x_offset + 5, y_pos), 
                                  font, 0.7, frame_dist_color, 2)
                    
                    # Increment person count for next column
                    person_count += 1
            
            # Draw frame number and FPS
            avg_fps = sum(self.fps_history) / max(1, len(self.fps_history))
            cv2.putText(frame, f"Frame: {frame_idx} | FPS: {avg_fps:.1f}", 
                      (frame.shape[1] - 250, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Return the visualized frame
        return {
            'output_frame': frame,
            **data
        }
    
    def reset(self) -> None:
        """Reset visualization state"""
        self.last_distances = {}
        self.fps_history = []
        self.last_frame_time = time.time()
        self.log_info("Visualizer reset")