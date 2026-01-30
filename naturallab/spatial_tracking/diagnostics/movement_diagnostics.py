import os
import time
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional

class MovementDiagnostics:
    """Diagnostic tools for debugging movement analysis issues"""
    
    def __init__(self, output_dir='movement_diagnostics'):
        """Initialize the diagnostics helper"""
        self.logger = logging.getLogger("motion_tracking.MovementDiagnostics")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'frames'), exist_ok=True)
        
        # Store history of positions per track
        self.position_history = {}
        
        # Create log file
        self.log_file = os.path.join(output_dir, 'movement_log.txt')
        with open(self.log_file, 'w') as f:
            f.write("Movement Analysis Log\n")
            f.write("====================\n\n")
    
    def log(self, message):
        """Write message to log file"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        self.logger.info(message)
    
    def log_position(self, track_id, frame_idx, 
                    image_pos, floor_pos=None, smoothed_pos=None, 
                    dist_to_last=None, is_jump=False):
        """Log position information for a track"""
        if track_id not in self.position_history:
            self.position_history[track_id] = []
        
        # Format positions for better readability
        img_pos_str = f"({image_pos[0]:.1f}, {image_pos[1]:.1f})" if image_pos is not None else "None"
        floor_pos_str = f"({floor_pos[0]:.1f}, {floor_pos[1]:.1f}, {floor_pos[2]:.1f})" if floor_pos is not None else "None"
        smoothed_pos_str = f"({smoothed_pos[0]:.1f}, {smoothed_pos[1]:.1f}, {smoothed_pos[2]:.1f})" if smoothed_pos is not None else "None"
        
        entry = {
            'frame_idx': frame_idx,
            'image_pos': image_pos,
            'floor_pos': floor_pos,
            'smoothed_pos': smoothed_pos,
            'dist_to_last': dist_to_last,
            'is_jump': is_jump
        }
        
        self.position_history[track_id].append(entry)
        
        # Log significant events immediately
        if is_jump or (dist_to_last is not None and dist_to_last > 100):
            jump_str = "JUMP DETECTED!" if is_jump else ""
            self.log(f"Track {track_id[:6]} Frame {frame_idx}: Dist={dist_to_last:.1f}mm {jump_str}")
            self.log(f"  Image: {img_pos_str}")
            self.log(f"  Floor: {floor_pos_str}")
            self.log(f"  Smoothed: {smoothed_pos_str}")
    
    def log_distance_added(self, track_id, frame_idx, distance_added, total_distance):
        """Log when distance is added to a track"""
        self.log(f"Track {track_id[:6]} Frame {frame_idx}: Added {distance_added:.1f}mm, Total={total_distance/1000:.2f}m")
    
    def generate_position_plot(self, track_id, last_n_frames=100):
        """Generate a plot of recent positions for a track"""
        try:
            import matplotlib.pyplot as plt
            
            if track_id not in self.position_history:
                return
            
            history = self.position_history[track_id]
            if len(history) < 2:
                return
            
            # Take only the last N frames
            history = history[-last_n_frames:]
            
            # Extract data
            frames = [entry['frame_idx'] for entry in history]
            floor_x = [entry['floor_pos'][0] if entry['floor_pos'] is not None else np.nan for entry in history]
            floor_z = [entry['floor_pos'][2] if entry['floor_pos'] is not None else np.nan for entry in history]
            
            # Calculate displacements between consecutive points
            displacements = []
            for i in range(1, len(history)):
                prev_pos = history[i-1]['floor_pos']
                curr_pos = history[i]['floor_pos']
                
                if prev_pos is not None and curr_pos is not None:
                    dx = curr_pos[0] - prev_pos[0]
                    dz = curr_pos[2] - prev_pos[2]
                    displacement = np.sqrt(dx*dx + dz*dz)
                    displacements.append(displacement)
                else:
                    displacements.append(np.nan)
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Plot bird's eye view (top-down XZ plane)
            ax1.plot(floor_x, floor_z, 'b-', label='Path')
            ax1.scatter(floor_x, floor_z, c=frames, cmap='viridis', label='Positions')
            
            # Mark jumps
            jumps = [i for i, entry in enumerate(history) if entry['is_jump']]
            if jumps:
                jump_x = [floor_x[i] for i in jumps]
                jump_z = [floor_z[i] for i in jumps]
                ax1.scatter(jump_x, jump_z, c='red', s=100, marker='x', label='Jumps')
            
            ax1.set_title(f'Track {track_id[:6]} - Floor Position (Bird\'s Eye View)')
            ax1.set_xlabel('X position (mm)')
            ax1.set_ylabel('Z position (mm)')
            ax1.grid(True)
            ax1.axis('equal')
            ax1.legend()
            
            # Plot displacements
            ax2.plot(frames[1:], displacements, 'g-')
            ax2.axhline(y=500, color='r', linestyle='--', label='Jump Threshold (500mm)')
            ax2.set_title(f'Frame-to-Frame Displacement')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Displacement (mm)')
            ax2.grid(True)
            ax2.legend()
            
            # Save plot
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'plots', f'track_{track_id[:6]}_positions.png')
            plt.savefig(plot_path)
            plt.close()
            
            self.log(f"Generated position plot for track {track_id[:6]}")
            return plot_path
            
        except Exception as e:
            self.log(f"Error generating position plot: {e}")
            return None
    
    def visualize_floor_positions(self, frame, tracks, movement_data, camera_matrix, dist_coeffs):
        """Add floor position visualization to frame"""
        import cv2
        import numpy as np
        
        # Create a copy of the frame
        output = frame.copy()
        
        try:
            # Draw floor grid (every 500mm)
            grid_size = 5000  # 5 meters
            grid_step = 500   # 0.5 meter
            
            grid_points = []
            for x in range(-grid_size, grid_size+1, grid_step):
                for z in range(-grid_size, grid_size+1, grid_step):
                    grid_points.append(np.array([x, 0, z], dtype=np.float32))
            
            # Project grid points to image
            for point in grid_points:
                try:
                    pixel_point = cv2.projectPoints(
                        point.reshape(1, 3),
                        np.zeros(3),
                        np.zeros(3),
                        camera_matrix,
                        dist_coeffs
                    )[0][0][0]
                    
                    cv2.circle(output, 
                              (int(pixel_point[0]), int(pixel_point[1])), 
                              2, (50, 50, 50), -1)
                except Exception as e:
                    continue
            
            # Draw floor positions for each track
            for track in tracks:
                track_id = track['id']
                color = track['color']
                
                if track_id in movement_data:
                    person_data = movement_data[track_id]
                    positions = list(person_data.get('position_history', []))
                    
                    if positions and len(positions) > 0:
                        # Get last position
                        last_pos = positions[-1]
                        
                        try:
                            # Project to image
                            pixel_point = cv2.projectPoints(
                                last_pos.reshape(1, 3),
                                np.zeros(3),
                                np.zeros(3),
                                camera_matrix,
                                dist_coeffs
                            )[0][0][0]
                            
                            # Draw floor position marker
                            cv2.circle(output, 
                                     (int(pixel_point[0]), int(pixel_point[1])), 
                                     8, color, -1)
                            
                            # Draw connecting line to person
                            bbox = track['bbox']
                            bottom_center = (int((bbox[0] + bbox[2]) / 2), int(bbox[3]))
                            cv2.line(output, bottom_center, 
                                    (int(pixel_point[0]), int(pixel_point[1])), 
                                    color, 2, cv2.LINE_AA)
                            
                            # Label with coordinates
                            label = f"({last_pos[0]:.0f}, {last_pos[2]:.0f})"
                            cv2.putText(output, label, 
                                      (int(pixel_point[0]) + 10, int(pixel_point[1]) + 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                        except Exception as e:
                            self.logger.debug(f"Error projecting point: {e}")
                            continue
        except Exception as e:
            self.log(f"Error in floor visualization: {e}")
        
        return output