import cv2
import numpy as np
import os
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

class DeepSORTDiagnostics:
    """Diagnostic tools for debugging DeepSORT tracking"""
    
    def __init__(self, feature_extractor=None, feature_gallery=None, output_dir="deepsort_diagnostics"):
        """
        Initialize diagnostics
        
        Args:
            feature_extractor: The AppearanceFeatureExtractor instance
            feature_gallery: The PersonFeatureGallery instance
            output_dir: Directory to save diagnostic outputs
        """
        self.logger = logging.getLogger("motion_tracking.DeepSORTDiagnostics")
        self.feature_extractor = feature_extractor
        self.feature_gallery = feature_gallery
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "gallery"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        
        # Create log file
        log_path = os.path.join(output_dir, "deepsort_log.txt")
        if os.path.exists(log_path):
            # Append timestamp to create a new log file
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(output_dir, f"deepsort_log_{timestamp}.txt")
            
        self.log_file = log_path
        with open(self.log_file, 'w') as f:
            f.write("DeepSORT Tracking Log\n")
            f.write("====================\n\n")
            
        self.log(f"Diagnostics initialized, saving to {output_dir}")
    
    def log(self, message: str) -> None:
        """Log a message to the log file and console"""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        # Write to log file
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"{log_message}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
        
        # Also log to console
        self.logger.info(message)
    
    def verify_model(self) -> bool:
        """Verify the feature extraction model is loaded correctly"""
        if self.feature_extractor is None:
            self.log("ERROR: No feature extractor provided to diagnostics")
            return False
            
        if not hasattr(self.feature_extractor, 'has_model'):
            self.log("ERROR: Feature extractor missing has_model attribute")
            return False
            
        if not self.feature_extractor.has_model:
            self.log("ERROR: Feature extractor model not successfully loaded")
            return False
            
        self.log("Feature extractor model verified")
        return True
    
    def log_feature_extraction(self, track_id: str, crop: np.ndarray, 
                              features: np.ndarray, frame_idx: int) -> None:
        """Log information about feature extraction for a track"""
        if crop is None or features is None:
            return
            
        # Save the crop image
        crop_path = os.path.join(self.output_dir, "features", 
                               f"track_{track_id[:6]}_frame_{frame_idx}.jpg")
        try:
            cv2.imwrite(crop_path, crop)
        except Exception as e:
            self.log(f"Error saving crop: {e}")
        
        # Log feature information
        feature_dim = features.shape[0] if features is not None else 0
        feature_min = np.min(features) if features is not None else 0
        feature_max = np.max(features) if features is not None else 0
        feature_mean = np.mean(features) if features is not None else 0
        
        self.log(f"Extracted features for track {track_id[:6]} at frame {frame_idx}: "
               f"dim={feature_dim}, range=[{feature_min:.3f}, {feature_max:.3f}], mean={feature_mean:.3f}")
    
    def log_track_creation(self, track_id: str, bbox: List[float], frame_idx: int) -> None:
        """Log creation of a new track"""
        self.log(f"Created new track {track_id[:6]} at frame {frame_idx}: bbox={[int(x) for x in bbox]}")
    
    def log_track_update(self, track_id: str, bbox: List[float], frame_idx: int) -> None:
        """Log update of an existing track"""
        if frame_idx % 30 == 0:  # Log less frequently to reduce spam
            self.log(f"Updated track {track_id[:6]} at frame {frame_idx}: bbox={[int(x) for x in bbox]}")
    
    def log_track_lost(self, track_id: str, frame_idx: int) -> None:
        """Log when a track is marked as lost"""
        self.log(f"Track {track_id[:6]} lost at frame {frame_idx}")
    
    def log_track_recovered(self, track_id: str, bbox: List[float], frame_idx: int, similarity: float) -> None:
        """Log recovery of a lost track"""
        self.log(f"Recovered track {track_id[:6]} at frame {frame_idx}: "
               f"bbox={[int(x) for x in bbox]}, similarity={similarity:.3f}")
    
    def log_reid_match(self, features: np.ndarray, matched_id: Optional[str], 
                      similarity: float, all_similarities: List[float]) -> None:
        """Log a ReID matching attempt"""
        if matched_id:
            self.log(f"ReID match: features -> track {matched_id[:6]}, similarity={similarity:.3f}")
        else:
            if all_similarities:
                top_sim = max(all_similarities) if all_similarities else 0
                self.log(f"ReID no match: highest similarity={top_sim:.3f} (below threshold)")
    
    def snapshot_gallery(self, frame_idx: int) -> None:
        """Create a snapshot of the feature gallery state"""
        if self.feature_gallery is None or frame_idx % 300 != 0:  # Every 300 frames
            return
            
        # Log gallery statistics
        track_counts = []
        for track_id, data in self.feature_gallery.gallery.items():
            feature_count = len(data['features'])
            inactive_time = self.feature_gallery.frame_count - data['last_seen']
            track_counts.append((track_id, feature_count, inactive_time))
        
        if not track_counts:
            return
            
        # Sort by feature count (most features first)
        track_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Log overview
        self.log(f"Gallery snapshot at frame {frame_idx}: {len(track_counts)} tracks")
        for track_id, count, inactive in track_counts[:5]:  # Log top 5
            self.log(f"  Track {track_id[:6]}: {count} features, {inactive} frames inactive")
    
    def generate_diagnostic_frame(self, frame: np.ndarray, active_tracks: List[Dict[str, Any]],
                                 lost_tracks: List[Dict[str, Any]], frame_idx: int) -> np.ndarray:
        """Generate a visualization frame with diagnostic information"""
        if frame is None:
            self.log("Error: Cannot generate diagnostic frame from None input")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Create a copy of the frame
        diagnostic_frame = frame.copy()
        
        # Draw active tracks
        for track in active_tracks:
            track_id = track['id']
            bbox = track['bbox']
            color = track['color']
            
            # Draw bounding box
            cv2.rectangle(diagnostic_frame, 
                        (int(bbox[0]), int(bbox[1])), 
                        (int(bbox[2]), int(bbox[3])), 
                        color, 2)
            
            # Draw track ID
            cv2.putText(diagnostic_frame, f"ID: {track_id[:6]}", 
                      (int(bbox[0]), int(bbox[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # If we have feature gallery, show feature count
            if self.feature_gallery and track_id in self.feature_gallery.gallery:
                feature_count = len(self.feature_gallery.gallery[track_id]['features'])
                cv2.putText(diagnostic_frame, f"feat: {feature_count}", 
                          (int(bbox[0]), int(bbox[1] - 30)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw lost tracks with dashed lines
        for track in lost_tracks:
            track_id = track['id']
            bbox = track['bbox']
            # Use yellow color for lost tracks
            color = (0, 255, 255)
            
            # Draw dashed bounding box
            for i in range(0, 8):
                pt1_x = int(bbox[0] + (bbox[2] - bbox[0]) * i / 8)
                pt1_y = int(bbox[1])
                pt2_x = int(bbox[0] + (bbox[2] - bbox[0]) * (i + 0.5) / 8)
                pt2_y = int(bbox[1])
                cv2.line(diagnostic_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
                
                pt1_x = int(bbox[0] + (bbox[2] - bbox[0]) * i / 8)
                pt1_y = int(bbox[3])
                pt2_x = int(bbox[0] + (bbox[2] - bbox[0]) * (i + 0.5) / 8)
                pt2_y = int(bbox[3])
                cv2.line(diagnostic_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
                
                pt1_x = int(bbox[0])
                pt1_y = int(bbox[1] + (bbox[3] - bbox[1]) * i / 8)
                pt2_x = int(bbox[0])
                pt2_y = int(bbox[1] + (bbox[3] - bbox[1]) * (i + 0.5) / 8)
                cv2.line(diagnostic_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
                
                pt1_x = int(bbox[2])
                pt1_y = int(bbox[1] + (bbox[3] - bbox[1]) * i / 8)
                pt2_x = int(bbox[2])
                pt2_y = int(bbox[1] + (bbox[3] - bbox[1]) * (i + 0.5) / 8)
                cv2.line(diagnostic_frame, (pt1_x, pt1_y), (pt2_x, pt2_y), color, 2)
            
            # Draw lost track ID
            cv2.putText(diagnostic_frame, f"Lost: {track_id[:6]}", 
                      (int(bbox[0]), int(bbox[1] - 10)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw time since update
            cv2.putText(diagnostic_frame, f"TSU: {track['time_since_update']}", 
                      (int(bbox[0]), int(bbox[1] - 30)),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw frame information
        cv2.putText(diagnostic_frame, f"Frame: {frame_idx}", 
                  (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(diagnostic_frame, f"Active: {len(active_tracks)}, Lost: {len(lost_tracks)}", 
                  (10, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save frame every 30 frames
        if frame_idx % 30 == 0:
            try:
                frame_path = os.path.join(self.output_dir, "frames", f"frame_{frame_idx}.jpg")
                cv2.imwrite(frame_path, diagnostic_frame)
            except Exception as e:
                self.log(f"Error saving diagnostic frame: {e}")
        
        return diagnostic_frame