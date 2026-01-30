import numpy as np
import cv2
import yaml

class DistanceMeasurer:
    def __init__(self, camera_calibration_file, floor_calibration_file):
        # Load camera calibration
        with open(camera_calibration_file) as f:
            camera_data = yaml.safe_load(f)
        self.camera_matrix = np.array(camera_data['camera_matrix'])
        self.dist_coeffs = np.array(camera_data['dist_coeff'])
        
        # Load floor calibration
        with open(floor_calibration_file) as f:
            floor_data = yaml.safe_load(f)
        self.floor_plane = np.array(floor_data['floor_plane'])
        
        # State variables
        self.points = []
        self.current_frame = None
        self.measuring = False
    
    def project_point_to_floor(self, image_point):
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
                return None
                
            t = -d / denominator
            if t < 0:  # Intersection behind camera
                return None
                
            intersection = ray_direction * t
            
            # Sanity check on the calculated position
            if abs(intersection[1]) > 3000:  # More than 3 meters up/down
                return None
                
            return intersection
            
        except Exception as e:
            print(f"Error in projection: {e}")
            return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.measuring:
            # Project click point to floor
            floor_point = self.project_point_to_floor(np.array([x, y]))
            if floor_point is not None:
                self.points.append((floor_point, (x, y)))
                self.draw_measurements()
    
    def draw_measurements(self):
        """Draw measurements on the frame"""
        if self.current_frame is None:
            return
            
        frame = self.current_frame.copy()
        
        # Draw all points
        for i, (_, image_point) in enumerate(self.points):
            cv2.circle(frame, image_point, 5, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i+1}", 
                       (image_point[0] + 10, image_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw lines and distances between point pairs (1-2, 3-4, etc.)
        for i in range(0, len(self.points) - 1, 2):
            pt1 = self.points[i]
            pt2 = self.points[i + 1]
            
            # Draw line
            cv2.line(frame, pt1[1], pt2[1], (0, 255, 0), 2)
            
            # Calculate midpoint for text placement
            mid_x = (pt1[1][0] + pt2[1][0]) // 2
            mid_y = (pt1[1][1] + pt2[1][1]) // 2
            
            # Calculate and draw distance with additional checks
            distance_mm = np.linalg.norm(pt2[0] - pt1[0])
            
            # Apply 10% correction factor
            distance_mm = distance_mm * 1.1
            
            # Apply same movement thresholds as tracking script
            min_movement = 5  # mm
            max_movement = 2000  # mm
            
            # Draw distance with background
            text = f"{distance_mm/1000:.2f}m"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            if not (min_movement < distance_mm < max_movement):
                cv2.putText(frame, "! Distance may be inaccurate !",
                           (mid_x - text_w//2, mid_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            cv2.rectangle(frame, 
                         (mid_x - text_w//2 - 5, mid_y - text_h - 5),
                         (mid_x + text_w//2 + 5, mid_y + 5),
                         (0, 0, 0), -1)
            cv2.putText(frame, text,
                       (mid_x - text_w//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "SPACE - Start/Stop measuring",
            "R - Reset points",
            "Q - Quit",
            f"Points: {len(self.points)} (Measuring pairs: 1-2, 3-4, ...)"
        ]
        
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show measuring state
        status = "MEASURING" if self.measuring else "PAUSED"
        cv2.putText(frame, status, (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 1,
                   (0, 255, 0) if self.measuring else (0, 0, 255), 2)
        
        cv2.imshow('Distance Measurement', frame)

def main():
    import os
    import argparse
    
    # Add command line arguments for input/output paths
    parser = argparse.ArgumentParser(description='Distance Measurement Tool')
    parser.add_argument('--camera-calib', type=str, default='calibration/camera_calibration.yaml',
                        help='Path to camera calibration file')
    parser.add_argument('--floor-calib', type=str, default='calibration/floor_calibration.yaml',
                        help='Path to floor calibration file')
    parser.add_argument('--input', type=str, default='calibration/distance_correction_input',
                        help='Path to input video or directory with distance measurement videos')
    args = parser.parse_args()
    
    # Check if calibration files exist
    if not os.path.exists(args.camera_calib):
        print(f"Error: Camera calibration file not found: {args.camera_calib}")
        print("Please run camera calibration first.")
        return
        
    if not os.path.exists(args.floor_calib):
        print(f"Error: Floor calibration file not found: {args.floor_calib}")
        print("Please run floor calibration first.")
        return
    
    # Initialize measurer with calibration files
    measurer = DistanceMeasurer(args.camera_calib, args.floor_calib)
    
    # Look for videos in the input directory
    if os.path.isdir(args.input):
        video_files = [f for f in os.listdir(args.input) if f.endswith(('.mp4', '.avi', '.m2ts', '.mov'))]
        if video_files:
            video_path = os.path.join(args.input, video_files[0])
            print(f"Using video file: {video_path}")
        else:
            print(f"No video files found in {args.input}")
            return
    else:
        # Use the input directly as a video path
        video_path = args.input
    
    # Video setup
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Create window and set mouse callback
    cv2.namedWindow('Distance Measurement')
    cv2.setMouseCallback('Distance Measurement', measurer.mouse_callback)
    
    print("\nDistance Measurement Instructions:")
    print("1. Press SPACE to start/stop measuring")
    print("2. Click points on the floor to measure distances")
    print("3. Press R to reset points")
    print("4. Press Q to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        measurer.current_frame = frame
        measurer.draw_measurements()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            measurer.measuring = not measurer.measuring
            print("Measuring:" if measurer.measuring else "Paused")
        elif key == ord('r'):
            measurer.points = []
            print("Points reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()