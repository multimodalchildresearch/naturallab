import numpy as np
import cv2
import yaml

class VideoFloorCalibrator:
    def __init__(self, calibration_file, min_positions=3):
        """Initialize calibrator for floor-based calibration"""
        # Load camera calibration
        with open(calibration_file) as f:
            calibration_data = yaml.safe_load(f)
        self.camera_matrix = np.array(calibration_data['camera_matrix'])
        self.dist_coeffs = np.array(calibration_data['dist_coeff'])
        
        # Chessboard parameters
        self.square_size = 172  # mm
        self.pattern_size = (7, 7)
        self.pattern_points = np.zeros((7 * 7, 3), np.float32)
        self.pattern_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2) * self.square_size
        
        # Calibration state
        self.min_positions = min_positions
        self.calibration_positions = []
        self.floor_plane = None
        
        # Debug info
        self.last_frame_had_detection = False
    
    def process_frame(self, frame):
        """Process a video frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Get pose of chessboard
            ret, rvec, tvec = cv2.solvePnP(
                self.pattern_points, 
                corners2, 
                self.camera_matrix, 
                self.dist_coeffs
            )
            
            if ret:
                self.last_frame_had_detection = True
                return True, corners2, tvec

        self.last_frame_had_detection = False
        return False, None, None
    
    def add_position(self, corners, tvec):
        """Add a new calibration position if it's different enough"""
        if self._is_new_position(corners):
            self.calibration_positions.append((corners, tvec))
            return True, len(self.calibration_positions)
        return False, len(self.calibration_positions)
    
    def _is_new_position(self, new_corners, min_distance_pixels=100):
        """Check if the new position is far enough from existing ones"""
        new_center = np.mean(new_corners, axis=0)
        
        for existing_corners, _ in self.calibration_positions:
            existing_center = np.mean(existing_corners, axis=0)
            distance = np.linalg.norm(new_center - existing_center)
            if distance < min_distance_pixels:
                return False
        
        return True
    
    def compute_floor_plane(self):
        """Compute floor plane from all calibration positions"""
        if len(self.calibration_positions) < self.min_positions:
            return False
            
        # Collect all corners in 3D space
        all_3d_points = []
        for corners, tvec in self.calibration_positions:
            for corner in corners:
                point_3d = self._image_to_floor_point(corner[0], tvec[2][0])
                if point_3d is not None:
                    all_3d_points.append(point_3d)
        
        if len(all_3d_points) < 3:
            return False
            
        # Fit plane to points
        points = np.array(all_3d_points)
        centroid = np.mean(points, axis=0)
        _, _, vh = np.linalg.svd(points - centroid)
        normal = vh[2]
        
        # Ensure normal points upward
        if normal[1] < 0:
            normal = -normal
            
        d = -np.dot(normal, centroid)
        self.floor_plane = np.array([*normal, d])
        return True
    
    def _image_to_floor_point(self, image_point, depth):
        """Convert image point to 3D point on floor given approximate depth"""
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
        
        return ray_direction * depth

def main():
    import os
    import argparse
    
    # Add command line arguments for input/output paths
    parser = argparse.ArgumentParser(description='Floor Calibration Tool')
    parser.add_argument('--camera-calib', type=str, default='calibration/camera_calibration.yaml',
                        help='Path to camera calibration file')
    parser.add_argument('--input', type=str, default='calibration/floor_calibration_input',
                        help='Path to input video or directory with floor calibration images')
    parser.add_argument('--output', type=str, default='calibration',
                        help='Directory to save floor calibration results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize calibrator with the camera calibration file
    calibrator = VideoFloorCalibrator(args.camera_calib)
    
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
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    print("\nFloor Calibration Instructions:")
    print("1. Press SPACE when the chessboard is in a good position")
    print("2. Need at least 3 different positions")
    print("3. Press 'f' to finish calibration")
    print("4. Press 'r' to reset calibration")
    print("5. Press 'q' to quit\n")

    calibration_file = os.path.join(args.output, 'floor_calibration.yaml')
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video, looping back")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        frame_count += 1
        if frame_count % 10 != 0:  # Process every 10th frame
            continue
        
        # Try to find chessboard
        success, corners, tvec = calibrator.process_frame(frame)
        
        # Create visualization frame
        vis_frame = frame.copy()
        
        # Draw existing stored positions in red
        for stored_corners, _ in calibrator.calibration_positions:
            cv2.drawChessboardCorners(vis_frame, (7, 7), stored_corners, True)
            center = np.mean(stored_corners, axis=0)[0].astype(np.int32)
            cv2.circle(vis_frame, tuple(center), 5, (0, 0, 255), -1)
        
        # If chessboard is detected in current frame, draw it in green
        if success:
            cv2.drawChessboardCorners(vis_frame, (7, 7), corners, True)
            current_center = np.mean(corners, axis=0)[0].astype(np.int32)
            cv2.circle(vis_frame, tuple(current_center), 5, (0, 255, 0), -1)
            
            # Add text to show if position is valid for capture
            if calibrator._is_new_position(corners):
                cv2.putText(vis_frame,
                           "Press SPACE to capture position",
                           (10, vis_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis_frame,
                           "Move chessboard further from previous positions",
                           (10, vis_frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
        
        # Draw status
        cv2.putText(vis_frame,
                   f"Positions: {len(calibrator.calibration_positions)}/{calibrator.min_positions}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        
        # Add debug info
        cv2.putText(vis_frame,
                   f"Detection: {'Yes' if success else 'No'}",
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0) if success else (0, 0, 255), 2)
        
        cv2.imshow('Floor Calibration', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and success:
            # Add current position
            added, num_positions = calibrator.add_position(corners, tvec)
            if added:
                print(f"Position {num_positions} captured!")
            else:
                print("Position too similar to existing ones")
        elif key == ord('f'):
            if calibrator.compute_floor_plane():
                print("\nFloor plane computed successfully!")
                # Save calibration data
                calibration_data = {
                    'floor_plane': calibrator.floor_plane.tolist()
                }
                with open(calibration_file, 'w') as f:
                    yaml.dump(calibration_data, f)
                print(f"Calibration data saved to {calibration_file}")
                break
            else:
                print("\nNeed more positions for calibration")
        elif key == ord('r'):
            calibrator = VideoFloorCalibrator('calibration_matrix.yaml')
            print("Calibration reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()