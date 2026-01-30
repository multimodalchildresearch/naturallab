import numpy as np
import cv2
import yaml
import os

class CameraCalibrator:
    def __init__(self, output_dir='calibration_frames'):
        self.pattern_size = (7, 7)
        self.pattern_points = np.zeros((7*7, 3), np.float32)
        self.pattern_points[:, :2] = np.mgrid[0:7, 0:7].T.reshape(-1, 2)
        
        self.object_points = []
        self.image_points = []
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def find_chessboard(self, frame):
        """Find chessboard corners in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners2, gray
        return False, None, gray
    
    def add_calibration_frame(self, frame):
        """Add frame to calibration set"""
        ret, corners, gray = self.find_chessboard(frame)
        if ret:
            self.object_points.append(self.pattern_points)
            self.image_points.append(corners)
            return True, corners
        return False, None
    
    def calibrate_camera(self, frame_size, output_file='camera_calibration.yaml'):
        """Perform camera calibration"""
        if len(self.object_points) < 10:
            print("Need at least 10 good frames for calibration")
            return False
            
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.object_points, self.image_points, frame_size, None, None
        )
        
        if ret:
            total_error = 0
            for i in range(len(self.object_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.object_points[i], rvecs[i], tvecs[i], mtx, dist
                )
                error = cv2.norm(self.image_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                total_error += error
            
            print(f"\nCalibration complete!")
            print(f"Average reprojection error: {total_error/len(self.object_points)}")
            
            calibration_data = {
                'camera_matrix': mtx.tolist(),
                'dist_coeff': dist.tolist()
            }
            with open(output_file, 'w') as f:
                yaml.dump(calibration_data, f)
            
            return True
        return False
    
def main():
    import os
    import argparse
    
    # Add command line arguments for input/output paths
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--input', type=str, default='calibration/camera_calibration_input',
                        help='Path to input video or directory with calibration images')
    parser.add_argument('--output', type=str, default='calibration',
                        help='Directory to save calibration results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    print("\nCamera Calibration Instructions:")
    print("1. Press SPACE when you see a good chessboard position")
    print("2. Need at least 10 good frames with different orientations")
    print("3. Include views where the chessboard is:")
    print("   - Near the edges of the frame")
    print("   - At different angles")
    print("   - At different distances")
    print("4. Press 'c' to run calibration")
    print("5. Press 'q' to quit")
    print("6. Use arrow keys to move through video:")
    print("   - Right arrow: Forward 1 second")
    print("   - Left arrow: Back 1 second\n")
    
    calibrator = CameraCalibrator(output_dir=os.path.join(args.output, 'calibration_frames'))
    
    # Try different video backends for m2ts
    backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
    cap = None
    
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
    
    for backend in backends:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            break
    
    if not cap or not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        success, corners, gray = calibrator.find_chessboard(frame)
        vis_frame = frame.copy()
        
        if success:
            cv2.drawChessboardCorners(vis_frame, calibrator.pattern_size, corners, True)
            status = "Chessboard detected"
        else:
            status = "No chessboard found"
        
        # Draw status
        cv2.putText(vis_frame,
                   f"Frames captured: {len(calibrator.image_points)}",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        cv2.putText(vis_frame,
                   status,
                   (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0) if success else (0, 0, 255), 2)
        
        # Show frame number
        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time = current_frame / fps
        cv2.putText(vis_frame,
                   f"Time: {current_time:.1f}s",
                   (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        
        cv2.imshow('Camera Calibration', vis_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and success:
            frame_path = f'calibration_frames/frame_{frame_count:03d}.jpg'
            cv2.imwrite(frame_path, frame)
            added, _ = calibrator.add_calibration_frame(frame)
            if added:
                print(f"Saved frame {frame_count}")
                frame_count += 1
        elif key == ord('c'):
            calibration_file_path = os.path.join(args.output, 'camera_calibration.yaml')
            if calibrator.calibrate_camera(gray.shape[::-1], calibration_file_path):
                print(f"Calibration data saved to {calibration_file_path}")
                break
        elif key == 83:  # Right arrow
            # Jump forward 1 second
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame + fps)
        elif key == 81:  # Left arrow
            # Jump backward 1 second
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - fps))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()