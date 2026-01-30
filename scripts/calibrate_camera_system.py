#!/usr/bin/env python3
"""
Camera System Calibration
=========================

Calibrate cameras for accurate real-world position measurements.
Supports both single cameras and multi-camera setups.

This calibration enables:
- Converting pixel coordinates to real-world positions
- Measuring actual distances traveled (in meters)
- Projecting detections onto a floor plane

Use Cases:
- Research labs with ceiling-mounted cameras
- Retail spaces for customer tracking
- Sports arenas for player position analysis
- Any space where real-world measurements are needed

Example Usage:
    # Step 1: Camera intrinsic calibration
    python calibrate_camera_system.py intrinsic \\
        --video chessboard_video.mp4 \\
        --output camera_intrinsics.yaml
    
    # Step 2: Floor plane calibration
    python calibrate_camera_system.py floor \\
        --video floor_chessboard_video.mp4 \\
        --camera-calib camera_intrinsics.yaml \\
        --output floor_calibration.yaml
    
    # Step 3: Verify calibration accuracy
    python calibrate_camera_system.py verify \\
        --video test_video.mp4 \\
        --camera-calib camera_intrinsics.yaml \\
        --floor-calib floor_calibration.yaml \\
        --known-distance 1000  # 1 meter reference
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def calibrate_intrinsic(args):
    """Perform camera intrinsic calibration."""
    import cv2
    import numpy as np
    import yaml
    
    video_path = Path(args.video)
    output_path = Path(args.output)
    
    print("=" * 60)
    print("Camera Intrinsic Calibration")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Chessboard size: {args.cols}x{args.rows}")
    print(f"Square size: {args.square_size}mm")
    print()
    
    # Chessboard parameters
    board_size = (args.cols - 1, args.rows - 1)  # Internal corners
    
    # Prepare object points (0,0,0), (1,0,0), (2,0,0), ...
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    
    # Arrays to store object points and image points
    object_points = []  # 3D points in real world space
    image_points = []   # 2D points in image plane
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 1
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video resolution: {frame_width}x{frame_height}")
    print(f"Total frames: {total_frames}")
    print()
    
    # Process frames
    print("Detecting chessboard corners...")
    print("(Processing every 30th frame to get diverse views)")
    
    frame_idx = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every Nth frame
        if frame_idx % 30 != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            object_points.append(objp)
            image_points.append(corners)
            detected_count += 1
            
            print(f"  Frame {frame_idx}: Chessboard detected ({detected_count} total)")
            
            # Save sample frame
            if args.save_frames:
                sample_dir = output_path.parent / "calibration_samples"
                sample_dir.mkdir(exist_ok=True)
                vis_frame = frame.copy()
                cv2.drawChessboardCorners(vis_frame, board_size, corners, ret)
                cv2.imwrite(str(sample_dir / f"frame_{frame_idx:06d}.jpg"), vis_frame)
        
        frame_idx += 1
        
        # Stop after enough detections
        if detected_count >= args.max_frames:
            break
    
    cap.release()
    
    if detected_count < 5:
        print(f"\nError: Only {detected_count} chessboard detections. Need at least 5.")
        return 1
    
    print(f"\nCalibrating with {detected_count} chessboard views...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, (frame_width, frame_height), None, None
    )
    
    if not ret:
        print("Error: Calibration failed")
        return 1
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(object_points)):
        imgpoints2, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i],
                                          camera_matrix, dist_coeffs)
        error = cv2.norm(image_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(object_points)
    
    print(f"\nCalibration complete!")
    print(f"Reprojection error: {mean_error:.4f} pixels")
    
    # Save calibration
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    calib_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "image_width": frame_width,
        "image_height": frame_height,
        "reprojection_error": float(mean_error),
        "num_calibration_images": detected_count
    }
    
    with open(output_path, "w") as f:
        yaml.dump(calib_data, f, default_flow_style=False)
    
    print(f"\nCalibration saved to: {output_path}")
    
    # Print camera matrix
    print("\nCamera Matrix:")
    print(f"  fx = {camera_matrix[0, 0]:.2f}")
    print(f"  fy = {camera_matrix[1, 1]:.2f}")
    print(f"  cx = {camera_matrix[0, 2]:.2f}")
    print(f"  cy = {camera_matrix[1, 2]:.2f}")
    
    return 0


def calibrate_floor(args):
    """Perform floor plane calibration."""
    import cv2
    import numpy as np
    import yaml
    
    video_path = Path(args.video)
    camera_calib_path = Path(args.camera_calib)
    output_path = Path(args.output)
    
    print("=" * 60)
    print("Floor Plane Calibration")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Camera calibration: {camera_calib_path}")
    print()
    
    # Load camera calibration
    with open(camera_calib_path) as f:
        camera_calib = yaml.safe_load(f)
    
    camera_matrix = np.array(camera_calib["camera_matrix"])
    dist_coeffs = np.array(camera_calib["dist_coeffs"])
    
    # Chessboard parameters
    board_size = (args.cols - 1, args.rows - 1)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    
    # Collect floor points from multiple chessboard positions
    all_floor_points = []
    
    cap = cv2.VideoCapture(str(video_path))
    
    print("Detecting chessboard on floor...")
    
    frame_idx = 0
    detected_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % 30 != 0:
            frame_idx += 1
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Solve PnP to get chessboard pose
            ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            
            if ret:
                # Transform object points to camera coordinates
                R, _ = cv2.Rodrigues(rvec)
                floor_points_cam = (R @ objp.T + tvec).T
                all_floor_points.extend(floor_points_cam)
                
                detected_count += 1
                print(f"  Frame {frame_idx}: Chessboard detected ({detected_count} positions)")
        
        frame_idx += 1
        
        if detected_count >= 5:
            break
    
    cap.release()
    
    if detected_count < 3:
        print(f"\nError: Only {detected_count} floor positions. Need at least 3.")
        return 1
    
    all_floor_points = np.array(all_floor_points)
    
    # Fit plane using SVD
    centroid = all_floor_points.mean(axis=0)
    centered = all_floor_points - centroid
    
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Normal to the plane
    
    # Ensure normal points upward (positive Z in camera frame)
    if normal[2] < 0:
        normal = -normal
    
    # Plane equation: ax + by + cz + d = 0
    d = -np.dot(normal, centroid)
    
    print(f"\nFloor plane equation: {normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + {d:.4f} = 0")
    
    # Save floor calibration
    floor_calib = {
        "plane_normal": normal.tolist(),
        "plane_d": float(d),
        "centroid": centroid.tolist(),
        "num_positions": detected_count
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        yaml.dump(floor_calib, f, default_flow_style=False)
    
    print(f"\nFloor calibration saved to: {output_path}")
    
    return 0


def verify_calibration(args):
    """Verify calibration accuracy."""
    import cv2
    import numpy as np
    import yaml
    
    print("=" * 60)
    print("Calibration Verification")
    print("=" * 60)
    
    # Load calibrations
    with open(args.camera_calib) as f:
        camera_calib = yaml.safe_load(f)
    
    with open(args.floor_calib) as f:
        floor_calib = yaml.safe_load(f)
    
    camera_matrix = np.array(camera_calib["camera_matrix"])
    dist_coeffs = np.array(camera_calib["dist_coeffs"])
    plane_normal = np.array(floor_calib["plane_normal"])
    plane_d = floor_calib["plane_d"]
    
    print(f"Camera calibration loaded")
    print(f"Floor calibration loaded")
    print(f"Known reference distance: {args.known_distance}mm")
    print()
    
    cap = cv2.VideoCapture(str(args.video))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read video frame")
        return 1
    
    # Interactive point selection
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Click two points on floor", frame)
            
            if len(points) == 2:
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow("Click two points on floor", frame)
    
    cv2.namedWindow("Click two points on floor")
    cv2.setMouseCallback("Click two points on floor", mouse_callback)
    cv2.imshow("Click two points on floor", frame)
    
    print("Click two points on the floor with known distance between them.")
    print("Press any key when done.")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) < 2:
        print("Error: Need two points")
        return 1
    
    # Project points to floor
    def project_to_floor(pixel_point):
        x, y = pixel_point
        
        # Undistort point
        pts = np.array([[[x, y]]], dtype=np.float32)
        pts_undist = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
        x_u, y_u = pts_undist[0, 0]
        
        # Create ray from camera through pixel
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        ray_dir = np.array([(x_u - cx) / fx, (y_u - cy) / fy, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # Intersect ray with floor plane
        denom = np.dot(plane_normal, ray_dir)
        if abs(denom) < 1e-6:
            return None
        
        t = -(plane_d) / denom
        floor_point = t * ray_dir
        
        return floor_point
    
    p1 = project_to_floor(points[0])
    p2 = project_to_floor(points[1])
    
    if p1 is None or p2 is None:
        print("Error: Could not project points to floor")
        return 1
    
    measured_distance = np.linalg.norm(p2 - p1)
    error = abs(measured_distance - args.known_distance)
    error_percent = (error / args.known_distance) * 100
    
    print(f"\nResults:")
    print(f"  Known distance: {args.known_distance:.1f}mm")
    print(f"  Measured distance: {measured_distance:.1f}mm")
    print(f"  Error: {error:.1f}mm ({error_percent:.1f}%)")
    
    if error_percent < 5:
        print("\n✓ Calibration accuracy is GOOD (< 5% error)")
    elif error_percent < 10:
        print("\n⚠ Calibration accuracy is ACCEPTABLE (5-10% error)")
    else:
        print("\n✗ Calibration accuracy is POOR (> 10% error)")
        print("  Consider recalibrating with more chessboard positions")
    
    # Suggest correction factor
    correction_factor = args.known_distance / measured_distance
    print(f"\nSuggested correction factor: {correction_factor:.4f}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Camera system calibration for real-world measurements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Intrinsic calibration
    intrinsic_parser = subparsers.add_parser("intrinsic",
                                             help="Camera intrinsic calibration")
    intrinsic_parser.add_argument("--video", "-v", required=True,
                                 help="Video of chessboard from various angles")
    intrinsic_parser.add_argument("--output", "-o", required=True,
                                 help="Output calibration file (YAML)")
    intrinsic_parser.add_argument("--cols", type=int, default=7,
                                 help="Chessboard columns (default: 7)")
    intrinsic_parser.add_argument("--rows", type=int, default=7,
                                 help="Chessboard rows (default: 7)")
    intrinsic_parser.add_argument("--square-size", type=float, default=25,
                                 help="Square size in mm (default: 25)")
    intrinsic_parser.add_argument("--max-frames", type=int, default=20,
                                 help="Max calibration frames (default: 20)")
    intrinsic_parser.add_argument("--save-frames", action="store_true",
                                 help="Save detected chessboard frames")
    
    # Floor calibration
    floor_parser = subparsers.add_parser("floor",
                                         help="Floor plane calibration")
    floor_parser.add_argument("--video", "-v", required=True,
                             help="Video of chessboard on floor")
    floor_parser.add_argument("--camera-calib", "-c", required=True,
                             help="Camera calibration file")
    floor_parser.add_argument("--output", "-o", required=True,
                             help="Output floor calibration file (YAML)")
    floor_parser.add_argument("--cols", type=int, default=7,
                             help="Chessboard columns (default: 7)")
    floor_parser.add_argument("--rows", type=int, default=7,
                             help="Chessboard rows (default: 7)")
    floor_parser.add_argument("--square-size", type=float, default=172,
                             help="Square size in mm (default: 172)")
    
    # Verification
    verify_parser = subparsers.add_parser("verify",
                                          help="Verify calibration accuracy")
    verify_parser.add_argument("--video", "-v", required=True,
                              help="Video for verification")
    verify_parser.add_argument("--camera-calib", "-c", required=True,
                              help="Camera calibration file")
    verify_parser.add_argument("--floor-calib", "-f", required=True,
                              help="Floor calibration file")
    verify_parser.add_argument("--known-distance", "-d", type=float, required=True,
                              help="Known reference distance in mm")
    
    args = parser.parse_args()
    
    if args.command == "intrinsic":
        return calibrate_intrinsic(args)
    elif args.command == "floor":
        return calibrate_floor(args)
    elif args.command == "verify":
        return verify_calibration(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
