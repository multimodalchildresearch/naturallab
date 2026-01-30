#!/usr/bin/env python3
"""
Track People in Video
=====================

A general-purpose script for tracking people in any video file.
Outputs movement trajectories, distances traveled, and interaction metrics.

Use Cases:
- Behavioral research (any population)
- Retail analytics (customer movement)
- Sports analysis (player tracking)
- Security/surveillance analysis
- Occupancy monitoring

Example Usage:
    # Basic tracking
    python track_people_in_video.py --input video.mp4 --output results/
    
    # With floor calibration for real-world distances
    python track_people_in_video.py --input video.mp4 --output results/ \\
        --camera-calib camera.yaml --floor-calib floor.yaml
    
    # With identity labels
    python track_people_in_video.py --input video.mp4 --output results/ \\
        --identities '{"Person A": "person wearing red", "Person B": "person in blue"}'
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    parser = argparse.ArgumentParser(
        description="Track people in video and extract movement metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument("--input", "-i", required=True,
                       help="Input video file or directory of videos")
    parser.add_argument("--output", "-o", required=True,
                       help="Output directory for results")
    
    # Detection settings
    parser.add_argument("--detector", choices=["yolo", "owl"], default="yolo",
                       help="Detection model to use (default: yolo)")
    parser.add_argument("--yolo-model", default="yolo11x.pt",
                       help="Path to YOLO model weights")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    
    # Calibration (optional, for real-world measurements)
    parser.add_argument("--camera-calib", 
                       help="Camera calibration file (YAML)")
    parser.add_argument("--floor-calib",
                       help="Floor calibration file (YAML)")
    parser.add_argument("--correction-factor", type=float, default=1.0,
                       help="Distance correction factor (default: 1.0)")
    
    # Identity matching (optional)
    parser.add_argument("--identities", type=str,
                       help='JSON dict of identity descriptions, e.g., \'{"Coach": "person in red", "Player": "person in white"}\'')
    parser.add_argument("--identity-file",
                       help="JSON file with identity descriptions")
    
    # Tracking settings
    parser.add_argument("--max-age", type=int, default=30,
                       help="Max frames to keep track alive without detection (default: 30)")
    parser.add_argument("--min-hits", type=int, default=3,
                       help="Min detections before track is confirmed (default: 3)")
    
    # Output options
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualization video")
    parser.add_argument("--save-frames", action="store_true",
                       help="Save sample frames with annotations")
    parser.add_argument("--frame-interval", type=int, default=100,
                       help="Interval for saving sample frames (default: 100)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum frames to process (for testing)")
    
    # Processing options
    parser.add_argument("--device", default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for detection (default: 1)")
    
    args = parser.parse_args()
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input not found: {args.input}")
        return 1
    
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load identity descriptions if provided
    identities = None
    if args.identities:
        identities = json.loads(args.identities)
    elif args.identity_file:
        with open(args.identity_file) as f:
            identities = json.load(f)
    
    print("=" * 60)
    print("NaturalLab - People Tracking Pipeline")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Detector: {args.detector}")
    print(f"Confidence threshold: {args.confidence}")
    if args.camera_calib:
        print(f"Camera calibration: {args.camera_calib}")
    if args.floor_calib:
        print(f"Floor calibration: {args.floor_calib}")
    if identities:
        print(f"Identities: {list(identities.keys())}")
    print()
    
    # Import tracking modules
    try:
        from naturallab.spatial_tracking.detection.yolo_detector import YOLODetector
        from naturallab.spatial_tracking.tracking.base_tracker import BaseTracker
        from naturallab.spatial_tracking.movement.floor_tracker import FloorTracker
        
        if identities:
            from naturallab.spatial_tracking.tracking.track_identity_matching import TrackIdentityMatcher
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure naturallab is installed: pip install -e .")
        return 1
    
    # Process videos
    if input_path.is_file():
        videos = [input_path]
    else:
        videos = list(input_path.glob("*.mp4")) + list(input_path.glob("*.avi"))
    
    print(f"Found {len(videos)} video(s) to process")
    
    for video_path in videos:
        print(f"\nProcessing: {video_path.name}")
        print("-" * 40)
        
        # Create output subdirectory for this video
        video_output = output_path / video_path.stem
        video_output.mkdir(exist_ok=True)
        
        # Initialize detector
        if args.detector == "yolo":
            detector = YOLODetector(
                model_path=args.yolo_model,
                confidence_threshold=args.confidence,
                device=args.device
            )
        else:
            from naturallab.spatial_tracking.detection.owl_detector import OWLDetector
            detector = OWLDetector(
                confidence_threshold=args.confidence,
                device=args.device
            )
        
        # Initialize tracker
        tracker = BaseTracker(
            max_age=args.max_age,
            min_hits=args.min_hits
        )
        
        # Initialize floor tracker if calibration available
        floor_tracker = None
        if args.camera_calib and args.floor_calib:
            floor_tracker = FloorTracker(
                camera_calib_path=args.camera_calib,
                floor_calib_path=args.floor_calib,
                correction_factor=args.correction_factor
            )
        
        # Process video
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"  Error: Could not open video")
            continue
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        
        # Storage for results
        all_tracks = []
        frame_idx = 0
        
        # Progress tracking
        from tqdm import tqdm
        max_frames = args.max_frames or total_frames
        
        for _ in tqdm(range(min(max_frames, total_frames)), desc="  Processing"):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people
            detections = detector.detect(frame)
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Store track data
            for track in tracks:
                track_data = {
                    "frame": frame_idx,
                    "track_id": track.track_id,
                    "x1": track.bbox[0],
                    "y1": track.bbox[1],
                    "x2": track.bbox[2],
                    "y2": track.bbox[3],
                    "confidence": track.confidence
                }
                
                # Add floor position if available
                if floor_tracker:
                    floor_pos = floor_tracker.project_to_floor(track.bbox)
                    if floor_pos is not None:
                        track_data["floor_x"] = floor_pos[0]
                        track_data["floor_y"] = floor_pos[1]
                        track_data["floor_z"] = floor_pos[2]
                
                all_tracks.append(track_data)
            
            # Save sample frames if requested
            if args.save_frames and frame_idx % args.frame_interval == 0:
                frame_path = video_output / "frames" / f"frame_{frame_idx:06d}.jpg"
                frame_path.parent.mkdir(exist_ok=True)
                # Draw tracks on frame
                annotated = frame.copy()
                for track in tracks:
                    x1, y1, x2, y2 = map(int, track.bbox)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"ID: {track.track_id}", 
                               (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(frame_path), annotated)
            
            frame_idx += 1
        
        cap.release()
        
        # Save results
        import pandas as pd
        
        if all_tracks:
            df = pd.DataFrame(all_tracks)
            df.to_csv(video_output / "tracks.csv", index=False)
            print(f"  Saved {len(df)} track records to tracks.csv")
            
            # Compute statistics per track
            stats = []
            for track_id in df["track_id"].unique():
                track_df = df[df["track_id"] == track_id]
                stat = {
                    "track_id": track_id,
                    "first_frame": track_df["frame"].min(),
                    "last_frame": track_df["frame"].max(),
                    "duration_frames": len(track_df),
                    "duration_seconds": len(track_df) / fps if fps > 0 else 0
                }
                
                # Add distance if floor tracking available
                if "floor_x" in track_df.columns:
                    import numpy as np
                    positions = track_df[["floor_x", "floor_y"]].values
                    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
                    stat["total_distance_mm"] = distances.sum()
                    stat["total_distance_m"] = distances.sum() / 1000
                
                stats.append(stat)
            
            stats_df = pd.DataFrame(stats)
            stats_df.to_csv(video_output / "track_statistics.csv", index=False)
            print(f"  Saved statistics for {len(stats_df)} tracks")
            
            # Identity matching if requested
            if identities:
                print("  Performing identity matching...")
                matcher = TrackIdentityMatcher(device=args.device)
                
                # Extract galleries
                galleries = matcher.extract_track_galleries(
                    video_path=str(video_path),
                    tracks_csv=str(video_output / "tracks.csv"),
                    output_dir=str(video_output / "galleries"),
                    frames_per_track=5
                )
                
                # Compute embeddings and match
                if galleries:
                    matcher.compute_track_embeddings(galleries=galleries)
                    matches = matcher.match_identities(identities)
                    
                    # Save matches
                    with open(video_output / "identity_matches.json", "w") as f:
                        json.dump({k: {str(tk): float(v) for tk, v in vals.items()} 
                                  for k, vals in matches.items()}, f, indent=2)
                    print(f"  Identity matches saved")
        
        print(f"  Results saved to: {video_output}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
