#!/usr/bin/env python
"""
Simple Neon Gaze Visualizer
--------------------------
This script generates visualizations of gaze data overlaid on Neon videos.
Uses frame indices in the gaze data to match with video frames.

Usage:
    python neon_gaze_visualizer.py --dir extracted_data
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import cv2
import time

def list_files(data_dir):
    """List video and potential gaze data files and have user select them"""
    # Find all video files
    video_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".mp4") and "_with_" not in filename:
            video_files.append(filename)
    
    if not video_files:
        print("No video files found in directory.")
        return None, None
    
    # List video files
    print("\nAvailable video files:")
    for i, filename in enumerate(video_files):
        print(f"  {i+1}. {filename}")
    
    # User selects video file
    while True:
        try:
            choice = int(input("\nSelect a video file by number: "))
            if 1 <= choice <= len(video_files):
                selected_video = video_files[choice-1]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Find all CSV files
    csv_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv") and not filename.endswith("_timestamps.csv"):
            csv_files.append(filename)
    
    if not csv_files:
        print("No CSV files found in directory.")
        return selected_video, None
    
    # List CSV files
    print("\nAvailable CSV files:")
    for i, filename in enumerate(csv_files):
        print(f"  {i+1}. {filename}")
    
    # User selects gaze file
    while True:
        try:
            choice = int(input("\nSelect a gaze data file by number: "))
            if 1 <= choice <= len(csv_files):
                selected_gaze = csv_files[choice-1]
                break
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    return selected_video, selected_gaze


def generate_gaze_visualization(video_path, gaze_path, output_path, circle_radius=10, 
                               filled=False, thickness=2, circle_color=(0, 0, 255)):
    """
    Generate visualization of gaze data overlaid on video using frame indices.
    
    Args:
        video_path: Path to video file
        gaze_path: Path to gaze data CSV
        output_path: Path for output video
        circle_radius: Radius of circle for gaze point
        filled: Whether to use filled circles (True) or outline circles (False)
        thickness: Thickness of circle outline (ignored if filled=True)
        circle_color: Color in BGR format
    """
    print(f"\nGenerating visualization...")
    print(f"Video: {video_path}")
    print(f"Gaze data: {gaze_path}")
    print(f"Output: {output_path}")
    print(f"Circle style: {'Filled' if filled else 'Outline'}")
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
    
    # Load gaze data
    try:
        gaze_df = pd.read_csv(gaze_path)
        print(f"Loaded gaze data: {len(gaze_df)} entries")
    except Exception as e:
        print(f"Error loading gaze data: {e}")
        cap.release()
        return
    
    # Identify frame index and coordinate columns
    frame_col = None
    x_col, y_col = None, None
    
    # Check for frame_index column
    for col in gaze_df.columns:
        if 'frame' in col.lower() and 'index' in col.lower():
            frame_col = col
            print(f"Found frame index column: {frame_col}")
            break
    
    if frame_col is None:
        print("Warning: Could not find a frame index column in the gaze data.")
        print("Available columns:", gaze_df.columns.tolist())
        frame_col = input("Enter the name of the frame index column (or press Enter to use row index): ")
        if not frame_col:
            print("Using row index as frame index.")
            gaze_df['frame_index'] = range(len(gaze_df))
            frame_col = 'frame_index'
    
    # Check for x,y coordinate columns
    potential_x_cols = ['x', 'gaze_x', 'eye_x']
    potential_y_cols = ['y', 'gaze_y', 'eye_y']
    
    for x_candidate in potential_x_cols:
        if x_candidate in gaze_df.columns:
            x_col = x_candidate
            break
    
    for y_candidate in potential_y_cols:
        if y_candidate in gaze_df.columns:
            y_col = y_candidate
            break
    
    if x_col is None or y_col is None:
        print("Warning: Could not identify coordinate columns.")
        print("Available columns:", gaze_df.columns.tolist())
        
        x_col = input("Enter the name of the X coordinate column: ")
        y_col = input("Enter the name of the Y coordinate column: ")
        
        if not x_col in gaze_df.columns or not y_col in gaze_df.columns:
            print(f"Error: One or both of the specified columns don't exist.")
            cap.release()
            return
    
    print(f"Using columns: {frame_col} for frame index, {x_col} and {y_col} for coordinates")
    
    # Create lookup dictionary for gaze data by frame index
    gaze_lookup = {}
    for _, row in gaze_df.iterrows():
        frame_idx = row[frame_col]
        
        # Convert to int if possible
        if pd.notna(frame_idx):
            try:
                frame_idx = int(frame_idx)
                gaze_lookup[frame_idx] = (row[x_col], row[y_col])
            except ValueError:
                pass
    
    print(f"Created lookup table for {len(gaze_lookup)} frames with gaze data")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    print("Processing frames...")
    frame_idx = 0
    progress_step = max(1, frame_count // 20)  # Show progress every 5%
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if we have gaze data for this frame
        if frame_idx in gaze_lookup:
            x, y = gaze_lookup[frame_idx]
            
            # Check if coordinates are valid
            if pd.notna(x) and pd.notna(y):
                # Ensure coordinates are within frame boundaries
                x = int(max(0, min(width-1, x)))
                y = int(max(0, min(height-1, y)))
                
                # Draw circle at gaze position based on style (filled or outline)
                if filled:
                    # Filled circle (red) with black border
                    cv2.circle(frame, (x, y), circle_radius, circle_color, -1)
                    cv2.circle(frame, (x, y), circle_radius, (0, 0, 0), 1)
                else:
                    # Outline circle with specified thickness
                    cv2.circle(frame, (x, y), circle_radius, circle_color, thickness)
        
        # Add frame counter for debugging
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Write the frame
        out.write(frame)
        
        # Show progress
        if frame_idx % progress_step == 0:
            print(f"Progress: {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Completed! Output saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Simple Neon Gaze Visualizer")
    parser.add_argument("--dir", type=str, required=True, 
                        help="Directory with extracted Neon data")
    parser.add_argument("--radius", type=int, default=10,
                        help="Radius of the circle used for gaze visualization (default: 10)")
    parser.add_argument("--filled", action="store_true",
                        help="Use filled circles instead of outline circles")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Thickness of circle outlines (default: 2, ignored if --filled is used)")
    parser.add_argument("--video", type=str,
                        help="Specific video filename (in data directory)")
    parser.add_argument("--gaze", type=str,
                        help="Specific gaze data filename (in data directory)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.dir):
        print(f"Error: Directory not found: {args.dir}")
        return 1
    
    # Get video and gaze files
    video_file, gaze_file = None, None
    
    if args.video and args.gaze:
        # Use specified files
        video_path = os.path.join(args.dir, args.video)
        gaze_path = os.path.join(args.dir, args.gaze)
        
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}")
            return 1
        
        if not os.path.exists(gaze_path):
            print(f"Error: Gaze file not found: {gaze_path}")
            return 1
        
        video_file = args.video
        gaze_file = args.gaze
    else:
        # Let user select files
        video_file, gaze_file = list_files(args.dir)
        
        if not video_file or not gaze_file:
            print("File selection failed.")
            return 1
    
    # Generate output filename
    output_file = os.path.join(args.dir, f"{os.path.splitext(video_file)[0]}_with_gaze.mp4")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Warning: Output file already exists: {output_file}")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            print("Aborting.")
            return 0
    
    # Process the files
    start_time = time.time()
    
    generate_gaze_visualization(
        os.path.join(args.dir, video_file),
        os.path.join(args.dir, gaze_file),
        output_file,
        circle_radius=args.radius,
        filled=args.filled,
        thickness=args.thickness
    )
    
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
