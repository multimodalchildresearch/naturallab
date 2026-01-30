#!/usr/bin/env python
"""
Neon Streams Combiner
---------------------
This script combines multiple video streams (Neon with gaze, Neon without gaze, depth visualization, 
RealSense color) into a single side-by-side video with audio.

Usage:
    python combine_neon_streams.py --dir extracted_data
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
import subprocess
from tempfile import NamedTemporaryFile

def get_video_properties(video_path):
    """Get basic properties of a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    props = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return props


def list_and_select_files(data_dir):
    """List video files and have user select them"""
    # Find and list video files
    video_files = {}
    
    # Find gaze visualization video
    gaze_videos = [f for f in os.listdir(data_dir) if f.endswith('.mp4') and ('_with_gaze' in f or '_with_' in f and 'gaze' in f)]
    print("\nAvailable gaze visualization videos:")
    for i, filename in enumerate(gaze_videos):
        print(f"  {i+1}. {filename}")
        video_files[f'gaze_{i+1}'] = filename
    
    if not gaze_videos:
        print("No gaze visualization videos found. Please run the gaze visualizer first.")
        return None
    
    # Select gaze video
    gaze_choice = int(input("\nSelect a gaze video by number: "))
    if 1 <= gaze_choice <= len(gaze_videos):
        selected_gaze = gaze_videos[gaze_choice-1]
    else:
        print("Invalid selection.")
        return None
    
    # Find Neon videos without gaze
    # Look for videos that appear to be Neon videos but don't have gaze visualization
    neon_videos = [f for f in os.listdir(data_dir) if f.endswith('.mp4') and 
                  ('neon' in f.lower() or 'eye' in f.lower()) and 
                  not any(g in f.lower() for g in ['_with_gaze', 'gaze', 'depth', 'color'])]
    
    print("\nAvailable Neon videos without gaze:")
    for i, filename in enumerate(neon_videos):
        print(f"  {i+1}. {filename}")
        video_files[f'neon_{i+1}'] = filename
    
    # Select Neon video without gaze
    neon_choice = int(input("\nSelect a Neon video without gaze by number: "))
    if 1 <= neon_choice <= len(neon_videos):
        selected_neon = neon_videos[neon_choice-1]
    else:
        print("Invalid selection.")
        return None
    
    # Find depth visualization video
    depth_videos = [f for f in os.listdir(data_dir) if 'depth' in f.lower() and f.endswith('.mp4')]
    print("\nAvailable depth visualization videos:")
    for i, filename in enumerate(depth_videos):
        print(f"  {i+1}. {filename}")
        video_files[f'depth_{i+1}'] = filename
    
    # Select depth video
    depth_choice = int(input("\nSelect a depth video by number: "))
    if 1 <= depth_choice <= len(depth_videos):
        selected_depth = depth_videos[depth_choice-1]
    else:
        print("Invalid selection.")
        return None
    
    # Find RealSense color video
    color_videos = [f for f in os.listdir(data_dir) if 'color' in f.lower() and f.endswith('.mp4')]
    # If none found, look for any other videos
    if not color_videos:
        color_videos = [f for f in os.listdir(data_dir) if f.endswith('.mp4') 
                        and f not in gaze_videos 
                        and f not in depth_videos
                        and f not in neon_videos]
    
    print("\nAvailable color/regular videos:")
    for i, filename in enumerate(color_videos):
        print(f"  {i+1}. {filename}")
        video_files[f'color_{i+1}'] = filename
    
    # Select color video
    color_choice = int(input("\nSelect a color video by number: "))
    if 1 <= color_choice <= len(color_videos):
        selected_color = color_videos[color_choice-1]
    else:
        print("Invalid selection.")
        return None
    
    # Find audio files (if any)
    audio_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    selected_audio = None
    
    if audio_files:
        print("\nAvailable audio files:")
        for i, filename in enumerate(audio_files):
            print(f"  {i+1}. {filename}")
            video_files[f'audio_{i+1}'] = filename
        
        # Select audio file (optional)
        audio_choice = input("\nSelect an audio file by number (or press Enter to use video audio): ")
        if audio_choice.strip():
            audio_choice = int(audio_choice)
            if 1 <= audio_choice <= len(audio_files):
                selected_audio = audio_files[audio_choice-1]
    
    return {
        'gaze': selected_gaze,
        'neon': selected_neon,  # New Neon video without gaze
        'depth': selected_depth,
        'color': selected_color,
        'audio': selected_audio
    }


def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def combine_videos(data_dir, selected_files, output_file):
    """Combine multiple videos into a side-by-side layout with audio"""
    gaze_path = os.path.join(data_dir, selected_files['gaze'])
    neon_path = os.path.join(data_dir, selected_files['neon'])  # New Neon video path
    depth_path = os.path.join(data_dir, selected_files['depth'])
    color_path = os.path.join(data_dir, selected_files['color'])
    
    # Get video properties
    gaze_props = get_video_properties(gaze_path)
    neon_props = get_video_properties(neon_path)  # New Neon video properties
    depth_props = get_video_properties(depth_path)
    color_props = get_video_properties(color_path)
    
    if not all([gaze_props, neon_props, depth_props, color_props]):
        print("Error: Could not read all video files.")
        return False
    
    print("\nVideo properties:")
    print(f"Gaze video: {gaze_props['width']}x{gaze_props['height']}, {gaze_props['fps']} FPS")
    print(f"Neon video: {neon_props['width']}x{neon_props['height']}, {neon_props['fps']} FPS")  # New line
    print(f"Depth video: {depth_props['width']}x{depth_props['height']}, {depth_props['fps']} FPS")
    print(f"Color video: {color_props['width']}x{color_props['height']}, {color_props['fps']} FPS")
    
    # Determine target size for each video pane
    # Make all videos the same height
    target_height = min(gaze_props['height'], neon_props['height'], depth_props['height'], color_props['height'])
    
    # Calculate scaled widths while maintaining aspect ratios
    gaze_width = int(gaze_props['width'] * (target_height / gaze_props['height']))
    neon_width = int(neon_props['width'] * (target_height / neon_props['height']))  # New width calculation
    depth_width = int(depth_props['width'] * (target_height / depth_props['height']))
    color_width = int(color_props['width'] * (target_height / color_props['height']))
    
    # Total output dimensions
    output_width = gaze_width + neon_width + depth_width + color_width  # Updated width
    output_height = target_height
    
    # Create a temporary file for the video without audio
    temp_output = NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    # Choose the fps that makes the most sense (use the highest for smoothness)
    output_fps = max(gaze_props['fps'], neon_props['fps'], depth_props['fps'], color_props['fps'])  # Added neon
    
    # Open video files
    gaze_cap = cv2.VideoCapture(gaze_path)
    neon_cap = cv2.VideoCapture(neon_path)  # New video capture
    depth_cap = cv2.VideoCapture(depth_path)
    color_cap = cv2.VideoCapture(color_path)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, output_fps, (output_width, output_height))
    
    # Determine number of frames to process
    # We'll use the shortest video to avoid blank frames
    total_frames = min(
        gaze_props['frame_count'], 
        neon_props['frame_count'],  # Added frame count
        depth_props['frame_count'], 
        color_props['frame_count']
    )
    
    # Process frames
    print(f"\nCombining videos ({total_frames} frames)...")
    progress_step = max(1, total_frames // 20)  # Show progress every 5%
    
    for frame_idx in range(int(total_frames)):
        # Read frames from each video
        ret_gaze, gaze_frame = gaze_cap.read()
        ret_neon, neon_frame = neon_cap.read()  # New frame read
        ret_depth, depth_frame = depth_cap.read()
        ret_color, color_frame = color_cap.read()
        
        # Check if any read failed
        if not all([ret_gaze, ret_neon, ret_depth, ret_color]):  # Added neon check
            print(f"\nReached end of at least one video at frame {frame_idx}")
            break
        
        # Resize frames to target dimensions
        gaze_frame = cv2.resize(gaze_frame, (gaze_width, target_height))
        neon_frame = cv2.resize(neon_frame, (neon_width, target_height))  # New resize
        depth_frame = cv2.resize(depth_frame, (depth_width, target_height))
        color_frame = cv2.resize(color_frame, (color_width, target_height))
        
        # Create empty combined frame
        combined_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Add labels to each frame
        cv2.putText(gaze_frame, "Neon with Gaze", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(neon_frame, "Neon without Gaze", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)  # New label
        cv2.putText(depth_frame, "Depth Visualization", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(color_frame, "Color Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Insert frames into combined image
        combined_frame[:, 0:gaze_width] = gaze_frame
        combined_frame[:, gaze_width:gaze_width+neon_width] = neon_frame  # Insert new frame
        combined_frame[:, gaze_width+neon_width:gaze_width+neon_width+depth_width] = depth_frame  # Updated position
        combined_frame[:, gaze_width+neon_width+depth_width:] = color_frame  # Updated position
        
        # Add frame counter
        cv2.putText(
            combined_frame,
            f"Frame: {frame_idx}",
            (10, output_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Write combined frame
        out.write(combined_frame)
        
        # Show progress
        if frame_idx % progress_step == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%)")
    
    # Release resources
    gaze_cap.release()
    neon_cap.release()  # Release new capture
    depth_cap.release()
    color_cap.release()
    out.release()
    
    print("Video combination complete. Now adding audio...")
    
    # Add audio to the video using FFmpeg
    if not check_ffmpeg():
        print("Warning: FFmpeg not found. Saving without audio.")
        # Just rename the temp file
        import shutil
        shutil.move(temp_output, output_file)
        return True
    
    # Determine audio source
    audio_source = None
    if selected_files['audio']:
        audio_source = os.path.join(data_dir, selected_files['audio'])
    else:
        # Extract audio from one of the videos
        print("Extracting audio from gaze video...")
        audio_source = NamedTemporaryFile(suffix='.wav', delete=False).name
        subprocess.run([
            'ffmpeg', '-y', '-i', gaze_path, '-q:a', '0', '-map', 'a', audio_source
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Add audio to output video
    try:
        subprocess.run([
            'ffmpeg', '-y', 
            '-i', temp_output, 
            '-i', audio_source, 
            '-c:v', 'copy', 
            '-c:a', 'aac', 
            '-map', '0:v:0', 
            '-map', '1:a:0', 
            '-shortest',
            output_file
        ], check=True)
        print(f"Successfully saved final video with audio to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error adding audio: {e}")
        # If failed, at least keep the video without audio
        import shutil
        shutil.move(temp_output, output_file)
        print(f"Saved video without audio to: {output_file}")
    
    # Clean up temporary files
    if os.path.exists(temp_output):
        os.remove(temp_output)
    
    if not selected_files['audio'] and os.path.exists(audio_source):
        os.remove(audio_source)
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Neon Streams Combiner")
    parser.add_argument("--dir", type=str, required=True, 
                        help="Directory with extracted Neon data")
    parser.add_argument("--output", type=str,
                        help="Output file path (default: combined_streams.mp4 in the data directory)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.dir):
        print(f"Error: Directory not found: {args.dir}")
        return 1
    
    # Set default output file if not specified
    if not args.output:
        args.output = os.path.join(args.dir, "combined_streams.mp4")
    
    # Let user select files
    selected_files = list_and_select_files(args.dir)
    if not selected_files:
        return 1
    
    # Check if output file already exists
    if os.path.exists(args.output):
        print(f"Warning: Output file already exists: {args.output}")
        choice = input("Overwrite? (y/n): ").strip().lower()
        if choice != 'y':
            print("Aborting.")
            return 0
    
    # Process the files
    start_time = time.time()
    
    success = combine_videos(args.dir, selected_files, args.output)
    
    end_time = time.time()
    if success:
        print(f"\nProcessing completed in {end_time - start_time:.2f} seconds.")
    else:
        print("\nVideo combination failed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())