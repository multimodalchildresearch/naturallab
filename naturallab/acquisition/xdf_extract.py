#!/usr/bin/env python
"""
XDF Extractor (Fixed)
-------------------
This script extracts all data streams from an XDF file into individual files.
Fixed version with better handling of different data formats.

Usage:
    python xdf_extract_fixed.py --file recording.xdf --outdir extracted_data
"""

import os
import sys
import argparse
import base64
import json
import time
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

# Ensure necessary packages are installed
try:
    import pyxdf
except ImportError:
    import subprocess
    print("Installing pyxdf...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyxdf"])
    import pyxdf

try:
    from tqdm import tqdm
except ImportError:
    import subprocess
    print("Installing tqdm for progress bars...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])
    from tqdm import tqdm

def extract_video_stream(stream, output_dir, name=None):
    """Extract a video stream from XDF to MP4 without saving individual frames"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting video stream: {stream_name}")
    
    # Output MP4 file
    output_file = os.path.join(output_dir, f"{stream_name}.mp4")
    
    # Extract timestamps and frame data
    timestamps = stream['time_stamps']
    frames_data = stream['time_series']
    
    if frames_data is None or len(frames_data) == 0:
        print(f"No data found in video stream: {stream_name}")
        return
    
    # Determine frame rate from timestamps
    frame_intervals = np.diff(timestamps)
    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1/30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")
    
    # Try to get the first frame to determine size
    first_frame = None
    for frame_index, frame_data in enumerate(frames_data):
        try:
            # Frames are stored as base64 encoded JPEG strings
            if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                frame_str = frame_data[0]
            elif isinstance(frame_data, list) and len(frame_data) > 0:
                frame_str = frame_data[0]
            else:
                frame_str = frame_data
                
            jpeg_data = base64.b64decode(frame_str)
            nparr = np.frombuffer(jpeg_data, np.uint8)
            first_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if first_frame is not None:
                break
        except Exception as e:
            print(f"Error decoding frame {frame_index}: {e}, trying next...")
    
    if first_frame is None:
        print(f"Could not decode any frames for {stream_name}")
        return
    
    # Get frame dimensions
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process frames with progress bar
    print(f"Processing {len(frames_data)} frames...")
    for i, frame_data in enumerate(tqdm(frames_data)):
        try:
            # Get the frame string
            if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                frame_str = frame_data[0]
            elif isinstance(frame_data, list) and len(frame_data) > 0:
                frame_str = frame_data[0]
            else:
                frame_str = frame_data
            
            # Decode base64 encoded JPEG
            jpeg_data = base64.b64decode(frame_str)
            nparr = np.frombuffer(jpeg_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                print(f"Warning: Could not decode frame {i}")
                continue
                
            # Write frame to video
            video_writer.write(frame)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
    
    # Release video writer
    video_writer.release()
    
    # Create a CSV with timestamps
    timestamp_file = os.path.join(output_dir, f"{stream_name}_timestamps.csv")
    timestamp_df = pd.DataFrame({
        'frame_index': range(len(timestamps)),
        'timestamp': timestamps,
        'datetime': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
    })
    timestamp_df.to_csv(timestamp_file, index=False)
    
    print(f"Video saved to: {output_file}")
    print(f"Timestamps saved to: {timestamp_file}")

def extract_audio_stream(stream, output_dir, name=None):
    """Extract an audio stream from XDF to WAV"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting audio stream: {stream_name}")
    
    # Create output WAV file
    output_file = os.path.join(output_dir, f"{stream_name}.wav")
    
    # Extract timestamps and audio data
    timestamps = stream['time_stamps']
    audio_data = stream['time_series']
    
    if audio_data is None or len(audio_data) == 0:
        print(f"No data found in audio stream: {stream_name}")
        return
    
    # Determine audio properties
    sample_rate = float(stream['info']['nominal_srate'][0])
    print(f"Sample rate: {sample_rate} Hz")
    
    # Get channel count
    channel_count = int(stream['info']['channel_count'][0])
    print(f"Channels: {channel_count}")
    
    # Try to import specialized audio libraries
    try:
        from scipy.io import wavfile
        import soundfile as sf
        audio_lib_available = True
    except ImportError:
        import subprocess
        print("Installing audio libraries...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy soundfile"])
        from scipy.io import wavfile
        import soundfile as sf
        audio_lib_available = True
    
    # Convert audio data into a continuous array
    try:
        # Audio data might be in chunks
        all_audio = []
        for chunk in tqdm(audio_data, desc="Processing audio chunks"):
            # Handle different types of chunks
            if isinstance(chunk, np.ndarray):
                all_audio.append(chunk)
            elif isinstance(chunk, list):
                # Convert list to numpy array
                all_audio.append(np.array(chunk))
            else:
                print(f"Warning: Unknown audio chunk type: {type(chunk)}")
        
        # Determine the shape and combine
        if len(all_audio) > 0:
            try:
                # Try to stack vertically
                audio_array = np.vstack(all_audio)
            except ValueError:
                # If shapes are incompatible, try concatenating
                print("Warning: Audio chunks have inconsistent shapes, trying concatenation")
                audio_array = np.concatenate([chunk.flatten() for chunk in all_audio])
                
                # Try to reshape to match channel count
                if channel_count > 1:
                    # Make sure length is divisible by channel count
                    length = (len(audio_array) // channel_count) * channel_count
                    audio_array = audio_array[:length].reshape(-1, channel_count)
        else:
            print("No audio chunks found")
            return
        
        print(f"Audio array shape: {audio_array.shape}")
        
        # Save as WAV file
        if audio_lib_available:
            if audio_array.dtype != np.int16 and audio_array.dtype != np.float32:
                # Convert to float32 in the range [-1, 1]
                audio_array = audio_array.astype(np.float32)
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
            
            sf.write(output_file, audio_array, int(sample_rate))
            print(f"Audio saved to: {output_file}")
        else:
            print("Could not save audio: audio libraries not available")
            
        # Create a CSV with timestamps
        timestamp_file = os.path.join(output_dir, f"{stream_name}_timestamps.csv")
        timestamp_df = pd.DataFrame({
            'chunk_index': range(len(timestamps)),
            'timestamp': timestamps,
            'datetime': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
        })
        timestamp_df.to_csv(timestamp_file, index=False)
        print(f"Timestamps saved to: {timestamp_file}")
        
    except Exception as e:
        print(f"Error processing audio stream: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        try:
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
            
            # Save numpy array if possible
            if 'audio_array' in locals() and audio_array is not None:
                np.save(np_file, audio_array)
                print(f"Fallback: Audio data saved as numpy file: {np_file}")
            
            # Save JSON metadata
            with open(fallback_file, 'w') as f:
                json.dump({
                    'sample_rate': sample_rate,
                    'channel_count': channel_count,
                    'timestamps': timestamps.tolist()
                }, f)
            print(f"Fallback: Audio metadata saved to {fallback_file}")
        except Exception as fallback_error:
            print(f"Fallback save also failed: {fallback_error}")

def extract_gaze_stream(stream, output_dir, name=None):
    """Extract a gaze stream from XDF to CSV with support for both API and LSL formats"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting gaze stream: {stream_name}")
    
    # Create output CSV file
    output_file = os.path.join(output_dir, f"{stream_name}.csv")
    
    # Extract timestamps and gaze data
    timestamps = stream['time_stamps']
    gaze_data = stream['time_series']
    
    # Properly check if data exists and has elements
    if gaze_data is None or len(gaze_data) == 0:
        print(f"No data found in gaze stream: {stream_name}")
        return
    
    # Print basic info about the data
    print(f"Gaze data type: {type(gaze_data)}")
    print(f"Gaze data shape: {gaze_data.shape if hasattr(gaze_data, 'shape') else 'unknown'}")
    print(f"First sample type: {type(gaze_data[0]) if len(gaze_data) > 0 else 'N/A'}")
    
    # Convert to proper array if needed
    if not isinstance(gaze_data, np.ndarray):
        try:
            gaze_data = np.array(gaze_data)
            print(f"Converted gaze data to numpy array with shape: {gaze_data.shape}")
        except Exception as e:
            print(f"Error converting gaze data to numpy array: {e}")
            
            # Fallback: Save what we can
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'sample_count': len(gaze_data)
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            return
    
    # Get channel count
    channel_count = int(stream['info']['channel_count'][0])
    print(f"Channel count from stream info: {channel_count}")
    
    # Determine format based on channel count
    if channel_count == 16:
        # This is the extended LSL format with 16 channels
        print("Detected 16-channel LSL format gaze data")
        column_names = [
            "x", "y", 
            "left_PupilDiameter", "left_EyeballCenterX", "left_EyeballCenterY", "left_EyeballCenterZ",
            "left_OpticalAxisX", "left_OpticalAxisY", "right_OpticalAxisZ", "right_PupilDiameter",
            "right_EyeballCenterX", "right_EyeballCenterY", "right_EyeballCenterZ",
            "right_OpticalAxisX", "right_OpticalAxisY", "right_OpticalAxisZ"
        ]
        format_type = "LSL"
    elif channel_count <= 5:
        # This is API format
        print("Detected API format gaze data")
        column_names = ["frame_index", "gaze_x", "gaze_y", "pupil_diameter_left", "pupil_diameter_right"]
        
        # Use only as many columns as we have in the data
        if len(gaze_data.shape) > 1:
            column_names = column_names[:gaze_data.shape[1]]
            # If we have more columns than expected, add generic names
            if gaze_data.shape[1] > len(column_names):
                column_names.extend([f"extra_{i}" for i in range(len(column_names), gaze_data.shape[1])])
        format_type = "API"
    else:
        # Generic handling for other channel counts
        print(f"Unknown gaze format with {channel_count} channels")
        column_names = [f"channel_{i}" for i in range(gaze_data.shape[1] if len(gaze_data.shape) > 1 else 1)]
        format_type = "UNKNOWN"
    
    # Create DataFrame with error handling
    try:
        # Make sure column counts match
        if len(gaze_data.shape) > 1 and gaze_data.shape[1] != len(column_names):
            print(f"Warning: Column count mismatch. Data has {gaze_data.shape[1]} columns, but {len(column_names)} column names.")
            # Adjust column names to match data shape
            if gaze_data.shape[1] < len(column_names):
                column_names = column_names[:gaze_data.shape[1]]
            else:
                column_names.extend([f"extra_{i}" for i in range(len(column_names), gaze_data.shape[1])])
        
        # Create DataFrame
        if len(gaze_data.shape) > 1:
            # Multi-column data
            df = pd.DataFrame(gaze_data, columns=column_names)
        else:
            # Single-column data
            df = pd.DataFrame({column_names[0]: gaze_data})
            
        if format_type == "LSL" and max(timestamps) < 1500000000:  # ~July 2017
            print(f"LSL gaze timestamps appear to be relative (max: {max(timestamps):.2f})")
            
            # Store original timestamps
            df['lsl_relative_timestamp'] = timestamps
            
            # Look for reference streams with absolute timestamps
            reference_time = None
            time_offset = None
            
            # Check IMU file first (most likely to have consistent absolute timestamps)
            imu_path = os.path.join(output_dir, "imu.csv")
            if os.path.exists(imu_path):
                try:
                    imu_df = pd.read_csv(imu_path)
                    if 'timestamp [ns]' in imu_df.columns:
                        # Get first timestamp in nanoseconds and convert to seconds
                        first_imu_time = imu_df['timestamp [ns]'].iloc[0] / 1e9
                        # Assuming first IMU timestamp corresponds roughly to first gaze timestamp
                        time_offset = first_imu_time - timestamps[0]
                        reference_time = first_imu_time
                        print(f"Using IMU reference timestamp: {datetime.fromtimestamp(first_imu_time).strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception as e:
                    print(f"Error reading IMU file: {e}")
            
            # Try fixations if IMU not available
            if reference_time is None:
                fix_path = os.path.join(output_dir, "fixations.csv")
                if os.path.exists(fix_path):
                    try:
                        fix_df = pd.read_csv(fix_path)
                        if 'start_timestamp [ns]' in fix_df.columns:
                            first_fix_time = fix_df['start_timestamp [ns]'].iloc[0] / 1e9
                            time_offset = first_fix_time - timestamps[0]
                            reference_time = first_fix_time
                            print(f"Using fixations reference timestamp: {datetime.fromtimestamp(first_fix_time).strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        print(f"Error reading fixations file: {e}")
            
            # Try saccades as last resort
            if reference_time is None:
                sac_path = os.path.join(output_dir, "saccades.csv")
                if os.path.exists(sac_path):
                    try:
                        sac_df = pd.read_csv(sac_path)
                        if 'start_timestamp [ns]' in sac_df.columns:
                            first_sac_time = sac_df['start_timestamp [ns]'].iloc[0] / 1e9
                            time_offset = first_sac_time - timestamps[0]
                            reference_time = first_sac_time
                            print(f"Using saccades reference timestamp: {datetime.fromtimestamp(first_sac_time).strftime('%Y-%m-%d %H:%M:%S')}")
                    except Exception as e:
                        print(f"Error reading saccades file: {e}")
            
            # Apply timestamp correction if we found a reference
            if time_offset is not None:
                print(f"Applying timestamp offset of {time_offset:.2f} seconds")
                
                # Apply offset to all timestamps
                adjusted_timestamps = timestamps + time_offset
                
                # Update timestamp columns
                df['timestamp'] = adjusted_timestamps
                df['timestamp [ns]'] = (adjusted_timestamps * 1e9).astype(np.int64)  # Add nanosecond version
                df['datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') 
                                   for ts in adjusted_timestamps]
                
                print(f"Adjusted timestamps range: {datetime.fromtimestamp(min(adjusted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(max(adjusted_timestamps)).strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                # Fallback if no reference streams found
                print("WARNING: No reference streams with absolute timestamps found.")
                print("Timestamps will remain as relative values and show dates from 1970.")
                print("Consider re-extracting this data with other streams that contain absolute timestamps.")
        
        # Reset frame_index for API format
        if format_type == "API" and 'frame_index' in df.columns:
            if not pd.isna(df['frame_index']).all():  # Make sure frame_index column has valid data
                # Reset frame index to start from 0
                if len(df) > 0 and not pd.isna(df['frame_index'].iloc[0]):
                    first_frame = df['frame_index'].iloc[0]
                    df['original_frame_index'] = df['frame_index'].copy()  # Preserve original
                    df['frame_index'] = df['frame_index'] - first_frame
                    print(f"Reset frame_index to start at 0 (original first frame: {first_frame})")
        
        # Add additional columns at the end
        if 'timestamp' not in df.columns:  # Only add if not already done by the adjustment code
            df['timestamp'] = timestamps
            df['datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
        df['format_type'] = format_type
        
        # Add data_type for LSL format
        if format_type == "LSL":
            df['data_type'] = 'GAZE'

        # Add empty event_value column for LSL format
        if format_type == "LSL":
            df['event_value'] = ''
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Gaze data saved to: {output_file}")
        print(f"Gaze format identified as: {format_type}")
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        try:
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'data_shape': gaze_data.shape,
                    'data_sample': gaze_data[0].tolist() if len(gaze_data) > 0 else [],
                    'format_type': format_type
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            
            # Also try to save as numpy file
            np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
            np.save(np_file, gaze_data)
            np_timestamps = os.path.join(output_dir, f"{stream_name}_timestamps.npy")
            np.save(np_timestamps, timestamps)
            print(f"Fallback: Data saved as numpy files: {np_file} and {np_timestamps}")
        except Exception as fallback_error:
            print(f"Fallback save also failed: {fallback_error}")

def extract_metadata_stream(stream, output_dir, name=None):
    """Extract a metadata stream from XDF to JSON"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting metadata stream: {stream_name}")
    
    # Create output JSON file
    output_file = os.path.join(output_dir, f"{stream_name}.json")
    
    # Extract timestamps and metadata
    timestamps = stream['time_stamps']
    metadata_entries = stream['time_series']
    
    if metadata_entries is None or len(metadata_entries) == 0:
        print(f"No data found in metadata stream: {stream_name}")
        return
    
    # Process metadata with progress bar
    metadata_list = []
    for i, entry in enumerate(tqdm(metadata_entries, desc="Processing metadata")):
        try:
            # Get the actual entry
            if isinstance(entry, np.ndarray) and entry.size > 0:
                entry_data = entry[0]
            elif isinstance(entry, list) and len(entry) > 0:
                entry_data = entry[0]
            else:
                entry_data = entry
            
            # Metadata is usually stored as JSON string
            if isinstance(entry_data, str):
                try:
                    metadata = json.loads(entry_data)
                except json.JSONDecodeError:
                    # Not JSON, use as is
                    metadata = entry_data
            else:
                metadata = entry_data
                
            # Add timestamp
            metadata_with_time = {
                'timestamp': timestamps[i],
                'datetime': datetime.fromtimestamp(timestamps[i]).strftime('%Y-%m-%d %H:%M:%S.%f'),
                'metadata': metadata
            }
            
            metadata_list.append(metadata_with_time)
            
        except Exception as e:
            print(f"Error processing metadata entry {i}: {e}")
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    print(f"Metadata saved to: {output_file}")

def extract_depth_stream(stream, output_dir, name=None, save_interval=30, include_csv=False):
    """Extract a depth stream from XDF to both MP4 visualization and raw depth data"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting depth stream: {stream_name}")
    
    # Create output directory
    depth_dir = os.path.join(output_dir, f"{stream_name}_depth")
    os.makedirs(depth_dir, exist_ok=True)
    
    # Output MP4 file for visualization
    output_file = os.path.join(output_dir, f"{stream_name}_visualization.mp4")
    
    # Extract timestamps and frame data
    timestamps = stream['time_stamps']
    frames_data = stream['time_series']
    
    if frames_data is None or len(frames_data) == 0:
        print(f"No data found in depth stream: {stream_name}")
        return
    
    # Determine frame rate from timestamps
    frame_intervals = np.diff(timestamps)
    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1/30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")
    
    # Check stream info for depth scale if available
    depth_scale = 0.001  # Default: 1mm = 0.001m
    
    # Process frames - first pass to get statistics and first frame
    depth_min_global = float('inf')
    depth_max_global = 0
    valid_depths = []
    
    # Find a good frame and collect statistics
    first_raw_depth = None
    for frame_index in range(min(10, len(frames_data))):
        try:
            frame_data = frames_data[frame_index]
            # Get the frame string
            if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                frame_str = frame_data[0]
            elif isinstance(frame_data, list) and len(frame_data) > 0:
                frame_str = frame_data[0]
            else:
                frame_str = frame_data
                
            # Decode the depth data
            png_data = base64.b64decode(frame_str)
            nparr = np.frombuffer(png_data, np.uint8)
            raw_depth = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if raw_depth is not None and raw_depth.size > 0:
                first_raw_depth = raw_depth
                
                # Collect depth statistics
                valid_mask = raw_depth > 0
                if np.any(valid_mask):
                    depth_min = np.min(raw_depth[valid_mask])
                    depth_max = np.max(raw_depth[valid_mask])
                    depth_min_global = min(depth_min_global, depth_min)
                    depth_max_global = max(depth_max_global, depth_max)
                    
                    # Sample some valid depths for percentiles
                    sample_size = min(10000, np.count_nonzero(valid_mask))
                    if sample_size > 0:
                        # Get random indices of valid pixels
                        valid_indices = np.where(valid_mask.flatten())[0]
                        sampled_indices = np.random.choice(valid_indices, sample_size, replace=False)
                        valid_depths.extend(raw_depth.flatten()[sampled_indices])
                break
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
    
    if first_raw_depth is None:
        print(f"Could not extract any valid depth frames from {stream_name}")
        return
    
    # Determine visualization range from collected statistics
    if valid_depths:
        valid_depths = np.array(valid_depths)
        p_low = np.percentile(valid_depths, 1)
        p_high = np.percentile(valid_depths, 99)
        
        # Use slightly expanded range for better visualization
        range_expand = (p_high - p_low) * 0.1
        vis_min = max(0, p_low - range_expand)
        vis_max = min(65535, p_high + range_expand)
        
        print(f"Using depth range for visualization: {vis_min:.1f}-{vis_max:.1f}")
        print(f"This corresponds to approximately {vis_min*depth_scale:.3f}m - {vis_max*depth_scale:.3f}m")
    else:
        # Fallback to simple min/max if no valid depths collected
        vis_min = depth_min_global if depth_min_global != float('inf') else 0
        vis_max = depth_max_global if depth_max_global != 0 else 10000
        print(f"Fallback depth range: {vis_min}-{vis_max}")
    
    # Get frame dimensions for video
    height, width = first_raw_depth.shape[:2]
    print(f"Frame dimensions: {width}x{height}")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process all frames with improved normalization
    print(f"Processing {len(frames_data)} frames...")
    print(f"Saving raw depth PNG every {save_interval} frames")
    
    frame_counter = 0
    for i, frame_data in enumerate(tqdm(frames_data)):
        try:
            # Get the frame string
            if isinstance(frame_data, np.ndarray) and frame_data.size > 0:
                frame_str = frame_data[0]
            elif isinstance(frame_data, list) and len(frame_data) > 0:
                frame_str = frame_data[0]
            else:
                frame_str = frame_data
                
            # Decode base64 encoded data
            png_data = base64.b64decode(frame_str)
            nparr = np.frombuffer(png_data, np.uint8)
            raw_depth = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            
            if raw_depth is None:
                print(f"Warning: Could not decode frame {i}")
                continue
            
            # Create improved visualization
            valid_mask = (raw_depth > 0)
            
            # Initialize black image for invalid areas
            color_frame = np.zeros((raw_depth.shape[0], raw_depth.shape[1], 3), dtype=np.uint8)
            
            if np.any(valid_mask):
                # Normalize the valid depths
                normalized = np.zeros_like(raw_depth, dtype=np.uint8)
                normalized[valid_mask] = np.clip(
                    ((raw_depth[valid_mask] - vis_min) / (vis_max - vis_min) * 255),
                    0, 255
                ).astype(np.uint8)
                
                # Apply colormap only to valid pixels
                colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
                color_frame[valid_mask] = colored[valid_mask]
                
                # Add depth scale text
                cv2.putText(
                    color_frame,
                    f"Range: {vis_min*depth_scale:.2f}m - {vis_max*depth_scale:.2f}m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
            
            # Save raw depth data at specified interval
            if i % save_interval == 0:
                depth_file = os.path.join(depth_dir, f"depth_{i:06d}.png")
                cv2.imwrite(depth_file, raw_depth)
                frame_counter += 1
                
                # Also save distance map as CSV if requested
                if include_csv and depth_scale != 0:
                    distance_map = raw_depth.astype(np.float32) * depth_scale
                    distance_file = os.path.join(depth_dir, f"distance_{i:06d}.csv")
                    np.savetxt(distance_file, distance_map, delimiter=',')
            
            # Write frame to video
            video_writer.write(color_frame)
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
    
    # Release video writer
    video_writer.release()
    
    # Create a CSV with timestamps
    timestamp_file = os.path.join(output_dir, f"{stream_name}_timestamps.csv")
    timestamp_df = pd.DataFrame({
        'frame_index': range(len(timestamps)),
        'timestamp': timestamps,
        'datetime': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
    })
    timestamp_df.to_csv(timestamp_file, index=False)
    
    print(f"Depth visualization saved to: {output_file}")
    print(f"Raw depth samples ({frame_counter} frames) saved to: {depth_dir}")
    print(f"Timestamps saved to: {timestamp_file}")
    print(f"Note: Raw depth values are in millimeters, scale by {depth_scale} to get meters")


def extract_generic_stream(stream, output_dir, name=None):
    """Extract a generic stream from XDF to CSV"""
    stream_name = name or stream['info']['name'][0]
    stream_type = stream['info']['type'][0]
    print(f"Extracting generic stream: {stream_name} (type: {stream_type})")
    
    # Create output CSV file
    output_file = os.path.join(output_dir, f"{stream_name}.csv")
    
    # Extract timestamps and data
    timestamps = stream['time_stamps']
    data_series = stream['time_series']
    
    if data_series is None or len(data_series) == 0:
        print(f"No data found in stream: {stream_name}")
        return
    
    try:
        # Print basic info about the data
        print(f"Data type: {type(data_series)}")
        print(f"First sample type: {type(data_series[0]) if len(data_series) > 0 else 'N/A'}")
        
        # Try to process data into rows for CSV
        rows = []
        
        # Process data
        for i, data in enumerate(data_series):
            try:
                # Create a row for this sample
                row = {'timestamp': timestamps[i]}
                
                # Extract the data values
                if isinstance(data, np.ndarray):
                    if data.size == 1:
                        # Single value
                        row['value'] = float(data)
                    else:
                        # Multiple values
                        for j, value in enumerate(data.flatten()):
                            row[f'channel_{j}'] = value
                elif isinstance(data, list):
                    # List of values
                    if len(data) == 1:
                        # Single value in a list
                        row['value'] = data[0]
                    else:
                        # Multiple values
                        for j, value in enumerate(data):
                            row[f'channel_{j}'] = value
                else:
                    # Single value
                    row['value'] = data
                
                rows.append(row)
            except Exception as e:
                print(f"Error processing row {i}: {e}")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Add datetime column
        df['datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in df['timestamp']]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error processing stream {stream_name}: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
        try:
            # Try to convert to list for JSON serialization
            data_list = []
            for item in data_series:
                if hasattr(item, 'tolist'):
                    data_list.append(item.tolist())
                elif isinstance(item, list):
                    data_list.append(item)
                else:
                    data_list.append(str(item))
                    
            with open(fallback_file, 'w') as f:
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'data': data_list
                }, f)
            print(f"Fallback: raw data saved to {fallback_file}")
        except Exception as json_error:
            print(f"Error saving JSON fallback: {json_error}")
            
            # Last resort: save as numpy
            try:
                np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
                np.save(np_file, data_series)
                np_timestamps = os.path.join(output_dir, f"{stream_name}_timestamps.npy")
                np.save(np_timestamps, timestamps)
                print(f"Fallback: Data saved as numpy files: {np_file} and {np_timestamps}")
            except Exception as np_error:
                print(f"Error saving numpy fallback: {np_error}")
            
def extract_imu_stream(stream, output_dir, name=None):
    """Extract IMU data from XDF to CSV following the specified format"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting IMU stream: {stream_name}")
    
    # Create output CSV file
    output_file = os.path.join(output_dir, "imu.csv")
    
    # Extract timestamps and IMU data
    timestamps = stream['time_stamps']
    imu_data = stream['time_series']
    
    if imu_data is None or len(imu_data) == 0:
        print(f"No data found in IMU stream: {stream_name}")
        return
    
    # Print basic info about the data
    print(f"IMU data type: {type(imu_data)}")
    print(f"IMU data shape: {imu_data.shape if hasattr(imu_data, 'shape') else 'unknown'}")
    print(f"First sample type: {type(imu_data[0]) if len(imu_data) > 0 else 'N/A'}")
    
    # Convert to proper array if needed
    if not isinstance(imu_data, np.ndarray):
        try:
            imu_data = np.array(imu_data)
            print(f"Converted IMU data to numpy array with shape: {imu_data.shape}")
        except Exception as e:
            print(f"Error converting IMU data to numpy array: {e}")
            
            # Fallback: Save what we can
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'sample_count': len(imu_data)
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            return
    
    # Get channel count
    channel_count = int(stream['info']['channel_count'][0])
    print(f"Channel count from stream info: {channel_count}")
    
    # Expected column names for IMU data
    imu_columns = [
        "gyro_x", "gyro_y", "gyro_z", 
        "accel_x", "accel_y", "accel_z", 
        "roll", "pitch", "yaw", 
        "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"
    ]
    
    # Adjust column names based on actual data
    if len(imu_data.shape) > 1:
        if imu_data.shape[1] < len(imu_columns):
            # Use only as many columns as we have in the data
            column_names = imu_columns[:imu_data.shape[1]]
            print(f"Warning: IMU data has fewer columns ({imu_data.shape[1]}) than expected ({len(imu_columns)})")
        elif imu_data.shape[1] > len(imu_columns):
            # If we have more columns than expected, add generic names
            column_names = imu_columns.copy()
            column_names.extend([f"extra_{i}" for i in range(len(imu_columns), imu_data.shape[1])])
            print(f"Warning: IMU data has more columns ({imu_data.shape[1]}) than expected ({len(imu_columns)})")
        else:
            column_names = imu_columns.copy()
    else:
        # Single column data (unlikely for IMU)
        column_names = [imu_columns[0]]
        print("Warning: IMU data appears to have only one column")
    
    try:
        # Create DataFrame
        if len(imu_data.shape) > 1:
            # Multi-column data
            df = pd.DataFrame(imu_data, columns=column_names)
        else:
            # Single-column data
            df = pd.DataFrame({column_names[0]: imu_data})
        
        # Add timestamp columns in nanoseconds as required by the format
        df['timestamp [ns]'] = (timestamps * 1e9).astype(np.int64)  # Convert to nanoseconds
        
        # Add a section_id and recording_id for compatibility
        df['section_id'] = 1
        df['recording_id'] = 1
        
        # Add datetime for readability
        df['datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
        
        # Reorder columns to match expected format
        ordered_cols = ['section_id', 'recording_id', 'timestamp [ns]']
        ordered_cols.extend([
            'gyro_x [deg/s]' if 'gyro_x' in df.columns else 'gyro_x', 
            'gyro_y [deg/s]' if 'gyro_y' in df.columns else 'gyro_y', 
            'gyro_z [deg/s]' if 'gyro_z' in df.columns else 'gyro_z',
            'acceleration_x [g]' if 'acceleration_x' in df.columns else 'accel_x [g]' if 'accel_x' in df.columns else 'accel_x',
            'acceleration_y [g]' if 'acceleration_y' in df.columns else 'accel_y [g]' if 'accel_y' in df.columns else 'accel_y',
            'acceleration_z [g]' if 'acceleration_z' in df.columns else 'accel_z [g]' if 'accel_z' in df.columns else 'accel_z',
            'roll [deg]' if 'roll' in df.columns else 'roll',
            'pitch [deg]' if 'pitch' in df.columns else 'pitch',
            'yaw [deg]' if 'yaw' in df.columns else 'yaw',
            'quaternion_w' if 'quaternion_w' in df.columns else 'quat_w' if 'quat_w' in df.columns else 'quaternion_w',
            'quaternion_x' if 'quaternion_x' in df.columns else 'quat_x' if 'quat_x' in df.columns else 'quaternion_x',
            'quaternion_y' if 'quaternion_y' in df.columns else 'quat_y' if 'quat_y' in df.columns else 'quaternion_y',
            'quaternion_z' if 'quaternion_z' in df.columns else 'quat_z' if 'quat_z' in df.columns else 'quaternion_z'
        ])
        
        # Add any extra columns
        extra_cols = [col for col in df.columns if col not in ordered_cols and col != 'datetime']
        ordered_cols.extend(extra_cols)
        ordered_cols.append('datetime')  # Add datetime at the end
        
        # Ensure all required columns exist in dataframe
        final_cols = []
        for col in ordered_cols:
            if col in df.columns:
                final_cols.append(col)
        
        # Rename columns to match expected format with units in column names
        if 'gyro_x' in df.columns and 'gyro_x [deg/s]' not in df.columns:
            df.rename(columns={'gyro_x': 'gyro_x [deg/s]'}, inplace=True)
        if 'gyro_y' in df.columns and 'gyro_y [deg/s]' not in df.columns:
            df.rename(columns={'gyro_y': 'gyro_y [deg/s]'}, inplace=True)
        if 'gyro_z' in df.columns and 'gyro_z [deg/s]' not in df.columns:
            df.rename(columns={'gyro_z': 'gyro_z [deg/s]'}, inplace=True)
            
        if 'accel_x' in df.columns and 'accel_x [g]' not in df.columns:
            df.rename(columns={'accel_x': 'accel_x [g]'}, inplace=True)
        if 'accel_y' in df.columns and 'accel_y [g]' not in df.columns:
            df.rename(columns={'accel_y': 'accel_y [g]'}, inplace=True)
        if 'accel_z' in df.columns and 'accel_z [g]' not in df.columns:
            df.rename(columns={'accel_z': 'accel_z [g]'}, inplace=True)
            
        if 'roll' in df.columns and 'roll [deg]' not in df.columns:
            df.rename(columns={'roll': 'roll [deg]'}, inplace=True)
        if 'pitch' in df.columns and 'pitch [deg]' not in df.columns:
            df.rename(columns={'pitch': 'pitch [deg]'}, inplace=True)
        if 'yaw' in df.columns and 'yaw [deg]' not in df.columns:
            df.rename(columns={'yaw': 'yaw [deg]'}, inplace=True)
        
        # Save to CSV with standardized column names
        final_df = df.copy()
        final_df.to_csv(output_file, index=False)
        print(f"IMU data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        try:
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'data_shape': imu_data.shape,
                    'data_sample': imu_data[0].tolist() if len(imu_data) > 0 else []
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            
            # Also try to save as numpy file
            np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
            np.save(np_file, imu_data)
            np_timestamps = os.path.join(output_dir, f"{stream_name}_timestamps.npy")
            np.save(np_timestamps, timestamps)
            print(f"Fallback: Data saved as numpy files: {np_file} and {np_timestamps}")
        except Exception as fallback_error:
            print(f"Fallback save also failed: {fallback_error}")

def extract_fixations_stream(stream, output_dir, name=None):
    """Extract fixations data from XDF to CSV following the specified format"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting fixations stream: {stream_name}")
    
    # Create output CSV file
    output_file = os.path.join(output_dir, "fixations.csv")
    
    # Extract timestamps and fixations data
    timestamps = stream['time_stamps']
    fixation_data = stream['time_series']
    
    if fixation_data is None or len(fixation_data) == 0:
        print(f"No data found in fixations stream: {stream_name}")
        return
    
    # Print basic info about the data
    print(f"Fixations data type: {type(fixation_data)}")
    print(f"Fixations data shape: {fixation_data.shape if hasattr(fixation_data, 'shape') else 'unknown'}")
    print(f"First sample type: {type(fixation_data[0]) if len(fixation_data) > 0 else 'N/A'}")
    
    # Convert to proper array if needed
    if not isinstance(fixation_data, np.ndarray):
        try:
            fixation_data = np.array(fixation_data)
            print(f"Converted fixations data to numpy array with shape: {fixation_data.shape}")
        except Exception as e:
            print(f"Error converting fixations data to numpy array: {e}")
            
            # Fallback: Save what we can
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'sample_count': len(fixation_data)
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            return
    
    # Get channel count
    channel_count = int(stream['info']['channel_count'][0])
    print(f"Channel count from stream info: {channel_count}")
    
    # Expected column names for fixation data
    # Expected column names for fixation data
    fixation_columns = [
        "fixation_id", "start_time_ns", "end_time_ns", "duration_ms",
        "mean_gaze_x", "mean_gaze_y", "azimuth_deg", "elevation_deg"
    ]
    
    # Adjust column names based on actual data
    if len(fixation_data.shape) > 1:
        if fixation_data.shape[1] < len(fixation_columns):
            # Use only as many columns as we have in the data
            column_names = fixation_columns[:fixation_data.shape[1]]
            print(f"Warning: Fixations data has fewer columns ({fixation_data.shape[1]}) than expected ({len(fixation_columns)})")
        elif fixation_data.shape[1] > len(fixation_columns):
            # If we have more columns than expected, add generic names
            column_names = fixation_columns.copy()
            column_names.extend([f"extra_{i}" for i in range(len(fixation_columns), fixation_data.shape[1])])
            print(f"Warning: Fixations data has more columns ({fixation_data.shape[1]}) than expected ({len(fixation_columns)})")
        else:
            column_names = fixation_columns.copy()
    else:
        # Single column data (unlikely for fixations)
        column_names = [fixation_columns[0]]
        print("Warning: Fixations data appears to have only one column")
    
    try:
        # Create DataFrame
        if len(fixation_data.shape) > 1:
            # Multi-column data
            df = pd.DataFrame(fixation_data, columns=column_names)
        else:
            # Single-column data
            df = pd.DataFrame({column_names[0]: fixation_data})
        
        # Add section_id and recording_id columns for compatibility
        df['section_id'] = 1
        df['recording_id'] = 1
        
        # Add timestamp column for when the fixation event was detected
        df['detected_timestamp'] = timestamps
        df['detected_datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
        
        # Ensure timestamp columns have correct format if they exist
        if 'start_timestamp_ns' in df.columns:
            # Check if timestamps look like seconds
            if df['start_timestamp_ns'].dtype == 'float64' and df['start_timestamp_ns'].max() < 1e12:
                print("Converting start_timestamp_ns from seconds to nanoseconds and int64")
                df['start_timestamp_ns'] = (df['start_timestamp_ns'] * 1e9).astype(np.int64)
            # Check if they are already large floats that should be integers
            elif df['start_timestamp_ns'].dtype == 'float64':
                print("Converting large float start_timestamp_ns to int64")
                # Add handling for potential NaNs if necessary before conversion
                # df['start_timestamp_ns'] = df['start_timestamp_ns'].fillna(-1).astype(np.int64) # Example NaN handling
                df['start_timestamp_ns'] = df['start_timestamp_ns'].astype(np.int64)
            # If already int64, potentially do nothing, or ensure it is int64
            elif df['start_timestamp_ns'].dtype != 'int64':
                df['start_timestamp_ns'] = df['start_timestamp_ns'].astype(np.int64)


        if 'end_timestamp_ns' in df.columns:
            # Check if timestamps look like seconds
            if df['end_timestamp_ns'].dtype == 'float64' and df['end_timestamp_ns'].max() < 1e12:
                print("Converting end_timestamp_ns from seconds to nanoseconds and int64")
                df['end_timestamp_ns'] = (df['end_timestamp_ns'] * 1e9).astype(np.int64)
            # Check if they are already large floats that should be integers
            elif df['end_timestamp_ns'].dtype == 'float64':
                print("Converting large float end_timestamp_ns to int64")
                # Add handling for potential NaNs if necessary before conversion
                # df['end_timestamp_ns'] = df['end_timestamp_ns'].fillna(-1).astype(np.int64) # Example NaN handling
                df['end_timestamp_ns'] = df['end_timestamp_ns'].astype(np.int64)
            # If already int64, potentially do nothing, or ensure it is int64
            elif df['end_timestamp_ns'].dtype != 'int64':
                df['end_timestamp_ns'] = df['end_timestamp_ns'].astype(np.int64)
        
        # Reorder columns to match expected format
        ordered_cols = ['section_id', 'recording_id', 'fixation_id', 
                         'start_timestamp [ns]', 'end_timestamp [ns]', 'duration [ms]',
                         'fixation_x [px]', 'fixation_y [px]', 'azimuth [deg]', 'elevation [deg]']
        
        # Rename columns to match expected format
        column_mapping = {
            'start_time_ns': 'start_timestamp [ns]',
            'end_time_ns': 'end_timestamp [ns]',
            'duration_ms': 'duration [ms]',
            'mean_gaze_x': 'fixation_x [px]',
            'mean_gaze_y': 'fixation_y [px]',
            'azimuth_deg': 'azimuth [deg]',
            'elevation_deg': 'elevation [deg]'
        }
        
        # Apply column renaming
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Ensure all required columns exist
        for col in ordered_cols:
            if col not in df.columns:
                # For missing columns, add with NaN values
                print(f"Warning: Adding missing column {col} with NaN values")
                df[col] = np.nan
        
        # Save to CSV with ordered columns
        final_cols = ordered_cols + ['detected_timestamp', 'detected_datetime']
        final_df = df[final_cols]
        final_df.to_csv(output_file, index=False)
        print(f"Fixations data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        try:
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'data_shape': fixation_data.shape,
                    'data_sample': fixation_data[0].tolist() if len(fixation_data) > 0 else []
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            
            # Also try to save as numpy file
            np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
            np.save(np_file, fixation_data)
            np_timestamps = os.path.join(output_dir, f"{stream_name}_timestamps.npy")
            np.save(np_timestamps, timestamps)
            print(f"Fallback: Data saved as numpy files: {np_file} and {np_timestamps}")
        except Exception as fallback_error:
            print(f"Fallback save also failed: {fallback_error}")

def extract_saccades_stream(stream, output_dir, name=None):
    """Extract saccades data from XDF to CSV following the specified format"""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting saccades stream: {stream_name}")
    
    # Create output CSV file
    output_file = os.path.join(output_dir, "saccades.csv")
    
    # Extract timestamps and saccades data
    timestamps = stream['time_stamps']
    saccade_data = stream['time_series']
    
    if saccade_data is None or len(saccade_data) == 0:
        print(f"No data found in saccades stream: {stream_name}")
        return
    
    # Print basic info about the data
    print(f"Saccades data type: {type(saccade_data)}")
    print(f"Saccades data shape: {saccade_data.shape if hasattr(saccade_data, 'shape') else 'unknown'}")
    print(f"First sample type: {type(saccade_data[0]) if len(saccade_data) > 0 else 'N/A'}")
    
    # Convert to proper array if needed
    if not isinstance(saccade_data, np.ndarray):
        try:
            saccade_data = np.array(saccade_data)
            print(f"Converted saccades data to numpy array with shape: {saccade_data.shape}")
        except Exception as e:
            print(f"Error converting saccades data to numpy array: {e}")
            
            # Fallback: Save what we can
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'sample_count': len(saccade_data)
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            return
    
    # Get channel count
    channel_count = int(stream['info']['channel_count'][0])
    print(f"Channel count from stream info: {channel_count}")
    
    # Expected column names for saccade data
    saccade_columns = [
        "saccade_id", "start_time_ns", "end_time_ns", "amplitude_angle_deg",
        "amplitude_pixels", "mean_velocity", "max_velocity", "duration_ms"
    ]
    
    # Adjust column names based on actual data
    if len(saccade_data.shape) > 1:
        if saccade_data.shape[1] < len(saccade_columns):
            # Use only as many columns as we have in the data
            column_names = saccade_columns[:saccade_data.shape[1]]
            print(f"Warning: Saccades data has fewer columns ({saccade_data.shape[1]}) than expected ({len(saccade_columns)})")
        elif saccade_data.shape[1] > len(saccade_columns):
            # If we have more columns than expected, add generic names
            column_names = saccade_columns.copy()
            column_names.extend([f"extra_{i}" for i in range(len(saccade_columns), saccade_data.shape[1])])
            print(f"Warning: Saccades data has more columns ({saccade_data.shape[1]}) than expected ({len(saccade_columns)})")
        else:
            column_names = saccade_columns.copy()
    else:
        # Single column data (unlikely for saccades)
        column_names = [saccade_columns[0]]
        print("Warning: Saccades data appears to have only one column")
    
    try:
        # Create DataFrame
        if len(saccade_data.shape) > 1:
            # Multi-column data
            df = pd.DataFrame(saccade_data, columns=column_names)
        else:
            # Single-column data
            df = pd.DataFrame({column_names[0]: saccade_data})
        
        # Add section_id and recording_id columns for compatibility
        df['section_id'] = 1
        df['recording_id'] = 1
        
        # Add timestamp column for when the saccade event was detected
        df['detected_timestamp'] = timestamps
        df['detected_datetime'] = [datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S.%f') for ts in timestamps]
        
        # Ensure timestamp columns have correct format if they exist
        if 'start_timestamp_ns' in df.columns:
            # Check if timestamps look like seconds
            if df['start_timestamp_ns'].dtype == 'float64' and df['start_timestamp_ns'].max() < 1e12:
                print("Converting start_timestamp_ns from seconds to nanoseconds and int64")
                df['start_timestamp_ns'] = (df['start_timestamp_ns'] * 1e9).astype(np.int64)
            # Check if they are already large floats that should be integers
            elif df['start_timestamp_ns'].dtype == 'float64':
                print("Converting large float start_timestamp_ns to int64")
                # Add handling for potential NaNs if necessary before conversion
                # df['start_timestamp_ns'] = df['start_timestamp_ns'].fillna(-1).astype(np.int64) # Example NaN handling
                df['start_timestamp_ns'] = df['start_timestamp_ns'].astype(np.int64)
            # If already int64, potentially do nothing, or ensure it is int64
            elif df['start_timestamp_ns'].dtype != 'int64':
                df['start_timestamp_ns'] = df['start_timestamp_ns'].astype(np.int64)


        if 'end_timestamp_ns' in df.columns:
            # Check if timestamps look like seconds
            if df['end_timestamp_ns'].dtype == 'float64' and df['end_timestamp_ns'].max() < 1e12:
                print("Converting end_timestamp_ns from seconds to nanoseconds and int64")
                df['end_timestamp_ns'] = (df['end_timestamp_ns'] * 1e9).astype(np.int64)
            # Check if they are already large floats that should be integers
            elif df['end_timestamp_ns'].dtype == 'float64':
                print("Converting large float end_timestamp_ns to int64")
                # Add handling for potential NaNs if necessary before conversion
                # df['end_timestamp_ns'] = df['end_timestamp_ns'].fillna(-1).astype(np.int64) # Example NaN handling
                df['end_timestamp_ns'] = df['end_timestamp_ns'].astype(np.int64)
            # If already int64, potentially do nothing, or ensure it is int64
            elif df['end_timestamp_ns'].dtype != 'int64':
                df['end_timestamp_ns'] = df['end_timestamp_ns'].astype(np.int64)
        
        # Reorder columns to match expected format
        ordered_cols = ['section_id', 'recording_id', 'saccade_id', 
                         'start_timestamp [ns]', 'end_timestamp [ns]', 'duration [ms]',
                         'amplitude [px]', 'amplitude [deg]', 
                         'mean_velocity [px/s]', 'peak_velocity [px/s]']
        
        # Rename columns to match expected format
        column_mapping = {
            'start_time_ns': 'start_timestamp [ns]',
            'end_time_ns': 'end_timestamp [ns]',
            'duration_ms': 'duration [ms]',
            'amplitude_pixels': 'amplitude [px]',
            'amplitude_angle_deg': 'amplitude [deg]',
            'mean_velocity': 'mean_velocity [px/s]',
            'max_velocity': 'peak_velocity [px/s]'
        }
        
        # Apply column renaming
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Ensure all required columns exist
        for col in ordered_cols:
            if col not in df.columns:
                # For missing columns, add with NaN values
                print(f"Warning: Adding missing column {col} with NaN values")
                df[col] = np.nan
        
        # Save to CSV with ordered columns
        final_cols = ordered_cols + ['detected_timestamp', 'detected_datetime']
        final_df = df[final_cols]
        final_df.to_csv(output_file, index=False)
        print(f"Saccades data saved to: {output_file}")
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback: save raw data and timestamps
        try:
            fallback_file = os.path.join(output_dir, f"{stream_name}_raw.json")
            with open(fallback_file, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump({
                    'timestamps': timestamps.tolist(),
                    'data_shape': saccade_data.shape,
                    'data_sample': saccade_data[0].tolist() if len(saccade_data) > 0 else []
                }, f)
            print(f"Fallback: Basic info saved to {fallback_file}")
            
            # Also try to save as numpy file
            np_file = os.path.join(output_dir, f"{stream_name}_data.npy")
            np.save(np_file, saccade_data)
            np_timestamps = os.path.join(output_dir, f"{stream_name}_timestamps.npy")
            np.save(np_timestamps, timestamps)
            print(f"Fallback: Data saved as numpy files: {np_file} and {np_timestamps}")
        except Exception as fallback_error:
            print(f"Fallback save also failed: {fallback_error}")

def extract_streams(xdf_file, output_dir, keep_raw_depth=True, depth_interval=30, include_csv=False):
    """Extract all streams from XDF file
    
    Args:
        xdf_file: Path to the XDF file
        output_dir: Directory to save extracted data
        keep_raw_depth: If True, keeps raw depth data for measurements. If False, deletes it after creating MP4
        depth_interval: Save raw depth PNG every N frames (default: 30)
        include_csv: Whether to include CSV distance maps (default: False)
    """
    print(f"Loading XDF file: {xdf_file}")
    print(f"Raw depth data will be {'kept' if keep_raw_depth else 'deleted'} after processing")
    print(f"Saving raw depth PNG every {depth_interval} frames")
    
    try:
        # Load XDF file
        streams, fileheader = pyxdf.load_xdf(xdf_file)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"XDF file loaded. Found {len(streams)} streams:")
        
        # List all streams
        for i, stream in enumerate(streams):
            name = stream['info']['name'][0]
            stream_type = stream['info']['type'][0]
            channel_count = int(stream['info']['channel_count'][0])
            sample_count = len(stream['time_series'])
            print(f"  {i+1}. {name} (Type: {stream_type}, Channels: {channel_count}, Samples: {sample_count})")
        
        # Track any depth raw folders for cleanup if needed
        depth_raw_dirs = []
        
        # Track if we've found certain specialized stream types
        found_imu = False
        found_fixations = False
        found_saccades = False
        
        # First pass - extract normal streams and identify special streams
        for stream in streams:
            try:
                name = stream['info']['name'][0]
                stream_type = stream['info']['type'][0]
                
                # Check for specialized streams by name and type
                if stream_type == 'IMU' or name.lower() in ['neonimu', 'neon_imu', 'imu']:
                    found_imu = True
                    extract_imu_stream(stream, output_dir)
                    continue
                    
                if name.lower() in ['neonfixations', 'neon_fixations', 'fixations']:
                    found_fixations = True
                    extract_fixations_stream(stream, output_dir)
                    continue
                    
                if name.lower() in ['neonsaccades', 'neon_saccades', 'saccades']:
                    found_saccades = True
                    extract_saccades_stream(stream, output_dir)
                    continue
                
                # Regular stream processing
                if stream_type == 'VideoStream':
                    extract_video_stream(stream, output_dir)
                elif stream_type == 'Depth' or stream_type == 'DepthData':
                    # Handle all depth-related streams with one function, passing the additional parameters
                    depth_dir = os.path.join(output_dir, f"{name}_depth")
                    if not keep_raw_depth:
                        depth_raw_dirs.append(depth_dir)
                    extract_depth_stream(stream, output_dir, save_interval=depth_interval, include_csv=include_csv)
                elif stream_type == 'Audio':
                    extract_audio_stream(stream, output_dir)
                elif stream_type == 'Gaze':
                    extract_gaze_stream(stream, output_dir)
                elif stream_type == 'DeviceInfo':
                    extract_metadata_stream(stream, output_dir)
                else:
                    # Try to infer stream type from name if not already identified
                    if 'fixation' in name.lower() and not found_fixations:
                        found_fixations = True
                        extract_fixations_stream(stream, output_dir)
                    elif 'saccade' in name.lower() and not found_saccades:
                        found_saccades = True
                        extract_saccades_stream(stream, output_dir)
                    elif 'imu' in name.lower() and not found_imu:
                        found_imu = True
                        extract_imu_stream(stream, output_dir)
                    else:
                        # Generic stream extractor for other types
                        extract_generic_stream(stream, output_dir)
                        
            except Exception as stream_error:
                print(f"Error extracting stream {name}: {stream_error}")
                import traceback
                traceback.print_exc()
                print("Continuing with next stream...")
        
        # Second pass - look for specialized streams by contents if not found by name/type
        if not found_imu or not found_fixations or not found_saccades:
            print("\nChecking for specialized streams by content pattern...")
            
            for stream in streams:
                try:
                    name = stream['info']['name'][0]
                    stream_type = stream['info']['type'][0]
                    channel_count = int(stream['info']['channel_count'][0])
                    
                    # Skip already processed specialized streams
                    if ((stream_type == 'IMU' or name.lower() in ['neonimu', 'neon_imu', 'imu']) or
                        (name.lower() in ['neonfixations', 'neon_fixations', 'fixations']) or
                        (name.lower() in ['neonsaccades', 'neon_saccades', 'saccades'])):
                        continue
                    
                    # Try to detect specialized streams by channel count and patterns
                    if not found_imu and channel_count >= 9 and channel_count <= 13:
                        # IMU typically has 9-13 channels (gyro xyz, accel xyz, quaternion wxyz, optional euler angles)
                        print(f"Stream '{name}' looks like it might contain IMU data (has {channel_count} channels)")
                        if input("Extract as IMU data? (y/n): ").lower().startswith('y'):
                            extract_imu_stream(stream, output_dir)
                            found_imu = True
                            continue
                    
                    if not found_fixations and channel_count >= 6 and channel_count <= 8:
                        # Fixations typically have 6-8 channels
                        print(f"Stream '{name}' looks like it might contain fixation data (has {channel_count} channels)")
                        if input("Extract as fixations data? (y/n): ").lower().startswith('y'):
                            extract_fixations_stream(stream, output_dir)
                            found_fixations = True
                            continue
                    
                    if not found_saccades and channel_count >= 7 and channel_count <= 8:
                        # Saccades typically have 7-8 channels
                        print(f"Stream '{name}' looks like it might contain saccade data (has {channel_count} channels)")
                        if input("Extract as saccades data? (y/n): ").lower().startswith('y'):
                            extract_saccades_stream(stream, output_dir)
                            found_saccades = True
                            continue
                
                except Exception as detect_error:
                    print(f"Error during stream type detection for {name}: {detect_error}")
                    continue
        
        # Clean up the raw depth folders if requested
        if not keep_raw_depth:
            for raw_dir in depth_raw_dirs:
                if os.path.exists(raw_dir):
                    import shutil
                    try:
                        print(f"Removing raw depth data folder: {raw_dir}")
                        shutil.rmtree(raw_dir)
                        print(f"Successfully removed: {raw_dir}")
                    except Exception as e:
                        print(f"Error removing directory {raw_dir}: {e}")
        else:
            print("Keeping all raw depth data for future measurement purposes")
        
        # Print summary of specialized streams
        print("\nSpecialized streams extraction summary:")
        print(f"- IMU data: {'Extracted' if found_imu else 'Not found'}")
        print(f"- Fixations data: {'Extracted' if found_fixations else 'Not found'}")
        print(f"- Saccades data: {'Extracted' if found_saccades else 'Not found'}")
        
        print(f"\nAll streams extracted to: {output_dir}")
        
    except Exception as e:
        print(f"Error extracting XDF file: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="XDF Extractor")
    parser.add_argument("--file", type=str, required=True, help="Path to XDF file")
    parser.add_argument("--outdir", type=str, default="extracted_data", help="Output directory (default: extracted_data)")
    parser.add_argument("--no-raw-depth", action="store_true", 
                        help="Delete raw depth data after creating MP4 (not recommended for measurements)")
    parser.add_argument("--depth-interval", type=int, default=1, 
                        help="Save raw depth PNG every N frames (default: 30, use 1 for all frames)")
    parser.add_argument("--include-csv", action="store_true", 
                        help="Include CSV distance maps (increases disk usage)")
    args = parser.parse_args()
    
    # Check if XDF file exists
    if not os.path.exists(args.file):
        print(f"Error: XDF file not found: {args.file}")
        return 1
    
    # Start extraction
    start_time = time.time()
    extract_streams(
        args.file, 
        args.outdir, 
        keep_raw_depth=not args.no_raw_depth,
        depth_interval=args.depth_interval,
        include_csv=args.include_csv
    )
    end_time = time.time()
    
    print(f"\nExtraction completed in {end_time - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
