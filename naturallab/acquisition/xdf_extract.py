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
import re
import time
import numpy as np
import pandas as pd
import cv2
from datetime import datetime

try:
    import pyxdf
except ImportError:
    pyxdf = None

try:
    from tqdm import tqdm
except ImportError as error:
    raise ImportError(
        "tqdm is required; install NaturalLab's core dependencies"
    ) from error


def _stream_name(stream):
    """Return an XDF stream name without exposing pyxdf's list wrapping."""
    try:
        return str(stream["info"]["name"][0])
    except (KeyError, IndexError, TypeError):
        return "<unknown>"


def _stream_type(stream):
    """Return an XDF stream type without exposing pyxdf's list wrapping."""
    try:
        return str(stream["info"]["type"][0])
    except (KeyError, IndexError, TypeError):
        return ""


def _is_declared_imu_stream(stream):
    """Return whether stream metadata explicitly identifies an IMU stream."""
    name = _stream_name(stream).lower()
    return _stream_type(stream).lower() == "imu" or name in {
        "neonimu",
        "neon_imu",
        "imu",
    }


def _safe_filename_component(value):
    """Make a deterministic, portable filename component."""
    component = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return component.strip("_") or "stream"


def _plan_imu_output_filenames(streams):
    """Choose non-colliding IMU filenames while retaining single-stream output."""
    if len(streams) == 1:
        return ["imu.csv"]

    filenames = []
    occurrences = {}
    for stream in streams:
        stem = f"imu_{_safe_filename_component(_stream_name(stream))}"
        occurrences[stem] = occurrences.get(stem, 0) + 1
        occurrence = occurrences[stem]
        suffix = "" if occurrence == 1 else f"_{occurrence}"
        filenames.append(f"{stem}{suffix}.csv")
    return filenames


def _prepare_extraction_output_dir(output_dir):
    """Create an empty output directory or reject stale/reused results."""
    output_dir = os.path.abspath(os.fspath(output_dir))
    if os.path.lexists(output_dir):
        if os.path.islink(output_dir) or not os.path.isdir(output_dir):
            raise RuntimeError(
                f"XDF output path is not a normal directory: {output_dir}"
            )
        with os.scandir(output_dir) as entries:
            if next(entries, None) is not None:
                raise RuntimeError(
                    f"XDF output directory is not empty: {output_dir}. "
                    "Choose a new empty directory so stale files cannot be "
                    "mistaken for this extraction."
                )
    else:
        os.makedirs(output_dir)
    return output_dir


def _unwrap_singleton(value):
    """Unwrap list/array containers used by pyxdf for scalar metadata."""
    while True:
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        return value


def _metadata_values_for_keys(payload, keys):
    """Recursively collect values for metadata keys from pyxdf structures."""
    values = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in keys:
                values.append(_unwrap_singleton(value))
            values.extend(_metadata_values_for_keys(value, keys))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            values.extend(_metadata_values_for_keys(value, keys))
    elif isinstance(payload, np.ndarray) and payload.dtype == object:
        for value in payload.reshape(-1):
            values.extend(_metadata_values_for_keys(value, keys))
    return values


def _validated_depth_scale(value, source):
    """Validate a metres-per-device-unit depth scale."""
    value = _unwrap_singleton(value)
    try:
        scale = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid depth scale from {source}: expected a number, got {value!r}"
        ) from error
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(
            f"invalid depth scale from {source}: expected a finite positive "
            f"metres-per-unit value, got {value!r}"
        )
    return scale


def _one_consistent_depth_scale(values, source):
    """Return one validated scale, rejecting conflicting metadata."""
    scales = [_validated_depth_scale(value, source) for value in values]
    if not scales:
        return None
    first = scales[0]
    if any(not np.isclose(first, scale, rtol=1e-9, atol=0.0) for scale in scales[1:]):
        raise RuntimeError(
            f"conflicting depth scales in {source}: "
            + ", ".join(f"{scale:.17g}" for scale in scales)
        )
    return first


def _decode_metadata_sample(sample):
    """Decode one JSON metadata sample, returning an empty mapping if unrelated."""
    sample = _unwrap_singleton(sample)
    if isinstance(sample, bytes):
        sample = sample.decode("utf-8")
    if isinstance(sample, str):
        try:
            sample = json.loads(sample)
        except json.JSONDecodeError:
            return {}
    return sample if isinstance(sample, dict) else {}


def _stream_family(name):
    """Return the device-family portion used to pair depth and metadata streams."""
    parts = [part for part in re.split(r"[^a-z0-9]+", name.lower()) if part]
    ignored = {"depth", "metadata", "deviceinfo", "device", "info", "color"}
    return "_".join(part for part in parts if part not in ignored)


def _resolve_depth_scale(depth_stream, all_streams=None):
    """Resolve the recorded hardware scale for one raw depth stream.

    Numeric metadata embedded in the depth stream is preferred. Older NaturalLab
    recordings stored the same value in a paired ``DeviceInfo`` stream, which is
    used only when it can be associated unambiguously.
    """
    stream_name = _stream_name(depth_stream)
    scale_keys = {"depth_scale_m_per_unit", "depth_scale"}
    embedded_values = _metadata_values_for_keys(
        depth_stream.get("info", {}),
        scale_keys,
    )
    embedded_scale = _one_consistent_depth_scale(
        embedded_values,
        f"{stream_name} stream metadata",
    )

    available_streams = list(all_streams or [])
    depth_stream_count = sum(
        _stream_type(candidate).lower() in {"depth", "depthdata"}
        for candidate in available_streams
    )
    metadata_records = []
    for candidate in available_streams:
        candidate_name = _stream_name(candidate)
        if (
            _stream_type(candidate).lower() != "deviceinfo"
            and "metadata" not in candidate_name.lower()
        ):
            continue
        values = []
        for sample in candidate.get("time_series", []):
            values.extend(
                _metadata_values_for_keys(
                    _decode_metadata_sample(sample),
                    scale_keys,
                )
            )
        scale = _one_consistent_depth_scale(
            values,
            f"{candidate_name} metadata stream",
        )
        if scale is not None:
            metadata_records.append((candidate_name, scale))

    depth_family = _stream_family(stream_name)
    paired_records = [
        record
        for record in metadata_records
        if depth_family and _stream_family(record[0]) == depth_family
    ]

    if embedded_scale is not None:
        for metadata_name, metadata_scale in paired_records:
            if not np.isclose(
                embedded_scale,
                metadata_scale,
                rtol=1e-9,
                atol=0.0,
            ):
                raise RuntimeError(
                    f"conflicting depth scales for {stream_name}: stream "
                    f"metadata has {embedded_scale:.17g} m/unit but "
                    f"{metadata_name} has {metadata_scale:.17g} m/unit"
                )
        return embedded_scale, "depth stream metadata"

    candidates = paired_records
    if (
        not candidates
        and len(metadata_records) == 1
        and depth_stream_count == 1
    ):
        candidates = metadata_records
    if candidates:
        scale = _one_consistent_depth_scale(
            [record[1] for record in candidates],
            f"metadata associated with {stream_name}",
        )
        sources = ", ".join(record[0] for record in candidates)
        return scale, f"DeviceInfo stream {sources}"

    if metadata_records:
        names = ", ".join(record[0] for record in metadata_records)
        raise RuntimeError(
            f"cannot associate a recorded depth scale with {stream_name}; "
            f"candidate metadata streams: {names}"
        )
    raise RuntimeError(
        f"no recorded depth scale is available for {stream_name}; metric depth "
        "cannot be produced safely. Record depth_scale_m_per_unit in the depth "
        "stream metadata or a paired DeviceInfo stream."
    )


def _decode_lsl_video_frame(frame_data, frame_index, stream_name):
    """Decode one base64 JPEG frame or fail without changing frame indices."""
    try:
        if isinstance(frame_data, np.ndarray):
            if frame_data.size == 0:
                raise ValueError("empty frame sample")
            frame_str = frame_data.reshape(-1)[0]
        elif isinstance(frame_data, list):
            if not frame_data:
                raise ValueError("empty frame sample")
            frame_str = frame_data[0]
        else:
            frame_str = frame_data
        jpeg_data = base64.b64decode(frame_str, validate=True)
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as error:
        raise RuntimeError(
            f"could not decode frame {frame_index} from {stream_name}: {error}"
        ) from error
    if frame is None:
        raise RuntimeError(
            f"could not decode frame {frame_index} from {stream_name}"
        )
    return frame


def extract_video_stream(stream, output_dir, name=None):
    """Extract a video stream while preserving 1:1 timestamp row alignment."""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting video stream: {stream_name}")

    output_file = os.path.join(output_dir, f"{stream_name}.mp4")
    partial_file = os.path.join(output_dir, f".{stream_name}.partial.mp4")
    timestamps = stream['time_stamps']
    frames_data = stream['time_series']

    if frames_data is None or len(frames_data) == 0:
        raise RuntimeError(f"no video frames found in stream: {stream_name}")
    if len(timestamps) != len(frames_data):
        raise RuntimeError(
            f"video/timestamp length mismatch for {stream_name}: "
            f"{len(frames_data)} frames and {len(timestamps)} timestamps"
        )

    try:
        timestamp_values = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"video timestamps are not numeric for {stream_name}"
        ) from error
    if not np.all(np.isfinite(timestamp_values)):
        raise RuntimeError(
            f"video timestamps are not finite for {stream_name}"
        )
    frame_intervals = np.diff(timestamp_values)
    if len(frame_intervals) and np.any(frame_intervals <= 0):
        raise RuntimeError(
            f"video timestamps are not strictly increasing for {stream_name}"
        )
    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1 / 30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")

    first_frame = _decode_lsl_video_frame(frames_data[0], 0, stream_name)
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(partial_file, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        video_writer.release()
        raise RuntimeError(f"could not create video file for {stream_name}")

    print(f"Processing {len(frames_data)} frames...")
    try:
        for frame_index, frame_data in enumerate(tqdm(frames_data)):
            frame = _decode_lsl_video_frame(
                frame_data,
                frame_index,
                stream_name,
            )
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"frame size changed at frame {frame_index} in "
                    f"{stream_name}"
                )
            video_writer.write(frame)
    except Exception:
        video_writer.release()
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise
    video_writer.release()
    os.replace(partial_file, output_file)

    timestamp_file = os.path.join(output_dir, f"{stream_name}_timestamps.csv")
    timestamp_df = pd.DataFrame({
        'frame_index': range(len(timestamp_values)),
        'timestamp': timestamp_values,
        'timestamp_domain': ['lsl'] * len(timestamp_values),
    })
    timestamp_df.to_csv(timestamp_file, index=False)

    print(f"Video saved to: {output_file}")
    print(f"Timestamps saved to: {timestamp_file}")

def extract_audio_stream(stream, output_dir, name=None):
    """Extract one sample-aligned LSL audio stream to WAV and timestamps."""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting audio stream: {stream_name}")

    output_file = os.path.join(output_dir, f"{stream_name}.wav")
    partial_file = os.path.join(output_dir, f".{stream_name}.partial.wav")
    timestamps = np.asarray(stream['time_stamps'], dtype=np.float64)
    audio_data = stream['time_series']

    if audio_data is None or len(audio_data) == 0:
        raise RuntimeError(f"no audio samples found in stream: {stream_name}")

    sample_rate_value = float(stream['info']['nominal_srate'][0])
    if not np.isfinite(sample_rate_value) or sample_rate_value <= 0:
        raise RuntimeError(
            f"invalid nominal sample rate for {stream_name}: {sample_rate_value!r}"
        )
    sample_rate = int(round(sample_rate_value))
    if not np.isclose(sample_rate, sample_rate_value):
        raise RuntimeError(
            f"non-integral nominal sample rate for {stream_name}: "
            f"{sample_rate_value!r}"
        )
    print(f"Sample rate: {sample_rate} Hz")

    channel_count = int(stream['info']['channel_count'][0])
    if channel_count <= 0:
        raise RuntimeError(
            f"invalid channel count for {stream_name}: {channel_count!r}"
        )
    print(f"Channels: {channel_count}")

    try:
        import soundfile as sf
    except ImportError as error:
        raise RuntimeError(
            "audio extraction requires soundfile; install "
            "NaturalLab's acquisition extra"
        ) from error

    try:
        audio_array = np.asarray(audio_data, dtype=np.float32)
        if audio_array.ndim == 1:
            audio_array = audio_array.reshape(-1, 1)
        if audio_array.ndim != 2 or audio_array.shape[1] != channel_count:
            raise RuntimeError(
                f"audio shape {audio_array.shape!r} does not match the declared "
                f"{channel_count} channel(s)"
            )
        if len(timestamps) != len(audio_array):
            raise RuntimeError(
                f"audio timestamp/sample mismatch for {stream_name}: "
                f"{len(timestamps)} timestamps for {len(audio_array)} samples"
            )
        if not np.all(np.isfinite(audio_array)):
            raise RuntimeError(f"audio contains non-finite samples: {stream_name}")
        if not np.all(np.isfinite(timestamps)):
            raise RuntimeError(f"audio contains non-finite timestamps: {stream_name}")
        if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(
                f"audio timestamps are not strictly increasing: {stream_name}"
            )

        print(f"Audio array shape: {audio_array.shape}")
        sf.write(partial_file, audio_array, sample_rate)
        os.replace(partial_file, output_file)
        print(f"Audio saved to: {output_file}")

        timestamp_file = os.path.join(output_dir, f"{stream_name}_timestamps.csv")
        timestamp_df = pd.DataFrame({
            'sample_index': range(len(timestamps)),
            'timestamp': timestamps,
            'timestamp_domain': ['lsl'] * len(timestamps),
        })
        timestamp_df.to_csv(timestamp_file, index=False)
        print(f"Timestamps saved to: {timestamp_file}")
        return output_file
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise

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
            
        # Reset frame_index for API format
        if format_type == "API" and 'frame_index' in df.columns:
            if not pd.isna(df['frame_index']).all():  # Make sure frame_index column has valid data
                # Reset frame index to start from 0
                if len(df) > 0 and not pd.isna(df['frame_index'].iloc[0]):
                    first_frame = df['frame_index'].iloc[0]
                    df['original_frame_index'] = df['frame_index'].copy()  # Preserve original
                    df['frame_index'] = df['frame_index'] - first_frame
                    print(f"Reset frame_index to start at 0 (original first frame: {first_frame})")
        
        timestamp_values = np.asarray(timestamps, dtype=np.float64)
        if len(timestamp_values) != len(df):
            raise RuntimeError(
                f"gaze timestamp/sample mismatch for {stream_name}: "
                f"{len(timestamp_values)} timestamps for {len(df)} samples"
            )
        if not np.all(np.isfinite(timestamp_values)):
            raise RuntimeError(f"gaze contains non-finite timestamps: {stream_name}")
        df['timestamp'] = timestamp_values
        df['timestamp_domain'] = 'lsl'
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
        
    except Exception:
        try:
            os.remove(output_file)
        except FileNotFoundError:
            pass
        raise

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

def extract_depth_stream(
    stream,
    output_dir,
    name=None,
    save_interval=30,
    include_csv=False,
    depth_scale_m_per_unit=None,
    depth_scale_source=None,
):
    """Extract raw depth plus metric derivatives using a recorded sensor scale."""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting depth stream: {stream_name}")

    # Extract timestamps and frame data
    timestamps = stream['time_stamps']
    frames_data = stream['time_series']

    if frames_data is None or len(frames_data) == 0:
        print(f"No data found in depth stream: {stream_name}")
        return
    if save_interval <= 0:
        raise ValueError("depth save interval must be a positive integer")

    if depth_scale_m_per_unit is None:
        depth_scale, resolved_source = _resolve_depth_scale(stream, [stream])
        depth_scale_source = depth_scale_source or resolved_source
    else:
        depth_scale = _validated_depth_scale(
            depth_scale_m_per_unit,
            depth_scale_source or f"{stream_name} extraction metadata",
        )
        depth_scale_source = depth_scale_source or "extraction metadata"
    print(
        "Depth scale: "
        f"{depth_scale!r} metres per raw device unit "
        f"(source: {depth_scale_source})"
    )

    # Create outputs only after metric conversion has been established.
    depth_dir = os.path.join(output_dir, f"{stream_name}_depth")
    os.makedirs(depth_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{stream_name}_visualization.mp4")

    # Determine frame rate from timestamps
    frame_intervals = np.diff(timestamps)
    avg_interval = np.mean(frame_intervals) if len(frame_intervals) > 0 else 1/30
    fps = 1.0 / avg_interval if avg_interval > 0 else 30
    print(f"Estimated frame rate: {fps:.2f} FPS")

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
    depth_metadata_file = os.path.join(
        output_dir,
        f"{stream_name}_depth_metadata.json",
    )
    depth_metadata = {
        "stream_name": stream_name,
        "raw_encoding": f"{first_raw_depth.dtype} PNG",
        "raw_value_unit": "device_depth_unit",
        "depth_scale_m_per_unit": depth_scale,
        "depth_scale_source": depth_scale_source,
        "metric_distance_unit": "metre",
        "distance_csv_unit": "metre" if include_csv else None,
    }
    with open(depth_metadata_file, "w", encoding="utf-8") as file_handle:
        json.dump(depth_metadata, file_handle, indent=2)
        file_handle.write("\n")

    print(f"Depth metadata saved to: {depth_metadata_file}")
    print(
        "Raw depth PNG values are device units; multiply by "
        f"{depth_scale!r} metres per unit for metric distances"
    )
    return depth_metadata


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
            
def extract_imu_stream(stream, output_dir, name=None, output_filename=None):
    """Extract one validated IMU stream without changing its LSL clock."""
    stream_name = name or stream['info']['name'][0]
    print(f"Extracting IMU stream: {stream_name}")

    # A direct/single-stream call keeps the historical ``imu.csv`` contract.
    output_filename = output_filename or "imu.csv"
    if os.path.basename(output_filename) != output_filename:
        raise ValueError("IMU output filename must not contain a directory")
    output_file = os.path.join(output_dir, output_filename)
    partial_file = os.path.join(output_dir, f".{output_filename}.partial")

    try:
        channel_count = int(stream['info']['channel_count'][0])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid channel count for IMU stream {stream_name}"
        ) from error
    if channel_count <= 0:
        raise RuntimeError(
            f"invalid channel count for IMU stream {stream_name}: "
            f"{channel_count!r}"
        )

    raw_imu_data = stream.get('time_series')
    if raw_imu_data is None or len(raw_imu_data) == 0:
        raise RuntimeError(f"no IMU samples found in stream: {stream_name}")
    try:
        imu_data = np.asarray(raw_imu_data, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"IMU samples are not a rectangular numeric array: {stream_name}"
        ) from error

    try:
        timestamps = np.asarray(stream['time_stamps'], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"IMU timestamps are not a numeric array: {stream_name}"
        ) from error

    # A single recorded sample may be represented as a flat channel vector.
    if imu_data.ndim == 1 and len(timestamps) == 1 and len(imu_data) == channel_count:
        imu_data = imu_data.reshape(1, channel_count)
    if imu_data.ndim != 2:
        raise RuntimeError(
            f"IMU data for {stream_name} must be a sample-by-channel matrix; "
            f"got shape {imu_data.shape!r}"
        )
    if imu_data.shape[1] != channel_count:
        raise RuntimeError(
            f"IMU channel mismatch for {stream_name}: metadata declares "
            f"{channel_count}, data has {imu_data.shape[1]}"
        )
    if timestamps.ndim != 1:
        raise RuntimeError(
            f"IMU timestamps for {stream_name} must be one-dimensional; "
            f"got shape {timestamps.shape!r}"
        )
    if len(timestamps) != len(imu_data):
        raise RuntimeError(
            f"IMU timestamp/sample mismatch for {stream_name}: "
            f"{len(timestamps)} timestamps for {len(imu_data)} samples"
        )
    if not np.all(np.isfinite(imu_data)):
        raise RuntimeError(f"IMU contains non-finite samples: {stream_name}")
    if not np.all(np.isfinite(timestamps)):
        raise RuntimeError(f"IMU contains non-finite timestamps: {stream_name}")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise RuntimeError(
            f"IMU timestamps are not strictly increasing: {stream_name}"
        )

    imu_columns = [
        "gyro_x [deg/s]",
        "gyro_y [deg/s]",
        "gyro_z [deg/s]",
        "accel_x [g]",
        "accel_y [g]",
        "accel_z [g]",
        "roll [deg]",
        "pitch [deg]",
        "yaw [deg]",
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
    ]
    column_names = imu_columns[:channel_count]
    column_names.extend(
        f"extra_{index}" for index in range(len(imu_columns), channel_count)
    )

    sensor_data = pd.DataFrame(imu_data, columns=column_names)
    metadata = pd.DataFrame(
        {
            "section_id": np.ones(len(timestamps), dtype=np.int64),
            "recording_id": np.ones(len(timestamps), dtype=np.int64),
            "timestamp": timestamps,
            "timestamp_domain": ["lsl"] * len(timestamps),
        }
    )
    final_df = pd.concat([metadata, sensor_data], axis=1)

    try:
        final_df.to_csv(partial_file, index=False)
        os.replace(partial_file, output_file)
    except Exception:
        try:
            os.remove(partial_file)
        except FileNotFoundError:
            pass
        raise

    print(f"IMU data saved to: {output_file}")
    return output_file

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
    if pyxdf is None:
        raise RuntimeError(
            "pyxdf is required for XDF extraction; install "
            "naturallab[acquisition]"
        )
    print(f"Loading XDF file: {xdf_file}")
    print(f"Raw depth data will be {'kept' if keep_raw_depth else 'deleted'} after processing")
    print(f"Saving raw depth PNG every {depth_interval} frames")
    
    try:
        # Load XDF file
        streams, fileheader = pyxdf.load_xdf(xdf_file)
        
        # A reused directory can mix files from different sessions or roles.
        output_dir = _prepare_extraction_output_dir(output_dir)
        
        print(f"XDF file loaded. Found {len(streams)} streams:")
        
        # List all streams
        for i, stream in enumerate(streams):
            name = stream['info']['name'][0]
            stream_type = stream['info']['type'][0]
            channel_count = int(stream['info']['channel_count'][0])
            sample_count = len(stream['time_series'])
            print(f"  {i+1}. {name} (Type: {stream_type}, Channels: {channel_count}, Samples: {sample_count})")

        declared_imu_streams = [
            stream for stream in streams if _is_declared_imu_stream(stream)
        ]
        imu_filenames = _plan_imu_output_filenames(declared_imu_streams)
        imu_filename_by_id = {
            id(stream): filename
            for stream, filename in zip(declared_imu_streams, imu_filenames)
        }
        
        # Track any depth raw folders for cleanup if needed
        depth_raw_dirs = []
        
        # Track if we've found certain specialized stream types
        found_imu = False
        found_fixations = False
        found_saccades = False
        extraction_errors = []
        imu_outputs = []
        depth_outputs = []
        
        # First pass - extract normal streams and identify special streams
        for stream in streams:
            name = "<unknown>"
            try:
                name = stream['info']['name'][0]
                stream_type = stream['info']['type'][0]
                
                # Check for specialized streams by name and type
                if _is_declared_imu_stream(stream):
                    found_imu = True
                    output_path = extract_imu_stream(
                        stream,
                        output_dir,
                        output_filename=imu_filename_by_id[id(stream)],
                    )
                    if output_path is not None:
                        imu_outputs.append((name, os.path.basename(output_path)))
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
                    depth_scale, scale_source = _resolve_depth_scale(
                        stream,
                        streams,
                    )
                    depth_result = extract_depth_stream(
                        stream,
                        output_dir,
                        save_interval=depth_interval,
                        include_csv=include_csv,
                        depth_scale_m_per_unit=depth_scale,
                        depth_scale_source=scale_source,
                    )
                    if depth_result is not None:
                        depth_outputs.append(depth_result)
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
                        output_path = extract_imu_stream(stream, output_dir)
                        if output_path is not None:
                            imu_outputs.append(
                                (name, os.path.basename(output_path))
                            )
                    else:
                        # Generic stream extractor for other types
                        extract_generic_stream(stream, output_dir)
                        
            except Exception as stream_error:
                print(f"Error extracting stream {name}: {stream_error}")
                import traceback
                traceback.print_exc()
                print("Continuing with next stream...")
                extraction_errors.append(f"{name}: {stream_error}")
        
        # Second pass - look for specialized streams by contents if not found by name/type
        if not found_imu or not found_fixations or not found_saccades:
            print("\nChecking for specialized streams by content pattern...")
            
            for stream in streams:
                name = "<unknown>"
                try:
                    name = stream['info']['name'][0]
                    stream_type = stream['info']['type'][0]
                    channel_count = int(stream['info']['channel_count'][0])
                    
                    # Skip already processed specialized streams
                    if (_is_declared_imu_stream(stream) or
                        (name.lower() in ['neonfixations', 'neon_fixations', 'fixations']) or
                        (name.lower() in ['neonsaccades', 'neon_saccades', 'saccades'])):
                        continue
                    
                    # Try to detect specialized streams by channel count and patterns
                    if not found_imu and channel_count >= 9 and channel_count <= 13:
                        # IMU typically has 9-13 channels (gyro xyz, accel xyz, quaternion wxyz, optional euler angles)
                        print(f"Stream '{name}' looks like it might contain IMU data (has {channel_count} channels)")
                        if input("Extract as IMU data? (y/n): ").lower().startswith('y'):
                            output_path = extract_imu_stream(stream, output_dir)
                            if output_path is not None:
                                imu_outputs.append(
                                    (name, os.path.basename(output_path))
                                )
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
                    extraction_errors.append(
                        f"specialized detection for {name}: {detect_error}"
                    )
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
                        extraction_errors.append(
                            f"raw-depth cleanup for {raw_dir}: {e}"
                        )
        else:
            print("Keeping all raw depth data for future measurement purposes")
        
        # Print summary of specialized streams
        print("\nSpecialized streams extraction summary:")
        if imu_outputs:
            imu_summary = ", ".join(
                f"{stream_name} -> {filename}"
                for stream_name, filename in imu_outputs
            )
            print(f"- IMU data: {len(imu_outputs)} stream(s) extracted: {imu_summary}")
        elif found_imu:
            print("- IMU data: Found, but no CSV was produced")
        else:
            print("- IMU data: Not found")
        if depth_outputs:
            depth_summary = ", ".join(
                f"{result['stream_name']} "
                f"({result['depth_scale_m_per_unit']!r} m/unit)"
                for result in depth_outputs
            )
            print(
                f"- Depth data: {len(depth_outputs)} stream(s) extracted: "
                f"{depth_summary}"
            )
        else:
            print("- Depth data: Not found or not safely extractable")
        print(f"- Fixations data: {'Extracted' if found_fixations else 'Not found'}")
        print(f"- Saccades data: {'Extracted' if found_saccades else 'Not found'}")
        
        if extraction_errors:
            summary = "; ".join(extraction_errors[:5])
            if len(extraction_errors) > 5:
                summary += f"; and {len(extraction_errors) - 5} more"
            raise RuntimeError(
                f"XDF extraction was incomplete ({len(extraction_errors)} "
                f"error(s)): {summary}"
            )

        print(f"\nAll streams extracted to: {output_dir}")
        
    except Exception as e:
        print(f"Error extracting XDF file: {e}")
        import traceback
        traceback.print_exc()
        raise

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
    try:
        extract_streams(
            args.file,
            args.outdir,
            keep_raw_depth=not args.no_raw_depth,
            depth_interval=args.depth_interval,
            include_csv=args.include_csv,
        )
    except Exception as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1
    end_time = time.time()
    
    print(f"\nExtraction completed in {end_time - start_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
