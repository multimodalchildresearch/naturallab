#!/usr/bin/env python
"""
LSL Streams Creator - Multi-Device Support with Direct IP Connection
--------------------------------------------------------------------
This script creates LSL streams from all sources without recording, supporting multiple Neon devices
with consistent role-based naming (Caregiver/Child) using direct IP addresses for reliable connection.
You can use LabRecorder to record the streams to XDF.

Usage:
    # Basic usage with IP addresses (recommended)
    python lsl_streams.py --caregiver-ip YOUR_IP_ADDRESS --child-ip YOUR_IP_ADDRESS
    
    # With cameras
    python lsl_streams.py --caregiver-ip YOUR_IP_ADDRESS --child-ip YOUR_IP_ADDRESS --rtsp-urls "rtsp://USERNAME:PASSWORD@cam1"
    
    # Auto-discovery fallback (if no IPs specified)
    python lsl_streams.py
"""

import os
import sys
import time
import base64
import argparse
import threading
import subprocess
import queue
import cv2
import numpy as np

import sys, os, platform
print("INTERPRETER:", sys.executable)
print("PYTHONPATH :", os.environ.get("PYTHONPATH", ""))

# Global flag for controlling the streams
running = True

# Ensure necessary packages are installed
try:
    import pylsl
except ImportError:
    print("Installing pylsl...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pylsl"])
    import pylsl

# Try to import PyAV for better audio streaming
try:
    import av
    PYAV_AVAILABLE = True
    print("PyAV library is available.")
except ImportError:
    PYAV_AVAILABLE = False
    print("PyAV library is not available. Install with: pip install av")
    print("Falling back to VLC for audio streaming.")

# Check if RealSense is available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
    print("RealSense library is available.")
except ImportError:
    REALSENSE_AVAILABLE = False
    print("RealSense library is not available. Install with: pip install pyrealsense2")

# Check if Pupil Labs realtime API is available
REALTIME_API_AVAILABLE = False
try:
    from pupil_labs.realtime_api.simple import discover_devices
    REALTIME_API_AVAILABLE = True
    print("Pupil Labs realtime API is available.")
except ImportError:
    print("Pupil Labs realtime API is not available. Install with: pip install pupil-labs-realtime-api")

def find_vlc_path():
    """Find the path to VLC executable"""
    # Common paths for VLC
    vlc_paths = [
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
    ]
    
    # First check common paths
    for path in vlc_paths:
        if os.path.exists(path):
            return path
    
    # Try which/where command
    try:
        if os.name == 'nt':  # Windows
            result = subprocess.run(["where", "vlc"], 
                                   stdout=subprocess.PIPE, text=True, 
                                   stderr=subprocess.PIPE)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
        else:  # Linux/Mac
            result = subprocess.run(["which", "vlc"], 
                                   stdout=subprocess.PIPE, text=True,
                                   stderr=subprocess.PIPE)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception as e:
        print(f"Error finding VLC in PATH: {e}")
    
    return None

#========================= RealSense to LSL =========================#
def stream_realsense_to_lsl():
    """Stream RealSense camera data to LSL with focus on raw depth data"""
    if not REALSENSE_AVAILABLE:
        print("RealSense library is not available. Skipping RealSense streaming.")
        return
        
    print("Starting RealSense to LSL streaming...")
    
    # Create LSL outlets for RealSense data
    color_info = pylsl.StreamInfo(
        name="RealSense_Color",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,  # Default FPS
        channel_format="string",  # Base64 encoded JPEG
        source_id="realsense_color"
    )
    color_outlet = pylsl.StreamOutlet(color_info)
    
    depth_info = pylsl.StreamInfo(
        name="RealSense_Depth",
        type="Depth",
        channel_count=1,
        nominal_srate=30,  # Default FPS
        channel_format="string",  # Base64 encoded PNG containing raw depth values
        source_id="realsense_depth"
    )
    # Add depth stream metadata
    desc = depth_info.desc()
    desc.append_child_value("content", "raw_depth")
    desc.append_child_value("depth_format", "uint16_mm")  # 16-bit millimeter scale
    
    depth_outlet = pylsl.StreamOutlet(depth_info)
    
    metadata_info = pylsl.StreamInfo(
        name="RealSense_Metadata",
        type="DeviceInfo",
        channel_count=1,
        nominal_srate=0,  # Irregular data
        channel_format="string",  # JSON string
        source_id="realsense_metadata"
    )
    metadata_outlet = pylsl.StreamOutlet(metadata_info)
    
    try:
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable color and depth streams
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        # Start the pipeline
        profile = pipeline.start(config)
        
        # Get device info
        device = profile.get_device()
        print(f"Connected to: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
        
        # Get depth scale for converting raw values to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print(f"Depth scale: {depth_scale}")
        
        # Send device metadata to LSL
        import json
        metadata = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            "depth_scale": depth_scale,  # Important for interpreting raw depth values
            "timestamp": time.time()
        }
        metadata_outlet.push_sample([json.dumps(metadata)])
        
        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)
        
        # Main loop - stream frames to LSL
        frame_count = 0
        start_time = time.time()
        
        print("Streaming RealSense frames to LSL...")
        
        while running:
            # Get frameset
            frames = pipeline.wait_for_frames()
            
            # Align frames
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                time.sleep(0.01)
                continue
            
            # Get current time for synchronization
            current_time = time.time()
            
            # Process color frame
            color_image = np.asanyarray(color_frame.get_data())
            
            # Compress and send color frame to LSL
            _, jpeg_color = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            color_base64 = base64.b64encode(jpeg_color.tobytes()).decode("utf-8")
            color_outlet.push_sample([color_base64], current_time)
            
            # Process depth frame - send RAW depth data rather than visualization
            depth_image = np.asanyarray(depth_frame.get_data())  # This is raw 16-bit depth
            
            # Compress raw depth and send to LSL - use PNG to preserve 16-bit values
            # PNG compression works well for depth maps and preserves the full 16-bit range
            _, png_depth = cv2.imencode(".png", depth_image)
            depth_base64 = base64.b64encode(png_depth.tobytes()).decode("utf-8")
            depth_outlet.push_sample([depth_base64], current_time)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"RealSense: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Error in RealSense streaming: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop pipeline
        try:
            pipeline.stop()
        except:
            pass
        
        print("RealSense streaming stopped")

#========================= Neon API to LSL (Multi-Device) =========================#
def stream_neon_api_to_lsl(device, device_id="Device1"):
    """Stream Neon gaze data to LSL using the Pupil Labs Realtime API"""
    if not REALTIME_API_AVAILABLE:
        print(f"Pupil Labs Realtime API is not available. Skipping Neon API streaming for {device_id}.")
        return
    
    print(f"Starting Neon API to LSL streaming for {device_id}...")
    
    # Create LSL outlet for Neon gaze data with device-specific naming
    gaze_info = pylsl.StreamInfo(
        name=f"NeonGaze_{device_id}",
        type="Gaze",
        channel_count=5,  # frame_index, gaze_x, gaze_y, pupil_diameter_left, pupil_diameter_right
        nominal_srate=30,  # Matched to video frame rate, not 200Hz
        channel_format="float32",
        source_id=f"neon_gaze_{device_id.lower()}"
    )
    
    # Add channel information with exact API field names
    channels = gaze_info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("label", "frame_index")
    channels.append_child("channel").append_child_value("label", "gaze_x")
    channels.append_child("channel").append_child_value("label", "gaze_y")
    channels.append_child("channel").append_child_value("label", "pupil_diameter_left")
    channels.append_child("channel").append_child_value("label", "pupil_diameter_right")
    
    gaze_outlet = pylsl.StreamOutlet(gaze_info)
    
    video_info = pylsl.StreamInfo(
        name=f"NeonVideo_{device_id}",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,  # Video frame rate
        channel_format="string",  # Base64 encoded JPEG
        source_id=f"neon_video_{device_id.lower()}"
    )
    video_outlet = pylsl.StreamOutlet(video_info)
    
    try:
        print(f"Connected to {device.phone_name} ({device_id})")
        
        # Main loop - stream Neon data to LSL
        frame_count = 0
        start_time = time.time()
        
        print(f"Streaming Neon data to LSL for {device_id}...")
        
        while running:
            # Get matched scene and gaze data
            scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
            
            # Get current time for synchronization
            current_time = time.time()
            
            # Forward the exact API gaze data to LSL, including frame_index
            gaze_data = [
                frame_count,  # frame_index
                gaze_sample.x,
                gaze_sample.y,
                gaze_sample.pupil_diameter_left,
                gaze_sample.pupil_diameter_right
            ]
            gaze_outlet.push_sample(gaze_data, current_time)
            
            # Process video frame
            frame = scene_sample.bgr_pixels
            
            # Compress and send to LSL
            _, jpeg_frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_str = base64.b64encode(jpeg_frame.tobytes()).decode("utf-8")
            video_outlet.push_sample([jpeg_str], current_time)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Neon API {device_id}: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Error in Neon API streaming for {device_id}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close device
        try:
            device.close()
        except:
            pass
        
        print(f"Neon API streaming stopped for {device_id}")

#========================= Eye Events to LSL (Child Only - Working Version) =========================#
def stream_eye_events_to_lsl():
    """Stream fixations and saccades from Neon to LSL using the Pupil Labs Realtime API
    This is the working version from the original code, simplified for child device only."""
    if not REALTIME_API_AVAILABLE:
        print("Pupil Labs Realtime API is not available. Skipping eye events streaming.")
        return
    
    print("Starting eye events (fixations and saccades) streaming for Child device...")
    
    # Import required modules from Pupil Labs Realtime API
    try:
        import asyncio
        from pupil_labs.realtime_api import Device, Network, receive_eye_events_data
        from pupil_labs.realtime_api.streaming.eye_events import (
            BlinkEventData,
            FixationEventData,
        )
    except ImportError:
        print("Error importing required modules for eye events streaming.")
        return
    
    # Create LSL outlet for fixations (Child device)
    fixation_info = pylsl.StreamInfo(
        name="NeonFixations_Child",
        type="Fixations",
        channel_count=8,
        nominal_srate=0,  # Irregular data
        channel_format="double64",  # double64 for nanosecond precisions on timestamps
        source_id="neon_fixations_child"
    )
    
    # Add channel labels
    channels = fixation_info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("label", "fixation_id")
    channels.append_child("channel").append_child_value("label", "start_timestamp_ns")
    channels.append_child("channel").append_child_value("label", "end_timestamp_ns")
    channels.append_child("channel").append_child_value("label", "duration_ms")
    channels.append_child("channel").append_child_value("label", "fixation_x_px")
    channels.append_child("channel").append_child_value("label", "fixation_y_px")
    channels.append_child("channel").append_child_value("label", "azimuth_deg")
    channels.append_child("channel").append_child_value("label", "elevation_deg")
    
    fixation_outlet = pylsl.StreamOutlet(fixation_info)
    
    # Create LSL outlet for saccades (Child device)
    saccade_info = pylsl.StreamInfo(
        name="NeonSaccades_Child",
        type="Saccades",
        channel_count=8,
        nominal_srate=0,  # Irregular data
        channel_format="double64",
        source_id="neon_saccades_child"
    )
    
    # Add channel labels
    channels = saccade_info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("label", "saccade_id")
    channels.append_child("channel").append_child_value("label", "start_timestamp_ns")
    channels.append_child("channel").append_child_value("label", "end_timestamp_ns")
    channels.append_child("channel").append_child_value("label", "amplitude_deg")
    channels.append_child("channel").append_child_value("label", "amplitude_px")
    channels.append_child("channel").append_child_value("label", "mean_velocity_px_s")
    channels.append_child("channel").append_child_value("label", "peak_velocity_px_s")
    channels.append_child("channel").append_child_value("label", "duration_ms")
    
    saccade_outlet = pylsl.StreamOutlet(saccade_info)
    
    # Create counters for event IDs
    fixation_id = 0
    saccade_id = 0
    
    # Define the async main function for handling eye events
    async def process_eye_events():
        nonlocal fixation_id, saccade_id
        
        print("Connecting to Neon device for eye events...")
        async with Network() as network:
            dev_info = await network.wait_for_new_device(timeout_seconds=5)
            
            if dev_info is None:
                print("No device could be found! Aborting eye events streaming.")
                return
                
            print(f"Found device for eye events: {dev_info.name}")
            
            async with Device.from_discovered_device(dev_info) as device:
                status = await device.get_status()
                sensor_eye_events = status.direct_eye_events_sensor()
                
                if not sensor_eye_events.connected:
                    print(f"Eye events sensor is not connected to {device}")
                    return
                    
                print(f"Connected to eye events sensor at {sensor_eye_events.url}")
                
                restart_on_disconnect = True
                
                # Process eye events as they arrive
                async for eye_event in receive_eye_events_data(
                    sensor_eye_events.url, run_loop=restart_on_disconnect
                ):
                    # Check if we should continue running
                    if not running:
                        break
                        
                    # Get current system time for LSL timestamp
                    current_time = time.time()
                    
                    # Process FixationEventData events
                    if isinstance(eye_event, FixationEventData):
                        # Fixation event (event_type = 1)
                        if eye_event.event_type == 1:
                            # Calculate duration in ms
                            duration_ms = (eye_event.end_time_ns - eye_event.start_time_ns) / 1e6
                            
                            # Create sample data - cast to float64 to ensure compatibility
                            fixation_data = [
                                float(fixation_id),
                                eye_event.start_time_ns,
                                eye_event.end_time_ns,
                                duration_ms,
                                float(eye_event.mean_gaze_x if hasattr(eye_event, 'mean_gaze_x') else 0.0),
                                float(eye_event.mean_gaze_y if hasattr(eye_event, 'mean_gaze_y') else 0.0),
                                float(eye_event.azimuth_deg if hasattr(eye_event, 'azimuth_deg') else 0.0),
                                float(eye_event.elevation_deg if hasattr(eye_event, 'elevation_deg') else 0.0)
                            ]
                        
                            # Send to LSL
                            fixation_outlet.push_sample(fixation_data, current_time)
                            fixation_id += 1
                            
                            if fixation_id % 10 == 0:
                                print(f"Streamed {fixation_id} fixations to LSL")
                        
                        # Saccade event (event_type = 0)
                        elif eye_event.event_type == 0:
                            # Calculate duration in ms
                            duration_ms = (eye_event.end_time_ns - eye_event.start_time_ns) / 1e6
                            
                            # Create sample data - cast to float64 to ensure compatibility
                            saccade_data = [
                                float(saccade_id),
                                eye_event.start_time_ns,
                                eye_event.end_time_ns,
                                float(eye_event.amplitude_angle_deg if hasattr(eye_event, 'amplitude_angle_deg') else 0.0),
                                float(eye_event.amplitude_pixels if hasattr(eye_event, 'amplitude_pixels') else 0.0),
                                float(eye_event.mean_velocity if hasattr(eye_event, 'mean_velocity') else 0.0),
                                float(eye_event.max_velocity if hasattr(eye_event, 'max_velocity') else 0.0),
                                float(duration_ms)
                            ]
                            
                            # Send to LSL
                            saccade_outlet.push_sample(saccade_data, current_time)
                            saccade_id += 1
                            
                            if saccade_id % 10 == 0:
                                print(f"Streamed {saccade_id} saccades to LSL")
    
    # Create a thread to run the asyncio event loop
    def run_async_loop():
        asyncio_thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_thread_loop)
        
        try:
            asyncio_thread_loop.run_until_complete(process_eye_events())
        except Exception as e:
            print(f"Error in eye events streaming: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Eye events streaming stopped, cleaning up resources...")
            try:
                asyncio_thread_loop.close()
            except:
                pass
    
    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=run_async_loop)
    asyncio_thread.daemon = True
    asyncio_thread.start()
    
    return asyncio_thread

#========================= IMU Data to LSL (Multi-Device) =========================#
def stream_imu_to_lsl(device, device_id="Device1"):
    """Stream IMU data from Neon to LSL"""
    if not REALTIME_API_AVAILABLE:
        print(f"Pupil Labs realtime API is not available. Skipping IMU streaming for {device_id}.")
        return
    
    print(f"Starting IMU data streaming for {device_id}...")
    
    # Create LSL outlet for IMU data with device-specific naming
    imu_info = pylsl.StreamInfo(
        name=f"NeonIMU_{device_id}",
        type="IMU",
        channel_count=13,
        nominal_srate=200,
        channel_format="float32",
        source_id=f"neon_imu_{device_id.lower()}"
    )
    
    # Add channel information
    channels = imu_info.desc().append_child("channels")
    channels.append_child("channel").append_child_value("label", "gyro_x")
    channels.append_child("channel").append_child_value("label", "gyro_y")
    channels.append_child("channel").append_child_value("label", "gyro_z")
    channels.append_child("channel").append_child_value("label", "accel_x")
    channels.append_child("channel").append_child_value("label", "accel_y")
    channels.append_child("channel").append_child_value("label", "accel_z")
    channels.append_child("channel").append_child_value("label", "roll")
    channels.append_child("channel").append_child_value("label", "pitch")
    channels.append_child("channel").append_child_value("label", "yaw")
    channels.append_child("channel").append_child_value("label", "quaternion_w")
    channels.append_child("channel").append_child_value("label", "quaternion_x")
    channels.append_child("channel").append_child_value("label", "quaternion_y")
    channels.append_child("channel").append_child_value("label", "quaternion_z")
    
    # Add unit information
    units = imu_info.desc().append_child("units")
    units.append_child("unit").append_child_value("gyro", "deg/s")
    units.append_child("unit").append_child_value("accel", "g")
    units.append_child("unit").append_child_value("angles", "deg")
    units.append_child("unit").append_child_value("quaternion", "normalized")
    
    imu_outlet = pylsl.StreamOutlet(imu_info)
    
    # Function to convert quaternion to Euler angles
    def quaternion_to_euler(w, x, y, z):
        """Convert quaternion to roll, pitch, yaw in degrees"""
        import math
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
            
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Convert to degrees
        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)
    
    try:
        print(f"Connected to {device.phone_name} for IMU data ({device_id})")
        
        # Main loop - stream IMU data to LSL
        packet_count = 0
        start_time = time.time()
        
        print(f"Streaming IMU data to LSL for {device_id}...")
        
        while running:
            # Get IMU data
            imu_sample = device.receive_imu_datum()
            
            # Get current time for synchronization
            current_time = time.time()
            
            if packet_count < 3:
                print(f"IMU data packet {packet_count} ({device_id}): {imu_sample}")
            
            # Extract data from the sample
            gyro_x = float(imu_sample.gyro_data.x)
            gyro_y = float(imu_sample.gyro_data.y)
            gyro_z = float(imu_sample.gyro_data.z)
            
            accel_x = float(imu_sample.accel_data.x)
            accel_y = float(imu_sample.accel_data.y)
            accel_z = float(imu_sample.accel_data.z)
            
            quat_w = float(imu_sample.quaternion.w)
            quat_x = float(imu_sample.quaternion.x)
            quat_y = float(imu_sample.quaternion.y)
            quat_z = float(imu_sample.quaternion.z)
            
            # Calculate Euler angles
            roll, pitch, yaw = quaternion_to_euler(quat_w, quat_x, quat_y, quat_z)
            
            # Create sample data
            imu_data = [
                gyro_x, gyro_y, gyro_z,
                accel_x, accel_y, accel_z,
                roll, pitch, yaw,
                quat_w, quat_x, quat_y, quat_z
            ]
            
            # Send to LSL
            imu_outlet.push_sample(imu_data, current_time)
            
            # Update packet count
            packet_count += 1
            
            # Print status periodically
            if packet_count % 100 == 0:
                elapsed = current_time - start_time
                rate = packet_count / elapsed if elapsed > 0 else 0
                print(f"IMU {device_id}: Streamed {packet_count} packets ({rate:.2f} Hz)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Error in IMU streaming for {device_id}: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Close device
        try:
            device.close()
        except:
            pass
            
        print(f"IMU streaming stopped for {device_id}")

#========================= RTSP Camera to LSL =========================#
def stream_rtsp_to_lsl(rtsp_url, stream_name="Camera"):
    """Stream RTSP camera to LSL"""
    print(f"Starting RTSP camera streaming: {rtsp_url}")
    
    # Create LSL outlet for video stream
    video_info = pylsl.StreamInfo(
        name=f"{stream_name}",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,  # Default FPS
        channel_format="string",  # Base64 encoded JPEG
        source_id=f"rtsp_{stream_name}"
    )
    video_outlet = pylsl.StreamOutlet(video_info)
    
    # Initialize camera
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open camera: {rtsp_url}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default if not specified
    
    # Update stream info with resolution
    video_info.desc().append_child_value("resolution", f"{width}x{height}")
    
    print(f"Streaming {stream_name}: {width}x{height} @ {fps} FPS")
    
    # Main loop - stream frames to LSL
    frame_count = 0
    start_time = time.time()
    
    try:
        while running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"{stream_name}: Video frame read failed. Reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)  # Reconnect on failure
                time.sleep(2)
                continue
            
            # Get current time for synchronization
            current_time = time.time()
            
            # Compress frame to JPEG and encode as base64 string
            _, jpeg_frame = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_str = base64.b64encode(jpeg_frame.tobytes()).decode("utf-8")
            
            # Send to LSL
            video_outlet.push_sample([jpeg_str], current_time)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"{stream_name}: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    except Exception as e:
        print(f"Error in RTSP streaming for {stream_name}: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Release camera
        cap.release()
        print(f"RTSP streaming stopped for {stream_name}")

#========================= Audio Streaming Functions =========================#
def stream_audio_to_lsl_pyav(url, stream_name="NeonAudio"):
    """Stream audio from RTSP to LSL using PyAV (improved method)"""
    print(f"Starting audio streaming from: {url} using PyAV")
    
    # Audio settings - will be updated from the actual stream
    sample_rate = 8000  # Default, will be updated when connected
    channels = 1       # Default, will be updated when connected
    
    # Create LSL outlet for audio
    audio_info = pylsl.StreamInfo(
        name=stream_name,
        type="Audio",
        channel_count=channels,
        nominal_srate=sample_rate,
        channel_format="float32",
        source_id=f"neon_audio_{stream_name.lower().replace(' ', '_')}"
    )
    audio_outlet = pylsl.StreamOutlet(audio_info)
    
    # Create audio queue for processing
    audio_queue = queue.Queue(maxsize=100)
    
    # Function to process audio frames
    def process_audio_frames():
        print("Audio processing thread started")
        try:
            while running:
                try:
                    # Get audio data from queue
                    samples = audio_queue.get(block=True, timeout=1.0)
                    
                    # Push to LSL
                    if samples.ndim > 1 and samples.shape[0] == channels:
                        samples = samples.T
                    
                    # Push each sample to LSL
                    if samples.ndim > 1:
                        audio_outlet.push_chunk(samples)
                    else:
                        for i in range(len(samples)):
                            audio_outlet.push_sample([samples[i]])
                        
                    time.sleep(0.001)
                    
                except queue.Empty:
                    continue
                    
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Audio processing thread ended")
    
    # Start the audio processing thread
    processing_thread = threading.Thread(target=process_audio_frames, daemon=True)
    processing_thread.start()
    
    # Connect to audio stream and capture frames
    try:
        # Open the RTSP stream
        container = av.open(url)
        
        # Get the audio stream
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        
        # Update the stream info with actual properties
        sample_rate = audio_stream.codec_context.sample_rate
        channels = audio_stream.codec_context.channels
        
        print(f"Connected to audio stream: {audio_stream}")
        print(f"Format: {audio_stream.codec_context.format.name}, "
              f"Sample rate: {sample_rate}, "
              f"Channels: {channels}")
        
        # Track frames for status updates
        frame_count = 0
        start_time = time.time()
        
        # Process frames
        for frame in container.decode(audio_stream):
            if not running:
                break
                
            # Convert audio samples to numpy array
            samples = frame.to_ndarray()
            
            # Convert to float32 in range [-1.0, 1.0]
            if samples.dtype == np.uint8:
                samples = (samples.astype(np.float32) - 128) / 128.0
            elif samples.dtype == np.int16:
                samples = samples.astype(np.float32) / 32768.0
            elif samples.dtype == np.int32:
                samples = samples.astype(np.float32) / 2147483648.0
            
            # Add to queue
            try:
                audio_queue.put(samples, block=True, timeout=1.0)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    qsize = audio_queue.qsize()
                    print(f"Audio: {frame_count} frames ({fps:.1f} FPS), Queue: {qsize}/{audio_queue.maxsize}")
                    
                    if qsize > audio_queue.maxsize * 0.8:
                        time.sleep(0.01)
                    
            except queue.Full:
                print("Warning: Audio buffer full, dropping frame")
        
    except Exception as e:
        print(f"Error in audio streaming thread: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Audio streaming stopped")

#========================= Main Function =========================#
def main():
    """Main function"""
    global running
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LSL Streams Creator - Multi-Device Support")
    parser.add_argument("--caregiver-ip", type=str, help="IP address of the caregiver's Neon device (e.g., YOUR_IP_ADDRESS)")
    parser.add_argument("--child-ip", type=str, help="IP address of the child's Neon device (e.g., YOUR_IP_ADDRESS)")
    parser.add_argument("--max-neon-devices", type=int, default=2, help="Maximum number of Neon devices to connect to (default: 2)")
    parser.add_argument("--no-realsense", action="store_true", help="Disable RealSense streaming")
    parser.add_argument("--no-neon", action="store_true", help="Disable Neon streaming")
    parser.add_argument("--no-audio", action="store_true", help="Disable audio streaming")
    parser.add_argument("--no-eye-events", action="store_true", help="Disable eye events (fixations/saccades) streaming")
    parser.add_argument("--no-imu", action="store_true", help="Disable IMU streaming")
    parser.add_argument("--audio-method", type=str, choices=["pyav", "vlc"], default="pyav", 
                       help="Method to use for audio streaming (default: pyav)")
    parser.add_argument("--rtsp-urls", type=str, help="Comma-separated list of additional RTSP camera URLs", default="rtsp://USERNAME:PASSWORD@YOUR_IP_ADDRESS/stream1,rtsp://USERNAME:PASSWORD@YOUR_IP_ADDRESS/stream1,rtsp://USERNAME:PASSWORD@YOUR_IP_ADDRESS/stream1,rtsp://USERNAME:PASSWORD@YOUR_IP_ADDRESS/stream1")
    parser.add_argument("--camera-names", type=str, help="Comma-separated list of camera names (optional, must match number of RTSP URLs)", default="Camera1,Camera2,Camera3,Camera4")
    args = parser.parse_args()
    
    # Store threads
    threads = []
    
    # Start RealSense streaming if enabled
    if not args.no_realsense and REALSENSE_AVAILABLE:
        realsense_thread = threading.Thread(target=stream_realsense_to_lsl)
        realsense_thread.daemon = True
        realsense_thread.start()
        threads.append(realsense_thread)
    
    # Start eye events streaming FIRST (before other Neon streams) - Child only
    if not args.no_eye_events and REALTIME_API_AVAILABLE:
        try:
            print("Starting eye events streaming for Child device...")
            eye_events_thread = stream_eye_events_to_lsl()
            if eye_events_thread:
                threads.append(eye_events_thread)
                print("✓ Eye events streaming started successfully")
        except Exception as e:
            print(f"Failed to start eye events streaming: {e}")
            print("Continuing without eye events streaming")
    
    # Start Neon device streaming if enabled
    if not args.no_neon and REALTIME_API_AVAILABLE:
        try:
            print("Connecting to Neon devices by IP address...")
            
            devices = []
            device_roles = {}
            
            # Import Device class for direct IP connection
            from pupil_labs.realtime_api.simple import Device
            
            # Try to connect to Caregiver device
            if args.caregiver_ip:
                try:
                    print(f"Connecting to Caregiver device at {args.caregiver_ip}...")
                    caregiver_device = Device(address=args.caregiver_ip, port="8080")
                    devices.append(caregiver_device)
                    device_roles[caregiver_device] = "Caregiver"
                    print(f"✓ Connected to Caregiver at {args.caregiver_ip}")
                except Exception as e:
                    print(f"✗ Failed to connect to Caregiver at {args.caregiver_ip}: {e}")
            
            # Try to connect to Child device
            if args.child_ip:
                try:
                    print(f"Connecting to Child device at {args.child_ip}...")
                    child_device = Device(address=args.child_ip, port="8080")
                    devices.append(child_device)
                    device_roles[child_device] = "Child"
                    print(f"✓ Connected to Child at {args.child_ip}")
                except Exception as e:
                    print(f"✗ Failed to connect to Child at {args.child_ip}: {e}")
            
            # If no IPs specified, try discovery as fallback
            if not args.caregiver_ip and not args.child_ip:
                print("No IPs specified, trying discovery as fallback...")
                try:
                    discovered_devices = discover_devices(10)
                    if discovered_devices:
                        print(f"Found {len(discovered_devices)} device(s) via discovery")
                        # Auto-assign roles for discovered devices
                        for i, device in enumerate(discovered_devices[:args.max_neon_devices]):
                            devices.append(device)
                            role = "Caregiver" if i == 0 else "Child" if i == 1 else f"Device{i+1}"
                            device_roles[device] = role
                            device_ip = getattr(device, 'phone_ip', 'unknown')
                            print(f"Auto-assigned {getattr(device, 'phone_name', 'Unknown')} ({device_ip}) as {role}")
                    else:
                        print("No devices found via discovery")
                except Exception as e:
                    print(f"Discovery failed: {e}")
                    print("Please specify device IPs with --caregiver-ip and --child-ip")
            
            # Process connected devices
            if devices and device_roles:
                print(f"Setting up streams for {len(devices)} device(s)...")
                
                for device, role in device_roles.items():
                    device_name = getattr(device, 'phone_name', 'Unknown')
                    device_ip = getattr(device, 'phone_ip', getattr(device, 'address', 'unknown'))
                    print(f"Setting up streams for {device_name} ({device_ip}) as {role}")
                    
                    # Start gaze and video streaming
                    neon_api_thread = threading.Thread(
                        target=stream_neon_api_to_lsl, 
                        args=(device, role)
                    )
                    neon_api_thread.daemon = True
                    neon_api_thread.start()
                    threads.append(neon_api_thread)
                    
                    # Start IMU streaming if enabled
                    if not args.no_imu:
                        try:
                            imu_thread = threading.Thread(
                                target=stream_imu_to_lsl,
                                args=(device, role)
                            )
                            imu_thread.daemon = True
                            imu_thread.start()
                            threads.append(imu_thread)
                        except Exception as e:
                            print(f"Failed to start IMU streaming for {role}: {e}")
                    
                    # Start audio streaming if enabled (for both devices with unique names)
                    if not args.no_audio:
                        device_ip = getattr(device, 'phone_ip', getattr(device, 'address', args.caregiver_ip if role == "Caregiver" else args.child_ip))
                        neon_audio_url = f"rtsp://{device_ip}:8086/?audio=on"
                        
                        if args.audio_method == "pyav" and PYAV_AVAILABLE:
                            audio_thread = threading.Thread(
                                target=stream_audio_to_lsl_pyav, 
                                args=(neon_audio_url, f"NeonAudio_{role}")
                            )
                            audio_thread.daemon = True
                            audio_thread.start()
                            threads.append(audio_thread)
                            print(f"Started audio streaming for {role} at {device_ip}")
                        else:
                            # VLC fallback would go here if implemented
                            print("VLC audio streaming not implemented in this version")
            else:
                print("No Neon devices connected. Continuing with other streams...")
                
        except Exception as e:
            print(f"Error connecting to Neon devices: {e}")
            print("Continuing without Neon devices...")
    
    # Start additional RTSP cameras if specified
    if args.rtsp_urls:
        rtsp_urls = args.rtsp_urls.split(',')
        
        # Parse camera names if provided
        camera_names = []
        if args.camera_names:
            camera_names = [name.strip() for name in args.camera_names.split(',')]
            
            # Check if number of names matches number of URLs
            if len(camera_names) != len(rtsp_urls):
                print(f"Warning: Number of camera names ({len(camera_names)}) doesn't match number of URLs ({len(rtsp_urls)})")
                print("Using default names for missing entries")
        
        for i, url in enumerate(rtsp_urls):
            url = url.strip()
            if url:
                # Use custom name if available, otherwise use default
                if i < len(camera_names) and camera_names[i]:
                    stream_name = camera_names[i]
                else:
                    stream_name = f"Camera{i+1}"
                    
                rtsp_thread = threading.Thread(target=stream_rtsp_to_lsl, args=(url, stream_name))
                rtsp_thread.daemon = True
                rtsp_thread.start()
                threads.append(rtsp_thread)
    
    # Print instructions
    print("\n=== LSL Streams Created (Multi-Device with Eye Events) ===")
    print("1. Open LabRecorder")
    print("2. Click 'Update' to see all streams")
    print("3. Select the streams you want to record")
    print("4. Click 'Start' to begin recording to XDF")
    print("\nStreams are named by role:")
    print("  - NeonGaze_Caregiver, NeonGaze_Child")
    print("  - NeonVideo_Caregiver, NeonVideo_Child")
    print("  - NeonIMU_Caregiver, NeonIMU_Child")
    print("  - NeonFixations_Child (child only)")
    print("  - NeonSaccades_Child (child only)")
    print("  - NeonAudio_Caregiver, NeonAudio_Child")
    print("\nRecommended usage:")
    print("  python lsl_streams.py --caregiver-ip X.X.X.X --child-ip Y.Y.Y.Y")
    print("\nRunning... Press Ctrl+C to stop")
    
    try:
        # Keep running until interrupted
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all streams...")
        running = False
    
    # Wait for all threads to finish
    for thread in threads:
        thread.join(timeout=2)
    
    print("All streams stopped")
    return 0

if __name__ == "__main__":
    sys.exit(main())
