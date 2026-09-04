#!/usr/bin/env python
"""
LSL Streams Creator - Multi-Device Support
------------------------------------------
This script publishes configured sensors as LSL streams for LabRecorder.
Explicit Neon addresses preserve the configured Caregiver/Child labels. Device
discovery uses neutral Device1/Device2 labels and never guesses participant
roles.
You can use LabRecorder to record the streams to XDF.

Usage:
    # Basic usage with IP addresses (recommended)
    python lsl_streams.py --caregiver-ip YOUR_IP_ADDRESS --child-ip YOUR_IP_ADDRESS
    
    # With cameras
    python lsl_streams.py --caregiver-ip YOUR_IP_ADDRESS --child-ip YOUR_IP_ADDRESS --rtsp-urls "rtsp://camera1/stream1"
    
    # Discovery fallback with neutral labels (if no IPs are specified)
    python lsl_streams.py
"""

import sys
import time
import base64
import argparse
import json
import re
import threading
from dataclasses import dataclass

import cv2
import numpy as np

# Global flag for controlling the streams
running = True

REALSENSE_FRAME_WIDTH = 640
REALSENSE_FRAME_HEIGHT = 480
REALSENSE_FPS = 15
AUDIO_READ_TIMEOUT_SECONDS = 5.0
AUDIO_TIME_ECHO_MEASUREMENTS = 100
SOURCE_STARTUP_TIMEOUT_SECONDS = 30.0


class WorkerStatus:
    """Thread-safe publication and failure state for one required source."""

    def __init__(self, name):
        self.name = name
        self.started = threading.Event()
        self.finished = threading.Event()
        self._failure = None
        self._lock = threading.Lock()

    @property
    def failure(self):
        with self._lock:
            return self._failure

    def mark_published(self):
        self.started.set()

    def mark_failed(self, message):
        with self._lock:
            if self._failure is None:
                self._failure = str(message)


def _mark_worker_published(worker_status):
    if worker_status is not None:
        worker_status.mark_published()


def _optional_float(record, attribute):
    """Represent an unavailable sensor field as missing, never as measured zero."""

    value = getattr(record, attribute, None)
    return float("nan") if value is None else float(value)


def _parse_rtsp_configuration(args, input_stream):
    """Validate RTSP inputs, optionally received over a private stdin pipe."""

    if args.rtsp_config_stdin:
        if args.rtsp_urls.strip() or args.camera_names.strip():
            raise ValueError(
                "--rtsp-config-stdin cannot be combined with --rtsp-urls or "
                "--camera-names"
            )
        line = input_stream.readline(1_000_001)
        if not line or len(line) > 1_000_000:
            raise ValueError(
                "--rtsp-config-stdin requires one JSON line no larger than 1 MB"
            )
        try:
            payload = json.loads(line)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "--rtsp-config-stdin did not contain valid JSON"
            ) from error
        if not isinstance(payload, dict) or set(payload) != {
            "rtsp_urls",
            "camera_names",
        }:
            raise ValueError(
                "--rtsp-config-stdin must contain only rtsp_urls and camera_names"
            )
        raw_urls = payload["rtsp_urls"]
        raw_names = payload["camera_names"]
        if not isinstance(raw_urls, list) or not all(
            isinstance(value, str) for value in raw_urls
        ):
            raise ValueError("rtsp_urls must be a JSON list of strings")
        if not isinstance(raw_names, list) or not all(
            isinstance(value, str) for value in raw_names
        ):
            raise ValueError("camera_names must be a JSON list of strings")
        rtsp_urls = [value.strip() for value in raw_urls]
        camera_names = [value.strip() for value in raw_names]
    else:
        rtsp_urls = (
            [url.strip() for url in args.rtsp_urls.split(",")]
            if args.rtsp_urls.strip()
            else []
        )
        camera_names = (
            [name.strip() for name in args.camera_names.split(",")]
            if args.camera_names.strip()
            else []
        )

    if any(not url for url in rtsp_urls):
        raise ValueError("RTSP camera URLs must not contain empty entries")
    if any(not name for name in camera_names):
        raise ValueError("camera names must not contain empty entries")
    if camera_names and len(camera_names) != len(rtsp_urls):
        raise ValueError("camera names must contain one name per RTSP URL")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("camera names must be unique")
    if rtsp_urls and not camera_names:
        camera_names = [f"Camera{index + 1}" for index in range(len(rtsp_urls))]
    return rtsp_urls, camera_names


def _discovered_neon_label(index):
    """Return a neutral label that makes no participant-role inference."""

    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("discovered Neon index must be a non-negative integer")
    return f"Device{index + 1}"


def _safe_worker_error(error):
    """Render a worker error without exposing RTSP locations or credentials."""

    message = str(error).strip() or type(error).__name__
    return re.sub(
        r"(?i)rtsp://(?:[^@\s,]+@)?[^\s,]+",
        "rtsp://[redacted]",
        message,
    )


def _run_managed_worker(worker_status, target, arguments):
    """Run a required source and retain its failure for the main thread."""

    try:
        target(*arguments, worker_status=worker_status)
    except Exception as error:
        worker_status.mark_failed(
            f"{type(error).__name__}: {_safe_worker_error(error)}"
        )
    finally:
        if not worker_status.started.is_set() and worker_status.failure is None:
            worker_status.mark_failed(
                "stopped before publishing its first sample"
            )
        elif running and worker_status.failure is None:
            worker_status.mark_failed("stopped unexpectedly")
        worker_status.finished.set()


def _launch_managed_worker(name, target, *arguments):
    status = WorkerStatus(name)
    thread = threading.Thread(
        target=_run_managed_worker,
        args=(status, target, arguments),
        daemon=True,
        name=f"naturallab-{name}",
    )
    thread.start()
    return status, thread


def _worker_failures(workers):
    return [status for status, _thread in workers if status.failure is not None]


def _wait_for_worker_startup(workers, timeout_seconds):
    deadline = time.monotonic() + timeout_seconds
    while running:
        failures = _worker_failures(workers)
        if failures:
            return failures
        if all(status.started.is_set() for status, _thread in workers):
            return []
        if time.monotonic() >= deadline:
            for status, _thread in workers:
                if not status.started.is_set():
                    status.mark_failed(
                        "did not publish a first sample within "
                        f"{timeout_seconds:g} seconds"
                    )
            return _worker_failures(workers)
        time.sleep(0.05)
    return []


def _print_worker_failures(failures):
    for status in failures:
        print(
            f"Error: requested source {status.name!r}: {status.failure}",
            file=sys.stderr,
        )


def _stop_threads(threads):
    global running

    running = False
    deadline = time.monotonic() + 2.0
    for thread in threads:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)

try:
    import pylsl
except ImportError:
    pylsl = None

# Check if RealSense is available
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    REALSENSE_AVAILABLE = False

# Check if Pupil Labs realtime API is available
REALTIME_API_AVAILABLE = False
try:
    from pupil_labs.realtime_api.simple import discover_devices
    REALTIME_API_AVAILABLE = True
except ImportError:
    pass

#========================= RealSense to LSL =========================#
def stream_realsense_to_lsl(worker_status=None):
    """Stream RealSense camera data to LSL with focus on raw depth data"""
    if not REALSENSE_AVAILABLE:
        raise RuntimeError(
            "RealSense support is unavailable; install pyrealsense2 or rerun "
            "with --no-realsense"
        )
        
    print("Starting RealSense to LSL streaming...")
    
    # Create LSL outlets for RealSense data
    color_info = pylsl.StreamInfo(
        name="RealSense_Color",
        type="VideoStream",
        channel_count=1,
        nominal_srate=REALSENSE_FPS,
        channel_format="string",  # Base64 encoded JPEG
        source_id="realsense_color"
    )
    color_outlet = pylsl.StreamOutlet(color_info)
    
    metadata_info = pylsl.StreamInfo(
        name="RealSense_Metadata",
        type="DeviceInfo",
        channel_count=1,
        nominal_srate=0,  # Irregular data
        channel_format="string",  # JSON string
        source_id="realsense_metadata"
    )
    metadata_outlet = pylsl.StreamOutlet(metadata_info)
    
    pipeline = None
    try:
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable color and depth streams
        config.enable_stream(
            rs.stream.color,
            REALSENSE_FRAME_WIDTH,
            REALSENSE_FRAME_HEIGHT,
            rs.format.bgr8,
            REALSENSE_FPS,
        )
        config.enable_stream(
            rs.stream.depth,
            REALSENSE_FRAME_WIDTH,
            REALSENSE_FRAME_HEIGHT,
            rs.format.z16,
            REALSENSE_FPS,
        )
        # Start the pipeline
        profile = pipeline.start(config)
        
        # Get device info
        device = profile.get_device()
        print(f"Connected to: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
        
        # Get depth scale for converting raw values to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
        if not np.isfinite(depth_scale) or depth_scale <= 0:
            raise RuntimeError(
                "RealSense returned an invalid depth scale; refusing to stream "
                f"unscaled depth values ({depth_scale!r})"
            )
        print(f"Depth scale: {depth_scale} metres per raw device unit")

        # Publish the measured scale with the depth stream itself. The separate
        # DeviceInfo sample remains for compatibility with existing recordings.
        depth_info = pylsl.StreamInfo(
            name="RealSense_Depth",
            type="Depth",
            channel_count=1,
            nominal_srate=REALSENSE_FPS,
            channel_format="string",  # Base64 PNG containing raw depth values
            source_id="realsense_depth",
        )
        desc = depth_info.desc()
        desc.append_child_value("content", "raw_depth")
        desc.append_child_value("depth_format", "uint16_device_units")
        desc.append_child_value(
            "depth_scale_m_per_unit",
            repr(depth_scale),
        )
        desc.append_child_value("metric_unit", "metre")
        depth_outlet = pylsl.StreamOutlet(depth_info)
        
        # Send device metadata to LSL
        import json
        metadata_timestamp = pylsl.local_clock()
        metadata = {
            "name": device.get_info(rs.camera_info.name),
            "serial": device.get_info(rs.camera_info.serial_number),
            # Keep the old key for readers of existing NaturalLab recordings and
            # add an explicit unit-bearing name for new readers.
            "depth_scale": depth_scale,
            "depth_scale_m_per_unit": depth_scale,
            "raw_depth_unit": "device_depth_unit",
            "metric_unit": "metre",
            "timestamp": metadata_timestamp,
            "timestamp_clock": "pylsl.local_clock",
        }
        metadata_outlet.push_sample([json.dumps(metadata)], metadata_timestamp)
        
        # Create align object to align depth frames to color frames
        align = rs.align(rs.stream.color)
        
        # Main loop - stream frames to LSL
        frame_count = 0
        start_time = time.monotonic()
        
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
            
            # Host-arrival timestamp in LSL's clock domain (not camera exposure time).
            lsl_timestamp = pylsl.local_clock()
            
            # Process color frame
            color_image = np.asanyarray(color_frame.get_data())
            
            # Compress and send color frame to LSL
            encoded_color, jpeg_color = cv2.imencode(
                ".jpg",
                color_image,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if not encoded_color:
                raise RuntimeError("could not encode a RealSense colour frame")
            color_base64 = base64.b64encode(jpeg_color.tobytes()).decode("utf-8")
            color_outlet.push_sample([color_base64], lsl_timestamp)
            
            # Process depth frame - send RAW depth data rather than visualization
            depth_image = np.asanyarray(depth_frame.get_data())  # This is raw 16-bit depth
            
            # Compress raw depth and send to LSL - use PNG to preserve 16-bit values
            # PNG compression works well for depth maps and preserves the full 16-bit range
            encoded_depth, png_depth = cv2.imencode(".png", depth_image)
            if not encoded_depth:
                raise RuntimeError("could not encode a RealSense depth frame")
            depth_base64 = base64.b64encode(png_depth.tobytes()).decode("utf-8")
            depth_outlet.push_sample([depth_base64], lsl_timestamp)
            if frame_count == 0:
                _mark_worker_published(worker_status)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"RealSense: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    finally:
        # Stop pipeline
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        
        print("RealSense streaming stopped")

#========================= Neon API to LSL (Multi-Device) =========================#
def stream_neon_api_to_lsl(device, device_id="Device1", worker_status=None):
    """Stream Neon gaze data to LSL using the Pupil Labs Realtime API"""
    if not REALTIME_API_AVAILABLE:
        raise RuntimeError(
            "Pupil Labs realtime API is unavailable; install the acquisition "
            "dependencies or rerun with --no-neon"
        )
    
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
        start_time = time.monotonic()
        
        print(f"Streaming Neon data to LSL for {device_id}...")
        
        while running:
            # Get matched scene and gaze data
            scene_sample, gaze_sample = device.receive_matched_scene_video_frame_and_gaze()
            
            # Host-arrival timestamp in LSL's clock domain (not sensor capture time).
            lsl_timestamp = pylsl.local_clock()
            
            # Forward the exact API gaze data to LSL, including frame_index
            gaze_data = [
                frame_count,  # frame_index
                gaze_sample.x,
                gaze_sample.y,
                gaze_sample.pupil_diameter_left,
                gaze_sample.pupil_diameter_right
            ]
            gaze_outlet.push_sample(gaze_data, lsl_timestamp)
            
            # Process video frame
            frame = scene_sample.bgr_pixels
            
            # Compress and send to LSL
            encoded, jpeg_frame = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if not encoded:
                raise RuntimeError("could not encode a Neon scene frame")
            jpeg_str = base64.b64encode(jpeg_frame.tobytes()).decode("utf-8")
            video_outlet.push_sample([jpeg_str], lsl_timestamp)
            if frame_count == 0:
                _mark_worker_published(worker_status)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"Neon API {device_id}: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    finally:
        # Close device
        try:
            device.close()
        except Exception:
            pass
        
        print(f"Neon API streaming stopped for {device_id}")

#========================= Eye Events to LSL (Child Only) =========================#
def stream_eye_events_to_lsl(child_ip, worker_status=None):
    """Stream fixations and saccades from the configured child Neon."""
    if not REALTIME_API_AVAILABLE:
        raise RuntimeError(
            "Pupil Labs realtime API is unavailable; install the acquisition "
            "dependencies or rerun with --no-eye-events"
        )
    
    print("Starting eye events (fixations and saccades) streaming for Child device...")
    
    # Import required modules from Pupil Labs Realtime API
    try:
        import asyncio
        from pupil_labs.realtime_api import Device, receive_eye_events_data
        from pupil_labs.realtime_api.streaming.eye_events import FixationEventData
    except ImportError as error:
        raise RuntimeError(
            "Pupil Labs eye-event support is unavailable; upgrade the realtime "
            "API or rerun with --no-eye-events"
        ) from error
    
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
        
        print(f"Connecting to configured Child device at {child_ip} for eye events...")
        async with Device(child_ip, 8080) as device:
                status = await device.get_status()
                sensor_eye_events = status.direct_eye_events_sensor()
                
                if not sensor_eye_events.connected:
                    raise RuntimeError("the Neon eye-event sensor is not connected")
                    
                print(f"Connected to eye events sensor at {sensor_eye_events.url}")
                
                restart_on_disconnect = True
                
                # Process eye events as they arrive
                async for eye_event in receive_eye_events_data(
                    sensor_eye_events.url, run_loop=restart_on_disconnect
                ):
                    # Check if we should continue running
                    if not running:
                        break
                        
                    # Host-arrival time in LSL's domain, not the eye-event sensor time.
                    lsl_timestamp = pylsl.local_clock()
                    
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
                                _optional_float(eye_event, "mean_gaze_x"),
                                _optional_float(eye_event, "mean_gaze_y"),
                                _optional_float(eye_event, "azimuth_deg"),
                                _optional_float(eye_event, "elevation_deg"),
                            ]
                        
                            # Send to LSL
                            fixation_outlet.push_sample(fixation_data, lsl_timestamp)
                            _mark_worker_published(worker_status)
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
                                _optional_float(eye_event, "amplitude_angle_deg"),
                                _optional_float(eye_event, "amplitude_pixels"),
                                _optional_float(eye_event, "mean_velocity"),
                                _optional_float(eye_event, "max_velocity"),
                                float(duration_ms)
                            ]
                            
                            # Send to LSL
                            saccade_outlet.push_sample(saccade_data, lsl_timestamp)
                            _mark_worker_published(worker_status)
                            saccade_id += 1
                            
                            if saccade_id % 10 == 0:
                                print(f"Streamed {saccade_id} saccades to LSL")
    
    # Run the asyncio loop in the managed worker created by ``main``.
    def run_async_loop():
        asyncio_thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_thread_loop)
        
        try:
            asyncio_thread_loop.run_until_complete(process_eye_events())
        finally:
            print("Eye events streaming stopped, cleaning up resources...")
            try:
                asyncio_thread_loop.close()
            except Exception:
                pass

    run_async_loop()

#========================= IMU Data to LSL (Multi-Device) =========================#
def stream_imu_to_lsl(device, device_id="Device1", worker_status=None):
    """Stream IMU data from Neon to LSL"""
    if not REALTIME_API_AVAILABLE:
        raise RuntimeError(
            "Pupil Labs realtime API is unavailable; install the acquisition "
            "dependencies or rerun with --no-imu"
        )
    
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
        start_time = time.monotonic()
        
        print(f"Streaming IMU data to LSL for {device_id}...")
        
        while running:
            # Get IMU data
            imu_sample = device.receive_imu_datum()
            
            # Host-arrival timestamp in LSL's clock domain, not sensor capture time.
            lsl_timestamp = pylsl.local_clock()
            
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
            imu_outlet.push_sample(imu_data, lsl_timestamp)
            if packet_count == 0:
                _mark_worker_published(worker_status)
            
            # Update packet count
            packet_count += 1
            
            # Print status periodically
            if packet_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                rate = packet_count / elapsed if elapsed > 0 else 0
                print(f"IMU {device_id}: Streamed {packet_count} packets ({rate:.2f} Hz)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    finally:
        # Close device
        try:
            device.close()
        except Exception:
            pass
            
        print(f"IMU streaming stopped for {device_id}")

#========================= RTSP Camera to LSL =========================#
def stream_rtsp_to_lsl(rtsp_url, stream_name="Camera", worker_status=None):
    """Stream RTSP camera to LSL"""
    print(f"Starting RTSP camera streaming for {stream_name} (URL hidden)")
    
    # Initialize camera
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("could not open the RTSP camera (URL hidden)")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    nominal_srate = (
        reported_fps if np.isfinite(reported_fps) and reported_fps > 0 else 0.0
    )

    # Create the outlet only after querying the source. Zero means irregular or
    # unknown sampling in LSL; inventing a frame rate would corrupt time models.
    video_info = pylsl.StreamInfo(
        name=f"{stream_name}",
        type="VideoStream",
        channel_count=1,
        nominal_srate=nominal_srate,
        channel_format="string",  # Base64 encoded JPEG
        source_id=f"rtsp_{stream_name}"
    )
    video_outlet = pylsl.StreamOutlet(video_info)
    
    # Update stream info with resolution
    video_info.desc().append_child_value("resolution", f"{width}x{height}")
    
    rate_label = f"{nominal_srate:g} FPS" if nominal_srate else "unknown FPS"
    print(f"Streaming {stream_name}: {width}x{height} @ {rate_label}")
    
    # Main loop - stream frames to LSL
    frame_count = 0
    start_time = time.monotonic()
    
    try:
        while running:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"{stream_name}: Video frame read failed. Reconnecting...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_url)  # Reconnect on failure
                time.sleep(2)
                if not cap.isOpened():
                    raise RuntimeError(
                        "the RTSP source stopped and reconnection failed "
                        "(URL hidden)"
                    )
                continue
            
            # Host-arrival/post-decode time in LSL's domain, not exposure time.
            lsl_timestamp = pylsl.local_clock()
            
            # Compress frame to JPEG and encode as base64 string
            encoded, jpeg_frame = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if not encoded:
                raise RuntimeError("could not encode an RTSP frame as JPEG")
            jpeg_str = base64.b64encode(jpeg_frame.tobytes()).decode("utf-8")
            
            # Send to LSL
            video_outlet.push_sample([jpeg_str], lsl_timestamp)
            if frame_count == 0:
                _mark_worker_published(worker_status)
            
            # Update frame count and print status periodically
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"{stream_name}: {frame_count} frames ({fps:.1f} FPS)")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
            
    finally:
        # Release camera
        cap.release()
        print(f"RTSP streaming stopped for {stream_name}")

#========================= Audio Streaming Functions =========================#
@dataclass
class _PreparedAudioStream:
    """Resources validated before an audio worker is reported as started."""

    device: object
    first_audio_frame: object
    first_samples: np.ndarray
    outlet: object
    sample_rate: int
    channel_count: int
    device_to_client_offset_seconds: float
    lsl_minus_unix_seconds: float
    stream_name: str


def _audio_source_timestamp_seconds(audio_frame):
    """Validate one RTCP-derived timestamp exposed by Pupil Labs."""
    timestamp = getattr(audio_frame, "timestamp_unix_seconds", None)
    try:
        timestamp = float(timestamp)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "the Pupil Labs audio frame has no absolute source timestamp"
        ) from error
    if not np.isfinite(timestamp) or timestamp <= 0:
        raise RuntimeError(
            "the Pupil Labs audio frame has an invalid absolute source timestamp"
        )
    return timestamp


def _audio_sample_rate(frame):
    """Read and validate the rate actually reported by the decoded source."""
    candidate = getattr(frame, "sample_rate", None)
    if candidate is not None and int(candidate) > 0:
        return int(candidate)
    raise RuntimeError("the decoded audio source did not report a sample rate")


def _layout_channel_count(layout):
    if layout is None:
        return None
    channels = getattr(layout, "channels", None)
    if channels is not None:
        try:
            count = len(channels)
        except TypeError:
            count = None
        if count:
            return int(count)
    count = getattr(layout, "nb_channels", None)
    if count is not None and int(count) > 0:
        return int(count)
    return None


def _audio_channel_count(frame):
    """Read and validate the channel count from the decoded frame/codec."""
    candidate = _layout_channel_count(getattr(frame, "layout", None))
    if candidate is not None and int(candidate) > 0:
        return int(candidate)
    raise RuntimeError("the decoded audio source did not report a channel count")


def _normalise_audio_samples(frame, channel_count):
    """Convert planar or packed PyAV audio into LSL sample-major float32."""
    samples = np.asarray(frame.to_ndarray())
    samples_per_channel = int(getattr(frame, "samples", 0) or 0)

    if samples.ndim == 1:
        if samples.size % channel_count:
            raise RuntimeError("audio sample buffer is not divisible by its channels")
        samples = samples.reshape(-1, channel_count)
    elif samples.ndim == 2:
        if samples_per_channel and samples.shape == (
            channel_count,
            samples_per_channel,
        ):
            samples = samples.T
        elif samples_per_channel and samples.shape == (
            1,
            samples_per_channel * channel_count,
        ):
            samples = samples.reshape(samples_per_channel, channel_count)
        elif samples.shape[1] == channel_count:
            pass
        elif samples.shape[0] == channel_count:
            samples = samples.T
        elif samples.size % channel_count == 0:
            samples = samples.reshape(-1, channel_count)
        else:
            raise RuntimeError("unsupported decoded audio buffer shape")
    else:
        raise RuntimeError("unsupported decoded audio buffer dimensions")

    if samples.dtype.kind == "u":
        midpoint = float(np.iinfo(samples.dtype).max + 1) / 2.0
        samples = (samples.astype(np.float32) - midpoint) / midpoint
    elif samples.dtype.kind == "i":
        scale = float(abs(np.iinfo(samples.dtype).min))
        samples = samples.astype(np.float32) / scale
    elif samples.dtype.kind == "f":
        samples = samples.astype(np.float32, copy=False)
    else:
        raise RuntimeError(f"unsupported decoded audio dtype: {samples.dtype}")

    if samples.shape[1] != channel_count or not samples.shape[0]:
        raise RuntimeError("decoded audio does not match the declared channel count")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("decoded audio contains non-finite samples")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _validated_time_echo(device):
    """Measure and validate the Neon-to-client clock mapping."""
    estimate_time_offset = getattr(device, "estimate_time_offset", None)
    if not callable(estimate_time_offset):
        raise RuntimeError(
            "Pupil Labs Time Echo is unavailable; upgrade "
            "pupil-labs-realtime-api and the Companion app"
        )
    estimates = estimate_time_offset(
        number_of_measurements=AUDIO_TIME_ECHO_MEASUREMENTS
    )
    if estimates is None:
        raise RuntimeError(
            "Pupil Labs Time Echo did not produce a clock-offset estimate"
        )

    try:
        offset_mean_ms = float(estimates.time_offset_ms.mean)
        offset_std_ms = float(estimates.time_offset_ms.std)
        roundtrip_mean_ms = float(estimates.roundtrip_duration_ms.mean)
        roundtrip_std_ms = float(estimates.roundtrip_duration_ms.std)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Pupil Labs Time Echo returned invalid statistics") from error
    statistics = (
        offset_mean_ms,
        offset_std_ms,
        roundtrip_mean_ms,
        roundtrip_std_ms,
    )
    if not all(np.isfinite(value) for value in statistics):
        raise RuntimeError("Pupil Labs Time Echo returned non-finite statistics")
    if offset_std_ms < 0 or roundtrip_mean_ms < 0 or roundtrip_std_ms < 0:
        raise RuntimeError("Pupil Labs Time Echo returned impossible statistics")
    return statistics


def _lsl_unix_clock_offset():
    """Map client Unix timestamps into pylsl's local-clock domain."""
    unix_before = time.time()
    lsl_time = pylsl.local_clock()
    unix_after = time.time()
    unix_midpoint = (unix_before + unix_after) / 2.0
    return lsl_time - unix_midpoint, (unix_after - unix_before) / 2.0


def _append_audio_metadata(
    audio_info,
    sample_rate,
    channel_count,
    *,
    offset_mean_ms,
    offset_std_ms,
    roundtrip_mean_ms,
    roundtrip_std_ms,
    clock_anchor_uncertainty_seconds,
):
    description = audio_info.desc()
    description.append_child_value("manufacturer", "Pupil Labs")
    description.append_child_value(
        "source_transport",
        "Pupil Labs Realtime API audio with RTCP absolute timestamps",
    )
    description.append_child_value("sample_rate_hz", str(sample_rate))
    description.append_child_value("channel_count", str(channel_count))
    description.append_child_value(
        "timestamp_mapping",
        "Neon Unix time + Time Echo offset mapped to the LSL local clock; "
        "chunk timestamp identifies the final sample",
    )
    description.append_child_value(
        "time_echo_measurements",
        str(AUDIO_TIME_ECHO_MEASUREMENTS),
    )
    description.append_child_value(
        "time_echo_offset_mean_ms",
        repr(offset_mean_ms),
    )
    description.append_child_value(
        "time_echo_offset_std_ms",
        repr(offset_std_ms),
    )
    description.append_child_value(
        "time_echo_roundtrip_mean_ms",
        repr(roundtrip_mean_ms),
    )
    description.append_child_value(
        "time_echo_roundtrip_std_ms",
        repr(roundtrip_std_ms),
    )
    description.append_child_value(
        "client_clock_anchor_uncertainty_seconds",
        repr(clock_anchor_uncertainty_seconds),
    )
    description.append_child_value(
        "timestamp_limitation",
        "Time Echo is measured at startup; within-session clock drift is not "
        "estimated, so validate high-precision studies with a shared sync event",
    )
    description.append_child_value(
        "dropped_frame_policy",
        "detect from source-timestamp discontinuity and stop acquisition",
    )
    description.append_child_value(
        "timeline_discontinuity_policy",
        "stop stream and report an error",
    )


def _receive_audio_frame(device):
    receive = getattr(device, "receive_audio_frame", None)
    if not callable(receive):
        raise RuntimeError(
            "the installed Pupil Labs realtime API has no timestamped audio "
            "interface; install pupil-labs-realtime-api>=1.7.1"
        )
    audio_frame = receive(timeout_seconds=AUDIO_READ_TIMEOUT_SECONDS)
    if audio_frame is None:
        raise RuntimeError("timed out waiting for a Pupil Labs audio frame")
    return audio_frame


def _prepare_audio_stream(device, stream_name):
    """Validate source timing and format before advertising an LSL outlet."""
    print(f"Validating timestamped Pupil Labs audio for {stream_name}")
    (
        offset_mean_ms,
        offset_std_ms,
        roundtrip_mean_ms,
        roundtrip_std_ms,
    ) = _validated_time_echo(device)
    lsl_minus_unix_seconds, clock_anchor_uncertainty = _lsl_unix_clock_offset()
    first_audio_frame = _receive_audio_frame(device)
    first_frame = getattr(first_audio_frame, "av_frame", None)
    if first_frame is None:
        raise RuntimeError("Pupil Labs audio did not provide a decoded audio frame")
    _audio_source_timestamp_seconds(first_audio_frame)
    sample_rate = _audio_sample_rate(first_frame)
    channel_count = _audio_channel_count(first_frame)
    first_samples = _normalise_audio_samples(first_frame, channel_count)

    audio_info = pylsl.StreamInfo(
        name=stream_name,
        type="Audio",
        channel_count=channel_count,
        nominal_srate=sample_rate,
        channel_format="float32",
        source_id=f"neon_audio_{stream_name.lower().replace(' ', '_')}",
    )
    _append_audio_metadata(
        audio_info,
        sample_rate,
        channel_count,
        offset_mean_ms=offset_mean_ms,
        offset_std_ms=offset_std_ms,
        roundtrip_mean_ms=roundtrip_mean_ms,
        roundtrip_std_ms=roundtrip_std_ms,
        clock_anchor_uncertainty_seconds=clock_anchor_uncertainty,
    )
    audio_outlet = pylsl.StreamOutlet(audio_info)

    audio_format = getattr(getattr(first_frame, "format", None), "name", "unknown")
    print(
        f"Validated {stream_name}: format={audio_format}, "
        f"sample_rate={sample_rate} Hz, channels={channel_count}, "
        f"Time Echo offset={offset_mean_ms:.3f}±{offset_std_ms:.3f} ms"
    )
    return _PreparedAudioStream(
        device=device,
        first_audio_frame=first_audio_frame,
        first_samples=first_samples,
        outlet=audio_outlet,
        sample_rate=sample_rate,
        channel_count=channel_count,
        device_to_client_offset_seconds=offset_mean_ms / 1000.0,
        lsl_minus_unix_seconds=lsl_minus_unix_seconds,
        stream_name=stream_name,
    )


def _push_prepared_audio_stream(prepared):
    """Push source-timestamped audio and stop on any detected frame loss."""
    expected_source_start = None
    frame_count = 0
    sample_count = 0
    start_time = time.monotonic()
    audio_frame = prepared.first_audio_frame
    samples = prepared.first_samples

    while running:
        frame = getattr(audio_frame, "av_frame", None)
        if frame is None:
            raise RuntimeError("Pupil Labs audio frame is missing decoded data")
        if not running:
            break
        frame_rate = _audio_sample_rate(frame)
        frame_channels = _audio_channel_count(frame)
        if frame_rate != prepared.sample_rate:
            raise RuntimeError(
                f"audio sample rate changed from {prepared.sample_rate} to {frame_rate}"
            )
        if frame_channels != prepared.channel_count:
            raise RuntimeError(
                "audio channel count changed from "
                f"{prepared.channel_count} to {frame_channels}"
            )

        source_start = _audio_source_timestamp_seconds(audio_frame)
        if expected_source_start is not None:
            discontinuity = source_start - expected_source_start
            tolerance = max(2.0 / prepared.sample_rate, 0.0001)
            if abs(discontinuity) > tolerance:
                raise RuntimeError(
                    "audio source timeline discontinuity detected: "
                    f"{discontinuity:+.6f} seconds"
                )

        source_last = source_start + ((samples.shape[0] - 1) / prepared.sample_rate)
        client_unix_last = (
            source_last + prepared.device_to_client_offset_seconds
        )
        lsl_timestamp = (
            client_unix_last + prepared.lsl_minus_unix_seconds
        )
        prepared.outlet.push_chunk(samples.tolist(), lsl_timestamp)

        expected_source_start = source_start + (
            samples.shape[0] / prepared.sample_rate
        )
        frame_count += 1
        sample_count += samples.shape[0]
        if frame_count % 100 == 0:
            elapsed = time.monotonic() - start_time
            print(
                f"{prepared.stream_name}: {sample_count} samples from "
                f"{frame_count} frames in {elapsed:.1f} seconds"
            )

        audio_frame = _receive_audio_frame(prepared.device)
        next_frame = getattr(audio_frame, "av_frame", None)
        if next_frame is None:
            raise RuntimeError("Pupil Labs audio frame is missing decoded data")
        samples = _normalise_audio_samples(next_frame, prepared.channel_count)


def _run_prepared_audio_stream(prepared):
    try:
        _push_prepared_audio_stream(prepared)
    except Exception as error:
        print(
            f"Audio stream {prepared.stream_name} stopped with an error: {error}",
            file=sys.stderr,
        )
    finally:
        print(f"Audio streaming stopped for {prepared.stream_name}")


def start_audio_stream_to_lsl(device, stream_name="NeonAudio"):
    """Validate timestamped Neon audio, then start its publish worker."""
    prepared = _prepare_audio_stream(device, stream_name)
    worker = threading.Thread(
        target=_run_prepared_audio_stream,
        args=(prepared,),
        daemon=True,
        name=f"audio-{stream_name}",
    )
    worker.start()
    return worker


def stream_audio_to_lsl(device, stream_name="NeonAudio"):
    """Validate and stream timestamped Neon audio in the current thread."""
    prepared = _prepare_audio_stream(device, stream_name)
    _push_prepared_audio_stream(prepared)


#========================= Main Function =========================#
def main():
    """Start exactly the requested sources or fail the complete acquisition."""
    global running

    running = True

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="LSL Streams Creator - Multi-Device Support"
    )
    parser.add_argument(
        "--caregiver-ip",
        type=str,
        help="IP address of the caregiver's Neon device",
    )
    parser.add_argument(
        "--child-ip",
        type=str,
        help="IP address of the child's Neon device",
    )
    parser.add_argument(
        "--max-neon-devices",
        type=int,
        default=2,
        help="Maximum number of discovered Neon devices (default: 2)",
    )
    parser.add_argument(
        "--no-realsense",
        action="store_true",
        help="Disable RealSense streaming",
    )
    parser.add_argument(
        "--no-neon",
        action="store_true",
        help="Disable Neon streaming",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable audio streaming",
    )
    eye_event_group = parser.add_mutually_exclusive_group()
    eye_event_group.add_argument(
        "--eye-events",
        dest="no_eye_events",
        action="store_false",
        help=(
            "Enable fixation/saccade streaming only after validating the "
            "selected device role"
        ),
    )
    eye_event_group.add_argument(
        "--no-eye-events",
        dest="no_eye_events",
        action="store_true",
        help="Disable fixation/saccade streaming (default)",
    )
    parser.set_defaults(no_eye_events=True)
    parser.add_argument(
        "--no-imu",
        action="store_true",
        help="Disable IMU streaming",
    )
    parser.add_argument(
        "--rtsp-urls",
        type=str,
        help="Comma-separated list of additional RTSP camera URLs",
        default="",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        help=(
            "Comma-separated camera names (optional; exactly one per RTSP URL)"
        ),
        default="",
    )
    parser.add_argument(
        "--rtsp-config-stdin",
        action="store_true",
        help=(
            "Read RTSP URLs and camera names as one JSON line from stdin. "
            "The recording window uses this so credentials do not appear in "
            "process arguments."
        ),
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=SOURCE_STARTUP_TIMEOUT_SECONDS,
        help=(
            "Maximum wait for every requested non-audio source to publish a "
            "first sample (default: 30)"
        ),
    )
    args = parser.parse_args()
    if pylsl is None:
        parser.error(
            "pylsl is required; install the NaturalLab acquisition extra"
        )
    if args.max_neon_devices <= 0:
        parser.error("--max-neon-devices must be positive")
    if (
        not np.isfinite(args.startup_timeout_seconds)
        or args.startup_timeout_seconds <= 0
    ):
        parser.error("--startup-timeout-seconds must be a positive number")
    if not args.no_neon and not args.no_eye_events and not args.child_ip:
        parser.error("--eye-events requires an explicit --child-ip")

    try:
        rtsp_urls, camera_names = _parse_rtsp_configuration(args, sys.stdin)
    except ValueError as error:
        parser.error(str(error))

    threads = []
    managed_workers = []
    audio_workers = []
    audio_start_failures = []
    startup_failures = []
    expected_audio_streams = set()
    if not args.no_neon and not args.no_audio:
        if args.caregiver_ip:
            expected_audio_streams.add("NeonAudio_Caregiver")
        if args.child_ip:
            expected_audio_streams.add("NeonAudio_Child")

    def launch_required(name, target, *arguments):
        worker = _launch_managed_worker(name, target, *arguments)
        managed_workers.append(worker)
        threads.append(worker[1])

    if not args.no_realsense:
        if REALSENSE_AVAILABLE:
            launch_required("RealSense colour and depth", stream_realsense_to_lsl)
        else:
            startup_failures.append(
                (
                    "RealSense colour and depth",
                    "pyrealsense2 is unavailable; install it or rerun with "
                    "--no-realsense",
                )
            )

    if not args.no_neon and not REALTIME_API_AVAILABLE:
        startup_failures.append(
            (
                "Neon",
                "the Pupil Labs realtime API is unavailable; install the "
                "acquisition dependencies or rerun with --no-neon",
            )
        )

    # Eye events use their own realtime-API connection.
    if (
        not args.no_neon
        and not args.no_eye_events
        and REALTIME_API_AVAILABLE
    ):
        launch_required(
            "Neon eye events for Child",
            stream_eye_events_to_lsl,
            args.child_ip,
        )

    if not args.no_neon and REALTIME_API_AVAILABLE:
        try:
            print("Connecting to Neon devices by IP address...")

            devices = []
            device_roles = {}

            # Import Device class for direct IP connection
            from pupil_labs.realtime_api.simple import Device

            if args.caregiver_ip:
                try:
                    print("Connecting to the configured Caregiver Neon...")
                    caregiver_device = Device(address=args.caregiver_ip, port="8080")
                    devices.append(caregiver_device)
                    device_roles[caregiver_device] = "Caregiver"
                    print("Connected to the configured Caregiver Neon")
                except Exception as error:
                    startup_failures.append(
                        (
                            "Neon Caregiver",
                            f"connection failed: {_safe_worker_error(error)}",
                        )
                    )

            if args.child_ip:
                try:
                    print("Connecting to the configured Child Neon...")
                    child_device = Device(address=args.child_ip, port="8080")
                    devices.append(child_device)
                    device_roles[child_device] = "Child"
                    print("Connected to the configured Child Neon")
                except Exception as error:
                    startup_failures.append(
                        (
                            "Neon Child",
                            f"connection failed: {_safe_worker_error(error)}",
                        )
                    )

            if not args.caregiver_ip and not args.child_ip:
                print("No IPs specified, trying discovery as fallback...")
                try:
                    discovered_devices = discover_devices(10)
                    if discovered_devices:
                        print(
                            f"Found {len(discovered_devices)} device(s) via discovery"
                        )
                        for i, device in enumerate(
                            discovered_devices[: args.max_neon_devices]
                        ):
                            devices.append(device)
                            role = _discovered_neon_label(i)
                            device_roles[device] = role
                            print(
                                "Labelled a discovered Neon as "
                                f"{role}; verify the physical device mapping "
                                "before recording"
                            )
                    else:
                        startup_failures.append(
                            ("Neon", "device discovery found no devices")
                        )
                except Exception as error:
                    startup_failures.append(
                        (
                            "Neon",
                            f"device discovery failed: {_safe_worker_error(error)}",
                        )
                    )

            if devices and device_roles:
                print(f"Setting up streams for {len(devices)} device(s)...")

                for device, role in device_roles.items():
                    print(f"Setting up Neon streams for role {role}")
                    launch_required(
                        f"Neon scene and gaze for {role}",
                        stream_neon_api_to_lsl,
                        device,
                        role,
                    )

                    if not args.no_imu:
                        launch_required(
                            f"Neon IMU for {role}",
                            stream_imu_to_lsl,
                            device,
                            role,
                        )

                    if not args.no_audio:
                        stream_name = f"NeonAudio_{role}"
                        expected_audio_streams.add(stream_name)
                        try:
                            audio_thread = start_audio_stream_to_lsl(
                                device,
                                stream_name,
                            )
                            if not audio_thread.is_alive():
                                raise RuntimeError(
                                    "audio worker stopped during startup"
                                )
                        except Exception as error:
                            audio_start_failures.append((stream_name, str(error)))
                            print(
                                f"✗ Audio did not start for {role}: {error}",
                                file=sys.stderr,
                            )
                        else:
                            threads.append(audio_thread)
                            audio_workers.append((stream_name, audio_thread))
                            print(
                                f"Started validated audio streaming for {role}"
                            )
            else:
                startup_failures.append(("Neon", "no Neon device connected"))

        except Exception as error:
            startup_failures.append(
                (
                    "Neon",
                    f"setup failed: {_safe_worker_error(error)}",
                )
            )

    for url, stream_name in zip(rtsp_urls, camera_names):
        launch_required(
            f"RTSP camera {stream_name}",
            stream_rtsp_to_lsl,
            url,
            stream_name,
        )

    active_audio_names = {
        stream_name
        for stream_name, worker in audio_workers
        if worker.is_alive()
    }
    missing_audio_streams = expected_audio_streams - active_audio_names
    if (
        not args.no_neon
        and not args.no_audio
        and not expected_audio_streams
    ):
        audio_start_failures.append(
            ("NeonAudio", "no Neon device connected for requested audio")
        )
        missing_audio_streams.add("NeonAudio")

    if startup_failures:
        for name, message in startup_failures:
            print(
                f"Error: requested source {name!r}: {message}",
                file=sys.stderr,
            )

    if audio_start_failures or missing_audio_streams or startup_failures:
        failed_names = ", ".join(sorted(missing_audio_streams))
        if audio_start_failures or missing_audio_streams:
            print(
                "Error: requested Neon audio failed validation for "
                f"{failed_names}. Fix audio or rerun with --no-audio.",
                file=sys.stderr,
            )
        _stop_threads(threads)
        return 1

    if not threads:
        print(
            "Error: no acquisition source started. Enable at least one "
            "available camera, Neon device, or RealSense device.",
            file=sys.stderr,
        )
        return 1

    failures = _wait_for_worker_startup(
        managed_workers,
        args.startup_timeout_seconds,
    )
    if failures:
        _print_worker_failures(failures)
        _stop_threads(threads)
        print(
            "Error: acquisition did not start; no partial sensor set was accepted.",
            file=sys.stderr,
        )
        return 1

    failed_audio_streams = [
        stream_name
        for stream_name, worker in audio_workers
        if not worker.is_alive()
    ]
    if failed_audio_streams:
        print(
            "Error: requested audio stopped during startup for "
            f"{', '.join(failed_audio_streams)}.",
            file=sys.stderr,
        )
        _stop_threads(threads)
        return 1

    # Print instructions
    print("\n=== Every requested NaturalLab LSL source published data ===")
    print("1. Open LabRecorder")
    print("2. Click 'Update' to see all streams")
    print("3. Select the streams you want to record")
    print("4. Click 'Start' to begin recording to XDF")
    if not args.no_neon:
        active_roles = sorted(set(device_roles.values()))
        print("\nActive Neon stream labels:")
        print("  - " + ", ".join(f"NeonGaze_{role}" for role in active_roles))
        print("  - " + ", ".join(f"NeonVideo_{role}" for role in active_roles))
        if not args.no_imu:
            print("  - " + ", ".join(f"NeonIMU_{role}" for role in active_roles))
        if not args.no_eye_events:
            print("  - NeonFixations_Child (validate device role before use)")
            print("  - NeonSaccades_Child (validate device role before use)")
        active_audio_streams = sorted(
            stream_name
            for stream_name, worker in audio_workers
            if worker.is_alive()
        )
        if active_audio_streams:
            print(f"  - {', '.join(active_audio_streams)}")
        elif not args.no_audio:
            print(
                "  - Audio was requested but no validated NeonAudio stream is active",
                file=sys.stderr,
            )
    print("\nRunning... Press Ctrl+C to stop")

    exit_code = 0
    try:
        while running:
            failures = _worker_failures(managed_workers)
            if failures:
                _print_worker_failures(failures)
                print(
                    "Error: a required source stopped; stopping the complete "
                    "acquisition.",
                    file=sys.stderr,
                )
                exit_code = 1
                running = False
                break
            failed_audio_streams = [
                stream_name
                for stream_name, worker in audio_workers
                if not worker.is_alive()
            ]
            if failed_audio_streams:
                print(
                    "Error: audio streaming stopped unexpectedly for "
                    f"{', '.join(failed_audio_streams)}; stopping acquisition.",
                    file=sys.stderr,
                )
                exit_code = 1
                running = False
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nStopping all streams...")
        running = False
    
    _stop_threads(threads)

    print("All streams stopped")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
