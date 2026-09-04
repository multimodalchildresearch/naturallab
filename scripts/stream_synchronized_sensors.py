#!/usr/bin/env python3
"""
Stream Multiple Sensors via LSL
================================

Create Lab Streaming Layer (LSL) streams from multiple sensor sources. Every
requested source must publish a first sample before startup succeeds. If a
source fails or stops afterward, the process stops with a nonzero status rather
than silently recording a partial sensor set.

Samples use host-arrival timestamps in LSL's local-clock domain and can be
recorded using LabRecorder. These timestamps are not camera exposure
timestamps; measure capture offset and drift before making cross-device timing
claims.

Supported Sensors:
- RTSP network cameras
- Pupil Labs Neon matched scene-video and gaze streams
- One Intel RealSense colour/raw-depth stream with recorded hardware scale

This source-checkout helper does not stream Neon audio, IMU, or eye events. Use
the installed ``naturallab record`` workflow when those streams are required.

Example Usage:
    # Stream from RTSP cameras
    python stream_synchronized_sensors.py \\
        --cameras "rtsp://camera1/stream,rtsp://camera2/stream" \\
        --camera-names "Front,Side"
    
    # Stream from Neon eye trackers
    python stream_synchronized_sensors.py \\
        --neon-ips "192.168.1.10,192.168.1.11" \\
        --neon-names "Participant1,Participant2"
    
    # Combined setup
    python stream_synchronized_sensors.py \\
        --cameras "rtsp://192.168.1.100/stream" \\
        --neon-ips "192.168.1.10"
"""

import argparse
import base64
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Global flag for clean shutdown
running = True


class WorkerStatus:
    """Thread-safe startup and failure state for one requested source."""

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

    def mark_started(self):
        self.started.set()

    def mark_failed(self, message):
        with self._lock:
            if self._failure is None:
                self._failure = str(message)


def _mark_started(worker_status):
    if worker_status is not None:
        worker_status.mark_started()


def _optional_float(value):
    """Preserve a real zero while representing an unavailable value as NaN."""

    return float("nan") if value is None else float(value)


def _run_worker(worker_status, target, arguments):
    """Run one source and convert every unexpected stop into a failure."""

    try:
        target(*arguments, worker_status=worker_status)
    except Exception as error:
        worker_status.mark_failed(f"{type(error).__name__}: {error}")
    finally:
        if not worker_status.started.is_set() and worker_status.failure is None:
            worker_status.mark_failed("stopped before publishing its first sample")
        elif running and worker_status.failure is None:
            worker_status.mark_failed("stopped unexpectedly")
        worker_status.finished.set()


def _launch_worker(name, target, *arguments):
    status = WorkerStatus(name)
    thread = threading.Thread(
        target=_run_worker,
        args=(status, target, arguments),
        daemon=True,
        name=f"naturallab-{name}",
    )
    thread.start()
    return status, thread


def _worker_failures(workers):
    return [status for status, _thread in workers if status.failure is not None]


def _stop_workers(workers):
    global running

    running = False
    deadline = time.monotonic() + 2.0
    for _status, thread in workers:
        remaining = max(0.0, deadline - time.monotonic())
        thread.join(timeout=remaining)


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
                        f"did not publish a first sample within {timeout_seconds:g} seconds"
                    )
            return _worker_failures(workers)
        time.sleep(0.05)
    return []


def _print_worker_failures(failures):
    for status in failures:
        print(f"ERROR: requested source {status.name!r}: {status.failure}", file=sys.stderr)


def parse_comma_separated(parser, value, option):
    """Parse one non-empty comma-separated CLI option."""
    if value is None:
        return []
    items = [item.strip() for item in value.split(",")]
    if not items or any(not item for item in items):
        parser.error(f"{option} must not contain empty entries")
    return items


def signal_handler(signum, frame):
    global running
    print("\nShutting down streams...")
    running = False


def stream_rtsp_camera(url, stream_name, quality=75, worker_status=None):
    """Stream RTSP camera to LSL."""
    import cv2
    import pylsl
    
    print(f"[{stream_name}] Starting RTSP stream (URL hidden)")
    
    # Open camera
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("could not open the RTSP camera")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    nominal_srate = (
        reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else 0.0
    )

    # Zero denotes irregular or unknown sampling in LSL. Do not invent a frame
    # rate when an RTSP backend cannot report one.
    info = pylsl.StreamInfo(
        name=stream_name,
        type="VideoStream",
        channel_count=1,
        nominal_srate=nominal_srate,
        channel_format="string",
        source_id=f"camera_{stream_name.lower().replace(' ', '_')}"
    )
    info.desc().append_child_value("resolution", f"{width}x{height}")
    outlet = pylsl.StreamOutlet(info)

    rate_label = f"{nominal_srate:g} FPS" if nominal_srate else "unknown FPS"
    print(f"[{stream_name}] Resolution: {width}x{height}; {rate_label}")
    
    frame_count = 0
    start_time = time.monotonic()
    
    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print(f"[{stream_name}] Reconnecting...")
                cap.release()
                time.sleep(2)
                if not running:
                    break
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    raise RuntimeError(
                        "the RTSP source stopped and reconnection failed"
                    )
                continue

            # Encode frame
            encoded, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if not encoded:
                raise RuntimeError("could not encode an RTSP frame as JPEG")
            b64_frame = base64.b64encode(jpeg.tobytes()).decode("utf-8")

            # Host-arrival/post-decode time in LSL's domain, not exposure time.
            outlet.push_sample([b64_frame], pylsl.local_clock())
            if frame_count == 0:
                _mark_started(worker_status)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed
                print(f"[{stream_name}] {frame_count} frames ({fps:.1f} FPS)")
    finally:
        cap.release()

    print(f"[{stream_name}] Stopped")


def stream_neon_device(ip_address, device_name, worker_status=None):
    """Stream Pupil Labs Neon to LSL."""
    import pylsl
    
    try:
        from pupil_labs.realtime_api.simple import Device
    except ImportError as error:
        raise RuntimeError(
            "pupil-labs-realtime-api is not installed"
        ) from error
    
    print(f"[{device_name}] Connecting to Neon at {ip_address}...")
    
    try:
        device = Device(address=ip_address, port="8080")
    except Exception as e:
        raise RuntimeError(f"could not connect to the Neon device: {e}") from e
    
    # Create gaze stream
    gaze_info = pylsl.StreamInfo(
        name=f"Gaze_{device_name}",
        type="Gaze",
        channel_count=5,
        nominal_srate=30,
        channel_format="float32",
        source_id=f"neon_gaze_{device_name.lower()}"
    )
    gaze_outlet = pylsl.StreamOutlet(gaze_info)
    
    # Create video stream
    video_info = pylsl.StreamInfo(
        name=f"Video_{device_name}",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id=f"neon_video_{device_name.lower()}"
    )
    video_outlet = pylsl.StreamOutlet(video_info)
    
    print(f"[{device_name}] Connected! Streaming...")
    
    import cv2
    import base64
    
    frame_count = 0
    start_time = time.monotonic()
    
    try:
        while running:
            scene, gaze = device.receive_matched_scene_video_frame_and_gaze()
            # Host-arrival time in LSL's domain, not sensor capture time.
            lsl_timestamp = pylsl.local_clock()
            
            # Push gaze data
            gaze_data = [
                float(frame_count),
                float(gaze.x),
                float(gaze.y),
                _optional_float(gaze.pupil_diameter_left),
                _optional_float(gaze.pupil_diameter_right),
            ]
            gaze_outlet.push_sample(gaze_data, lsl_timestamp)
            
            # Push video frame
            encoded, jpeg = cv2.imencode(
                ".jpg",
                scene.bgr_pixels,
                [cv2.IMWRITE_JPEG_QUALITY, 75],
            )
            if not encoded:
                raise RuntimeError("could not encode a Neon scene frame as JPEG")
            b64_frame = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            video_outlet.push_sample([b64_frame], lsl_timestamp)
            if frame_count == 0:
                _mark_started(worker_status)
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed
                print(f"[{device_name}] {frame_count} frames ({fps:.1f} FPS)")
    finally:
        device.close()
    print(f"[{device_name}] Stopped")


def stream_realsense(device_name="RealSense", worker_status=None):
    """Stream Intel RealSense to LSL."""
    import pylsl
    
    try:
        import pyrealsense2 as rs
    except ImportError as error:
        raise RuntimeError("pyrealsense2 is not installed") from error
    
    print(f"[{device_name}] Initializing...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    profile = pipeline.start(config)
    device = profile.get_device()
    depth_scale = float(device.first_depth_sensor().get_depth_scale())
    if not math.isfinite(depth_scale) or depth_scale <= 0:
        pipeline.stop()
        raise RuntimeError(
            "RealSense returned an invalid hardware depth scale: "
            f"{depth_scale!r}"
        )
    device_name_value = device.get_info(rs.camera_info.name)
    serial_number = device.get_info(rs.camera_info.serial_number)
    print(
        f"[{device_name}] Connected to {device_name_value}; "
        f"serial {serial_number}; depth scale {depth_scale} metres per raw unit"
    )
    
    # Create color stream
    color_info = pylsl.StreamInfo(
        name=f"{device_name}_Color",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id="realsense_color"
    )
    color_outlet = pylsl.StreamOutlet(color_info)
    
    # Create depth stream
    depth_info = pylsl.StreamInfo(
        name=f"{device_name}_Depth",
        type="Depth",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id="realsense_depth"
    )
    depth_description = depth_info.desc()
    depth_description.append_child_value("content", "raw_depth")
    depth_description.append_child_value(
        "depth_format",
        "uint16_device_units",
    )
    depth_description.append_child_value(
        "depth_scale_m_per_unit",
        repr(depth_scale),
    )
    depth_description.append_child_value("metric_unit", "metre")
    depth_outlet = pylsl.StreamOutlet(depth_info)

    metadata_info = pylsl.StreamInfo(
        name=f"{device_name}_Metadata",
        type="DeviceInfo",
        channel_count=1,
        nominal_srate=0,
        channel_format="string",
        source_id="realsense_metadata",
    )
    metadata_outlet = pylsl.StreamOutlet(metadata_info)
    metadata_timestamp = pylsl.local_clock()
    metadata_outlet.push_sample(
        [
            json.dumps(
                {
                    "name": device_name_value,
                    "serial": serial_number,
                    "depth_scale": depth_scale,
                    "depth_scale_m_per_unit": depth_scale,
                    "raw_depth_unit": "device_depth_unit",
                    "metric_unit": "metre",
                    "timestamp": metadata_timestamp,
                    "timestamp_clock": "pylsl.local_clock",
                }
            )
        ],
        metadata_timestamp,
    )
    
    print(f"[{device_name}] Streaming...")
    
    import cv2
    import numpy as np
    
    frame_count = 0
    start_time = time.monotonic()
    
    try:
        while running:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Host-arrival time in LSL's domain, not sensor capture time.
            lsl_timestamp = pylsl.local_clock()

            # Color frame
            color_image = np.asanyarray(color_frame.get_data())
            encoded_color, jpeg = cv2.imencode(
                ".jpg",
                color_image,
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            if not encoded_color:
                raise RuntimeError("could not encode a RealSense colour frame")
            color_outlet.push_sample(
                [base64.b64encode(jpeg.tobytes()).decode()],
                lsl_timestamp,
            )

            # Depth frame (PNG to preserve the raw 16-bit device values)
            depth_image = np.asanyarray(depth_frame.get_data())
            encoded_depth, png = cv2.imencode(".png", depth_image)
            if not encoded_depth:
                raise RuntimeError("could not encode a RealSense depth frame")
            depth_outlet.push_sample(
                [base64.b64encode(png.tobytes()).decode()],
                lsl_timestamp,
            )
            if frame_count == 0:
                _mark_started(worker_status)

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed
                print(f"[{device_name}] {frame_count} frames ({fps:.1f} FPS)")
    finally:
        pipeline.stop()
    print(f"[{device_name}] Stopped")


def main():
    global running

    running = True
    parser = argparse.ArgumentParser(
        description="Stream sensor data with host-side timestamps via LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Camera options
    parser.add_argument(
        "--cameras",
        type=str,
        help="Comma-separated RTSP camera URLs",
    )
    parser.add_argument(
        "--camera-names",
        type=str,
        help="Comma-separated camera names (must match --cameras)",
    )
    parser.add_argument(
        "--camera-quality",
        type=int,
        default=75,
        help="JPEG quality for camera streams (default: 75)",
    )
    
    # Neon options
    parser.add_argument(
        "--neon-ips",
        type=str,
        help="Comma-separated Neon device IPs",
    )
    parser.add_argument(
        "--neon-names",
        type=str,
        help="Comma-separated Neon device names",
    )
    
    # RealSense options
    parser.add_argument(
        "--realsense",
        action="store_true",
        help="Enable one RealSense colour/raw-depth source",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=30.0,
        help=(
            "Maximum wait for every requested source to publish its first "
            "sample (default: 30)"
        ),
    )
    
    args = parser.parse_args()

    camera_urls = parse_comma_separated(parser, args.cameras, "--cameras")
    camera_names = parse_comma_separated(
        parser,
        args.camera_names,
        "--camera-names",
    )
    if camera_names and not camera_urls:
        parser.error("--camera-names requires --cameras")
    if camera_names and len(camera_names) != len(camera_urls):
        parser.error("--camera-names must contain one name per camera URL")
    if not 1 <= args.camera_quality <= 100:
        parser.error("--camera-quality must be between 1 and 100")

    neon_ips = parse_comma_separated(parser, args.neon_ips, "--neon-ips")
    neon_names = parse_comma_separated(
        parser,
        args.neon_names,
        "--neon-names",
    )
    if neon_names and not neon_ips:
        parser.error("--neon-names requires --neon-ips")
    if neon_names and len(neon_names) != len(neon_ips):
        parser.error("--neon-names must contain one name per Neon IP")
    if (
        not math.isfinite(args.startup_timeout_seconds)
        or args.startup_timeout_seconds <= 0
    ):
        parser.error("--startup-timeout-seconds must be a positive number")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    workers = []
    
    print("=" * 60)
    print("NaturalLab - Multi-Sensor LSL Streaming")
    print("=" * 60)
    print()
    
    # Start camera streams
    if camera_urls:
        names = [f"Camera{i+1}" for i in range(len(camera_urls))]
        if camera_names:
            names = camera_names
        if len(set(names)) != len(names):
            parser.error("camera names must be unique")
        
        for url, name in zip(camera_urls, names):
            workers.append(
                _launch_worker(
                    f"RTSP camera {name}",
                    stream_rtsp_camera,
                    url,
                    name,
                    args.camera_quality,
                )
            )
    
    # Start Neon streams
    if neon_ips:
        names = [f"Neon{i+1}" for i in range(len(neon_ips))]
        if neon_names:
            names = neon_names
        if len(set(names)) != len(names):
            parser.error("Neon names must be unique")
        
        for ip, name in zip(neon_ips, names):
            workers.append(
                _launch_worker(
                    f"Neon {name}",
                    stream_neon_device,
                    ip,
                    name,
                )
            )
    
    # Start RealSense stream
    if args.realsense:
        workers.append(
            _launch_worker(
                "RealSense",
                stream_realsense,
                "RealSense",
            )
        )
    
    if not workers:
        print("No sensors configured. Use --help for options.", file=sys.stderr)
        return 1

    failures = _wait_for_worker_startup(
        workers,
        args.startup_timeout_seconds,
    )
    if failures:
        _print_worker_failures(failures)
        _stop_workers(workers)
        print("Acquisition did not start; no partial sensor set was accepted.")
        return 1
    if not running:
        _stop_workers(workers)
        print("Acquisition stopped during startup.")
        return 0

    print()
    print("=" * 60)
    print("Every requested stream published data. Open LabRecorder to record XDF.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    exit_code = 0
    try:
        while running:
            failures = _worker_failures(workers)
            if failures:
                _print_worker_failures(failures)
                print(
                    "A requested source stopped; stopping the complete "
                    "acquisition.",
                    file=sys.stderr,
                )
                exit_code = 1
                break
            time.sleep(0.2)
    except KeyboardInterrupt:
        running = False

    print("\nWaiting for streams to close...")
    _stop_workers(workers)
    print("All streams stopped.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
