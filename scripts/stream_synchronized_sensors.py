#!/usr/bin/env python3
"""
Stream Synchronized Sensors via LSL
====================================

Create Lab Streaming Layer (LSL) streams from multiple sensor sources
with automatic synchronization. Outputs can be recorded using LabRecorder.

Supported Sensors:
- RTSP network cameras (IP cameras, webcams)
- Pupil Labs Neon eye trackers
- Intel RealSense depth cameras
- Custom sensors (via extension)

Use Cases:
- Multi-camera behavioral recording
- Eye tracking studies
- Motion capture with depth sensing
- Synchronized multi-modal data collection
- Remote monitoring and recording

Example Usage:
    # Stream from RTSP cameras
    python stream_synchronized_sensors.py \\
        --cameras "rtsp://user:pass@camera1/stream,rtsp://user:pass@camera2/stream" \\
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
import sys
import time
import threading
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Global flag for clean shutdown
running = True


def signal_handler(signum, frame):
    global running
    print("\nShutting down streams...")
    running = False


def stream_rtsp_camera(url, stream_name, quality=75):
    """Stream RTSP camera to LSL."""
    import cv2
    import base64
    import pylsl
    
    print(f"[{stream_name}] Starting RTSP stream: {url}")
    
    # Create LSL outlet
    info = pylsl.StreamInfo(
        name=stream_name,
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id=f"camera_{stream_name.lower().replace(' ', '_')}"
    )
    outlet = pylsl.StreamOutlet(info)
    
    # Open camera
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[{stream_name}] ERROR: Could not open camera")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[{stream_name}] Resolution: {width}x{height}")
    
    frame_count = 0
    start_time = time.time()
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print(f"[{stream_name}] Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(url)
            continue
        
        # Encode frame
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        b64_frame = base64.b64encode(jpeg.tobytes()).decode("utf-8")
        
        # Push to LSL
        outlet.push_sample([b64_frame], time.time())
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"[{stream_name}] {frame_count} frames ({fps:.1f} FPS)")
    
    cap.release()
    print(f"[{stream_name}] Stopped")


def stream_neon_device(ip_address, device_name):
    """Stream Pupil Labs Neon to LSL."""
    import pylsl
    
    try:
        from pupil_labs.realtime_api.simple import Device
    except ImportError:
        print(f"[{device_name}] ERROR: pupil-labs-realtime-api not installed")
        return
    
    print(f"[{device_name}] Connecting to Neon at {ip_address}...")
    
    try:
        device = Device(address=ip_address, port="8080")
    except Exception as e:
        print(f"[{device_name}] ERROR: Could not connect: {e}")
        return
    
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
    start_time = time.time()
    
    while running:
        try:
            scene, gaze = device.receive_matched_scene_video_frame_and_gaze()
            current_time = time.time()
            
            # Push gaze data
            gaze_data = [
                float(frame_count),
                float(gaze.x),
                float(gaze.y),
                float(gaze.pupil_diameter_left or 0),
                float(gaze.pupil_diameter_right or 0)
            ]
            gaze_outlet.push_sample(gaze_data, current_time)
            
            # Push video frame
            _, jpeg = cv2.imencode(".jpg", scene.bgr_pixels, [cv2.IMWRITE_JPEG_QUALITY, 75])
            b64_frame = base64.b64encode(jpeg.tobytes()).decode("utf-8")
            video_outlet.push_sample([b64_frame], current_time)
            
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = current_time - start_time
                fps = frame_count / elapsed
                print(f"[{device_name}] {frame_count} frames ({fps:.1f} FPS)")
                
        except Exception as e:
            print(f"[{device_name}] Error: {e}")
            time.sleep(0.1)
    
    device.close()
    print(f"[{device_name}] Stopped")


def stream_realsense(device_name="RealSense"):
    """Stream Intel RealSense to LSL."""
    import pylsl
    import base64
    
    try:
        import pyrealsense2 as rs
    except ImportError:
        print(f"[{device_name}] ERROR: pyrealsense2 not installed")
        return
    
    print(f"[{device_name}] Initializing...")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    try:
        pipeline.start(config)
    except Exception as e:
        print(f"[{device_name}] ERROR: Could not start: {e}")
        return
    
    # Create color stream
    color_info = pylsl.StreamInfo(
        name=f"{device_name}_Color",
        type="VideoStream",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id=f"realsense_color"
    )
    color_outlet = pylsl.StreamOutlet(color_info)
    
    # Create depth stream
    depth_info = pylsl.StreamInfo(
        name=f"{device_name}_Depth",
        type="Depth",
        channel_count=1,
        nominal_srate=30,
        channel_format="string",
        source_id=f"realsense_depth"
    )
    depth_outlet = pylsl.StreamOutlet(depth_info)
    
    print(f"[{device_name}] Streaming...")
    
    import cv2
    import numpy as np
    
    frame_count = 0
    start_time = time.time()
    
    while running:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        current_time = time.time()
        
        # Color frame
        color_image = np.asanyarray(color_frame.get_data())
        _, jpeg = cv2.imencode(".jpg", color_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        color_outlet.push_sample([base64.b64encode(jpeg.tobytes()).decode()], current_time)
        
        # Depth frame (PNG to preserve 16-bit)
        depth_image = np.asanyarray(depth_frame.get_data())
        _, png = cv2.imencode(".png", depth_image)
        depth_outlet.push_sample([base64.b64encode(png.tobytes()).decode()], current_time)
        
        frame_count += 1
        if frame_count % 100 == 0:
            elapsed = current_time - start_time
            fps = frame_count / elapsed
            print(f"[{device_name}] {frame_count} frames ({fps:.1f} FPS)")
    
    pipeline.stop()
    print(f"[{device_name}] Stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Stream synchronized sensor data via LSL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Camera options
    parser.add_argument("--cameras", type=str,
                       help="Comma-separated RTSP camera URLs")
    parser.add_argument("--camera-names", type=str,
                       help="Comma-separated camera names (must match --cameras)")
    parser.add_argument("--camera-quality", type=int, default=75,
                       help="JPEG quality for camera streams (default: 75)")
    
    # Neon options
    parser.add_argument("--neon-ips", type=str,
                       help="Comma-separated Neon device IPs")
    parser.add_argument("--neon-names", type=str,
                       help="Comma-separated Neon device names")
    
    # RealSense options
    parser.add_argument("--realsense", action="store_true",
                       help="Enable RealSense depth camera")
    
    args = parser.parse_args()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    threads = []
    
    print("=" * 60)
    print("NaturalLab - Synchronized Sensor Streaming")
    print("=" * 60)
    print()
    
    # Start camera streams
    if args.cameras:
        urls = [u.strip() for u in args.cameras.split(",")]
        names = [f"Camera{i+1}" for i in range(len(urls))]
        if args.camera_names:
            names = [n.strip() for n in args.camera_names.split(",")]
        
        for url, name in zip(urls, names):
            t = threading.Thread(target=stream_rtsp_camera, 
                               args=(url, name, args.camera_quality))
            t.daemon = True
            t.start()
            threads.append(t)
    
    # Start Neon streams
    if args.neon_ips:
        ips = [ip.strip() for ip in args.neon_ips.split(",")]
        names = [f"Neon{i+1}" for i in range(len(ips))]
        if args.neon_names:
            names = [n.strip() for n in args.neon_names.split(",")]
        
        for ip, name in zip(ips, names):
            t = threading.Thread(target=stream_neon_device, args=(ip, name))
            t.daemon = True
            t.start()
            threads.append(t)
    
    # Start RealSense stream
    if args.realsense:
        t = threading.Thread(target=stream_realsense)
        t.daemon = True
        t.start()
        threads.append(t)
    
    if not threads:
        print("No sensors configured. Use --help for options.")
        return 1
    
    print()
    print("=" * 60)
    print("Streams active! Use LabRecorder to record to XDF.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    # Wait for threads
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    print("\nWaiting for streams to close...")
    for t in threads:
        t.join(timeout=2)
    
    print("All streams stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
