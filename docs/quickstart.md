# Quick Start Guide

Get up and running with NaturalLab in 5 minutes.

## Installation

```bash
# Clone repository
git clone https://github.com/anonymous/naturallab.git
cd naturallab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install package
pip install -e ".[all]"
```

## Download Models

```bash
# YOLOv11 for person detection
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

# OSNet for person re-identification (optional, for better tracking)
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal', 'osnet_ain_x1_0_msmt17.pth')"
```

## Quick Examples

### Track People in Video

```bash
python scripts/track_people_in_video.py \
    --input your_video.mp4 \
    --output results/
```

**Output:**
- `tracks.csv` - Position data for each person, each frame
- `track_statistics.csv` - Summary per person

### Detect Custom Objects

```bash
# Organize reference images: images/category_name/image.jpg

# Create prototypes
python scripts/detect_custom_objects.py create-prototypes \
    --images images/ \
    --output prototypes.h5

# Detect in video
python scripts/detect_custom_objects.py detect \
    --input video.mp4 \
    --prototypes prototypes.h5 \
    --output detections/
```

### Stream Sensors via LSL

```bash
# Start camera streams
python scripts/stream_synchronized_sensors.py \
    --cameras "rtsp://192.168.1.100/stream" \
    --camera-names "MainCamera"

# Then open LabRecorder to record
```

## Common Workflows

### Workflow 1: Basic People Tracking

```
Video → YOLO Detection → DeepSORT Tracking → CSV Output
```

```bash
python scripts/track_people_in_video.py -i video.mp4 -o output/
```

### Workflow 2: Calibrated Distance Measurement

```
Calibration → Video → Tracking → Real-world Positions
```

```bash
# Calibrate once
python scripts/calibrate_camera_system.py intrinsic -v calib.mp4 -o camera.yaml
python scripts/calibrate_camera_system.py floor -v floor.mp4 -c camera.yaml -o floor.yaml

# Track with real distances
python scripts/track_people_in_video.py -i video.mp4 -o output/ \
    --camera-calib camera.yaml --floor-calib floor.yaml
```

### Workflow 3: Multi-Sensor Recording

```
Sensors → LSL Streams → LabRecorder → XDF File → Analysis
```

```bash
# Terminal 1: Start streams
python scripts/stream_synchronized_sensors.py --cameras "rtsp://cam1,rtsp://cam2"

# Use LabRecorder GUI to record to .xdf file
```

## Next Steps

- Read [General Applications](general_applications.md) for detailed use cases
- See [API Documentation](api.md) for programmatic usage
- Check `scripts/` for more example scripts

## Troubleshooting

### CUDA Out of Memory

```bash
# Use CPU instead
python scripts/track_people_in_video.py --device cpu ...

# Or use smaller model
--yolo-model yolov8n.pt  # nano model
```

### Camera Connection Failed

```bash
# Test RTSP URL first
ffplay rtsp://user:pass@ip/stream

# Check network connectivity
ping camera_ip
```

### Low Detection Accuracy

```bash
# Lower confidence threshold
--confidence 0.3

# Use larger model
--yolo-model yolo11x.pt
```
