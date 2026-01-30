# General Applications Guide

NaturalLab was designed for developmental research but its components are applicable to many domains. This guide covers how to use the system for various applications.

## Table of Contents
1. [People Tracking](#people-tracking)
2. [Custom Object Detection](#custom-object-detection)
3. [Multi-Sensor Data Acquisition](#multi-sensor-data-acquisition)
4. [Camera Calibration](#camera-calibration)

---

## People Tracking

Track people in video and extract movement metrics without any domain-specific setup.

### Quick Start

```bash
python scripts/track_people_in_video.py \
    --input your_video.mp4 \
    --output results/
```

### Applications

| Domain | Use Case |
|--------|----------|
| **Retail** | Customer flow analysis, dwell time, path optimization |
| **Sports** | Player tracking, formation analysis, distance covered |
| **Security** | Occupancy monitoring, crowd analysis |
| **Healthcare** | Patient mobility assessment, fall detection |
| **Research** | Behavioral observation, interaction analysis |

### Output Files

- `tracks.csv` - Frame-by-frame position data
- `track_statistics.csv` - Per-person summary (duration, distance)
- `identity_matches.json` - Identity assignments (if configured)

### With Real-World Measurements

For actual distance measurements (in meters), calibrate your camera:

```bash
# 1. Calibrate camera intrinsics
python scripts/calibrate_camera_system.py intrinsic \
    --video chessboard_video.mp4 \
    --output camera.yaml

# 2. Calibrate floor plane
python scripts/calibrate_camera_system.py floor \
    --video floor_chessboard_video.mp4 \
    --camera-calib camera.yaml \
    --output floor.yaml

# 3. Track with calibration
python scripts/track_people_in_video.py \
    --input video.mp4 \
    --camera-calib camera.yaml \
    --floor-calib floor.yaml \
    --output results/
```

### With Identity Labeling

Label tracked individuals using natural language descriptions:

```bash
python scripts/track_people_in_video.py \
    --input video.mp4 \
    --output results/ \
    --identities '{
        "Coach": "person wearing red shirt",
        "Player1": "person in white jersey number 10",
        "Player2": "person in white jersey number 7"
    }'
```

---

## Custom Object Detection

Detect your own objects without training a model - just provide reference images.

### Quick Start

```bash
# Step 1: Organize reference images by category
# reference_images/
#   product_A/
#     image1.jpg
#     image2.jpg
#   product_B/
#     image1.jpg
#     ...

# Step 2: Create prototypes
python scripts/detect_custom_objects.py create-prototypes \
    --images reference_images/ \
    --output prototypes.h5

# Step 3: Detect in video/images
python scripts/detect_custom_objects.py detect \
    --input video.mp4 \
    --prototypes prototypes.h5 \
    --output detections/
```

### Applications

| Domain | Use Case |
|--------|----------|
| **Inventory** | Product counting, shelf monitoring |
| **Quality Control** | Defect detection, part verification |
| **Wildlife** | Species identification, animal counting |
| **Research** | Experimental object tracking |
| **Art/Cultural** | Artifact identification, style matching |

### How It Works

1. **First Stage**: OWL-ViT detects general object regions
2. **Second Stage**: CLIP matches detections to your prototypes
3. **Output**: Category labels with confidence scores

### Tips for Good Prototypes

- **Multiple angles**: 3-5 images per object from different viewpoints
- **Consistent lighting**: Match expected deployment conditions
- **Clear backgrounds**: Avoid cluttered prototype images
- **Size variation**: Include close-up and distant views

---

## Multi-Sensor Data Acquisition

Synchronize and record multiple sensors using Lab Streaming Layer (LSL).

### Quick Start

```bash
# Stream from IP cameras
python scripts/stream_synchronized_sensors.py \
    --cameras "rtsp://192.168.1.100/stream,rtsp://192.168.1.101/stream" \
    --camera-names "Front,Side"
```

Then open [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder) to record all streams to XDF format.

### Supported Sensors

| Sensor | Data Types | Sample Rate |
|--------|------------|-------------|
| RTSP Cameras | Video (JPEG) | 30 Hz |
| Pupil Labs Neon | Gaze, Video, IMU | 200/30 Hz |
| Intel RealSense | RGB, Depth | 30 Hz |
| Custom (via API) | Any | Variable |

### Why LSL?

- **Automatic synchronization**: All streams share a common time base
- **Millisecond precision**: Clock sync across devices
- **Standard format**: XDF files work with Python, MATLAB, R
- **Extensible**: Add custom sensors with minimal code

### Recording Workflow

1. Start sensor streams with the script
2. Open LabRecorder and click "Update"
3. Select streams to record
4. Click "Start" to begin recording
5. Click "Stop" when done
6. Process XDF file with your analysis pipeline

### Example: Multi-Camera Setup

```bash
# 4-camera setup with named streams
python scripts/stream_synchronized_sensors.py \
    --cameras "rtsp://user:pass@cam1/stream,rtsp://user:pass@cam2/stream,rtsp://user:pass@cam3/stream,rtsp://user:pass@cam4/stream" \
    --camera-names "North,South,East,West"
```

---

## Camera Calibration

Calibrate cameras for accurate real-world measurements.

### When You Need Calibration

- Measuring actual distances (meters, feet)
- Converting pixel positions to floor coordinates
- Multi-camera 3D reconstruction
- Correcting lens distortion

### Calibration Steps

#### 1. Prepare Chessboard

Print a chessboard pattern:
- Default: 7x7 squares
- Recommended square size: 25mm for handheld, 100-200mm for floor

#### 2. Intrinsic Calibration

Captures camera lens properties (focal length, distortion):

```bash
# Record video moving chessboard through camera's field of view
# Get 15-20 different positions and angles

python scripts/calibrate_camera_system.py intrinsic \
    --video chessboard_recording.mp4 \
    --output camera_calibration.yaml \
    --square-size 25  # mm
```

#### 3. Floor Calibration

Defines the ground plane for position projection:

```bash
# Record video with chessboard placed on floor at 3-5 positions

python scripts/calibrate_camera_system.py floor \
    --video floor_chessboard.mp4 \
    --camera-calib camera_calibration.yaml \
    --output floor_calibration.yaml \
    --square-size 172  # mm (use larger chessboard for floor)
```

#### 4. Verify Accuracy

```bash
python scripts/calibrate_camera_system.py verify \
    --video test_video.mp4 \
    --camera-calib camera_calibration.yaml \
    --floor-calib floor_calibration.yaml \
    --known-distance 1000  # mm (1 meter reference)
```

### Calibration Tips

- **Good lighting**: Avoid shadows on chessboard
- **Sharp images**: Ensure chessboard corners are crisp
- **Full coverage**: Move chessboard to all areas of view
- **Flat surface**: Chessboard must be rigid and flat
- **Multiple positions**: More positions = better accuracy

---

## Python API

For programmatic use, import modules directly:

```python
# People tracking
from naturallab.spatial_tracking.detection.yolo_detector import YOLODetector
from naturallab.spatial_tracking.tracking.base_tracker import BaseTracker

detector = YOLODetector(model_path="yolo11x.pt")
tracker = BaseTracker()

detections = detector.detect(frame)
tracks = tracker.update(detections, frame)

# Object detection
from naturallab.gaze_analysis.object_detection.owlv2 import OWLv2Detector

detector = OWLv2Detector()
results = detector.detect(image, queries=["toy", "book", "cup"])

# Identity matching
from naturallab.spatial_tracking.tracking.track_identity_matching import TrackIdentityMatcher

matcher = TrackIdentityMatcher()
matches = matcher.match_identities({
    "Person A": "adult wearing blue",
    "Person B": "child in red shirt"
})
```

---

## Getting Help

- **Issues**: Open a GitHub issue
- **Documentation**: See other docs in this folder
- **Examples**: Check the `scripts/` directory
