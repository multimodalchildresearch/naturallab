# NaturalLab: Multi-Modal Tracking and Analysis System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive, open-source system for automated tracking and behavioral analysis. Originally developed for developmental research, NaturalLab provides general-purpose tools for multi-camera spatial tracking, custom object detection, and synchronized multi-sensor data acquisition.

## 🎯 Key Features

- **Multi-Camera Spatial Tracking**: Track people with automatic identity assignment using vision-language models
- **Real-World Measurements**: Calibrated floor projection for actual distance calculations (meters, not pixels)
- **Zero-Shot Object Detection**: Detect custom objects without training - just provide reference images
- **Multi-Sensor Synchronization**: LSL-based acquisition with millisecond precision across cameras, eye trackers, and more
- **Modular Design**: Use individual components or the complete pipeline

## 🌐 Applications

| Domain | Use Cases |
|--------|-----------|
| **Research** | Behavioral observation, developmental studies, interaction analysis |
| **Retail** | Customer flow, dwell time, path optimization |
| **Sports** | Player tracking, formation analysis, distance metrics |
| **Healthcare** | Patient mobility, fall detection, rehabilitation |
| **Security** | Occupancy monitoring, crowd analysis |
| **Manufacturing** | Quality control, inventory tracking |

## 📋 System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended, but CPU works too)
- **Hardware** (varies by use case):
  - Any IP/RTSP cameras or webcams
  - Optional: Pupil Labs Neon eye trackers
  - Optional: Intel RealSense depth cameras

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anonymous/naturallab.git
cd naturallab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install with all dependencies
pip install -e ".[all]"

# Or install specific components:
pip install -e ".[tracking]"      # Spatial tracking only
pip install -e ".[gaze]"          # Gaze analysis only
pip install -e ".[acquisition]"   # Data acquisition only
```

### Download Model Weights

```bash
# Download YOLOv11 model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

# Download OSNet ReID model for tracking
python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal', 'osnet_ain_x1_0_msmt17.pth')"
```

## ⚡ Quick Start Scripts

### Track People in Any Video

```bash
python scripts/track_people_in_video.py \
    --input your_video.mp4 \
    --output results/
```

Output: `tracks.csv` with frame-by-frame positions, `track_statistics.csv` with per-person summaries.

### Detect Custom Objects (Zero-Shot)

```bash
# 1. Create prototypes from reference images
python scripts/detect_custom_objects.py create-prototypes \
    --images reference_images/ \
    --output prototypes.h5

# 2. Detect in video
python scripts/detect_custom_objects.py detect \
    --input video.mp4 \
    --prototypes prototypes.h5 \
    --output detections/
```

### Stream & Record Multiple Sensors

```bash
# Start LSL streams
python scripts/stream_synchronized_sensors.py \
    --cameras "rtsp://cam1/stream,rtsp://cam2/stream" \
    --camera-names "Front,Side"

# Then use LabRecorder to record to XDF
```

### Calibrate for Real-World Distances

```bash
# Camera intrinsics
python scripts/calibrate_camera_system.py intrinsic \
    --video chessboard.mp4 --output camera.yaml

# Floor plane
python scripts/calibrate_camera_system.py floor \
    --video floor.mp4 --camera-calib camera.yaml --output floor.yaml

# Now tracking outputs real meters!
python scripts/track_people_in_video.py \
    --input video.mp4 \
    --camera-calib camera.yaml \
    --floor-calib floor.yaml \
    --output results/
```

## 📁 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA ACQUISITION                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ RTSP Cameras │  │ Neon Eye     │  │ Additional Sensors       │  │
│  │ (4x 30Hz)    │  │ Trackers     │  │ (RealSense, IMU, etc.)   │  │
│  └──────┬───────┘  └──────┬───────┘  └────────────┬─────────────┘  │
│         │                 │                        │                │
│         └─────────────────┼────────────────────────┘                │
│                           ▼                                         │
│              ┌────────────────────────┐                             │
│              │ Lab Streaming Layer    │                             │
│              │ (LSL Synchronization)  │                             │
│              └───────────┬────────────┘                             │
│                          ▼                                          │
│              ┌────────────────────────┐                             │
│              │ XDF Recording          │                             │
│              │ (LabRecorder)          │                             │
│              └────────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      OFFLINE PROCESSING                              │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 SPATIAL TRACKING PATHWAY                     │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │   │
│  │  │ YOLO      │→ │ DeepSORT  │→ │ Identity  │→ │ Floor    │ │   │
│  │  │ Detection │  │ Tracking  │  │ Matching  │  │ Tracking │ │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   GAZE ANALYSIS PATHWAY                      │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌──────────┐ │   │
│  │  │ OWL-ViT   │→ │ CLIP      │→ │ Prototype │→ │ Gaze     │ │   │
│  │  │ Detection │  │ Embedding │  │ Matching  │  │ Metrics  │ │   │
│  │  └───────────┘  └───────────┘  └───────────┘  └──────────┘ │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 📖 Usage

### 1. Camera Calibration

Before running the tracking pipeline, calibrate your cameras:

```bash
# Intrinsic calibration (camera parameters)
python -m naturallab.spatial_tracking.calibration.camera_calib \
    --video calibration_video.mp4 \
    --output camera_calibration.yaml

# Floor plane calibration
python -m naturallab.spatial_tracking.calibration.floor_calib \
    --video floor_calibration_video.mp4 \
    --camera-calib camera_calibration.yaml \
    --output floor_calibration.yaml
```

### 2. Data Acquisition

Start LSL streams for synchronized recording:

```bash
# Start all streams (cameras + eye trackers)
python -m naturallab.acquisition.lsl_streams \
    --caregiver-ip 192.168.0.120 \
    --child-ip 192.168.0.121 \
    --rtsp-urls "rtsp://user:pass@camera1/stream,rtsp://user:pass@camera2/stream"
```

Then use LabRecorder to capture all streams to XDF format.

### 3. Spatial Tracking Pipeline

Process recorded videos to extract movement data:

```bash
python -m naturallab.spatial_tracking.pipeline.tracker_pipeline \
    --input session_video.mp4 \
    --camera-calib camera_calibration.yaml \
    --floor-calib floor_calibration.yaml \
    --output output_directory/
```

### 4. Gaze Analysis

Analyze eye-tracking data with custom object detection:

```python
from naturallab.gaze_analysis.object_detection.two_stage import TwoStageDetector

# Initialize detector with your prototype images
detector = TwoStageDetector(
    first_stage_labels={"toy": ["toy", "object"], "hand": ["hand", "finger"]},
    second_stage_labels={"toy": ["blocks", "letters", "toys"]},
    prototype_path="prototypes.h5"
)

# Detect objects in frame
detections = detector.forward(image)
```

## 📊 Output Data

### Spatial Tracking Outputs

| File | Description |
|------|-------------|
| `floor_positions.csv` | 3D positions projected onto floor plane |
| `inter_person_distances.csv` | Frame-by-frame distances between participants |
| `track_statistics.json` | Summary statistics per tracked individual |
| `deepsort_tracks.csv` | Raw tracking data with bounding boxes |

### Gaze Analysis Outputs

| File | Description |
|------|-------------|
| `detections.csv` | Object detections with prototype matches |
| `gaze_metrics.csv` | Computed gaze metrics (dwell time, switches, etc.) |

## 🔧 Configuration

### Identity Matching

Configure participant identification in `identities.json`:

```json
{
    "Caregiver": "bird's eye view of a full-sized adult person",
    "Child": "bird's eye view of a very small child, toddler, or infant"
}
```

### Object Prototypes

Create prototypes for custom object detection:

```python
from naturallab.gaze_analysis.object_detection.utils import create_prototypes

# Create prototype embeddings from reference images
create_prototypes(
    image_dir="reference_images/",
    output_path="prototypes.h5"
)
```

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@article{anonymous2026naturallab,
  title={NaturalLab: An Open-Source Multi-Modal System for Automated 
         Behavioral Analysis in Naturalistic Developmental Research},
  author={Anonymous},
  journal={Behavior Research Methods},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This system builds upon several excellent open-source projects:

- [YOLOv11](https://github.com/ultralytics/ultralytics) for object detection
- [DeepSORT](https://github.com/nwojke/deep_sort) for multi-object tracking
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) for person re-identification
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32) for open-vocabulary detection
- [CLIP](https://github.com/openai/CLIP) for vision-language alignment
- [Lab Streaming Layer](https://github.com/sccn/labstreaminglayer) for data synchronization

## 📞 Support

For questions and support, please open an issue on GitHub.
