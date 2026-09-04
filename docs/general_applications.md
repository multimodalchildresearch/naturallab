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

### Output Files

- `tracks.csv` - Frame-by-frame position data
- `track_statistics.csv` - Per-person summary (duration, distance)
- `identity_matches.json` - Identity assignments (if configured)

### With Real-World Measurements

For actual floor-distance measurements, run the three automatic calibration
stages for that camera. Pattern dimensions are OpenCV internal corners; an
8-by-8-square board has 7 by 7 internal corners.

```bash
# 1. Calibrate camera intrinsics
naturallab calibrate intrinsic \
    --video recordings/camera-01-intrinsic.mp4 \
    --camera-id camera-01 \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/intrinsic

# 2. Calibrate the floor after fixing the camera in its study position
naturallab calibrate floor \
    --video recordings/camera-01-floor.mp4 \
    --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/floor

# 3. Verify on a separate recording without moving the camera
naturallab calibrate verify \
    --video recordings/camera-01-verification.mp4 \
    --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
    --floor calibration/camera-01/floor/floor.yaml \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/verification

# 4. Track with the accepted calibration
python scripts/track_people_in_video.py \
    --input video.mp4 \
    --camera-calib calibration/camera-01/intrinsic/intrinsics.yaml \
    --floor-calib calibration/camera-01/floor/floor.yaml \
    --output results/
```

See [Automatic camera and floor calibration](calibration_workflow.md) for the
recording protocol and acceptance checks.

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

Start with photographs of the actual objects rather than a new training run.
NaturalLab groups those photographs by folder name and applies the resulting
reference-image prototypes to videos or still images.

### Quick Start

```bash
# Step 1: Organize reference images by category
# private-study-data/reference_images/
#   product_A/
#     image1.jpg
#     image2.jpg
#   product_B/
#     image1.jpg
#     ...

# Step 2: Create prototypes
python scripts/detect_custom_objects.py create-prototypes \
    --images private-study-data/reference_images/ \
    --output private-study-data/prototypes.h5

# Step 3: Detect in video/images
python scripts/detect_custom_objects.py detect \
    --input private-study-data/video.mp4 \
    --prototypes private-study-data/prototypes.h5 \
    --output private-study-data/detection-run-01/ \
    --save-frames
```

Use several clear views of each object and test them on a separate recording.
The [object detector setup guide](object_detection_guide.md) gives the full
photograph, tuning, visual-review, and external-training workflow.

---

## Multi-Sensor Data Acquisition

Record multiple sensors in one XDF container using Lab Streaming Layer (LSL).
The adapters use host-arrival timestamps in LSL's local-clock domain. These are
not camera exposure timestamps, so timing-sensitive studies must measure
capture offset and drift for their actual hardware and network.

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
| RTSP Cameras | Video (JPEG) | Camera-reported; irregular if unavailable |
| Pupil Labs Neon | Matched gaze and scene video | 30/30 Hz nominal |
| Intel RealSense | RGB, Depth | 30 Hz standalone; 15 Hz GUI/package path |
| Custom (via API) | Any | Variable |

The rate column describes source-reported metadata or configured nominal rates,
not measured delivery guarantees. Record observed rates, gaps, and timestamp
provenance for the actual devices used.

### Why LSL?

- **Common recording container**: LabRecorder stores the LSL streams and their
  host-side timestamps from LSL's local-clock domain together in one XDF file
- **Measured precision only**: Current camera adapters timestamp frames after
  receipt and decoding, not at camera exposure; validate capture offsets and
  clock drift before making cross-device timing claims
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
    --cameras "rtsp://cam1/stream,rtsp://cam2/stream,rtsp://cam3/stream,rtsp://cam4/stream" \
    --camera-names "North,South,East,West"
```

For authenticated cameras, prefer the recording GUI: it URL-encodes the
credentials, keeps the password session-only, and redacts it from displayed
URLs and logs. This avoids saving it in the configuration or shell history;
the credentialed URL still exists in the child process arguments and memory
while streaming, so access to the acquisition workstation should remain
restricted.

---

## Camera Calibration

Calibrate each camera independently for accurate floor measurements. The
supported workflow is automatic: it does not ask a user to choose frames,
click endpoints, or invent a scale correction.

### When You Need Calibration

- Measuring actual distances (meters, feet)
- Converting pixel positions to floor coordinates
- Correcting lens distortion
- Preparing one part of a later multi-camera 3D reconstruction

### Calibration Steps

#### 1. Prepare Chessboard

Use a rigid, flat chessboard and measure one square. Commands take the number
of **internal corners**, not the number of black-and-white squares. For
example, an 8-by-8-square board is passed as
`--inner-cols 7 --inner-rows 7`. Use the measured square side length for
`--square-size-mm`.

#### 2. Intrinsic Calibration

Record at least 20 sharp, varied views covering the image centre, edges, and
corners, with different distances and tilts:

```bash
naturallab calibrate intrinsic \
    --video recordings/camera-01-intrinsic.mp4 \
    --camera-id camera-01 \
    --input-rotation none \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/intrinsic \
    --save-frames
```

#### 3. Floor Calibration

Fix the camera in its final study position. Record the board lying flat and
stationary for several seconds at five or more widely separated floor
locations:

```bash
naturallab calibrate floor \
    --video recordings/camera-01-floor.mp4 \
    --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/floor \
    --save-frames
```

#### 4. Verify Accuracy

Without moving the camera, make a separate recording with the board at new
floor locations:

```bash
naturallab calibrate verify \
    --video recordings/camera-01-verification.mp4 \
    --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
    --floor calibration/camera-01/floor/floor.yaml \
    --inner-cols 7 --inner-rows 7 \
    --square-size-mm 30 \
    --output-dir calibration/camera-01/verification \
    --save-frames
```

The command reports measured-versus-known board spans, absolute and percentage
error, and a `PASS`, `WARNING`, or `FAIL` screening result. Verification never
refits the calibration.

### Important Boundary

Per-camera intrinsics and floor planes are sufficient for metric points that
are assumed to touch the floor. They are not sufficient to triangulate
arbitrary 3D skeleton joints. Four-camera 3D additionally needs relative
camera extrinsics in one room frame, synchronized frames, cross-view
correspondence, and a triangulation/validation stage. See
[Multiview 3D readiness](multiview_3d_readiness.md).

### Recording Tips

- **Good lighting**: Avoid shadows on chessboard
- **Sharp images**: Ensure chessboard corners are crisp
- **Full coverage**: Move chessboard to all areas of view
- **Flat surface**: Chessboard must be rigid and flat
- **Stable geometry**: Do not move the camera after floor calibration
- **Independent check**: Do not reuse the floor-calibration video for final
  verification

---

## Python API

Use the stable input and artifact contracts when integrating NaturalLab into a
larger analysis. Model construction and complete study execution have
additional dependencies and are documented in the
[researcher workflow](researcher_workflow.md).

```python
from naturallab.media import VideoFileSource
from naturallab.spatial_tracking.calibration import (
    load_calibration_bundle_file,
)
from naturallab.spatial_tracking.multiview import load_room_registration

# Any conventional video can be exposed as timestamped frame packets.
source = VideoFileSource("session.mp4", source_id="camera-01")
first_packet = next(iter(source))
frame = first_packet.image  # OpenCV BGR array

# Calibration and room-registration files are strictly validated on load.
bundle = load_calibration_bundle_file(
    "calibration/camera-01/calibration-bundle.yaml"
)
registration = load_room_registration(
    "calibration/shared-room/room-registration.yaml"
)
camera_to_room = registration.registration_for(
    "camera-01",
    camera_id=bundle.camera_id,
)
```

Detection can consume the frame without calibration. Metric floor projection
must additionally apply the loaded bundle, while cross-view conversion must
validate and apply `camera_to_room`. Do not combine tracks merely because they
occur at similar coordinates; cross-view identity evidence remains explicit.

---

## Getting Help

- **Issues**: Open an issue in the
  [NaturalLab issue tracker](https://github.com/multimodalchildresearch/naturallab/issues)
- **Documentation**: See other docs in this folder
- **Examples**: Check the `scripts/` directory
