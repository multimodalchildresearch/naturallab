# NaturalLab lab setup checklist

Use this checklist to install NaturalLab, connect the recording devices,
calibrate the cameras automatically, and complete one acceptance recording.
For existing video files, start with the [software quick start](quickstart.md).

## 1. Prepare the rig

- Assign stable IDs such as `camera-01`, `camera-02`, `neon-child`, and
  `depth-centre`.
- Mount every room camera rigidly and set its final resolution, orientation,
  focus, zoom, and exposure.
- Make sure adjacent cameras share enough visible floor area for the calibration
  board.
- Photograph or diagram the final camera positions.
- Prepare a rigid, flat chessboard. Count its **internal corners** and measure
  one square in millimetres.
- Give all network devices stable IP addresses. Use wired networking where
  possible.
- Choose a recording volume with enough free space for XDF, extracted videos,
  raw depth, calibration outputs, and backups.

Do not move a camera or change its lens, crop, focus, zoom, orientation, or
resolution after calibration.

## 2. Install NaturalLab and LabRecorder

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/multimodalchildresearch/naturallab.git
cd naturallab

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[spatial,gaze,acquisition,qwen]"

naturallab --version
naturallab doctor --profile all
ffmpeg -version
```

Install [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder/releases)
on the acquisition computer. RealSense use also requires a compatible
`pyrealsense2` installation.

Record the versions of Python, NaturalLab, FFmpeg, LabRecorder, `liblsl`, the
GPU driver/CUDA stack, and every model used.

## 3. Check each device

### RTSP cameras

Test every camera separately:

```bash
ping CAMERA_IP
ffplay "rtsp://USER:PASSWORD@CAMERA_IP/STREAM_PATH"
```

Confirm the intended image, orientation, resolution, frame rate, focus, and
exposure. Keep authenticated URLs out of manifests, logs, screenshots, shell
scripts, and version control.

### Pupil Labs Neon

- Connect each phone and the acquisition computer to the recording network.
- Record the device ID and stable IP address.
- Complete the manufacturer-required wearer calibration.
- Verify scene video and gaze in the manufacturer preview.
- Record the device-to-role assignment for the session.

### Optional RealSense

- Record the serial number and depth scale.
- Preview colour and depth in the final position.
- Check the working area for invalid or saturated depth.
- Preserve raw 16-bit depth.

## 4. Start the streams and make a disposable XDF

Replace all placeholders and omit unused argument groups:

```bash
python scripts/stream_synchronized_sensors.py \
  --cameras "RTSP_URL_1,RTSP_URL_2,RTSP_URL_3,RTSP_URL_4" \
  --camera-names "camera-01,camera-02,camera-03,camera-04" \
  --neon-ips "NEON_CHILD_IP,NEON_CAREGIVER_IP" \
  --neon-names "child,caregiver" \
  --realsense
```

In LabRecorder:

1. Click **Update**.
2. Select each required stream exactly once.
3. Confirm unique names, advancing counters, plausible rates, decodable images,
   and no reconnect loop.
4. Record a short disposable XDF.
5. Stop recording and confirm that the XDF is non-empty.

Expected stream names:

| Input | LSL stream name |
|---|---|
| RTSP camera | Matching `--camera-names` value |
| Neon gaze | `Gaze_<neon-name>` |
| Neon scene video | `Video_<neon-name>` |
| RealSense colour | `RealSense_Color` |
| RealSense depth | `RealSense_Depth` |

Record one sharp event visible to all cameras at the beginning and end of every
acceptance or study recording. Measure and retain the resulting timing offset
and drift. The current host-arrival timestamps are not sufficient by themselves
for a millisecond-synchronization claim.

## 5. Run automatic camera calibration

The supported calibration path is automatic chessboard-corner detection. No
manual corner clicking is required. Repeat the intrinsic, floor, and
verification steps for every fixed camera.

The example below uses a board with 7-by-7 internal corners and 30 mm squares.
Replace those values with the measured board specification.

### 5.1 Record and calculate intrinsics

Record at least 20 sharp board poses covering the centre, edges, corners,
different sizes, and different tilts. Then run:

```bash
naturallab calibrate intrinsic \
  --video recordings/calibration/camera-01-intrinsic.mp4 \
  --camera-id camera-01 \
  --input-rotation none \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/intrinsic \
  --save-frames
```

Check `intrinsic-report.json` and the annotated frames.

### 5.2 Record and calculate the floor plane

Without moving the camera, record the board flat and stationary for about three
seconds at five or more widely separated floor locations. Then run:

```bash
naturallab calibrate floor \
  --video recordings/calibration/camera-01-floor.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/floor \
  --save-frames
```

Check the generated calibration bundle, report, and annotated frames.

### 5.3 Verify with a separate recording

Record the board at at least three new, spatially separated floor locations.
Do not reuse the floor-calibration video.

```bash
naturallab calibrate verify \
  --video recordings/calibration/camera-01-verification.mp4 \
  --bundle calibration/camera-01/floor/calibration-bundle.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/verification \
  --save-frames
```

Check `verification.json` and the annotated frames. Re-record a failed
calibration; do not add a manual distance multiplier.

### 5.4 Register multiple cameras to the shared floor

Record the same stationary floor-board placements in every camera. Copy
[`examples/shared_board_extrinsics.yaml`](../examples/shared_board_extrinsics.yaml),
set the camera IDs and relative file paths, then run:

```bash
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room \
  --save-frames
```

Accept the result only when `extrinsics-report.json` reports `status: pass` and
the annotated correspondences are correct. This step registers the shared floor;
it does not validate elevated 3D skeleton accuracy. See
[multiview 3D readiness](multiview_3d_readiness.md) before using triangulated 3D.

For all calibration options and acceptance fields, use the
[calibration workflow reference](calibration_workflow.md).

## 6. Make the two-minute acceptance recording

Record all study streams in LabRecorder for two minutes. During the recording:

- perform the shared visible event at the beginning and end;
- move a consenting adult or non-identifiable stand-in through near, middle,
  far, overlap, entry, exit, turn, pause, and brief-occlusion conditions;
- include two known gaze-reference objects if Neon is part of the study; and
- monitor CPU, memory, network, disk throughput, dropped frames, and reconnects.

After stopping, retain the XDF and calculate its checksum:

```bash
ls -lh recordings/acceptance-001.xdf
shasum -a 256 recordings/acceptance-001.xdf  # macOS
sha256sum recordings/acceptance-001.xdf      # Linux
```

## 7. Extract and inspect the acceptance recording

Keep the original XDF immutable and backed up.

```bash
python -m naturallab.acquisition.xdf_extract \
  --file recordings/acceptance-001.xdf \
  --outdir extracted/acceptance-001 \
  --depth-interval 1
```

Confirm that every required stream is present for the expected duration,
videos open, frame counts and durations are plausible, timestamps are
monotonic, raw depth is retained, and the start/end timing residuals meet the
study's predeclared limits.

## 8. Run a first processing smoke test

Replace the input path with the path reported by the extractor:

```bash
python scripts/track_people_in_video.py \
  --input extracted/acceptance-001/camera-01.mp4 \
  --output results/acceptance-local \
  --detector yolo \
  --tracker kalman \
  --yolo-model yolo11n.pt \
  --device auto \
  --max-frames 300 \
  --save-frames
```

Confirm that `tracks.csv`, `track_statistics.csv`, `run_metadata.json`, and
annotated frames are created and inspect them for obvious failures.

For the optional Qwen/DeepSORT setup:

```bash
export NATURALLAB_VLM_BASE_URL="https://your-approved-service.example/v1"
export NATURALLAB_VLM_API_KEY="..."
naturallab doctor --profile qwen

python scripts/track_people_in_video.py \
  --input extracted/acceptance-001/camera-01.mp4 \
  --output results/acceptance-qwen \
  --detector qwen \
  --tracker deepsort \
  --max-frames 300
```

The default DeepSORT setup stops when its verified ReID model cannot be loaded.
Only use `--allow-reid-fallback` after explicitly accepting the warning for that
run.

## 9. Retain the setup record

Keep the following with the study:

- device IDs, device-to-role mapping, rig photograph, network/configuration
  record, and software/model versions;
- original XDF/video checksums, recording times, selected stream list, observed
  rates, gap/drop report, and timing residuals;
- intrinsic, floor, verification, and room-registration outputs, reports, and
  annotated frames; and
- processing commands, calibration hashes, model/preset identifiers, and
  `run_metadata.json`.

Use pseudonymous session IDs. Store consent records, identity keys, credentials,
and authenticated URLs outside the repository.

Do not start participant recording until this checklist passes on the exact
computer, network, mounted rig, and storage volume that will be used.

For failures, use the [calibration workflow](calibration_workflow.md),
[software quick start](quickstart.md), or
[researcher workflow](researcher_workflow.md) instead of changing thresholds
without recording the change.
