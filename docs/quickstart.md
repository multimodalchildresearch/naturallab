# Software quick start

This guide verifies a NaturalLab source checkout and runs individual components
on existing footage. For mounting cameras, configuring RTSP and Neon devices,
recording with LSL/LabRecorder, calibrating a fixed rig, and accepting the first
test recording, follow the
[laboratory setup and first-recording guide](lab_setup_guide.md).

NaturalLab's component commands are usable now. The study manifest is currently
a validation, planning, and resume contract; it does not provide a generic
end-to-end `study run` command.

## 1. Install and inspect the environment

Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/multimodalchildresearch/naturallab.git
cd naturallab

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Core media, calibration, configuration, and CLI support
python -m pip install -e .

naturallab --version
naturallab doctor --profile core
```

Use `python3.11` if Python 3.12 is unavailable. Install only the optional groups
needed by the intended command:

```bash
python -m pip install -e ".[spatial]"      # tracking and floor-position support
python -m pip install -e ".[yolo]"         # optional, separately licensed YOLO path
python -m pip install -e ".[gaze]"         # object and gaze analysis
python -m pip install -e ".[acquisition]"  # LSL, XDF, and timestamped Neon streams
python -m pip install -e ".[qwen]"         # Qwen client + DeepSORT/OSNet
python -m pip install -e ".[all]"          # all packaged extras
```

Then run the matching read-only preflight, for example:

```bash
naturallab doctor --profile spatial
naturallab doctor --profile yolo
naturallab doctor --profile gaze
naturallab doctor --profile acquisition
```

Intel RealSense requires a separately available, platform-compatible
`pyrealsense2`. NaturalLab does not install
[LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder) or launch
the Qwen model service. For acquisition, install an official LabRecorder build
and record its exact release/build as described in the
[laboratory setup guide](lab_setup_guide.md). `doctor` does not contact devices
or model endpoints and does not download weights.

## 2. Track people in an existing video

The simplest local smoke test uses YOLO with the Kalman tracker:

```bash
python scripts/track_people_in_video.py \
  --input your_video.mp4 \
  --output results \
  --detector yolo \
  --tracker kalman \
  --yolo-model yolo11n.pt \
  --device auto \
  --max-frames 300 \
  --save-frames
```

Ultralytics may download a named YOLO checkpoint when it is absent. For an
offline or frozen analysis, pre-stage the exact file and pass its path with
`--yolo-model`. Results are written below `results/<video-name>/`:

- `tracks.csv` contains per-frame boxes, track IDs, source timestamps, and
  observation/prediction status;
- `track_statistics.csv` contains per-track summaries;
- `run_metadata.json` records the detector and processing provenance; and
- annotated frames are written when `--save-frames` is selected.

NaturalLab refuses to reuse a non-empty per-video result directory. Pass
`--overwrite` only to replace that video's complete result set, including old
CSV files, identities, and annotated frames.

Supplying no calibration produces image-space tracking only. For floor
coordinates, first complete and independently verify the camera calibration,
then provide both canonical files:

```bash
python scripts/track_people_in_video.py \
  --input your_video.mp4 \
  --output results-calibrated \
  --camera-calib calibration/camera-01/intrinsic/intrinsics.yaml \
  --floor-calib calibration/camera-01/floor/floor.yaml
```

## 3. Use the supported Qwen/DeepSORT client path

This path requires an already deployed OpenAI-compatible service offering the
exact model `Qwen/Qwen3.6-27B`. The repository contains its client adapter; it
does not provide or launch the 27-billion-parameter service. Measure performance
for the exact model, prompt, service configuration, and data used in your study.

Grounding transmits complete frames and role assignment transmits cropped
track images. Use only an approved HTTPS endpoint covered by the study's
consent and data-protection arrangements. Plain HTTP is suitable only for a
service bound to the same machine. A researcher can explicitly accept remote
plaintext transport with `NATURALLAB_ALLOW_INSECURE_VLM_HTTP=1`, but that is
not a safe default for identifiable recordings.

```bash
export NATURALLAB_VLM_BASE_URL="https://your-approved-service.example/v1"
export NATURALLAB_VLM_API_KEY="..."  # omit when the service requires no key

naturallab doctor --profile qwen

python scripts/track_people_in_video.py \
  --input your_video.mp4 \
  --output results-qwen \
  --detector qwen \
  --tracker deepsort \
  --qwen-cadence 10 \
  --max-frames 300
```

The doctor checks local configuration but does not contact the endpoint; the
tracking command is the first end-to-end service request. The packaged
preset pins an official OSNet-AIN x1.0 checkpoint by immutable revision, byte
count, and SHA-256. Its first construction downloads and verifies the file when
needed. Set `NATURALLAB_REID_MODEL_PATH` to a pre-staged copy of those exact
bytes on an offline cluster.

ReID acquisition or startup failure emits a warning and stops. Only after
reviewing it may a researcher explicitly accept the reduced-capability
histogram backend by adding `--allow-reid-fallback`. Provenance records whether
fallback was allowed and used.

Role assignment can be requested from representative completed-track crops:

```bash
python scripts/track_people_in_video.py \
  --input your_video.mp4 \
  --output results-qwen-roles \
  --detector qwen \
  --tracker deepsort \
  --identities '{"participant":"the person completing the task","facilitator":"the person presenting the materials"}'
```

The packaged preset contains no study-specific roles: every role and
description comes from `--identities`. The role assigner may abstain. A local
track ID or role prediction is not, by itself, a cross-camera identity.

## 4. Detect study-specific objects

This path uses photographs of the lab's actual objects and does not require a
conventional model-training run. Arrange the photographs as one directory per
category:

```text
private-study-data/reference_images/
├── ball/
│   ├── ball_01.jpg
│   └── ball_02.jpg
└── book/
    └── book_01.jpg
```

Create prototypes, then apply them to a video, image, or image directory:

```bash
python scripts/detect_custom_objects.py create-prototypes \
  --images private-study-data/reference_images \
  --output private-study-data/prototypes.h5 \
  --device auto

python scripts/detect_custom_objects.py detect \
  --input private-study-data/your_video.mp4 \
  --prototypes private-study-data/prototypes.h5 \
  --output private-study-data/object-run-01 \
  --device auto \
  --save-frames
```

Inspect `annotated_frames/`, `detections.csv`, and, for video,
`detection_summary.csv`. Prototype quality and thresholds must be evaluated on
held-out study footage; this smoke test does not establish detection accuracy.
The [object detector setup guide](object_detection_guide.md) explains how to
take useful reference photographs, tune on a separate clip, and prepare an
external training handoff when prototypes are insufficient. NaturalLab does
not currently provide a `train` command.

## 5. Run automatic calibration

Calibration is click-free but requires deliberately recorded board footage.
Commands use OpenCV **internal corners**. An 8-by-8-square board therefore has
7-by-7 internal corners. Replace the example 30 mm square size with the measured
value.

```bash
naturallab calibrate intrinsic \
  --video recordings/camera-01-intrinsic.mp4 \
  --camera-id camera-01 \
  --input-rotation none \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/intrinsic \
  --save-frames

naturallab calibrate floor \
  --video recordings/camera-01-floor.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/floor \
  --save-frames

# This must be a separate video at new board locations.
naturallab calibrate verify \
  --video recordings/camera-01-verification.mp4 \
  --bundle calibration/camera-01/floor/calibration-bundle.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30.0 \
  --output-dir calibration/camera-01/verification \
  --save-frames
```

For multiple fixed views, start from
[`examples/shared_board_extrinsics.yaml`](../examples/shared_board_extrinsics.yaml)
after every camera has a matching bundle and independent verification:

```bash
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room \
  --save-frames
```

Read the [automatic calibration workflow](calibration_workflow.md) before
recording and use shared-room output only after its report passes.

## 6. Validate a study manifest

Copy [`examples/study_manifest.yaml`](../examples/study_manifest.yaml), replace
its illustrative paths, use pseudonymous identifiers, and save the real copy in
the ignored `private-study-data/` directory (or the study's access-controlled
data store) if it contains private paths or participant-related metadata. Then
inspect the contract:

```bash
naturallab study validate private-study-data/session.yaml
naturallab study plan private-study-data/session.yaml
naturallab study status private-study-data/session.yaml
```

These commands do not run analysis or create state. Concrete applications must
connect selected step names to executors through the library API. See the
[researcher workflow](researcher_workflow.md) for safe resume, gaze assignment,
multimodal alignment, registration, and fusion contracts.

## Recording and hardware setup

Do not infer a complete acquisition procedure from the component examples
above. Follow the [laboratory setup guide](lab_setup_guide.md) for camera
mounting, RTSP and Neon setup, LSL/LabRecorder stream checks, host-arrival timing
limitations, disk testing, XDF extraction, independent calibration acceptance,
and data safety.

## Troubleshooting

### A command is missing

Activate the same virtual environment used for installation and rerun the
matching doctor profile. The `track_people_in_video.py`, object-detection, and
sensor-streaming entry points are source-checkout scripts rather than installed
console commands.

### CUDA runs out of memory

Use `--device cpu`, a smaller explicitly recorded YOLO checkpoint, or a machine
with sufficient GPU memory. Changing a model after validation creates a new
analysis configuration and must be recorded as such.

### A camera or XDF stream is missing

Use the checks in the [laboratory setup guide](lab_setup_guide.md). A passing
software doctor report does not test network devices or LabRecorder content.

### Calibration fails

Inspect the generated report and annotated frames. Verify board rigidity,
internal-corner count, measured square size, image rotation, focus, exposure,
coverage, and camera stability. Re-record before relaxing a quality gate, and
never introduce a post-hoc distance multiplier to make verification pass.

### Qwen doctor passes but tracking cannot connect

The doctor only validates the local endpoint configuration. Confirm service
reachability, credentials, exact model ID, OpenAI-compatible request/response
schema, and timeout with the service operator.

Report reproducible software problems through the
[NaturalLab issue tracker](https://github.com/multimodalchildresearch/naturallab/issues),
without attaching identifiable participant data, credentials, or private local
paths.
