# NaturalLab

NaturalLab is a research pipeline for person tracking, room-scale movement,
egocentric object detection, gaze analysis, and multimodal sensor data.

The repository is being consolidated into task-oriented workflows. The current
foundation is usable for a guided recording setup, environment checks, external
video input, automatic camera calibration, person tracking, Qwen person
grounding and role assignment, reference-image object detection, explicit
multiview registration and fusion, gaze/object alignment, and resumable study
contracts. Concrete end-to-end study executors and a guided analysis UI remain
active development work.

## Start here

- [Laboratory setup and first-recording guide](docs/lab_setup_guide.md): a
  beginner-oriented walkthrough from connecting cameras to a router through a
  checked test recording and automatic calibration.
- [Object detector setup](docs/object_detection_guide.md): how to photograph
  study objects, build and check reference-image prototypes, and decide when a
  separately trained detector is warranted.
- [Software quick start](docs/quickstart.md): install the package and run
  individual components on existing footage.
- [Researcher workflow](docs/researcher_workflow.md): library contracts for
  arbitrary sources and views, Qwen/DeepSORT, gaze, multimodal alignment,
  registration, fusion, and verified resume.

The first two guides distinguish an operational smoke test from scientific
validation. In particular, host-arrival timestamps are not proof of
millisecond-accurate capture synchronization, and a floor-only registration is
not a validated person-volume 3D reconstruction.

## What can be run now

| Workflow | Current entry point | Status |
|---|---|---|
| Check an installation | `naturallab doctor` | Ready |
| Configure cameras/sensors and open LabRecorder | `naturallab record` | Ready on a desktop with Tk; four shared-credential camera rows |
| Validate, plan, or inspect a study manifest | `naturallab study` | Ready, read-only CLI |
| Read arbitrary video, image sequences, or Python frame iterables | `naturallab.media` | Ready as a library API |
| Track people in a video with YOLO, OWLv2, or Qwen | `scripts/track_people_in_video.py` | Compatibility CLI |
| Assign configured roles to tracks with Qwen | `--identities` on the tracking script | Compatibility CLI |
| Automatically calibrate intrinsics, floor plane, shared-room geometry, and verification | `naturallab calibrate` | Ready, click-free CLI |
| Validate and consume versioned calibration artifacts | `naturallab.spatial_tracking.calibration` | Ready as a library API |
| Register/fuse arbitrary calibrated views | `naturallab.spatial_tracking.multiview` | Ready as a library API |
| Build reference-image prototypes and detect them in video or images | `scripts/detect_custom_objects.py` | Compatibility CLI |
| Assign gaze and align timestamped modalities | `naturallab.gaze_analysis` | Ready as a library API |
| Stream or extract timestamped sensor data | `naturallab.acquisition` | Hardware-specific, optional |

The compatibility analysis scripts are available from a source checkout.
Calibration is also installed as a console command; its source-checkout script
calls the same implementation. See the
[researcher workflow guide](docs/researcher_workflow.md) for the study manifest,
Qwen/DeepSORT preset, camera registration and fusion boundaries, gaze
assignment, multimodal alignment, and safe resume behavior.

## Install and check

NaturalLab requires Python 3.10 or newer. Python 3.11 or 3.12 is the
recommended researcher environment for the current model and sensor stacks.

```bash
git clone https://github.com/multimodalchildresearch/naturallab.git
cd naturallab

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Lightweight core: media, calibration contracts, CLI, and Qwen HTTP client
python -m pip install -e .

naturallab --version
naturallab doctor --profile core
```

Install only the workflow dependencies you need:

```bash
python -m pip install -e ".[spatial]"
python -m pip install -e ".[gaze]"
python -m pip install -e ".[acquisition]"
python -m pip install -e ".[qwen]"  # Qwen + required DeepSORT/OSNet runtime

# Development checks
python -m pip install -e ".[all,dev]"
```

The acquisition extra covers LSL, XDF, Pupil Labs, and PyAV support.
Intel RealSense additionally requires a platform-compatible `pyrealsense2`
installation. NaturalLab commands do not install packages at runtime.

## External footage is a first-class input

Analysis is not coupled to NaturalLab acquisition. A component can consume a
regular video, an ordered image directory, or any iterable that yields frames:

```python
from naturallab.media import (
    ImageDirectorySource,
    IterableFrameSource,
    VideoFileSource,
)

video = VideoFileSource("session.mp4", source_id="ceiling-01")
images = ImageDirectorySource("exported_frames", fps=30)
live_or_custom = IterableFrameSource(frame_generator, fps=30)

for packet in video:
    process(
        packet.image,
        timestamp_ns=packet.timestamp_ns,
        color_space=packet.metadata.get("color_space"),
    )
```

`VideoFileSource` yields OpenCV BGR arrays, while `ImageDirectorySource`
yields Pillow RGB images. A custom iterable may yield any image type. Consumers
must normalize the image type and color space they require; built-in sources
declare known color spaces in `packet.metadata["color_space"]`. Numbered image
filenames are naturally ordered (`frame1`, `frame2`, `frame10`).

Timestamps are optional for frame-level detection. Workflows that calculate
duration or synchronize modalities must require timestamps at their own
boundary. The video-based compatibility scripts retain container timestamps
when OpenCV exposes them and explicitly mark nominal-FPS fallbacks.

## Track people in any video

The local baseline uses YOLO plus a Kalman tracker. It is selected explicitly;
it is not an automatic fallback from the Qwen/DeepSORT path:

```bash
python scripts/track_people_in_video.py \
  --input session.mp4 \
  --output results \
  --detector yolo \
  --tracker kalman \
  --device auto
```

The current operational Qwen path uses an OpenAI-compatible service hosting the
exact model `Qwen/Qwen3.6-27B`:

```bash
export NATURALLAB_VLM_BASE_URL="https://your-institutional-service.example/v1"
export NATURALLAB_VLM_API_KEY="..."  # omit when the service needs no key

naturallab doctor --profile qwen

python scripts/track_people_in_video.py \
  --input session.mp4 \
  --output results \
  --detector qwen \
  --tracker deepsort \
  --qwen-cadence 10
```

The repository provides the client adapter; it does not launch or download the
27-billion-parameter model. The client sends complete frames for grounding and
cropped track images for role assignment. Use only an institutionally approved
HTTPS endpoint whose data handling is covered by the study's consent and data
protection arrangements; a loopback HTTP endpoint is appropriate only when the
service runs on the same machine. Non-loopback HTTP is rejected unless the
researcher explicitly sets `NATURALLAB_ALLOW_INSECURE_VLM_HTTP=1`. Qwen
detections use normalized `xyxy` coordinates,
strict JSON validation, and nullable confidence rather than invented scores.
Reported scores below `--confidence` are removed; detections whose score is
null are retained and remain visibly null. Frames between Qwen calls are marked
as temporal predictions and do not age the track. Consequently, `--max-age`
counts detector updates rather than cadence-skipped video frames on this path.
Model performance depends on the prompt, serving configuration, and target data;
evaluate the exact deployed configuration before reporting accuracy.

To assign semantic roles from several crops of each completed track:

```bash
python scripts/track_people_in_video.py \
  --input session.mp4 \
  --output results \
  --detector qwen \
  --tracker deepsort \
  --identities '{"child":"the infant participant","caregiver":"the adult participant"}'
```

The operational preset accepts at most five evidence images per track and
exposes that limit as `components.role_assigner.evidence_images_per_track`.
Supplying more is rejected rather than silently discarding evidence.

Outputs are written below `results/<video-name>/`:

- `tracks.csv`: frame, source timestamp, track ID, bounding box, confidence,
  prediction flag, and optional floor coordinates.
- `track_statistics.csv`: track span, separate observed/predicted counts, and
  first-to-last elapsed time plus distance totals with an explicit unit field
  when calibration is available.
- `run_metadata.json`: detector settings, processed-frame count, and persisted
  detection provenance.
- `identity_matches.json`: role, abstention, reason, and Qwen provenance when
  role assignment was requested.

## Calibration and camera views

Per-camera calibration has three separately invoked automatic stages, plus an
optional shared-room stage for temporally corresponding stationary-board
footage from fixed-camera views.
Pattern dimensions are OpenCV **internal corners**: an 8-by-8-square board has
7 by 7 internal corners.

```bash
naturallab calibrate intrinsic \
  --video camera-01-intrinsic.mp4 \
  --camera-id camera-01 \
  --input-rotation 90_cw \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/intrinsic

naturallab calibrate floor \
  --video camera-01-floor.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/floor

naturallab calibrate verify \
  --video camera-01-verification.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --floor calibration/camera-01/floor/floor.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/verification

# After every view has a bundle and shared-board video:
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room
```

No frame selection or measurement clicking is required. Intrinsics use
automatic diverse-view selection and true OpenCV reprojection RMS. Floor
calibration identifies stationary, separated placements and uses complete PnP
poses. Verification automatically presents measured-versus-known board spans
through the fixed calibration. It never refits the verification data or
suggests a distance correction multiplier.

Use one bundle per camera view. Each view can produce its own calibrated
metrics independently. Combining identities or trajectories across views
additionally requires a shared room registration. NaturalLab can recover it
from a manifest of shared stationary-board recordings; it never infers
geometry from camera count alone.

Versioned artifacts reject hidden distance correction factors and mismatched
camera IDs, hashes, image sizes, rotations, or coordinate frames. The tracking
compatibility script can still read the repository's older `dist_coeffs` and
`plane_normal`/`plane_d` files.

Read the full [automatic calibration workflow](docs/calibration_workflow.md)
before recording. Per-camera floor calibration does not establish the relative
camera poses required for 3D skeleton triangulation; see the
[multiview 3D readiness assessment](docs/multiview_3d_readiness.md).

## Reference-image object detection

Reference images are organized as one directory per category:

```text
private-study-data/reference_images/
├── ball/
│   ├── ball_01.jpg
│   └── ball_02.jpg
└── book/
    └── book_01.jpg
```

Create prototypes and apply them to a video, single image, or image directory:

```bash
python scripts/detect_custom_objects.py create-prototypes \
  --images private-study-data/reference_images \
  --output private-study-data/prototypes.h5 \
  --device auto

python scripts/detect_custom_objects.py detect \
  --input private-study-data/scene_video.mp4 \
  --prototypes private-study-data/prototypes.h5 \
  --output private-study-data/detection-run-01 \
  --device auto \
  --save-frames
```

This pathway writes `detections.csv` and, for video input,
`detection_summary.csv`; annotated review frames are optional. Video detections
include the source timestamp in seconds and nanoseconds plus whether it came
from container PTS or nominal FPS. Read the
[object detector setup guide](docs/object_detection_guide.md) before collecting
reference photos or deciding to train a separate model.

## Qwen preset and reproducibility

The packaged preset
`naturallab/config/presets/qwen36_27b_quality.yaml` records the current
operational model, deployment-defined service precision, Qwen grounding and
role assignment, detection cadence, and the intended DeepSORT temporal backend. The
`build_spatial_pipeline()` factory validates this preset and constructs Qwen
grounding, strict DeepSORT/OSNet tracking, and Qwen role assignment. The
compatibility tracking script keeps Kalman as its default and exposes the same
preset-driven DeepSORT path with `--tracker deepsort`.

The preset-driven factory uses the official OSNet-AIN x1.0 MSMT17 checkpoint
pinned to one immutable `kaiyangzhou/osnet` revision, byte size, and SHA-256. On first
use it downloads the 17.3 MB file into the NaturalLab cache, verifies it, loads
every non-classifier backbone parameter, and runs a finite 512-D embedding
preflight. Set `NATURALLAB_REID_MODEL_PATH` only to use an existing copy of
those exact bytes, or `NATURALLAB_REID_CACHE_DIR` to relocate the cache.

`naturallab doctor --profile qwen` stays read-only. A missing cache entry is a
warning because runtime can acquire it; an invalid explicit model path is a
failure. If acquisition or model startup fails, DeepSORT warns and stops
without fallback. After reading that warning, a researcher may explicitly
accept the reduced-capability HSV histogram backend for one run:

```python
components = build_spatial_pipeline(allow_reid_fallback=True)
```

The tracking CLI exposes the equivalent `--allow-reid-fallback` only with
`--tracker deepsort`. Provenance records whether fallback was allowed and
whether it was actually used. A model that fails after tracking starts never
switches to 48-D histograms, which prevents mixed feature dimensions in a
gallery.

Every Qwen adapter result includes secret-free provenance: model ID, prompt
version, endpoint identity, precision, and the cadence actually used. The
tracking script persists detection provenance in `run_metadata.json` and role
provenance in `identity_matches.json`. API keys and URL credentials are never
written to provenance or doctor output.

## Development checks

```bash
pytest -q
ruff check naturallab/cli.py naturallab/doctor.py \
  naturallab/media naturallab/spatial_tracking/calibration \
  naturallab/spatial_tracking/vlm
```

The remaining implementation order is:

1. Stable end-to-end executors for the study manifest steps.
2. Cross-view identity evidence and fusion QC.
3. Stable installed console subcommands and task-oriented demo data.
4. A guided analysis UI beyond the existing recording window.

NaturalLab is licensed under the MIT License. Clone the public repository from
[multimodalchildresearch/naturallab](https://github.com/multimodalchildresearch/naturallab)
and report reproducible software problems through its
[issue tracker](https://github.com/multimodalchildresearch/naturallab/issues).
Do not attach identifiable participant data, credentials, or private local
paths to an issue.
