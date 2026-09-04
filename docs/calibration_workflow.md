# Automatic camera and floor calibration

NaturalLab calibration is a click-free workflow with three per-camera stages
and one optional multi-camera stage:

1. estimate camera intrinsics;
2. estimate the floor plane after the camera is in its final position;
3. recover a shared-room registration when the same stationary board is
   visible in multiple fixed cameras;
4. verify the fixed calibration on a separately recorded board video.

Each stage takes one explicit video path. The software does not choose the
first file in a directory, ask the researcher to accept frames, or derive a
distance correction factor.

The installed command is `naturallab calibrate`. From a source checkout,
`python scripts/calibrate_camera_system.py` exposes the same implementation.

## Before recording

Use a rigid, flat chessboard and measure one square with a ruler or calliper.
NaturalLab describes the pattern using **internal corners**, following OpenCV:

- an 8-by-8-square board has 7 by 7 internal corners;
- pass that board as `--inner-cols 7 --inner-rows 7`;
- `--square-size-mm` is the side length of one square, not the full board.

The commands below use a 30 mm square only as an example. Always enter the
measured square size of the board used for the recording.

For every stage:

- keep the camera, lens, zoom, focus, and processing resolution identical to
  the study recording;
- lock focus and exposure when the camera permits it;
- keep the complete board sharp and visible;
- avoid motion blur, reflections, deep shadows, and bent paper;
- provide the video file itself, not a directory containing several videos.

Common video containers readable by the local OpenCV build are suitable.
MP4/H.264 is a practical default. Calibration uses decoded image content; it
does not require footage captured by NaturalLab.

## Rotation and image geometry

`--input-rotation` records the right-angle rotation applied after decoding:

- `none`
- `90_cw`
- `180`
- `90_ccw`

Choose one policy:

1. retain the raw camera video and declare its rotation; or
2. create physically rotated video files and use `none`.

Do not rotate the file and also declare the same rotation. The intrinsic
artifact stores the post-rotation image size and rotation. The floor,
verification, and tracking stages inherit that stored rotation and reject an
artifact-pair mismatch or wrong post-rotation image size. Video metadata alone
cannot prove that a physically wrong 180-degree choice—or a wrong rotation on
a square frame—is correct. Inspect the annotated images and rely on the
independent metric verification to catch that class of mistake.

## Step 1: record and estimate intrinsics

Record the board moving through the camera's field of view. Include:

- centre, all four edges, and all four corners of the image;
- several board sizes by moving nearer and farther from the camera;
- several horizontal and vertical tilts;
- at least 20 sharp, visibly different views;
- a brief pause at useful views when the video is compressed heavily.

The camera may be mounted already, but it does not have to be in its final room
position for intrinsic calibration. Do not change its lens, focus, zoom, image
resolution, or digital crop afterward.

```bash
naturallab calibrate intrinsic \
  --video recordings/camera-01-intrinsic.mp4 \
  --camera-id camera-01 \
  --input-rotation 90_cw \
  --inner-cols 7 \
  --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/intrinsic \
  --save-frames
```

The command screens approximately one frame per second, refines detected
corners to sub-pixel precision, selects spatially and perspectivally diverse
views, and removes individual views above the configured reprojection limit.
It also requires centre coverage along both image axes, near/far scale change,
and directly observed out-of-plane perspective around both board axes. This
prevents translated but fronto-parallel footage from passing merely because it
has a very low reprojection RMS. It fails rather than silently accepting a
poor or geometrically degenerate model.

Outputs:

| File | Meaning |
|---|---|
| `intrinsics.yaml` | Canonical schema-v1 intrinsic artifact |
| `intrinsic-report.json` | Source path, size, modification time and SHA-256; selected frames; coverage/perspective gates; true OpenCV RMS; per-view errors; and internal holdout diagnostic |
| `selected-views.csv` | One row per accepted view |
| `intrinsic-selected-views/` | Optional annotated detections |

The OpenCV RMS is a diagnostic, not a universal pass/fail guarantee. Compare any
new result with its image resolution, selected-view coverage, annotated frames,
and downstream independent verification.

Useful controls:

```text
--sample-seconds 1
--target-views 28
--minimum-views 16
--maximum-view-rms-pixels 3
--minimum-center-span-fraction 0.20
--minimum-scale-ratio 1.20
--minimum-perspective-change 0.02
--minimum-tilted-views 4
```

Do not relax a quality limit merely to force a recording to pass. Inspect the
annotated frames and re-record first.

The current implementation uses OpenCV's standard pinhole camera model with
Brown–Conrady distortion. OpenCV fisheye calibration is not implemented here.
Do not use this command for a fisheye or unusually strong wide-angle lens
without first validating that this lens model is appropriate.

## Step 2: record and estimate the floor plane

Put the camera in its final fixed study position before recording. Any later
camera movement invalidates the floor plane.

Place the rigid board flat on the actual study floor:

- use at least five widely separated locations when possible;
- cover both image axes and near/far floor regions;
- hold each placement completely still for at least three sampling intervals
  (approximately three seconds with the defaults);
- keep the full internal-corner grid visible;
- move the board between placements, then stop before the next hold;
- do not tilt or hold the board above the floor.

```bash
naturallab calibrate floor \
  --video recordings/camera-01-floor.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --inner-cols 7 \
  --inner-rows 7 \
  --square-size-mm 30 \
  --minimum-placements 3 \
  --output-dir calibration/camera-01/floor \
  --save-frames
```

The floor stage inherits the input rotation and image geometry from
`intrinsics.yaml`. It detects stationary runs, retains sharp and spatially
separated placements, solves the complete board pose, transforms every known
board point with

```text
X_camera = R X_board + t
```

and fits one metric plane by SVD. It rejects board poses behind the camera,
large reprojection errors, inconsistent board normals, and placements that do
not support one flat surface. One invalid selected pose is recorded as rejected
while the other placements continue; the stage fails only when too few valid
placements remain or the retained set fails the common-plane gates.

Outputs:

| File | Meaning |
|---|---|
| `floor.yaml` | Canonical floor artifact bound to the exact intrinsic SHA-256 |
| `calibration-bundle.yaml` | Validated intrinsic/floor pair |
| `floor-report.json` | Selected placements, plane residuals, pose consistency, and warnings |
| `floor-internal-measurements.csv` | Leave-one-placement-out board-span measurements |
| `floor-selected-placements/` | Optional annotated placements |

The leave-one-placement-out result measures consistency within this recording.
It is not an independent accuracy result.

`calibration-bundle.yaml` can be used directly by verification:

```bash
naturallab calibrate verify \
  --video recordings/camera-01-verification.mp4 \
  --bundle calibration/camera-01/floor/calibration-bundle.yaml \
  --inner-cols 7 \
  --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/verification
```

## Step 3: recover a shared room for multiple fixed cameras

Run this stage only when two or more cameras recorded the same stationary
board placements and did not move afterward. It is not required for
single-camera floor tracking.

Create one YAML manifest that binds every view to the exact video and
`calibration-bundle.yaml` used for it. A copyable template is available at
`examples/shared_board_extrinsics.yaml`:

```yaml
schema_version: '1.0'
kind: shared_board_extrinsics_input
rig_id: room-a
anchor_view_id: camera-03
room_coordinate_frame: rig/room-a/floor/camera-03-x
room_frame_mode: floor_aligned_anchor
board:
  internal_columns: 7
  internal_rows: 7
  square_size_mm: 30
sampling:
  sample_seconds: 1
  stationary_motion_pixels: 20
  minimum_stationary_samples: 2
  minimum_placement_separation_pixels: 50
  minimum_shared_placements: 3
  maximum_shared_placements: 12
  time_tolerance_seconds: 0.15
views:
  - view_id: camera-01
    video: extracted/shared-board/camera-01.mp4
    calibration_bundle: calibration/camera-01/floor/calibration-bundle.yaml
    timestamp_csv: extracted/shared-board/camera-01_timestamps.csv
  - view_id: camera-03
    video: extracted/shared-board/camera-03.mp4
    calibration_bundle: calibration/camera-03/floor/calibration-bundle.yaml
    timestamp_csv: extracted/shared-board/camera-03_timestamps.csv
```

For videos extracted from one XDF, supply the matching timestamp CSV generated
by the extractor for every view. The loader subtracts the anchor view's first
timestamp from each view's first timestamp and uses that relative start offset.
Do not trim or re-encode the MP4s after extraction. For externally produced
videos, omit `timestamp_csv` and set `time_offset_seconds` explicitly for every
view; an omitted offset defaults to zero and is valid only when video starts are
already aligned within `time_tolerance_seconds`.

Then run:

```bash
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room \
  --save-frames
```

The command detects placements visible in every view, matches their
timestamps, resolves every valid checkerboard symmetry, estimates
fixed-intrinsic stereo transforms, and creates a floor-aligned room frame with
positive `z` above the floor. It validates within-recording
leave-one-placement-out corner transfer, triangulated board lengths,
placement-level transform stability, and agreement between the separately
estimated per-camera floor planes.

Outputs:

| File | Meaning |
|---|---|
| `room-registration.yaml` | Strict transforms from each camera frame into one shared metric room frame |
| `extrinsics-report.json` | Exact matrices, hashes, symmetry choices, quality limits, within-recording LOPO errors, and scope |
| `shared-observations.csv` | Selected frame/timestamp/corner-order evidence for every view and placement |
| `annotated-placements/<view>/` | Optional visual check of every selected board detection |

Each transform is bound to the canonical SHA-256 of that view's floor
artifact. A quality failure writes `candidate-room-registration.yaml` and the
evidence, then exits with code 3; it does not publish the normal operational
filename. `PASS` establishes the declared planar geometry. When all targets
lie on the floor, the report remains
`volumetric_validated: false`: the result supports shared-floor fusion and
provisional triangulation but has not certified accuracy at head or hand
height. Camera motion after the shared recording invalidates that camera's
transform.

## Step 4: record and run independent verification

Make a new video after calibration without moving the camera. Put the same
measured board flat at several locations that were not used for the floor
recording and hold each placement still. At least three spatially distinct
stationary placements are required by default, and their centres must span at
least 10% of both image axes.

```bash
naturallab calibrate verify \
  --video recordings/camera-01-verification.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --floor calibration/camera-01/floor/floor.yaml \
  --inner-cols 7 \
  --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/verification \
  --save-frames
```

Verification does not refit the floor plane. It automatically detects the
board, projects its four directly observed boundary spans through the fixed
calibration, and compares the reconstructed distances with their known metric
lengths.

The console presents:

- mean known and reconstructed distance;
- mean absolute error in millimetres and percent;
- 90th-percentile and maximum error;
- an operational `PASS`, `WARNING`, or `FAIL` label.

The default decision requires each of the overall mean, the 90th-percentile
edge error, and the worst placement's mean to be at most 3% for `PASS` or at
most 5% for `WARNING`; otherwise the result is `FAIL`. A bad location therefore
cannot be hidden by averaging many good ones. These are operational screening
thresholds, not universal scientific acceptance criteria. A study should
define its own tolerable error before data analysis.

Outputs:

| File | Meaning |
|---|---|
| `verification.json` | Exact metrics, thresholds, hashes, provenance, and status |
| `measurements.csv` | Every automatically measured edge |
| `annotated-placements/` | Optional detected corners and reconstructed lengths |

NaturalLab never suggests or stores a distance correction multiplier. A failed
verification means the geometry, board specification, rotation, camera
stability, or recording must be corrected.

For shell scripts and cluster jobs, `PASS` and `WARNING` are completed commands
with exit code 0. `FAIL` still writes `verification.json` and
`measurements.csv`, then exits with code 3. Input, processing, or output errors
exit with code 2.

## Use the result for tracking

Pass the two canonical files to the tracking compatibility command:

```bash
python scripts/track_people_in_video.py \
  --input recordings/session.mp4 \
  --output results \
  --camera-calib calibration/camera-01/intrinsic/intrinsics.yaml \
  --floor-calib calibration/camera-01/floor/floor.yaml
```

Tracking applies the recorded input rotation, checks the exact image size, and
validates that the floor artifact is bound to the supplied intrinsic hash.

Repeat the per-camera stages separately for every camera. Per-camera
calibration is sufficient for floor contact points and trajectories. It is not
by itself a multi-camera 3D skeleton calibration; see
[Multiview 3D readiness](multiview_3d_readiness.md).

## Failure messages

Quality and input failures are detected before calibration artifacts are
written. Individual files use atomic replacement, although the complete
multi-file output directory is not one filesystem transaction. The commands
stop when they find:

- an unreadable or empty video;
- invalid frame rate or changing frame dimensions;
- too few intrinsic detections or accepted views;
- a high-error intrinsic view remaining at the minimum view count;
- an intrinsic/floor hash, camera, rotation, or image-size mismatch;
- too few stationary and separated floor placements;
- too few stationary placements visible in every multi-camera view;
- an ambiguous cross-camera checkerboard symmetry;
- a shared-room stereo, held-out transfer, floor-agreement, or board-length
  quality gate;
- invalid/behind-camera PnP geometry;
- a set of placements that does not describe one flat plane;
- viewing rays that intersect the plane behind the camera.

Existing output files are not replaced unless `--overwrite` is explicit.
