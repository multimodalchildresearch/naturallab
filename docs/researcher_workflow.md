# Researcher workflow: sources, views, tracking, gaze, and resume

NaturalLab components are intentionally usable outside the NaturalLab capture
stack. The unit of input is an ordered sequence of images, optionally with
timestamps—not a particular camera brand, file layout, or acquisition session.

This guide describes the current library contracts and the study manifest. The
`naturallab study` commands below inspect configuration and state only. They do
not launch model services or execute analysis steps.

For recording and calibrating a camera, follow the separate
[automatic calibration workflow](calibration_workflow.md). For the distinction
between per-camera floor geometry and multiview skeleton reconstruction, see
[multiview 3D readiness](multiview_3d_readiness.md).

## Choose the boundary you actually need

| Task | Required input | Calibration needed? | Cross-view identity needed? |
|---|---|---:|---:|
| Object or person detection | Images in processing order | No | No |
| 2D tracking | Ordered images; timestamps recommended | No | No |
| Per-view floor trajectories and distance | One calibrated camera view | Yes, for that view | No |
| Registered room coordinates | Any explicit set of calibrated views | Floor calibration plus a rigid view-to-room transform for each view | No |
| Fused cross-view trajectory | Registered observations with synchronized timestamps | Yes | Yes, an explicit shared identity |
| 3D skeleton triangulation | Synchronized 2D joints from overlapping views | Shared camera extrinsics and intrinsics; floor planes alone are insufficient | Yes |
| Gaze-to-object assignment | Gaze points and object boxes in the same named view and time base | Image dimensions only for normalized gaze | No |
| Multimodal alignment | Timestamped records and declared tolerances | No | No |

The default is to retain evidence per view. Room registration changes
coordinates; it does not imply that two local tracks are the same person.
Fusion is a separate, opt-in operation. The current room-registration/fusion
API operates on reconstructed floor points; it does not triangulate image-space
skeleton joints.

## 1. Describe one study session

Start from
[`examples/study_manifest.yaml`](../examples/study_manifest.yaml). A manifest
has four required top-level fields:

- `schema_version`: currently `"1.0"`.
- `study_id` and `session_id`: stable identifiers using letters, digits,
  `.`, `_`, and `-`.
- `views`: one or more researcher-chosen view names. There is no fixed camera
  count.
- `steps`: explicit selected/skipped steps, dependencies, inputs, outputs, and
  JSON-compatible configuration.

All relative paths are resolved relative to the manifest file. Every selected
step must declare at least one output so completion can be verified. A view
requires `media`; calibration, role labels, object inputs, and gaze inputs are
optional and should be declared only when that view uses them.

Validate and inspect the session before running anything:

```bash
naturallab study validate examples/study_manifest.yaml
naturallab study plan examples/study_manifest.yaml
naturallab study status examples/study_manifest.yaml

# The same reports are available for scripts and job schedulers.
naturallab study validate examples/study_manifest.yaml --json
naturallab study plan examples/study_manifest.yaml --json
naturallab study status examples/study_manifest.yaml --json
```

`validate` checks the strict schema and dependency graph. `plan` shows selected
steps in dependency order without resolving them to generic executors.
`status` reads the adjacent `study_manifest.run-state.json`, or reports the
initial pending/skipped state when that file does not exist. Use `--state PATH`
to inspect a state file elsewhere. None of these commands creates or updates
state.

The manifest is a reproducibility contract, not an assertion that the example
paths exist. Input existence and content fingerprints are enforced when a
`WorkflowRunner` executes concrete, injected step functions.

## 2. Use footage from any compatible source

The built-in adapters cover a conventional video, a naturally ordered image
directory, and an arbitrary Python iterable:

```python
from naturallab.media import (
    ImageDirectorySource,
    IterableFrameSource,
    VideoFileSource,
)

video = VideoFileSource("session.mp4", source_id="ceiling-left")
images = ImageDirectorySource("exported-frames", fps=30)
custom = IterableFrameSource(frame_generator, source_id="live-adapter", fps=30)

for packet in video:
    consume(
        packet.image,
        frame_index=packet.frame_index,
        timestamp_ns=packet.timestamp_ns,
        metadata=packet.metadata,
    )
```

An iterable can also yield `FramePacket` values when an upstream system already
has timestamps. Detection needs image content only. Duration, gaze assignment,
and multimodal synchronization need a defensible timestamp source; synthesized
nominal-FPS timestamps are marked in packet metadata.

Image representation is part of the adapter boundary. `VideoFileSource`
produces OpenCV BGR arrays, `ImageDirectorySource` produces Pillow RGB images,
and a custom source may produce either. Normalize image type and color space
before passing it to a model that expects a specific representation.

## 3. Build the current operational Qwen and DeepSORT path

The packaged `qwen36_27b_quality` preset fixes the operational path to the exact
model `Qwen/Qwen3.6-27B` for person grounding and post-tracking role
assignment. It constructs Qwen person detection, DeepSORT temporal tracking,
and a Qwen role assigner from one schema-validated configuration:

Grounding requests contain complete frames and role-assignment requests contain
cropped track images. Use an institutionally approved HTTPS service covered by
the study's consent and data-protection arrangements. Use HTTP only for a
loopback service on the same machine. Remote HTTP is rejected unless the
researcher explicitly opts in with
`NATURALLAB_ALLOW_INSECURE_VLM_HTTP=1`.

```bash
export NATURALLAB_VLM_BASE_URL="https://your-approved-service.example/v1"
export NATURALLAB_VLM_API_KEY="..."  # omit for a service without credentials
naturallab doctor --profile qwen
```

```python
from naturallab.media import VideoFileSource
from naturallab.spatial_tracking.pipeline import build_spatial_pipeline

components = build_spatial_pipeline()  # qwen36_27b_quality
source = VideoFileSource("session.mp4", source_id="ceiling-left")

for packet in source:
    ok, output_frame, data = components.pipeline.process_frame(
        packet.image,
        frame_idx=packet.frame_index,
    )
    if not ok:
        raise RuntimeError(f"tracking failed at frame {packet.frame_index}")
    tracks = data["tracks"]
```

The factory does not download or launch Qwen. The configured endpoint must
offer an OpenAI-compatible chat-completions API for the exact model.

For ReID, the preset identifies the official OSNet-AIN x1.0 MSMT17 artifact by
repository, immutable revision, full filename, byte size, and SHA-256. The
first production construction downloads that 17.3 MB checkpoint into
`~/.cache/naturallab/reid`, verifies its bytes, loads the complete non-classifier
backbone, and preflights a 256-high by 128-wide input to a finite normalized
512-D embedding. Set `NATURALLAB_REID_CACHE_DIR` to move the cache. For an
offline shared copy, set `NATURALLAB_REID_MODEL_PATH`; the override must still
match the exact pinned size and hash.

If download, integrity validation, loading, device selection, or startup
preflight fails, construction emits a visible warning and stops. It does not
silently substitute appearance features. Only after reviewing that message may
a researcher opt into the reduced-capability histogram backend for that run:

```python
components = build_spatial_pipeline(allow_reid_fallback=True)
```

The tracking script exposes the same policy as `--tracker deepsort
--allow-reid-fallback`. The opt-in and the backend actually used are both
recorded in provenance. Once model-backed tracking begins, an inference failure
stops the run rather than mixing 512-D model embeddings with 48-D histograms.

Qwen is called on the configured cadence. Intermediate DeepSORT updates are
marked as temporal predictions, and cadence-skipped frames do not consume
detector-age or gallery-expiry budget. Qwen confidence is nullable and remains
nullable rather than being invented.

`components.role_assigner` is separate from the per-frame pipeline. Apply it to
representative evidence crops from a completed track when assigning roles such
as `child` and `caregiver`. Its output can abstain. Do not equate a local
DeepSORT track ID with a cross-camera identity. The preset exposes and enforces
an upper bound of five evidence images per track; excess evidence is rejected
instead of being silently truncated.

For tests, inject `transport`, `deep_sort_factory`, `feature_extractor`, or
`feature_gallery` into `build_spatial_pipeline`. Injected runtimes do not
download a checkpoint. Production construction has no automatic detector or
tracker fallback.

## 4. Keep arbitrary camera sets explicit

Create one `ViewRegistration` for every view that will enter room coordinates.
Each registration names its camera, exact source floor-calibration artifact,
source camera frame, room frame, units, and a proper rigid 4×4 transform:

```python
from naturallab.spatial_tracking.multiview import (
    RoomRegistration,
    TrajectoryObservation,
    ViewRegistration,
    process_multiview_trajectories,
)

# `left_floor` and `right_floor` are the validated floor-calibration
# artifacts that produced the ray/plane intersections for these views.
# `left_camera_to_room` and `right_camera_to_room` are independently measured
# rigid transforms, not values inferred by this API.
left = ViewRegistration(
    view_id="room_left",
    camera_id="ceiling-left-01",
    source_coordinate_frame=left_floor.coordinate_frame,
    source_floor_calibration_sha256=left_floor.sha256,
    room_coordinate_frame="nursery-room",
    units="mm",
    transform_to_room=left_camera_to_room,
    provenance={"method": "surveyed-control-points"},
)
right = ViewRegistration(
    view_id="room_right",
    camera_id="ceiling-right-02",
    source_coordinate_frame=right_floor.coordinate_frame,
    source_floor_calibration_sha256=right_floor.sha256,
    room_coordinate_frame="nursery-room",
    units="mm",
    transform_to_room=right_camera_to_room,
)
room = RoomRegistration(
    room_coordinate_frame="nursery-room",
    units="mm",
    views=(left, right),
)
```

Registrations round-trip as strict JSON/YAML artifacts. Manifest-style
per-view files can be loaded with `load_view_registration(path)` and composed
into a `RoomRegistration`; a bundled room artifact can be loaded directly with
`load_room_registration(path)`. Both loaders validate schema, kind, rigid
geometry, units, camera/view uniqueness, and floor-calibration hashes.

There is no assumed number or layout of views. Camera IDs, units, source
frames, source floor-calibration SHA-256 digests, and target room frames must
match exactly. The digest is required and lowercase, so a transform cannot be
silently reused with a different floor calibration. Scaled, reflected,
singular, or otherwise non-rigid transforms are rejected.

These transforms are supplied by the researcher; this API validates and
applies them but does not estimate them from the videos. They register
already-computed floor points. The ray/plane intersection is a complete 3D
point in the camera's OpenCV frame; its `z` is normally nonzero. The tracking
CSV therefore retains `floor_x`, `floor_y`, and `floor_z`. A 4×4 registration
of those camera-frame floor points is not a camera projection matrix and must
not be treated as sufficient calibration for above-floor 3D joints.

Pass local floor observations through registration while leaving fusion off:

```python
# `row` is one tracking CSV record. Its three floor fields are in the exact
# coordinate frame named by the floor-calibration artifact.
observations = [
    TrajectoryObservation(
        view_id="room_left",
        camera_id="ceiling-left-01",
        track_id="local-track-7",
        timestamp_ns=1_000_000_000,
        floor_point=(row.floor_x, row.floor_y, row.floor_z),
        coordinate_frame=left_floor.coordinate_frame,
        source_floor_calibration_sha256=left_floor.sha256,
        units="mm",
        shared_identity="child",
    ),
]

per_view = process_multiview_trajectories(observations, room)
assert per_view.fusion_enabled is False
metrics = per_view.per_view_metrics
```

`per_view_metrics` groups by source view and local track. This is the safe
default even after points have been transformed into the room frame.

Enable fusion only when the shared identity is independently justified and the
timestamp tolerance is scientifically acceptable:

```python
fused = process_multiview_trajectories(
    observations_from_all_views,
    room,
    fuse=True,
    timestamp_tolerance_ns=50_000_000,
)
```

Fusion never guesses correspondence from local track IDs, camera count, or
geometric proximity. It resolves eligible samples globally by smallest
timestamp difference and raises when equally near alternatives make the
correspondence ambiguous. Fused records retain source view, camera, track,
source floor-calibration hash, registration hash, and provenance. Views or
observations without a shared identity remain in the per-view result and are
not fused.

## 5. Assign gaze to synchronized object observations

Gaze and object boxes must share a view ID and time base. Pixel gaze works
directly; normalized gaze additionally requires the image dimensions:

```python
from naturallab.gaze_analysis import (
    GazeSample,
    ObjectObservation,
    assign_gaze_to_objects,
)

gaze = [
    GazeSample(
        sample_id="gaze-0001",
        timestamp_seconds=1.020,
        view_id="wearable",
        x=0.42,
        y=0.55,
        coordinate_space="normalized",
    )
]
objects = [
    ObjectObservation(
        observation_id="box-0042",
        timestamp_seconds=1.000,
        view_id="wearable",
        bbox_xyxy=(700, 400, 1000, 750),
        category="toy",
        track_id="toy-3",
    )
]

assignments = assign_gaze_to_objects(
    gaze,
    objects,
    image_sizes={"wearable": (1920, 1080)},
    timestamp_tolerance_seconds=0.05,
)
```

The nearest object frame is used only within the explicit tolerance. Invalid
gaze, missing synchronized frames, gaze outside all boxes, and overlapping
boxes are explicit abstention reasons. The default overlap policy is
`"abstain"`; use `"smallest_box"` only when that documented assumption is
appropriate.

## 6. Align arbitrary modalities without hiding missing data

Represent each modality as timestamped records and choose one anchor stream:

```python
from naturallab.gaze_analysis import TimedRecord, align_streams

anchors = [
    TimedRecord(
        stream_id="gaze",
        record_id="gaze-0001",
        timestamp_seconds=1.020,
        values={"object": "toy"},
    )
]
streams = {
    "people": [
        TimedRecord(
            stream_id="people",
            record_id="track-frame-30",
            timestamp_seconds=1.000,
            values={"role": "child"},
        )
    ],
    "audio": [],
}

aligned = align_streams(
    anchors,
    streams,
    tolerance_seconds={"people": 0.10, "audio": 0.05},
    required_stream_ids=("people", "audio"),
)
```

Alignment is deterministic nearest-timestamp matching. Ties prefer the earlier
timestamp and then record ID. A stream outside tolerance remains `None`;
required missing streams are listed in `missing_required_streams` rather than
being dropped or imputed.

## 7. Resume only verified work

Concrete applications connect manifest step names to executor functions:

```python
from naturallab.workflows import (
    StepExecutionContext,
    WorkflowRunner,
    load_manifest,
)

manifest = load_manifest("session.yaml")
runner = WorkflowRunner(manifest)

def run_tracking(context: StepExecutionContext) -> None:
    # Run the project-specific adapter. It must create every declared output.
    run_tracking_adapter(context)

result = runner.run(
    {
        "track_people": run_tracking,
        "per_view_metrics": run_metrics_adapter,
        "detect_objects": run_object_adapter,
        "assign_gaze": run_gaze_adapter,
        "align_multimodal": run_alignment_adapter,
    }
)
```

The names in the executor mapping must cover every selected step that is not
reusable. NaturalLab records state atomically beside the manifest by default.
A step is reused only when all of these still match:

- its canonical configuration fingerprint;
- declared view inputs and step inputs;
- completed dependency outputs;
- every declared output path and its current content fingerprint.

A normal function return is not enough: missing declared outputs mark the step
failed. Changed inputs, configuration, dependency outputs, or output content
force re-execution. The CLI currently exposes validation, planning, and status;
there is intentionally no generic `study run` command until the repository's
concrete researcher step executors are wired end to end.

## Session checklist

Before a production analysis:

1. Record the original time base and color space for every input.
2. Run and independently verify each camera's
   [automatic intrinsic and floor calibration](calibration_workflow.md).
3. Compute per-view metrics first.
4. Validate view-to-room transforms before enabling room registration.
5. Enable fusion only with explicit shared identities and timestamp tolerance.
6. Keep gaze/object overlap abstentions and missing modalities in the output.
7. Save model ID, preset version, prompt versions, calibration hashes, and run
   state with the derived data.
8. Do not label output as a 3D skeleton unless shared camera extrinsics,
   synchronization residuals, triangulation quality, and metric validation are
   all recorded.
