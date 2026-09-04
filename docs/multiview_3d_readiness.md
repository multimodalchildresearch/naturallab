# Multiview geometry and 3D readiness

## Current scope

NaturalLab can register an explicit set of calibrated camera views into one
metric room coordinate frame and can fuse synchronized floor observations when
a shared cross-view identity is supplied. The registration and fusion APIs do
not assume a fixed number, brand, or layout of cameras.

The automatic `naturallab calibrate extrinsics` workflow currently requires the
same stationary chessboard placements to be visible in every configured view.
It estimates each non-anchor camera directly relative to the selected anchor.
For a rig whose cameras overlap only in pairs, calibrate the connected overlap
graph with an external procedure and provide the resulting rigid transforms to
the registration API.

A board recorded only on the floor validates **planar floor geometry**. It does
not by itself validate 3D reconstruction at hand, torso, or head height. Dynamic
3D also requires synchronization whose offset and drift have been measured.

## Required calibration inputs

For each camera, retain:

- the intrinsic matrix and lens-distortion coefficients for the exact image
  geometry used during analysis;
- a rigid transform between the camera and one shared room frame;
- the coordinate-frame name, units, and calibration artifact hashes;
- a measured timing relationship to the other cameras for moving subjects.

Per-camera floor calibration is sufficient for points that are known to touch
the floor, such as a foot contact or the bottom centre of a person box. It is
not sufficient for an arbitrary point above the floor.

Skeleton triangulation additionally needs synchronized 2D joint observations,
cross-view person and joint correspondence, and a projection matrix for each
camera:

```text
P_i = K_i [R_i | t_i]
```

All transforms must refer to the same room frame, and observed pixels must be
undistorted before triangulation.

## Create a shared-room registration

First complete and verify intrinsic and floor calibration separately for every
fixed camera. Then prepare the shared-board manifest from
[`examples/shared_board_extrinsics.yaml`](../examples/shared_board_extrinsics.yaml)
and run:

```bash
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room \
  --save-frames
```

The output `room-registration.yaml` contains one rigid 4-by-4
`transform_to_room` per view. Each transform is bound to the source floor
calibration by SHA-256. The generated report records the selected observations,
quality limits, planar validation scope, and whether each check passed.

Do not reuse a registration after a camera, lens, focus, zoom, image rotation,
resolution, or digital crop changes. Recalibrate the affected view and repeat
shared-room verification.

## What NaturalLab implements

- `naturallab calibrate extrinsics` estimates shared-room transforms from
  stationary-board footage with common visibility across all configured views.
- `naturallab.spatial_tracking.multiview.registration` validates and applies
  explicit rigid transforms to reconstructed floor points.
- `naturallab.spatial_tracking.multiview.fusion` combines registered floor
  observations only when explicit shared identities and time tolerances are
  provided.

The current fusion path does not infer cross-view identity from local track IDs
or geometric proximity. It also does not triangulate image-space joints. The
current pose estimator's model-relative `z` value is not metric room depth.

## Verification before 3D analysis

Before reporting multiview 3D measurements:

1. Record a raised calibration target, calibration wand, or independently
   registered depth reference throughout the working volume.
2. Measure camera offset and clock drift with visible synchronization events or
   a shared capture clock.
3. Check reprojection error and reconstructed known lengths at multiple heights
   and positions that were not used to fit the calibration.
4. Reject poorly conditioned ray intersections and ambiguous identities rather
   than forcing a reconstruction.
5. Record the exact calibration hashes, synchronization residual, software
   revision, and acceptance thresholds with the analysis.

Use a connected overlap graph when one target cannot be seen by every camera.
A large ChArUco or AprilGrid target with unique IDs is preferable for that setup.

## Using a depth camera

An aligned RGB-depth camera can lift central-view 2D joints to metric 3D when
valid foreground-associated depth is available. It can also serve as a
validation reference or room-frame anchor after it is registered to the other
cameras.

Retain and verify:

- colour and depth intrinsics;
- depth-to-colour extrinsics;
- the metric depth scale used for each recording;
- the rigid transform into the shared room frame;
- synchronization with every other camera;
- raw depth at the temporal resolution used for reconstruction.

A depth camera does not automatically determine the outer cameras' poses. Its
depth measurements should be independently checked before they are treated as
3D ground truth.
