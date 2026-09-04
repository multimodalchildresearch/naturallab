# Set up a NaturalLab room and make the first recording

This guide is for a researcher setting up the room for the first time. It
starts with cables and cameras, then walks through a disposable test recording.
You do not need to understand the internal software components to follow it.

If your lab already has video files, skip the recording sections and use the
[software quick start](quickstart.md). NaturalLab analysis is not limited to
footage recorded with NaturalLab.

## What the room should look like

The room cameras and recording computer share one local network:

```text
Room camera 1 ── Ethernet ──┐
Room camera 2 ── Ethernet ──┤
Room camera 3 ── Ethernet ──┼── Router / managed lab network ── Ethernet ── Recording PC
Room camera 4 ── Ethernet ──┤
Neon phones ───── Wi-Fi ────┘

RealSense depth camera ───── USB ─────────────────────────── Recording PC
```

The router creates the shared network. It does not have to be connected to the
internet during recording. Avoid a guest Wi-Fi network because guest networks
often prevent devices from seeing one another.

If the router does not have enough sockets, add a network switch between it and
the wired devices. A switch by itself normally does not create addresses or
provide Wi-Fi, so use one alone only when local IT has supplied a managed
network or fixed addresses.

A normal network cable does not necessarily power a camera. Use the camera's
power adapter, or a compatible Power over Ethernet (PoE) switch or injector.

Choose the input arrangement that matches the lab:

| What the lab has | How to connect it | What NaturalLab can do |
|---|---|---|
| Fixed IP cameras with RTSP | Ethernet to the shared router or managed lab network | Record, calibrate, and analyze them |
| Existing video files from another system | Copy the files to the analysis computer | Run detection and tracking without NaturalLab recording |
| Neon eye trackers | Wi-Fi on the same local network as the recording PC | Add scene video and gaze streams |
| RealSense depth camera | USB directly to the recording PC | Add colour and raw depth streams |
| Another camera or live image source | Record a normal video, export ordered images, or add a software adapter | Use the analysis components independently of the recorder |

## 1. Gather and label the equipment

You need:

- one recording computer with enough free storage;
- one router with enough LAN sockets, or a router plus a network switch, and
  one cable for the computer;
- one cable and a suitable power source for each room camera;
- fixed room cameras that provide an RTSP video stream;
- optional Neon eye trackers and an optional RealSense depth camera; and
- a rigid, flat chessboard for automatic camera calibration.

Put a physical label on every camera, for example `camera-01` through
`camera-04`. Use the same names in NaturalLab. Also note where each camera is
mounted, such as “left wall” or “ceiling opposite door.”

## 2. Connect everything to one network

1. Connect every room camera to a LAN port on the router. If more ports are
   needed, connect a switch to the router and the cameras to the switch.
2. Connect the recording computer by cable to that same router or attached
   switch.
3. If Neon phones are used, connect them to the router's normal Wi-Fi network,
   not its guest network.
4. Connect a RealSense directly to the recording computer by USB.
5. Turn everything on and check the link lights on the camera and router ports.

NaturalLab does not choose network addresses for the devices. Use the router's
device list or each manufacturer's setup tool to find them. Ask local IT to
create a DHCP reservation for every room camera and Neon phone so its address
stays the same.

Keep a small setup record outside the public repository:

| Physical label | Room position | Network address | Name in NaturalLab |
|---|---|---|---|
| Camera 1 | left wall | `192.168.1.101` | `camera-01` |
| Camera 2 | right wall | `192.168.1.102` | `camera-02` |

The numbers above are examples only. Also record the camera username and stream
path in the lab's password manager. Do not put passwords in study manifests,
screenshots, issue reports, or version control.

## 3. Position and check the cameras

Mount the cameras before calibration. Each intended activity area should be
visible clearly and people should remain large enough to detect. Reduce glare,
strong backlighting, and large hidden areas where practical.

For multiple-camera room registration, plan several floor positions where the
same chessboard can be seen by **every camera included in that calibration**.
Pairwise overlap alone is not sufficient for the current automatic command.

Set the final resolution, frame rate, orientation, focus, zoom, and exposure.
Then fasten the mounts. Moving a camera or changing any of those image settings
after calibration means that camera must be calibrated again.

Before opening NaturalLab, verify one camera at a time in the manufacturer's
viewer or another RTSP viewer. Confirm that:

- the address, username, password, and stream path work;
- the image is upright, sharp, and shows the intended area; and
- the displayed resolution and frame rate are the values you plan to use.

Add the remaining cameras only after the first one works.

If Neon is used, also complete the manufacturer's wearer calibration, check
scene video and gaze in the manufacturer's preview, and write down which
physical unit is worn by the child and which by the caregiver. Never infer
those roles from discovery order.

If RealSense is used, note its serial number, preview both colour and depth in
the final position, and check that the activity area is not blank, saturated,
or outside its useful depth range. Preserve the raw depth stream and the depth
scale with the study record.

## 4. Install the recording software once

This is a one-time computer setup and can be completed by local technical
support. Python 3.11 or 3.12 is recommended.

```bash
git clone https://github.com/multimodalchildresearch/naturallab.git
cd naturallab

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[acquisition]"
naturallab doctor --profile acquisition
```

For this acquisition setup, resolve every missing-package warning or failure
reported by `doctor` before recording.
The guided recording window is currently supported as an end-to-end path on
macOS and Linux; its process controls have not yet been validated on Windows.

Install [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder/releases)
separately on the recording computer. RealSense use also requires a compatible
`pyrealsense2` installation. The computer needs Python Tk support and a desktop
session to display the NaturalLab recording window.

## 5. Configure the NaturalLab recording window

Activate the same environment and open the recorder:

```bash
source .venv/bin/activate
naturallab record
```

The bundled streaming script is selected automatically. On the
**Configuration** tab:

1. Choose the installed LabRecorder application.
2. Choose the folder where recordings will be stored.
3. Leave the bundled LSL script path unchanged.

On the main tab:

1. For each room camera, enter only its address, such as `192.168.1.101`, give
   it the matching physical name, and tick **Enable**.
2. Enter the shared camera username, password, and stream path. The password is
   kept for the current session and is not saved in the NaturalLab settings.
3. Enter the two Neon addresses and roles if they are used. Otherwise tick
   **Disable Neon**.
4. Tick **Disable RealSense**, **Disable Audio**, or **Disable IMU** for anything
   the lab is not recording.
5. Leave **Disable Eye Events** selected unless the lab has independently
   verified that those events come from the intended Neon device.
6. Click **Create Multi-Device LabRecorder Config**, then **Save
   Configuration**.

The current recording window has four camera rows and one shared set of RTSP
credentials and stream path. For a different number of cameras, or cameras that
need different credentials or paths, use the command-line option described in
[Other camera configurations](#other-camera-configurations).

## 6. Make a disposable one-camera recording

Test with one enabled camera first and no participants.

1. Click **Start LSL Streaming**.
2. Click **Open LabRecorder**.
3. In LabRecorder, click **Update**.
4. Confirm that the camera appears once and its sample counter advances.
5. Select that stream, click **Start**, and record for 20–30 seconds.
6. Click **Stop** and confirm that a non-empty `.xdf` file was created.

If the stream does not appear, first check camera power, cable link lights, the
saved address, and whether the computer and camera are on the same non-guest
network. Then verify the camera in its own viewer again.

Repeat this test after enabling each additional camera or sensor. Do not begin
with all devices at once; adding them one at a time makes failures easy to find.

## 7. Make the full acceptance recording

Once every device works separately, enable the final set and make a two-minute
recording with no research participants. During it:

- make one sharp event visible to every camera near the beginning and another
  near the end, such as a bright light switched on briefly;
- have a consenting adult or non-identifiable stand-in walk through the near,
  middle, far, overlap, entrance, and briefly hidden parts of the room;
- show the intended study objects at realistic locations and angles; and
- watch for stopped counters, reconnect messages, or storage warnings.

The visible events allow the lab to measure timing offset and drift. Recording
all streams into one XDF file does not by itself prove that camera exposures
were synchronized.

Keep the original XDF unchanged and backed up. Extract a copy for inspection:

```bash
python -m naturallab.acquisition.xdf_extract \
  --file recordings/exp1/block_Acceptance.xdf \
  --outdir extracted/acceptance-001 \
  --depth-interval 1
```

Use the exact XDF path shown in LabRecorder. With the configuration created by
the NaturalLab window, the usual pattern below the selected recording folder
is `exp<number>/block_<name>.xdf`; the command above is an example for
experiment 1 and the **Acceptance** block.

Open every extracted video. Check that the expected view is present for the
full recording and that playback, duration, and frame count look plausible.

## 8. Calibrate each fixed room camera automatically

Calibration uses a chessboard and automatic corner detection. There is no
manual frame selection or corner clicking.

Count the chessboard's internal corners and measure one square. An 8-by-8
square board has 7-by-7 internal corners. The commands below use 7-by-7
internal corners and 30 mm squares only as an example.

For each camera, make three separate recordings without moving the camera:

1. **Camera recording:** move the board through the centre, edges, and corners
   of the image with different distances and tilts. Capture at least 20 clear
   board positions.
2. **Floor recording:** place the board flat and still for about three seconds
   at five or more widely separated floor positions.
3. **Check recording:** place it flat at at least three different floor
   positions that were not used in the floor recording.

If these recordings were made in LabRecorder, extract each XDF into a separate
folder before calibration. For example:

```bash
python -m naturallab.acquisition.xdf_extract \
  --file recordings/exp2/block_Main.xdf \
  --outdir extracted/calibration/camera-01-intrinsic \
  --depth-interval 1
```

Repeat that extraction for the floor and check XDF files, choosing a different
output folder each time. The extracted MP4 takes the stream name entered in the
recording window. A stream named `camera-01`, for example, becomes
`camera-01.mp4`. If a different recorder already produced normal video files,
use those files directly and skip XDF extraction.

Play one extracted video before running the commands. Set `--input-rotation` to
`none`, `90_cw`, `180`, or `90_ccw` so the decoded image is upright. Use the
same choice for that camera throughout the study.

Run the three matching commands:

```bash
naturallab calibrate intrinsic \
  --video extracted/calibration/camera-01-intrinsic/camera-01.mp4 \
  --camera-id camera-01 \
  --input-rotation none \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/intrinsic \
  --save-frames

naturallab calibrate floor \
  --video extracted/calibration/camera-01-floor/camera-01.mp4 \
  --intrinsics calibration/camera-01/intrinsic/intrinsics.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/floor \
  --save-frames

naturallab calibrate verify \
  --video extracted/calibration/camera-01-verification/camera-01.mp4 \
  --bundle calibration/camera-01/floor/calibration-bundle.yaml \
  --inner-cols 7 --inner-rows 7 \
  --square-size-mm 30 \
  --output-dir calibration/camera-01/verification \
  --save-frames
```

Replace the example paths, board dimensions, and square size. Open the saved
annotated frames and reports. Accept the camera only when the verification
report passes and the marked corners are visibly correct. Re-record a failed
calibration instead of adding a manual correction factor.

To place multiple cameras in one shared room coordinate system, record the same
stationary board placements simultaneously in every included camera. Extract
that shared XDF once; its output folder will contain one MP4 per named camera.
It will also contain one matching `<camera-name>_timestamps.csv` file. Keep the
MP4s and these CSV files together, without trimming or re-encoding them.

In the copied manifest, point each `video` entry at the matching MP4 and each
`timestamp_csv` entry at its matching CSV. NaturalLab then calculates the
relative video start offsets automatically. Use a numeric `time_offset_seconds`
instead only for externally produced videos whose alignment has been
established separately; zero is safe only when their starts are already aligned
within the manifest's time tolerance. Then run:

```bash
naturallab calibrate extrinsics \
  --manifest calibration/shared-board.yaml \
  --output-dir calibration/shared-room \
  --save-frames
```

Start from
[`examples/shared_board_extrinsics.yaml`](../examples/shared_board_extrinsics.yaml).
Use `room-registration.yaml` only when `extrinsics-report.json` says
`"status": "pass"` and the saved board correspondences look correct in every
camera.
The detailed [automatic calibration reference](calibration_workflow.md)
explains file preparation and report fields. Floor registration supports room
movement measurements; read [multiview 3D readiness](multiview_3d_readiness.md)
before attempting elevated 3D skeleton reconstruction.

## 9. Choose what to analyze

| Research goal | Is additional model training required? | Start here |
|---|---:|---|
| Detect and track people in video | No | [Software quick start](quickstart.md#2-track-people-in-an-existing-video) |
| Measure a person's movement on the floor | No; verified camera calibration is required | [Calibration section](#8-calibrate-each-fixed-room-camera-automatically) |
| Assign roles such as child and caregiver | No new training; the approved Qwen service is required | [Qwen/DeepSORT path](quickstart.md#3-use-the-current-operational-qwendeepsort-path) |
| Find the lab's own toys or materials | Usually not at first; use reference-image prototypes | [Object detector setup](object_detection_guide.md) |
| Detect a new category reliably when prototypes are insufficient | Usually yes; training is currently external to NaturalLab | [Training handoff](object_detection_guide.md#when-reference-images-are-not-enough) |

People tracking already uses pretrained detectors. You do not train a model for
ordinary movement tracking. Calibration is needed only when the output must be
expressed as positions or distances on the real floor.

## 10. Repeat this short check before every study session

- Confirm that camera mounts and image settings have not changed.
- Confirm that every device has its recorded address and intended role.
- Open `naturallab record`, enter the session-only camera password, and start
  the selected streams.
- In LabRecorder, click **Update** and verify that every required stream appears
  exactly once and advances.
- Make a brief disposable recording and inspect it before participants enter.
- Preserve the original recording, configuration record, calibration reports,
  model names, and processing command with the study data.

Recalibrate a camera whenever it is moved or its lens, zoom, focus, crop,
orientation, or resolution changes.

## Other camera configurations

The simpler source-checkout script accepts any matching list of camera URLs and
names. It is useful when cameras do not share credentials or when more than four
views are needed:

```bash
python scripts/stream_synchronized_sensors.py \
  --cameras "RTSP_URL_1,RTSP_URL_2" \
  --camera-names "camera-01,camera-02"
```

There must be exactly one name for every URL. Add optional `--neon-ips` and
matching `--neon-names`, or `--realsense`, only after the camera-only recording
works. Credentialed URLs can appear in the shell's process list and history, so
prefer the recording window for cameras that share one login.
