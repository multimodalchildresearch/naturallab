# Prepare objects for NaturalLab detection

NaturalLab offers two different starting points:

- **People and movement:** use a pretrained people detector. No additional
  training is required for ordinary person tracking.
- **Your lab's own toys or materials:** first try reference-image prototypes.
  You photograph the objects; NaturalLab uses those photographs without a
  conventional training run.

A separately trained model is only the next step when reference images are not
reliable enough for the study. NaturalLab does not currently include a complete
annotation and model-training workflow.

## Choose the simplest suitable path

| Goal | What you prepare | Training needed? |
|---|---|---:|
| Track people through a room | A representative test video | No |
| Label tracks as roles such as child or caregiver | Clear role descriptions and an approved Qwen service | No new training |
| Find the exact toys or materials used in the study | Reference photographs grouped by category | Usually no |
| Detect a broad new object category across changing rooms and objects | A labeled image dataset and an external training workflow | Usually yes |

Prototype detections are independent observations in each image or video
frame. They are not object tracks, unique-object counts, or proof that a person
interacted with an object.

## 1. Decide the labels before taking photographs

Write down the categories needed by the study, for example:

- `red_ball`
- `picture_book`
- `stacking_cup`

Use short, stable folder names with letters, numbers, hyphens, or underscores.
Do not mix visually unrelated objects in one category. If the distinction
matters to the research question, give the objects separate categories.

If the study only needs one particular physical ball, photograph that exact
ball. If the study needs to recognize many different balls, include every
intended kind and test whether one combined category remains reliable.

## 2. Make a useful reference-photo set

A practical first attempt is 5–10 photographs per category. This is a starting
point, not an accuracy guarantee. Each photograph should contain one intended
object and make it easy to see.

Include:

- the front, back, sides, and any commonly visible tilted view;
- near and far appearances similar to the real camera view;
- the room lighting used during recording;
- a few realistic partial occlusions; and
- more than one background, so the background does not become the only visual
  clue.

Start with clear, tightly framed photographs, then add a few crops taken from
the actual room cameras. Remove blurry images and images where the target is
tiny or hidden beyond recognition.

Keep a separate short test recording that is **not** used for the reference
photos. Otherwise the first check will be unrealistically easy.

## 3. Arrange the folders

Inside the NaturalLab checkout, make a folder called `private-study-data` and
one immediate reference-image subfolder per final label. NaturalLab's
`.gitignore` excludes this private folder from ordinary Git commits:

```text
private-study-data/
├── reference_images/
│   ├── red_ball/
│   │   ├── front.jpg
│   │   ├── side.jpg
│   │   └── room_view.jpg
│   ├── picture_book/
│   │   ├── cover.jpg
│   │   └── angled.jpg
│   └── stacking_cup/
│       ├── upright.jpg
│       └── side.jpg
└── prototype-check.mp4
```

The Git rule prevents an easy accident; it is not encryption, access control,
or a backup. Keep this folder on approved study storage and follow the lab's
data-protection rules. Adding, removing, or replacing a category means
rebuilding the prototype file.

## 4. Build the prototype file

Install the object-analysis components once:

```bash
python -m pip install -e ".[gaze]"
```

Then create the prototypes:

```bash
python scripts/detect_custom_objects.py create-prototypes \
  --images private-study-data/reference_images \
  --output private-study-data/prototypes.h5 \
  --device auto
```

The command prints the category folders it found. Check that every expected
label appears. Every discovered reference image must decode and produce an
embedding; one failure rejects the complete build instead of averaging only the
survivors. The HDF5 file is replaced atomically only after the whole build
succeeds, so a failed rebuild leaves the prior file unchanged. The first run
may download the required pretrained model files; an offline lab should
download and preserve them before data collection.

Keep the resulting `private-study-data/prototypes.h5` with the reference
photographs and study configuration. Rebuilding it after changing photographs
creates a different analysis input.

## 5. Test on footage the prototypes have not seen

Start with a short, non-participant clip that includes:

- the target objects both near and far from the camera;
- ordinary empty-room views;
- other objects that look similar; and
- realistic handling, turning, and partial hiding.

Run:

```bash
python scripts/detect_custom_objects.py detect \
  --input private-study-data/prototype-check.mp4 \
  --prototypes private-study-data/prototypes.h5 \
  --output private-study-data/prototype-check-default \
  --device auto \
  --save-frames \
  --frame-interval 30
```

Open `private-study-data/prototype-check-default/annotated_frames/` and look
through the marked frames. The result folder also contains:

- `detections.csv`, with one row per accepted detection; and
- `detection_summary.csv`, with counts and the average prototype match score
  for each category.

For a successfully processed input, an empty table means that no candidate
passed the chosen thresholds. The command rejects an empty image folder and a
video that lacks a trustworthy frame count or decodes fewer requested frames
than its container reports. A failed run publishes no partial result folder.
Match scores help compare settings for this exact setup; they are not
calibrated probabilities.

For a folder of still images, use the folder as `--input`. With
`--save-frames`, NaturalLab writes one annotated copy per input image.

## 6. Help the detector look for the right kind of region

The final output labels always come from the prototype folder names. Optional
search phrases can make the first object search more specific:

```bash
python scripts/detect_custom_objects.py detect \
  --input private-study-data/prototype-check.mp4 \
  --prototypes private-study-data/prototypes.h5 \
  --output private-study-data/prototype-check-search-phrases \
  --categories '{"red_ball":["toy ball","red ball"],"picture_book":["book","child picture book"]}' \
  --save-frames
```

Keep the names on the left consistent with the folder names for readability.
The phrases on the right tell the first search stage what kinds of objects may
be present; they do not create new final labels.

Use a new output folder name for every configuration. NaturalLab refuses to
write into a non-empty result folder so annotated frames from two settings
cannot be mixed accidentally.

## 7. Adjust settings only on a dedicated tuning set

Begin with the defaults. If the annotated frames show a consistent problem,
change one setting at a time:

| What you see | First change to try |
|---|---|
| The object is never proposed at all | Add a clearer search phrase or lower `--threshold` slightly |
| The object is proposed but rejected as the wrong prototype | Improve the reference photos or lower `--match-threshold` slightly |
| Many unrelated objects are accepted | Raise `--match-threshold` or use more distinctive reference photos |
| Processing is too slow for an initial check | Increase `--frame-skip`, then return to the planned setting for final analysis |

Do not keep adjusting settings on the final evaluation recordings. Choose them
with a tuning set, record them, and then measure performance on different held-
out footage.

A useful manual check is a small table of representative frames: mark whether
each expected object was found, whether its label was correct, and whether any
unrelated object was marked. This provides study-specific evidence even when
no external ground-truth dataset exists.

## When reference images are not enough

Consider a trained detector when:

- objects are very small, frequently hidden, or visually similar;
- the intended category contains many different-looking objects;
- misses and false detections cannot both be reduced with better references
  and fixed thresholds; or
- the study requires a frozen detector validated across multiple rooms or
  camera systems.

Training is currently performed outside NaturalLab. A sound handoff has these
steps:

1. Record representative material from every planned camera type, distance,
   room condition, and object appearance.
2. Split complete sessions into training, tuning, and final test sets. Do this
   before extracting nearby frames, so almost identical frames cannot appear in
   both training and testing.
3. Draw a bounding box and category label around every target object in the
   selected frames. Write a short labeling rule for ambiguous and partly hidden
   cases.
4. Train a detector in a maintained external framework. For a YOLO route, use
   the current official
   [Ultralytics training guide](https://docs.ultralytics.com/modes/train/) and
   [detection-dataset format](https://docs.ultralytics.com/datasets/detect/).
   Review NaturalLab's [third-party notices](../THIRD_PARTY_NOTICES.md) before
   selecting that separately licensed route.
5. Freeze the chosen weights and settings, then evaluate once on the untouched
   test sessions. Report misses, wrong labels, and false detections for each
   camera condition that matters to the study.
6. Preserve the dataset version, label definitions, split, software version,
   training configuration, weights, and evaluation report.

NaturalLab's current people-tracking script accepts a YOLO weights path with
`--yolo-model`, but it keeps only class index `0` and interprets that class as
`person`. A custom people model therefore needs `person` at class index `0`:

```bash
python scripts/track_people_in_video.py \
  --input session.mp4 \
  --output results \
  --detector yolo \
  --tracker kalman \
  --yolo-model path/to/best.pt
```

Do not use that command to track an arbitrary object category: it would treat
class `0` as a person and produce person-style movement output. Connecting a
separately trained arbitrary-object detector currently requires a small code
adapter and its own validation; there is no stable `naturallab train` or custom
detector plug-in command yet.

## What to keep with a study

Keep these items together under a non-identifying study/session ID:

- the label list and plain-language definition of each category;
- the reference photographs and generated `prototypes.h5` file;
- the exact detection command and thresholds;
- the tuning and held-out test recordings or their controlled references;
- the manual review table and known failure conditions; and
- for a trained model, the dataset split, annotation rules, software version,
  training configuration, final weights, and evaluation report.

Do not place study images, room images, annotations, participant-derived
results, identities, or confidential metadata in the public repository or a
public issue report, whether or not a person seems identifiable.
