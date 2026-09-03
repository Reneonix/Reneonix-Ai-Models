# Scripts/videos

Live-video-file inference scripts.

**Single-model (YOLO-only) benchmarking:**
```
python video_predict.py [path/to/video.mp4]         # plain inference
python video_predict_sahi.py [path/to/video.mp4]      # SAHI (6-tile) inference
```

**Two-model pipelines** (YOLO detects/localizes only; a separate ResNet classifies each crop;
a tracker gives each physical object a stable ID so counting is per-object, not per-frame
detection) — three variants trading accuracy for speed:
```
python yr_sahi_botsort.py [video]   # 6-tile SAHI + BoT-SORT+ReID - best small-object/dense-scene accuracy, slowest
python yr_byte.py [video]           # no SAHI + ByteTrack - fastest, worse on small/distant objects
python yr_byte_sahi.py [video]      # 6-tile SAHI + ByteTrack - middle ground
```

**Two-model, lazy + quality-gated classification:**
```
python yr_byte_lazy.py [video]      # yr_byte.py, but classifies each track ONCE and caches
```
Same YOLO + ByteTrack + ResNet pipeline as `yr_byte.py`, but with two upgrades over simply
reclassifying every frame: (1) a GOOD CROP gate - a crop is only worth classifying if it's big
enough, sharp, not overlapping another tracked box (a cheap occlusion proxy), and from a
confident detection, otherwise it's skipped and retried next frame; (2) PROBABILITY
ACCUMULATION instead of hard vote-counting - every good crop's full softmax vector is summed
into that track's evidence, so a confident frame counts for more than an uncertain one, and a
track locks once its leading class is clearly ahead (or a hard attempt cap is hit). A track
that goes too long with zero evidence is force-classified once, gate bypassed, so nothing is
left permanently unlabeled. Solves "FPS drops as object count rises" the same way as before
(most objects cost ~1 ResNet call over their whole life, not one per frame), now with more
trustworthy evidence per call. The printed report shows exactly how many ResNet calls were
saved, and how many crops the quality gate rejected.

**Two-model, recommended architecture** (fixes real over-counting seen with `yr_byte_lazy.py` -
238 counted vs. ~41 real objects on a busy conveyor):
```
python yr_botsort_lazy.py [video]   # BoT-SORT+ReID + quality-gated lazy classification + count-once-at-a-line
```
`yr_byte_lazy.py`'s over-count wasn't a classifier problem - it was ByteTrack (motion/IoU-only)
losing and re-issuing track IDs under fast motion/occlusion, so one physical object got counted
several times. This script swaps in BoT-SORT+ReID (appearance-based tracking, same tracker as
`yr_sahi_botsort.py`, plain-YOLO not SAHI-tiled) for real identity persistence, uses the same
good-crop-gated, probability-accumulating lazy classification as `yr_byte_lazy.py`, and adds a
third, independent safeguard: a track is only added to the final count the moment it crosses a
configurable counting line (`COUNTING_LINE_AXIS`/`COUNTING_LINE_POSITION`), and never more than
once - even a reused ID can't be double-counted. Also lowers `CONF` and raises `TRACK_BUFFER` to
reduce the detection-flicker/short-gap causes of ID loss in the first place, and runs ResNet in
FP16. Includes live tracking diagnostics (`[NEW TRACK]`/`[TRACK LOST]`/`[VIDEO END]` events,
per-track lifetimes, and a CSV log) for verifying the tracker itself is behaving, not just the
final numbers - see the script's own module docstring.
**Set `COUNTING_LINE_AXIS`/`COUNTING_LINE_POSITION` to match your actual camera framing/belt
direction before trusting the counts** - an object that never crosses the line isn't counted,
by design.

All scripts open a live OpenCV window (paced to the source video's own FPS), and print a
benchmark report (inference/tracking/classification latency, end-to-end latency, achievable
FPS, and object/detection counts per class) on quit (`q`).

`yolo26n-reid.onnx` lives here alongside `yr_sahi_botsort.py` and `yr_botsort_lazy.py` - its
ReID appearance-embedding model, referenced by a relative path (`REID_MODEL =
"yolo26n-reid.onnx"`), so run either script from inside this folder (or keep this file next to
it if you move it elsewhere).

The five `yr_*.py` scripts import shared ROI-cropping helpers from `../roi/roi_utils.py`
(added via `sys.path.insert(...)` at the top of each, since `Scripts/` was split into
`camera/`, `videos/`, `images/`, `roi/` subfolders - see `../roi/README.md` for the ROI
selector tool itself).

To point any of these at a different experiment, change its `WEIGHTS` / `YOLO_WEIGHTS` /
`RESNET_WEIGHTS` (and `RESULTS_CSV`, where present) constant to the new
`experiments/exp00N_.../weights/best.pt` path.
