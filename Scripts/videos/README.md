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

**Two-model, lazy classification:**
```
python yr_byte_lazy.py [video]      # yr_byte.py, but classifies each track ONCE and caches
```
Same YOLO + ByteTrack + ResNet pipeline as `yr_byte.py`, but instead of reclassifying every
tracked object every frame, it classifies once per track and locks the result (immediately if
one prediction is confident enough, otherwise after a few agreeing votes, capped either way) -
solves "FPS drops as object count rises," since a busier conveyor no longer means a
proportionally bigger ResNet batch every frame. The printed report shows exactly how many
ResNet calls were saved vs. `yr_byte.py`'s classify-every-frame approach. Same object counts,
same accuracy - just far fewer redundant classifications.

All scripts open a live OpenCV window (paced to the source video's own FPS), and print a
benchmark report (inference/tracking/classification latency, end-to-end latency, achievable
FPS, and object/detection counts per class) on quit (`q`).

`yolo26n-reid.onnx` lives here alongside `yr_sahi_botsort.py` - its ReID appearance-embedding
model, referenced by a relative path (`REID_MODEL = "yolo26n-reid.onnx"`), so run that script
from inside this folder (or keep this file next to it if you move it elsewhere).

The four `yr_*.py` scripts import shared ROI-cropping helpers from `../roi/roi_utils.py`
(added via `sys.path.insert(...)` at the top of each, since `Scripts/` was split into
`camera/`, `videos/`, `images/`, `roi/` subfolders - see `../roi/README.md` for the ROI
selector tool itself).

To point any of these at a different experiment, change its `WEIGHTS` / `YOLO_WEIGHTS` /
`RESNET_WEIGHTS` (and `RESULTS_CSV`, where present) constant to the new
`experiments/exp00N_.../weights/best.pt` path.
