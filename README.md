# Waste Identification

Object detection + classification for waste-material identification on a conveyor/pour setup.
Two datasets, two tasks:

- **6-class detection**: **aluminium, plastic, metal, stone, ceramic, glass** — the main
  pipeline (exp001, exp003, exp005, exp006).
- **Single-class "glass" detection** — a narrower glass-cullet detector (exp002), trained
  separately on its own dataset.
- **5-class AGI detection + classification** — a separate external dataset/taxonomy
  (`ferrous, plastic, stone, ceramic, glass`, no `aluminium`), its own two-model pipeline
  (exp007 detector + exp008 classifier), never mixed with the 6-class pipeline above.

## Project structure

```
Wastes_identification/
├── data/
│   ├── raw/                       flat, unsplit copy of exp001_exp003_exp006 - see its own README
│   ├── versions/
│   │   ├── exp001_exp003_exp006/  7,389 train / 1,847 val, 6-class - used by exp001, exp003,
│   │   │                            AND exp006's fine-tuning stage
│   │   ├── exp002/                2,350 train / 587 val, single-class "glass" - used by exp002
│   │   ├── exp004/                294,959 / 73,448 crops, 6-class classification (ImageFolder) - used by exp004
│   │   ├── exp005/                7,389 train / 1,847 val, same split as exp001_exp003_exp006 - used by exp005
│   │   ├── exp006/                1,631 train / 408 val, 6-class - exp006's base-training stage
│   │   ├── exp007_AGI/            23,545 train / 5,887 val, 5-class (ferrous/plastic/stone/ceramic/glass) - used by exp007
│   │   └── exp008_AGI/            2,287,754 / 569,306 crops, 5-class classification (ImageFolder) - used by exp008
│   └── scripts/                   dataset build/import/convert scripts (see below)
├── src/
│   ├── exp001_yolov8l.py           exp001's recipe (also targets exp009 if re-run) - yolov8l, differential LR
│   ├── exp002_yolo26s.py           exp002's recipe (also targets exp010 if re-run) - yolo26s, single-class glass
│   ├── exp003_yolo26l_p2.py        exp003's recipe - yolo26l + P2 head architecture experiment
│   ├── exp004_resnet50.py          exp004's recipe - ResNet-50 second-stage material classifier
│   ├── exp005_rtdetr.py            exp005's recipe - RT-DETR-L (Baidu), via Ultralytics' own RTDETR class
│   ├── exp006_yolo26s_finetuned.py exp006's recipe (also targets exp011 if re-run) - yolo26s, TWO-STAGE
│   ├── exp007_yolov8l_agi.py       exp007's recipe - yolov8l, same differential-LR methodology as
│   │                                  exp001, trained on the separate AGI dataset (5 classes)
│   └── exp008_resnet50_agi.py      exp008's recipe - ResNet-50 second-stage classifier for exp007,
│                                      trained on cropped AGI-dataset boxes (5 classes)
├── experiments/
│   ├── exp001_yolov8l/            weights/, config.yaml, wandb/ (local sync cache)
│   ├── exp002_yolo26s/            same - imported from a laptop-trained run
│   ├── exp003_yolo26l_p2/         same - YOLO26+P2 architecture test
│   ├── exp004_resnet50/           same - ResNet-50 classifier
│   ├── exp005_rtdetr_l/           same - RT-DETR-L
│   ├── exp006_yolo26s_finetuned/  same, PLUS a base_stage/ subfolder (weights/, config.yaml) for
│   │                                the two-stage pipeline's first stage - see below
│   ├── exp007_yolov8l_AGI/        same - trained on the separate AGI dataset (5 classes, no
│   │                                aluminium) - "AGI" deliberately kept in every folder name
│   │                                this experiment touches, see below
│   ├── exp008_resnet50_AGI/       same - ResNet-50 classifier pairing with exp007, trained on
│   │                                cropped exp007_AGI boxes (data/versions/exp008_AGI)
│   └── testing/                   NOT versioned experiments - stock/reference models for sanity-testing:
│       ├── rtdetr/                 stock rtdetr-l.pt (COCO-pretrained) + testing.py
│       └── rfdetr/                 Google's real CircularNet ONNX model + testing.py (see its own docstring)
├── results/
│   ├── exp001/ … exp006/          metrics, curves, confusion matrix, batch preview images, config.json
│   │                                (exp006/base_stage/ holds stage 1's own results separately)
│   ├── testing_images/             drop test images here for Scripts/images/*.py to process
│   └── predicted_images/           one expNNN/ folder per model, populated by Scripts/images/*.py
├── models/                        deliverable copies of chosen weights (exp001_yolov8l_best.pt, ...)
├── Scripts/                        inference / live demo / benchmarking / ROI tools (see below) -
│   ├── camera/                       live-webcam inference (single-model)
│   ├── videos/                       live-video-file inference (single-model + yr_*.py pipelines)
│   ├── images/                       batch-image inference across every trained model at once
│   └── roi/                          shared region-of-interest selector
├── videos/                        demo videos for inference testing
├── notebooks/                     exploration/debugging
├── requirements.txt
└── README.md
```

**exp005 (RF-DETR / "CircleNet") was tried and removed 2026-08-28** after three unrecoverable
training crashes — two silent hangs, one confirmed native access-violation segfault in
`python311.dll` roughly 5 minutes after its first epoch finished, every time with zero
checkpoint ever saved. The `rfdetr` pip package is gone. **The exp005 number was reclaimed**
for RT-DETR (Baidu's *different*, unrelated DETR-family architecture, trained via
Ultralytics' own `RTDETR` class - no separate package, no crashes) - `experiments/exp005_rtdetr_l/`
is that run, not the removed one.

**exp007 and exp008 are a deliberate exception to the naming convention below**: they train on a
separate external dataset (`data/raw/AGI/` → `data/versions/exp007_AGI`, 5 classes - `ferrous,
plastic, stone, ceramic, glass`, no separate `aluminium`), and **every folder they touch keeps
"AGI" in its name** (`data/versions/exp007_AGI`, `data/versions/exp008_AGI`,
`experiments/exp007_yolov8l_AGI`, `experiments/exp008_resnet50_AGI`, `results/exp007_AGI`,
`results/exp008_AGI`, `models/exp007_yolov8l_AGI_best.pt`, `models/exp008_resnet50_AGI_best.pt`)
so neither is ever confused with the main 6-class pipeline. exp008 is exp007's second-stage
classifier, the same two-model pattern as exp004 pairs with exp001/exp003/exp005/exp006 - do
**not** pair exp007/exp008 with exp004, or exp001/exp003/exp005/exp006 with exp008, across the
taxonomy boundary. Their metrics are **not comparable** to exp001/exp003/exp004/exp005/exp006 -
different classes, different data.

**exp006 is a two-stage pipeline**, imported from a laptop-trained run (`KAVIYA/` at the
project root): base-trained on its own smaller dataset (`data/versions/exp006`), then
**fine-tuned** on this project's own main dataset (`data/versions/exp001_exp003_exp006` -
verified byte-identical to what was `KAVIYA/fine_tune_dataset/`). Its
`experiments/exp006_yolo26s_finetuned/weights/` holds the final fine-tuned model (the
deliverable); `base_stage/weights/` holds the intermediate base-trained checkpoint, kept for
provenance only.

## The naming convention (read this before training something new)

Every experiment folder is named `exp{id}_{model}` — e.g. `exp001_yolov8l`. **Dataset
provenance deliberately does NOT live in this folder name** (an earlier version of this
convention appended it, e.g. `exp001_yolov8l_exp001_exp003` - dropped as redundant/noisy since
it's already recorded in full inside that folder's own `config.yaml` and in its wandb run name)
- to see everything you've ever trained, just look at the `experiments/` folder:

```
experiments/
├── exp001_yolov8l/                -> yolov8l, trained on data/versions/exp001_exp003_exp006 (see config.yaml)
├── exp002_yolo26s/                -> yolo26s, single-class glass, trained on data/versions/exp002
├── exp003_yolo26l_p2/             -> yolo26l + P2 head, trained on data/versions/exp001_exp003_exp006
├── exp004_resnet50/                -> resnet50, trained on data/versions/exp004
├── exp005_rtdetr_l/                -> RT-DETR-L, trained on data/versions/exp005
└── exp006_yolo26s_finetuned/       -> yolo26s, two-stage: data/versions/exp006 then
                                        data/versions/exp001_exp003_exp006 (fine-tune)
```

Each `src/*.py` script still computes a separate `RUN_NAME` (`exp{id}_{model}_{dataset}`,
full provenance) used only for its wandb run name and artifact naming - the local
`experiments/`/`models/` folder/file names use the shorter `EXP_FOLDER` (`exp{id}_{model}`).

**Dataset folders are named after the experiment(s) that actually use them**, not an opaque
version counter — `data/versions/exp001_exp003_exp006` because exp001, exp003, AND exp006's
fine-tuning stage all train on that exact same dataset; `data/versions/exp004` because only
exp004 uses it (even though it's *derived from* exp001_exp003_exp006 by cropping). If a
dataset is ever shared by more experiments later, extend the name further, don't rename what's
already there — a dataset folder's name is append-only once the first experiment using it
exists, same spirit as rule 1 below applied to the name itself, not just the content. (This
already happened once: the folder was `v1`, then `exp001_exp003`, now `exp001_exp003_exp006`.)

Two rules that make this reliable long-term:

1. **A dataset version is immutable once used in an experiment.** If you get new images later,
   that's never "add them into an existing dataset folder" — it's a new folder with its own
   README documenting what changed. This means an experiment's dataset can never silently
   drift after the fact.
2. **Every experiment gets the next sequential ID.** Taken so far: 1–8 (5 was tried and
   abandoned once, then reclaimed - see above; 7 and 8 are the AGI-dataset runs, a deliberate
   naming exception - see above). `exp009` and `exp010` are already reserved
   (`src/exp001_yolov8l.py`'s and `src/exp002_yolo26s.py`'s own future-rerun targets) —
   `exp011` is the actual next free one (`src/exp006_yolo26s_finetuned.py`'s target, bumped
   from its original exp008 once that number was claimed by the AGI ResNet run instead). Note
   the filename of a script always says the experiment it *originally* produced, not
   whatever future run its `EXP_ID` constant currently targets.

### What to do when you want to train something new

1. If you have new data: create `data/versions/<name>/` (named after the experiment(s) that
   will use it — check step 2 first to know the ID) with its own `images/{train,val}` +
   `labels/{train,val}`, `data.yaml`, and a `README.md` describing the source/counts/date.
   Never edit an existing version's files in place.
2. Copy the closest existing `src/*.py` script as a template and edit the identity constants
   at the top:
   ```python
   EXP_ID = 9                # next free number - check experiments/ to confirm
   MODEL_NAME = "yolov8l"    # whatever architecture you're training
   DATASET_VERSION = "exp001_exp003_exp006"   # which data/versions/ folder to train on
   ```
   Everything else (the experiment folder name, `data.yaml` path, results folder) is derived
   from these automatically — you don't hand-type any paths.
3. Run it. When training finishes, weights + config land in `experiments/exp00N_.../`, and
   plots/metrics/results.csv are automatically moved into `results/exp00N/`, and `best.pt` is
   automatically copied into `models/` — you don't have to sort any of that out by hand.
4. If a run crashes or gets interrupted, just re-run the same command — it auto-detects an
   existing checkpoint at its target experiment folder and resumes from it.

## Setup

1. Install PyTorch with CUDA **first**, matching your GPU (this project targets an RTX 5080,
   which needs the cu128 build):

   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

   Don't skip this step or install it via `requirements.txt` — the plain PyPI wheels are
   CPU-only, and `pip install`/`--upgrade ultralytics` can silently pull them back in later
   too, replacing a working CUDA install. If that happens, just re-run the command above.

2. Install everything else:

   ```
   pip install -r requirements.txt
   ```

3. Verify CUDA is actually being used:

   ```
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

## Training

```
cd src
python exp001_yolov8l.py            # exp001's recipe (targets exp009 if re-run): yolov8l, differential LR, from COCO weights
python exp002_yolo26s.py            # exp002's recipe (targets exp010 if re-run): yolo26s, single-class glass
python exp003_yolo26l_p2.py         # exp003's recipe: yolo26l + P2 head, small-object detection
python exp004_resnet50.py           # exp004's recipe: ResNet-50, second-stage material classifier
python exp005_rtdetr.py             # exp005's recipe: RT-DETR-L (Baidu), via Ultralytics' own RTDETR class
python exp006_yolo26s_finetuned.py  # exp006's recipe (targets exp008 if re-run): yolo26s, TWO-STAGE (base + fine-tune)
python exp007_yolov8l_agi.py        # exp007's recipe: yolov8l, same differential-LR methodology as exp001, on the AGI dataset (5 classes)
python exp008_resnet50_agi.py       # exp008's recipe: ResNet-50 second-stage classifier pairing with exp007, on cropped AGI boxes (5 classes)
```

`exp001_yolov8l.py` uses a differential learning rate: the first 5 backbone layers are frozen
(generic low-level filters), the rest of the backbone/neck trains at a low LR, and the
detection head trains at a high LR — a cosine schedule anneals both down over the run.
`exp003_yolo26l_p2.py` trains a newer architecture (YOLO26 with an added P2/160×160 detection
head, for better small-object detection) at a single uniform LR. `exp004_resnet50.py` is stage
two of a two-model pipeline: YOLO detects/localizes an object, then this classifier looks at
just that cropped region to assign the final material label (heavy color/lighting
augmentation, since the whole point is robustness to glare/lighting the detector's own head
struggled with). `exp002_yolo26s.py` is a narrower single-class detector (glass cullet only),
trained on its own smaller dataset. `exp005_rtdetr.py` fine-tunes RT-DETR (a DETR-family
transformer - no anchor boxes/NMS, unlike the CNN-based YOLO variants) via **Ultralytics
itself** (confirmed to share YOLO's `.train()`/callback API) - not to be confused with the
removed RF-DETR attempt, an unrelated architecture from a different company (Roboflow, not
Baidu) with its own separate package. `exp006_yolo26s_finetuned.py` runs base training then
fine-tuning as one script, in sequence, each stage independently resume-safe.

`exp007_yolov8l_agi.py` reuses exp001's exact differential-LR methodology unchanged, trained
instead on the separate AGI dataset (batch bumped to 16, with backbone/head LR restored to
exp001's original pre-batch-8-scaling values of 0.0005/0.01 to match - see the script's own
comments).

All seven track losses/metrics to wandb per epoch, and auto-resume from a crash (see the
naming-convention section above for the exact workflow; `exp005_rtdetr.py`'s Ultralytics-based
resume works the same as every other YOLO/RTDETR script here).

| | exp001 (yolov8l) | exp003 (yolo26l + P2) | exp006 (yolo26s, fine-tuned) |
|---|---|---|---|
| Precision | 0.945 | 0.947 | 0.944 |
| Recall | 0.932 | 0.930 | 0.927 |
| mAP50 | 0.975 | 0.977 | 0.975 |
| mAP50-95 | 0.935 | **0.951** | 0.943 |

All three columns are measured on the **exact same held-out validation set** (1,847 images) -
verified byte-identical file lists for exp006's dataset, not just a similar-sounding number.
exp003 improved mAP50-95 across every class, most notably `stone` (0.971) and `ceramic`
(0.980) — consistent with the P2 head's purpose of catching smaller/harder objects the plain
P3–P5 model missed. exp006 lands between exp001 and exp003, despite being a smaller/faster
architecture (yolo26s vs. yolov8l/yolo26l) - the base-training stage on its own separate
1,631-image dataset likely helped before ever seeing this project's own data.

**exp007 (yolov8l, AGI dataset, 5 classes)**: precision 0.934, recall 0.934, mAP50 0.973,
mAP50-95 0.931, 100/100 epochs in 6h58m. **Not comparable to the table above** - different
classes (`ferrous, plastic, stone, ceramic, glass`, no separate aluminium), different data,
different validation set.

| | exp002 (yolo26s, glass-only) | exp004 (resnet50 classifier) | exp006 stage 1 (base, own val set) |
|---|---|---|---|
| Precision | 0.694 | — | 0.994 |
| Recall | 0.553 | — | 0.981 |
| mAP50 | 0.602 | — | 0.994 |
| mAP50-95 | 0.370 | — | 0.720 |
| Val accuracy | — | 0.9937 | — |

exp002's numbers are meaningfully lower than the main 6-class detectors — glass cullet
fragments are a genuinely harder single-class task (irregular shapes, no consistent silhouette
the way a whole bottle has) on a much smaller dataset (2,350 vs. 7,389 train images). exp006's
stage-1 column is measured on its *own* 408-image validation split (not the shared one above)
— not directly comparable to the other columns; see `results/exp006/config.json`.

**Known gotcha (fixed, but worth knowing):** if you ever see a `PermissionError` inside
`wandb_log_train_end` right after a run finishes, it means the results-split loop tried to move
the `wandb/` folder itself — `wandb.finish()` doesn't synchronously release its background
service's file handles, so that folder must never be touched by the move loop. Every training
script explicitly skips it now.

**Another one, learned the hard way:** an overnight unattended run can die silently with zero
trace in the terminal if (a) Windows Update forces a restart mid-run, or (b) a DataLoader
worker deadlocks on Windows (this project hit this with `torchvision`/ResNet's DataLoader).
Before leaving a long run unattended: pause Windows Update, and if you see a run's log stop
advancing for an unusually long time with the GPU still showing utilization but no new logged
steps, don't assume it's just slow — check `nvidia-smi`, the Windows Application/System event
logs (`Get-WinEvent`), and the log timestamps together. A training process can also die with an
orphaned background process (e.g. wandb's own sync service) still holding a file lock inside
its experiment folder afterward - check `Get-Process` broadly (not just `python`) before trying
to rename/delete that folder.

## Testing reference/stock models (not fine-tuned)

`experiments/testing/` holds two models kept for comparison, neither trained by this project:

```
python experiments/testing/rtdetr/testing.py [video]   # stock rtdetr-l.pt (COCO-pretrained, 80 generic classes)
python experiments/testing/rfdetr/testing.py [video]    # Google's real CircularNet ONNX model (50 classes, own taxonomy)
```

Both take an optional video path argument (defaults to a demo video) and print a live
annotated window + latency benchmark on quit (`q`). The CircularNet one downloads/reimplements
Google's own verified preprocessing/postprocessing logic (from the real
`tensorflow/models` source, not guessed) and runs locally via `onnxruntime` (CPU - the CUDA
execution provider needs a separately-installed `cudnn64_9.dll` not present on this machine).

## Inference & benchmarking

`Scripts/` is split into four subfolders, each with its own README:

```
Scripts/
├── camera/    live-webcam inference (single-model)
├── videos/    live-video-file inference (single-model + the two-model yr_*.py pipelines)
├── images/    batch-image inference across EVERY trained model at once (see below)
└── roi/       shared region-of-interest selector used by videos/yr_*.py
```

**`Scripts/camera/`** (live webcam) and **`Scripts/videos/`** (video files) - single-model:
```
python video_predict.py [path/to/video.mp4]         # live video, plain inference (exp003 weights)
python video_predict_sahi.py [path/to/video.mp4]      # live video, SAHI (tiled) inference (exp003 weights)
python live_prediction.py [camera_index]                # live webcam, plain inference (exp001 weights)
python live_prediction_sahi.py [camera_index]           # live webcam, SAHI inference (exp001 weights)
```

**`Scripts/videos/`** also has the **two-model pipelines** (YOLO detects/localizes only; a
separate ResNet classifies each crop; a tracker gives each physical object a stable ID so
counting is per-object, not per-frame detection) — three variants trading accuracy for speed:
```
python yr_sahi_botsort.py [video]   # 6-tile SAHI + BoT-SORT+ReID - best small-object/dense-scene accuracy, slowest
python yr_byte.py [video]           # no SAHI + ByteTrack - fastest, worse on small/distant objects
python yr_byte_sahi.py [video]      # 6-tile SAHI + ByteTrack - middle ground
```
Each reports per-model latency (YOLO/SAHI, tracker, ResNet) separately, combined end-to-end
latency/achievable FPS, and final unique-object counts per class (majority vote of each
tracked object's ResNet predictions across its lifetime, not one count per frame it was
visible in).

**`Scripts/images/`** - batch inference over `results/testing_images/` against **every model
in the project**, not just one at a time:
```
python image_predict.py         # plain inference
python image_predict_sahi.py     # SAHI (6-tile) inference
```
Each has a `MODEL SELECTION` block with one candidate line per experiment (exp001–exp006) -
exactly one uncommented at a time - and writes annotated output to
`results/predicted_images/<expNNN>/`, with the output folder and validation-accuracy lookup
both derived automatically from whichever `WEIGHTS` path is active. Prints per-image
resolution + inference latency plus an aggregate benchmark. **Standing rule**: every new
experiment gets its own candidate line added to both scripts as part of that same training
work - see `Scripts/images/README.md` for the full explanation, including how `exp004`
(a classifier, not a detector) is handled differently there.

**`Scripts/roi/`**: `python roi.py [video]` opens the first frame, lets you drag a rectangle
over the exact area you want detection restricted to, and saves it to
`Scripts/roi/roi_config.json` — all three `yr_*.py` pipelines in `videos/` pick this up
automatically (cropping every frame to it before running detection) if the file exists. A
light-grey dotted rectangle marks the active region on screen while running. `python roi.py
--clear` removes it.

- The `video_predict*`/`live_prediction*` scripts print a live OpenCV window (paced to the
  source's own FPS) plus an inference-time / end-to-end-latency benchmark report on quit (`q`).
- The `yr_*.py` pipelines each print YOLO/detector accuracy pulled from their weights'
  `results.csv` (validation-set mAP50/precision/recall) alongside the live benchmark - no
  ground truth exists for the demo videos, so that's a training-time reference number, not a
  live measurement.

To point any of these at a different experiment later, just change its `YOLO_WEIGHTS` /
`RESNET_WEIGHTS` (and `RESULTS_CSV`, where present) constant to the new
`experiments/exp00N_.../weights/best.pt` path.

## Notes

- GPU: developed against an RTX 5080 (16GB VRAM). `device=0` (or the equivalent for the
  framework in use) is hardcoded in the training scripts — training will fail loudly rather
  than silently fall back to CPU.
- If a training run gets interrupted, check the run's `weights/last.pt` before resuming — a
  hard crash can leave it corrupted, in which case `weights/best.pt` (saved less frequently,
  only on improvement) is the fallback to resume from.
- Windows is case-insensitive for folder names — renaming/creating folders that only differ by
  case (e.g. `Models` vs `models`) can silently collide. Worth knowing if you ever reorganize
  this layout further.
- `opencv-python` vs `opencv-python-headless`: never let both be installed at once — one
  silently overwrites the other's files on Windows, and uninstalling either after that can
  delete `cv2` entirely. This bit the project once via a since-removed dependency (see
  `requirements.txt`'s history) - worth remembering if any future package pulls in
  `opencv-python-headless` again.
- `raw.githubusercontent.com` specifically has hung/blocked when downloading `.py` files on
  this machine (plain `.csv` from the same repo worked fine) - matches this project's
  recurring AV/security-software interference pattern. `cdn.jsdelivr.net/gh/<owner>/<repo>@<ref>/<path>`
  mirrors any GitHub file and worked every time as a fallback.
