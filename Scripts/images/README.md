# Scripts/images

Batch single/multi-image inference across **every trained model in this project at once** -
the image-input counterpart to `../videos/video_predict.py` / `video_predict_sahi.py`.

```
python image_predict.py         # plain inference
python image_predict_sahi.py     # SAHI (6-tile) inference
```

- **Input**: every image file in `results/testing_images/` (any of `.jpg .jpeg .png .bmp .webp`).
- **Output**: an annotated copy of each, saved to `results/predicted_images/<expNNN>/` -
  `<original_stem>_plain<ext>` from `image_predict.py`, `<original_stem>_sahi<ext>` from
  `image_predict_sahi.py`. `<expNNN>` is derived **automatically** from whichever `WEIGHTS`
  path is active (see below) - never hand-typed.
- **Benchmark printed for every image**: resolution + inference latency (forward-pass only,
  or the batched 6-tile forward pass for the SAHI script), plus an aggregate summary
  (avg/median/min/max) at the end, alongside the active model's own training-time validation
  accuracy (read straight from `results/<expNNN>/results.csv` - never hand-typed either).

## Switching models

Both scripts have a `MODEL SELECTION` block near the top with **one candidate line per
experiment** - exactly one must be uncommented at a time:

```python
MODEL_TYPE = "yolo"      # "yolo" | "rtdetr" | "resnet_whole_image" (or "resnet_tiled" in the SAHI script)

WEIGHTS = "d:/.../experiments/exp001_yolov8l/weights/best.pt"
# WEIGHTS = "d:/.../experiments/exp002_yolo26s/weights/best.pt"
# WEIGHTS = "d:/.../experiments/exp003_yolo26l_p2/weights/best.pt"
# WEIGHTS = "d:/.../experiments/exp004_resnet50/weights/best.pt"   # MODEL_TYPE = "resnet_whole_image"
# WEIGHTS = "d:/.../experiments/exp005_rtdetr_l/weights/best.pt"    # MODEL_TYPE = "rtdetr"
# WEIGHTS = "d:/.../experiments/exp006_yolo26s_finetuned/weights/best.pt"
```

To test a different model: comment the currently-active `WEIGHTS` line, uncomment the one you
want, **and set `MODEL_TYPE` to match** (most experiments are `"yolo"`; `exp005` needs
`"rtdetr"` since it's loaded via Ultralytics' `RTDETR` class, not `YOLO`; `exp004` needs
`"resnet_whole_image"`/`"resnet_tiled"` - see below). Everything else (output folder, accuracy
lookup) follows automatically from the `WEIGHTS` path itself.

## Why exp004 is handled differently

`exp004` is a **classifier** (ResNet-50), not a detector - it cannot draw bounding boxes on a
raw multi-object image at all (it expects an already-cropped single-object region as input,
exactly like the second stage of the `../videos/yr_*.py` two-model pipelines). It's still
included, as an honest reference point, not a real detection result:

- `image_predict.py` (`MODEL_TYPE="resnet_whole_image"`): classifies the **entire image** as
  one object and stamps that single predicted label across the frame.
- `image_predict_sahi.py` (`MODEL_TYPE="resnet_tiled"`): classifies **each of the 6 tiles
  separately** (a crude, untrained approximation of localization via tiling) and labels each
  tile with its own prediction.

Neither is a real detection count - useful mainly for seeing why the two-model (YOLO detects,
ResNet classifies each crop) pipeline in `../videos/yr_*.py` exists in the first place.

## Standing project rule - do this for every new model

Whenever a new experiment is trained and added to this project, **as part of that same piece
of work** (not a separate later step):

1. Add its own candidate `WEIGHTS` line (commented) to **both** `image_predict.py` and
   `image_predict_sahi.py`, with a `MODEL_TYPE` note if it isn't a plain YOLO checkpoint.
2. Nothing else needs manual setup - `results/predicted_images/<expNNN>/` is created
   automatically the first time either script runs with that model active.

This keeps `results/predicted_images/` always holding one folder per experiment that's ever
existed, each with a plain and a SAHI sample prediction, without anyone needing to remember a
separate checklist.
