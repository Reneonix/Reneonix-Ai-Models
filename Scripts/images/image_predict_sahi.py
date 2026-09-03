"""
Batch image inference (SAHI-tiled) + benchmark report - the image-input counterpart to
video_predict_sahi.py.

Same SAHI technique as video_predict_sahi.py (each image sliced into a TILE_ROWS x TILE_COLS
grid, all tiles run as one batched forward pass, per-class NMS merges tile-overlap duplicates)
but over every image in results/testing_images/ instead of video frames. Tile boxes are
recomputed PER IMAGE (not once, like the video scripts do) since test images can be different
resolutions from each other, unlike a single video's fixed frame size.

Saves an annotated copy of each image into
results/predicted_images/<expNNN>/<original_stem>_sahi<ext>, and prints per-image resolution +
batched inference latency plus an aggregate benchmark summary.

MODEL SELECTION: exactly ONE of the candidates in the "MODEL SELECTION" block below must be
active (uncommented) at a time - comment the current one, uncomment a different one, then
rerun. The output folder (results/predicted_images/expNNN/) and the validation-accuracy
lookup (results/expNNN/results.csv) are both derived AUTOMATICALLY from whichever WEIGHTS path
is active - nothing else needs to change when you switch models.

STANDING PROJECT RULE: whenever a new experiment is trained, add its own candidate line here
(and in image_predict.py) - see Scripts/images/README.md.

exp004 (resnet50) is a CLASSIFIER, not a detector - "SAHI" for it here means something
different: each tile is classified SEPARATELY (a crude, non-trained approximation of
localization via tiling) rather than the whole image getting one label like image_predict.py
does - useful for comparison, not a real detection result.
"""

from ultralytics import YOLO, RTDETR
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import batched_nms
import cv2
import time
import csv
import os
import re
import glob
import statistics as stats

# ---------------- MODEL SELECTION ----------------
MODEL_TYPE = "yolo"      # "yolo" | "rtdetr" | "resnet_tiled" - must match whichever WEIGHTS line is active below

WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp001_yolov8l/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp002_yolo26s/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"   # MODEL_TYPE = "resnet_tiled"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp005_rtdetr_l/weights/best.pt"    # MODEL_TYPE = "rtdetr"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp006_yolo26s_finetuned/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - not comparable to the 6-class models above
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp008_resnet50_AGI/weights/best.pt"   # MODEL_TYPE = "resnet_tiled" - AGI classifier, 5 classes, pairs with exp007

PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
INPUT_DIR = f"{PROJECT_ROOT}/results/testing_images"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CONF = 0.5
IMGSZ = 640              # per-tile inference size (matches training imgsz)
RESNET_IMG_SIZE = 224    # matches exp004_resnet50.py's transform
DEVICE = 0               # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
TORCH_DEVICE = f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"

TILE_COLS = 3          # 3x2 = 6 tiles per image - same grid this project's video scripts use
TILE_ROWS = 2
TILE_OVERLAP = 0.2
NMS_IOU = 0.5

RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

CLASS_COLORS = [
    (233, 180, 86), (0, 159, 230), (115, 158, 0), (66, 228, 240), (167, 121, 204), (0, 94, 213),
]

# ---------------- auto-derived from WEIGHTS - do not hand-edit ----------------
_match = re.search(r"exp(\d{3})", WEIGHTS)
if not _match:
    raise ValueError(f"Could not determine experiment ID (expNNN) from WEIGHTS path: {WEIGHTS}")
EXP_ID = f"exp{_match.group(1)}"
# most results/ folders are named exactly EXP_ID (e.g. results/exp001/); a few (e.g. exp007_AGI)
# carry a suffix - glob for whichever actually exists so both styles resolve automatically.
_results_dirs = glob.glob(f"{PROJECT_ROOT}/results/{EXP_ID}*") or [f"{PROJECT_ROOT}/results/{EXP_ID}"]
_results_dirs = [d for d in _results_dirs if os.path.isdir(d)] or [f"{PROJECT_ROOT}/results/{EXP_ID}"]
RESULTS_CSV = f"{_results_dirs[0]}/results.csv"
OUTPUT_DIR = f"{PROJECT_ROOT}/results/predicted_images/{EXP_ID}"


def load_validation_accuracy():
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    if "metrics/precision(B)" in last:
        return {
            "precision": float(last["metrics/precision(B)"]), "recall": float(last["metrics/recall(B)"]),
            "mAP50": float(last["metrics/mAP50(B)"]), "mAP50-95": float(last["metrics/mAP50-95(B)"]),
            "epoch": int(last["epoch"]),
        }
    return {"val_acc": float(last["val_acc"]), "epoch": int(last["epoch"])}


def summarize(label, values_ms):
    print(f"  {label}:")
    print(f"    avg: {stats.mean(values_ms):.2f} ms  ({1000 / stats.mean(values_ms):.1f} FPS)")
    print(f"    median: {stats.median(values_ms):.2f} ms")
    print(f"    min: {min(values_ms):.2f} ms")
    print(f"    max: {max(values_ms):.2f} ms")


def compute_tile_boxes(width, height, cols, rows, overlap_ratio):
    tile_w = int(round((width / cols) * (1 + overlap_ratio)))
    tile_h = int(round((height / rows) * (1 + overlap_ratio)))
    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)

    x_step = (width - tile_w) / (cols - 1) if cols > 1 else 0
    y_step = (height - tile_h) / (rows - 1) if rows > 1 else 0

    boxes = []
    for r in range(rows):
        y1 = min(int(round(r * y_step)), height - tile_h)
        for c in range(cols):
            x1 = min(int(round(c * x_step)), width - tile_w)
            boxes.append((x1, y1, x1 + tile_w, y1 + tile_h))
    return boxes


def sahi_infer(model, frame, tile_boxes):
    tiles = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in tile_boxes]

    t0 = time.time()
    results = model.predict(tiles, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    batched_inference_ms = (time.time() - t0) * 1000

    all_boxes, all_scores, all_cls = [], [], []
    raw_count = 0
    for (x1, y1, x2, y2), res in zip(tile_boxes, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        raw_count += len(res.boxes)
        xyxy = res.boxes.xyxy.clone()
        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        all_boxes.append(xyxy)
        all_scores.append(res.boxes.conf)
        all_cls.append(res.boxes.cls)

    if not all_boxes:
        return (torch.empty((0, 4)), torch.empty(0), torch.empty(0), 0, batched_inference_ms)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    cls = torch.cat(all_cls)
    keep = batched_nms(boxes, scores, cls, NMS_IOU)
    return boxes[keep], scores[keep], cls[keep], raw_count, batched_inference_ms


def draw_detections(frame, boxes, scores, cls, class_names):
    annotated = frame.copy()
    for box, score, c in zip(boxes, scores, cls):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = CLASS_COLORS[int(c) % len(CLASS_COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{class_names[int(c)]} {float(score):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def draw_tile_grid(frame, tile_boxes):
    for (x1, y1, x2, y2) in tile_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)


def load_resnet(weights_path):
    ckpt = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(TORCH_DEVICE).eval()
    return model, class_names


def classify_tiles(model, class_names, frame, tile_boxes):
    """"SAHI" for a classifier: each tile is classified SEPARATELY (batched into one forward
    pass) instead of one label for the whole image - a crude, untrained approximation of
    localization via tiling, not a real detection result. See module docstring."""
    crops = torch.zeros(len(tile_boxes), 3, RESNET_IMG_SIZE, RESNET_IMG_SIZE)
    for i, (x1, y1, x2, y2) in enumerate(tile_boxes):
        tile_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(tile_rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        crops[i] = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    crops = (crops.to(TORCH_DEVICE) - RESNET_MEAN.to(TORCH_DEVICE)) / RESNET_STD.to(TORCH_DEVICE)

    t0 = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(crops), dim=1)
        confs, preds = probs.max(dim=1)
    inference_ms = (time.time() - t0) * 1000

    annotated = frame.copy()
    draw_tile_grid(annotated, tile_boxes)
    for (x1, y1, x2, y2), pred, conf in zip(tile_boxes, preds, confs):
        label = f"{class_names[int(pred)]} {float(conf):.2f}"
        cv2.putText(annotated, label, (x1 + 4, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return annotated, inference_ms


def main():
    image_paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, "*")) if p.lower().endswith(IMG_EXTENSIONS)
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Model ({MODEL_TYPE}): {WEIGHTS}")
    print(f"Input images: {INPUT_DIR}  ({len(image_paths)} found)")
    print(f"Output folder: {OUTPUT_DIR}\n")

    if MODEL_TYPE == "yolo":
        model = YOLO(WEIGHTS)
    elif MODEL_TYPE == "rtdetr":
        model = RTDETR(WEIGHTS)
    elif MODEL_TYPE == "resnet_tiled":
        model, class_names = load_resnet(WEIGHTS)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    inference_times = []
    resolutions = []
    detection_counts = []

    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"WARNING: could not read {img_path}, skipping")
            continue
        h, w = frame.shape[:2]
        resolutions.append((w, h))
        tile_boxes = compute_tile_boxes(w, h, TILE_COLS, TILE_ROWS, TILE_OVERLAP)

        if MODEL_TYPE == "resnet_tiled":
            annotated, inference_ms = classify_tiles(model, class_names, frame, tile_boxes)
            n_det = len(tile_boxes)   # one "prediction" per tile - not a real detection count
        else:
            boxes, scores, cls, raw_count, inference_ms = sahi_infer(model, frame, tile_boxes)
            annotated = draw_detections(frame, boxes, scores, cls, model.names)
            draw_tile_grid(annotated, tile_boxes)
            n_det = len(boxes)

        inference_times.append(inference_ms)
        detection_counts.append(n_det)

        stem, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_sahi{ext}")
        cv2.imwrite(out_path, annotated)

        print(f"{os.path.basename(img_path)}: {w}x{h}  tiles={len(tile_boxes)}  "
              f"inference={inference_ms:.2f}ms  detections={n_det}  -> {out_path}")

    if not inference_times:
        print("No images were processed.")
        return

    acc = load_validation_accuracy()

    print("\n" + "=" * 60)
    print("IMAGE BENCHMARK REPORT (SAHI-tiled)")
    print("=" * 60)
    print(f"Model: {WEIGHTS}")
    print(f"Images processed: {len(inference_times)}")
    unique_res = sorted(set(resolutions))
    print(f"Input resolution(s): {', '.join(f'{w}x{h}' for w, h in unique_res)}")
    print(f"Tiles per image: {TILE_COLS}x{TILE_ROWS} = {TILE_COLS * TILE_ROWS}  ({int(TILE_OVERLAP * 100)}% overlap)")

    print(f"\nModel accuracy (validation set, epoch {acc['epoch']}):")
    if "precision" in acc:
        print(f"  Precision:  {acc['precision']:.4f}")
        print(f"  Recall:     {acc['recall']:.4f}")
        print(f"  mAP50:      {acc['mAP50']:.4f}")
        print(f"  mAP50-95:   {acc['mAP50-95']:.4f}")
    else:
        print(f"  Val accuracy: {acc['val_acc']:.4f}")

    print()
    summarize(f"Batched model inference time ({TILE_COLS * TILE_ROWS} tiles/image, 1 batched forward pass)", inference_times)
    print("=" * 60)


if __name__ == "__main__":
    main()
