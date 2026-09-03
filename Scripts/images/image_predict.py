"""
Batch image inference + benchmark report - the image-input counterpart to video_predict.py.

Processes every image in results/testing_images/, saves an annotated copy of each into
results/predicted_images/<expNNN>/<original_stem>_plain<ext>, and prints per-image resolution
+ inference latency plus an aggregate benchmark summary.

MODEL SELECTION: exactly ONE of the candidates in the "MODEL SELECTION" block below must be
active (uncommented) at a time - comment the current one, uncomment a different one, then
rerun. The output folder (results/predicted_images/expNNN/) and the validation-accuracy
lookup (results/expNNN/results.csv) are both derived AUTOMATICALLY from whichever WEIGHTS path
is active - nothing else needs to change when you switch models.

STANDING PROJECT RULE: whenever a new experiment is trained, add its own candidate line here
(and in image_predict_sahi.py) - see Scripts/images/README.md.

exp004 (resnet50) is a CLASSIFIER, not a detector - it cannot draw bounding boxes on a raw
multi-object image (it expects an already-cropped single-object region as input, same as the
second stage of the yr_*.py two-model pipelines). Included anyway as an honest reference point:
MODEL_TYPE="resnet_whole_image" classifies the ENTIRE image as a single object and stamps that
one predicted label across the frame - useful for seeing why the two-model (YOLO+ResNet)
pipeline exists, not a real detection result.
"""

from ultralytics import YOLO, RTDETR
import torch
import torch.nn as nn
from torchvision.models import resnet50
import cv2
import time
import csv
import os
import re
import glob
import statistics as stats

# ---------------- MODEL SELECTION ----------------
MODEL_TYPE = "yolo"      # "yolo" | "rtdetr" | "resnet_whole_image" - must match whichever WEIGHTS line is active below

WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp001_yolov8l/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp002_yolo26s/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"   # MODEL_TYPE = "resnet_whole_image"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp005_rtdetr_l/weights/best.pt"    # MODEL_TYPE = "rtdetr"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp006_yolo26s_finetuned/weights/best.pt"
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - not comparable to the 6-class models above
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp008_resnet50_AGI/weights/best.pt"   # MODEL_TYPE = "resnet_whole_image" - AGI classifier, 5 classes, pairs with exp007

PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
INPUT_DIR = f"{PROJECT_ROOT}/results/testing_images"
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

CONF = 0.5
IMGSZ = 640              # YOLO/RTDETR inference size (matches training imgsz)
RESNET_IMG_SIZE = 224    # matches exp004_resnet50.py's transform
DEVICE = 0               # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
TORCH_DEVICE = f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"

RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

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
    """Pull the active model's final validation metrics straight from its own training
    results.csv (last row) - always matches whichever WEIGHTS candidate is active above."""
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    if "metrics/precision(B)" in last:
        return {
            "precision": float(last["metrics/precision(B)"]), "recall": float(last["metrics/recall(B)"]),
            "mAP50": float(last["metrics/mAP50(B)"]), "mAP50-95": float(last["metrics/mAP50-95(B)"]),
            "epoch": int(last["epoch"]),
        }
    # exp004 (resnet) results.csv has a different schema (classification, not detection)
    return {"val_acc": float(last["val_acc"]), "epoch": int(last["epoch"])}


def summarize(label, values_ms):
    print(f"  {label}:")
    print(f"    avg: {stats.mean(values_ms):.2f} ms  ({1000 / stats.mean(values_ms):.1f} FPS)")
    print(f"    median: {stats.median(values_ms):.2f} ms")
    print(f"    min: {min(values_ms):.2f} ms")
    print(f"    max: {max(values_ms):.2f} ms")


def load_resnet(weights_path):
    ckpt = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(TORCH_DEVICE).eval()
    return model, class_names


def classify_whole_image(model, class_names, frame_bgr):
    """Classifies the ENTIRE frame as one object (no localization at all) - see module
    docstring for why this is only a reference point, not a real detection result."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(TORCH_DEVICE).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = (tensor - RESNET_MEAN.to(TORCH_DEVICE)) / RESNET_STD.to(TORCH_DEVICE)
    t0 = time.time()
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)
        conf, pred = probs.max(dim=1)
    inference_ms = (time.time() - t0) * 1000

    annotated = frame_bgr.copy()
    label = f"{class_names[int(pred)]} {float(conf):.2f}  (WHOLE-IMAGE classification, not a detection)"
    cv2.putText(annotated, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
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
    elif MODEL_TYPE == "resnet_whole_image":
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

        if MODEL_TYPE == "resnet_whole_image":
            annotated, inference_ms = classify_whole_image(model, class_names, frame)
            n_det = 1   # the one whole-image "prediction" - not a real detection count
        else:
            results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
            inference_ms = results[0].speed["inference"]
            annotated = results[0].plot(line_width=BOX_THICKNESS)
            n_det = len(results[0].boxes) if results[0].boxes is not None else 0

        inference_times.append(inference_ms)
        detection_counts.append(n_det)

        stem, ext = os.path.splitext(os.path.basename(img_path))
        out_path = os.path.join(OUTPUT_DIR, f"{stem}_plain{ext}")
        cv2.imwrite(out_path, annotated)

        print(f"{os.path.basename(img_path)}: {w}x{h}  inference={inference_ms:.2f}ms  "
              f"detections={n_det}  -> {out_path}")

    if not inference_times:
        print("No images were processed.")
        return

    acc = load_validation_accuracy()

    print("\n" + "=" * 60)
    print("IMAGE BENCHMARK REPORT (plain, no SAHI)")
    print("=" * 60)
    print(f"Model: {WEIGHTS}")
    print(f"Images processed: {len(inference_times)}")
    unique_res = sorted(set(resolutions))
    print(f"Input resolution(s): {', '.join(f'{w}x{h}' for w, h in unique_res)}")

    print(f"\nModel accuracy (validation set, epoch {acc['epoch']}):")
    if "precision" in acc:
        print(f"  Precision:  {acc['precision']:.4f}")
        print(f"  Recall:     {acc['recall']:.4f}")
        print(f"  mAP50:      {acc['mAP50']:.4f}")
        print(f"  mAP50-95:   {acc['mAP50-95']:.4f}")
    else:
        print(f"  Val accuracy: {acc['val_acc']:.4f}")

    print()
    summarize("Model inference time (forward pass only)", inference_times)
    print("=" * 60)


if __name__ == "__main__":
    main()
