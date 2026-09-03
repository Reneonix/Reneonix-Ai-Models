"""
Two-model pipeline: plain YOLO (single-pass, no SAHI) for detection/
localization only + ByteTrack for persistent per-object tracking + ResNet-50
for the final classification of each tracked object's cropped region.

This is the lighter counterpart to yr_sahi_botsort.py:

  - NO SAHI tiling: one plain forward pass on the full frame, not 6 tiles
    batched together. Faster, but worse at catching small/distant objects
    (SAHI's whole point was upscaling tiles so small objects appear bigger).
  - ByteTrack instead of BoT-SORT+ReID: motion/IoU-only association, no
    appearance-embedding model. Cheaper (no ONNX ReID network to run every
    frame), simpler, and pairs naturally with plain detection - SAHI's dense
    per-tile detections were the main case where appearance-based matching
    earned its extra cost (many similar-looking objects close together);
    without SAHI there are typically fewer simultaneous detections per frame,
    so plain motion/IoU matching is usually adequate.

WHY YOLO+RESNET AT ALL (unchanged from the SAHI version):
exp001's own classification head shows near-zero confusion between real
classes on held-out validation - but live testing showed reflective/glossy
materials (aluminium, ceramic, plastic) occasionally misclassified as glass.
This pipeline hands the actual class decision to a dedicated ResNet trained
specifically on cropped object patches with heavy lighting/color-jitter
augmentation (src/exp004_resnet50.py, 99.4% val accuracy), while YOLO is used
purely to localize (its own predicted class is discarded for the final label).

The final report counts UNIQUE tracked objects (majority vote of that
object's ResNet predictions across every frame it was seen in), not one
count per frame it happened to be visible.

Press 'q' (video window focused) to stop; report + object counts print either way.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.engine.results import Boxes
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from collections import defaultdict, Counter
import numpy as np
import cv2
import time
import csv
import os
import sys
import argparse
import statistics as stats

# roi_utils.py lives in the sibling Scripts/roi/ folder, not next to this script -
# Scripts/ was reorganized into camera/, videos/, images/, roi/ subfolders (see each folder's own README.md)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roi"))
from roi_utils import load_roi, crop_to_roi, clamp_roi_to_frame, draw_dotted_rect

# ---------------- CONFIG ----------------
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp001_yolov8l/weights/best.pt"
YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - RESNET_WEIGHTS below (exp004) is a 6-class classifier with a different taxonomy (aluminium/metal split, no "ferrous"), so its per-class labels won't line up cleanly with this YOLO model's own classes if you swap to it

YOLO_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp001/results.csv"
RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/high_exposure.mp4"   # default - overridden by a CLI argument if given

CONF = 0.5    # minimum YOLO detection confidence to keep a box (localization only)
IMGSZ = 640          # YOLO inference size (matches training imgsz)
DEVICE = 0           # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
RESNET_IMG_SIZE = 224   # matches exp004_resnet50.py's transform

TRACKER_YAML = "bytetrack.yaml"   # ships with Ultralytics; loaded the same way model.track() does

# DEVICE (bare int) is what Ultralytics' own model.predict(device=...) expects, matching every
# other script in this project - but raw PyTorch calls (torch.load's map_location, tensor.to())
# need an actual device string, not a bare int, so this is used for those instead.
TORCH_DEVICE = f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"


def load_validation_accuracy(results_csv, precision_key="metrics/precision(B)", recall_key="metrics/recall(B)",
                              map50_key="metrics/mAP50(B)", map5095_key="metrics/mAP50-95(B)"):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return {
        "precision": float(last[precision_key]), "recall": float(last[recall_key]),
        "mAP50": float(last[map50_key]), "mAP50-95": float(last[map5095_key]),
        "epoch": int(last["epoch"]),
    }


def summarize(label, values_ms):
    print(f"  {label}:")
    print(f"    avg: {stats.mean(values_ms):.2f} ms  ({1000 / stats.mean(values_ms):.1f} FPS)")
    print(f"    median: {stats.median(values_ms):.2f} ms")
    print(f"    min: {min(values_ms):.2f} ms")
    print(f"    max: {max(values_ms):.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO (plain) detect -> ByteTrack -> ResNet classify pipeline")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def yolo_infer(model, frame):
    """Single plain forward pass over the full frame (no tiling). Ultralytics' own postprocessing
    already applies NMS internally for a single-image call, so no extra merge step is needed
    here (unlike the SAHI version, where duplicates can arise ACROSS separately-processed tiles)."""
    t0 = time.time()
    results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    yolo_ms = (time.time() - t0) * 1000

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0), yolo_ms
    return res.boxes.xyxy, res.boxes.conf, res.boxes.cls, yolo_ms


def load_resnet(weights_path):
    ckpt = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    class_names = ckpt["class_names"]   # ImageFolder's own (alphabetical) order - saved correctly
                                          # by exp004_resnet50.py, never assume/hardcode an order here
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(TORCH_DEVICE).eval()
    return model, class_names


RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(TORCH_DEVICE)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(TORCH_DEVICE)


def classify_crops(resnet_model, frame, boxes_xyxy):
    """Crops every tracked box out of the frame, batches them into ONE ResNet forward pass, and
    returns (predicted_class_idx, confidence, total_ms). BGR->RGB conversion matters: ImageFolder
    (used in exp004_resnet50.py) loads images via PIL as RGB, so feeding raw OpenCV BGR crops here
    without converting would silently mismatch the color channels the model was trained on.

    Uses cv2.resize (fast, C-level) per crop + ONE batched GPU-side normalize, instead of a
    per-crop torchvision transform pipeline - meaningfully cheaper with many objects per frame.
    Timing covers the whole crop->resize->normalize->forward-pass cost, not just the final GPU
    call - a forward-pass-only timer understates the real per-frame classification cost."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0.0

    t0 = time.time()
    h, w = frame.shape[:2]
    crops = np.zeros((len(boxes_xyxy), RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3), dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy.astype(int)):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops[i] = cv2.resize(crop_rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    batch_tensor = torch.from_numpy(crops).to(TORCH_DEVICE).permute(0, 3, 1, 2).float() / 255.0
    batch_tensor = (batch_tensor - RESNET_MEAN) / RESNET_STD

    with torch.no_grad():
        outputs = resnet_model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)
    total_ms = (time.time() - t0) * 1000
    return preds.cpu().numpy(), confs.cpu().numpy(), total_ms


def build_tracker():
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_YAML)))
    return BYTETracker(args=cfg)


def update_tracker(tracker, boxes, scores, cls, frame):
    """Feeds our own YOLO detections into Ultralytics' real BYTETracker (the same class
    model.track() uses internally) by wrapping them in the exact Boxes format it expects:
    (N,6) = [x1,y1,x2,y2,conf,cls]. Returns tracked (xyxy, track_id, conf, cls) arrays - empty
    arrays if nothing is currently tracked."""
    if len(boxes) == 0:
        det = np.zeros((0, 6), dtype=np.float32)
    else:
        det = torch.cat([boxes, scores.unsqueeze(1), cls.unsqueeze(1)], dim=1).cpu().numpy()

    det_boxes = Boxes(det, orig_shape=frame.shape[:2])
    tracks = tracker.update(det_boxes, frame)
    if len(tracks) == 0:
        return (np.zeros((0, 4)), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0))

    tracked_xyxy = tracks[:, :4]
    track_ids = tracks[:, 4].astype(int)
    track_conf = tracks[:, 5]
    track_cls = tracks[:, 6]
    return tracked_xyxy, track_ids, track_conf, track_cls


def draw_tracked_detections(frame, boxes, track_ids, resnet_preds, resnet_confs, class_names, class_colors):
    annotated = frame.copy()
    for box, tid, pred, conf in zip(boxes, track_ids, resnet_preds, resnet_confs):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = class_colors[int(pred) % len(class_colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"#{tid} {class_names[int(pred)]} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


CLASS_COLORS = [   # BGR - indexed by ResNet's own class order, not YOLO's
    (233, 180, 86), (167, 121, 204), (0, 94, 213), (115, 158, 0), (0, 159, 230), (66, 228, 240),
]


def main():
    video_path = os.path.abspath(parse_args().video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    print(f"Using video: {video_path}")

    yolo_model = YOLO(YOLO_WEIGHTS)
    resnet_model, class_names = load_resnet(RESNET_WEIGHTS)
    tracker = build_tracker()
    print(f"ResNet class order: {class_names}\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    roi = load_roi()
    if roi:
        rx, ry, rw, rh = roi
        print(f"Using saved ROI: x={rx} y={ry} w={rw} h={rh} (Scripts/roi_config.json) - "
              f"detection restricted to this region only\n")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if roi:
        width, height = roi[2], roi[3]   # everything downstream (tracking, display) now
                                           # operates on the ROI-cropped frame, not the full frame
    frame_budget = 1.0 / src_fps
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}\n")

    # WARM-UP: first-ever calls to YOLO/CUDA and the ResNet GPU kernels each carry a one-time
    # JIT/compilation cost - running the full pipeline on the real first frame here, before the
    # timed/displayed loop starts, keeps that cost out of both the benchmark stats and the
    # visible playback (otherwise the video visibly crawls for its first few seconds). No
    # tracker.reset() afterward: this frame's real detections seed real tracks, which is
    # harmless (those objects get counted normally) - the one tradeoff is this exact frame isn't
    # separately displayed/benchmarked, negligible for any video longer than a handful of frames.
    print("Warming up (first-call JIT/compile costs)...")
    ok, warm_frame = cap.read()
    if ok:
        warm_frame = crop_to_roi(warm_frame, roi)
        for _ in range(2):
            wb, ws, wc, _ = yolo_infer(yolo_model, warm_frame)
            wt_xyxy, wt_ids, _, _ = update_tracker(tracker, wb, ws, wc, warm_frame)
            classify_crops(resnet_model, warm_frame, wt_xyxy)
    print("Warm-up done.\n")

    window = f"YOLO+ByteTrack+ResNet - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    yolo_times, track_times, resnet_times, e2e_times = [], [], [], []
    resnet_batch_sizes = []   # how many objects were batched into each frame's ResNet call
    # track_id -> Counter of ResNet class predictions seen for that physical object, across
    # every frame it was tracked in - the majority vote is the object's final counted class.
    track_votes = defaultdict(Counter)

    prev_time = time.time()
    frame_count = 0
    while True:
        loop_start = time.time()

        ok, raw_frame = cap.read()
        if not ok:
            break
        frame = crop_to_roi(raw_frame, roi)
        frame_count += 1

        boxes, scores, cls, yolo_ms = yolo_infer(yolo_model, frame)
        yolo_times.append(yolo_ms)

        t0 = time.time()
        tracked_xyxy, track_ids, track_conf, _ = update_tracker(tracker, boxes, scores, cls, frame)
        track_times.append((time.time() - t0) * 1000)

        resnet_preds, resnet_confs, resnet_ms = classify_crops(resnet_model, frame, tracked_xyxy)
        if resnet_ms:
            resnet_times.append(resnet_ms)
            resnet_batch_sizes.append(len(tracked_xyxy))

        for tid, pred in zip(track_ids, resnet_preds):
            track_votes[int(tid)][int(pred)] += 1

        annotated = draw_tracked_detections(frame, tracked_xyxy, track_ids, resnet_preds,
                                             resnet_confs, class_names, CLASS_COLORS)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}  Tracked: {len(track_ids)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # If an ROI is active, show the full uncropped frame with the processed/annotated ROI
        # crop pasted back into its original spot, boundary marked with a light grey dotted
        # rectangle - lets you see the active detection area in context instead of only ever
        # seeing the cropped-down region filling the whole window.
        if roi:
            roi_box = clamp_roi_to_frame(raw_frame, roi)
            display_frame = raw_frame.copy()
            rx1, ry1, rx2, ry2 = roi_box
            display_frame[ry1:ry2, rx1:rx2] = annotated
            draw_dotted_rect(display_frame, (rx1, ry1), (rx2, ry2), color=(211, 211, 211), thickness=1)
        else:
            display_frame = annotated

        cv2.imshow(window, display_frame)
        display_end = time.time()
        e2e_times.append((display_end - loop_start) * 1000)

        remaining_ms = max(1, int((frame_budget - (display_end - loop_start)) * 1000))
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not yolo_times:
        print("No frames were processed.")
        return

    yolo_acc = load_validation_accuracy(YOLO_RESULTS_CSV)

    print("\n" + "=" * 70)
    print("YOLO (PLAIN) + BYTETRACK + RESNET PIPELINE - BENCHMARK & COUNT REPORT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}  |  Source FPS: {src_fps:.2f}  |  Frames processed: {frame_count}")

    print(f"\nYOLO detector accuracy (validation set, epoch {yolo_acc['epoch']}) - localization reference only:")
    print(f"  Precision: {yolo_acc['precision']:.4f}  Recall: {yolo_acc['recall']:.4f}  "
          f"mAP50: {yolo_acc['mAP50']:.4f}  mAP50-95: {yolo_acc['mAP50-95']:.4f}")
    print(f"ResNet classifier accuracy (validation set): 0.9937 (see results/exp004/results.csv)")

    print("\n--- LATENCY (separate) ---")
    summarize("YOLO detection (plain, single forward pass)", yolo_times)
    summarize("ByteTrack update", track_times)
    if resnet_times:
        print(f"  ResNet input size: {RESNET_IMG_SIZE}x{RESNET_IMG_SIZE}")
        print(f"  ResNet batch inference size (objects/frame): avg {stats.mean(resnet_batch_sizes):.1f}  "
              f"min {min(resnet_batch_sizes)}  max {max(resnet_batch_sizes)}")
        summarize("ResNet classification (1 batched forward pass per frame's tracked objects)", resnet_times)
    else:
        print("  ResNet classification: no objects were ever tracked - nothing to report.")

    print("\n--- END-TO-END (combined) ---")
    summarize("Full per-frame pipeline (capture + YOLO + track + crop + ResNet + draw + display)", e2e_times)
    achievable_fps = 1000 / stats.mean(e2e_times)
    print(f"\nMax achievable FPS (unthrottled, back-to-back): {achievable_fps:.1f}")
    print(f"Source video FPS: {src_fps:.2f}  ->  "
          f"{'pipeline kept up with real time' if achievable_fps >= src_fps else 'pipeline is SLOWER than source FPS - live playback lagged behind real time'}.")

    print("\n--- OBJECT COUNTS (unique tracked objects, majority-vote class) ---")
    final_counts = Counter()
    for tid, votes in track_votes.items():
        best_class_idx = votes.most_common(1)[0][0]
        final_counts[class_names[best_class_idx]] += 1

    total_objects = sum(final_counts.values())
    for name in sorted(class_names):
        print(f"  {name:12s}: {final_counts.get(name, 0)}")
    print(f"  {'TOTAL':12s}: {total_objects}")
    print("=" * 70)


if __name__ == "__main__":
    main()
