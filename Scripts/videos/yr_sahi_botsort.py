"""
Two-model pipeline: YOLO (SAHI-tiled) for detection/localization only +
BoT-SORT-with-ReID (DeepSORT-style appearance-based tracking) for persistent
per-object tracking + ResNet-50 for the final classification of each tracked
object's cropped region.

WHY THIS EXISTS:
exp003's own classification head shows near-zero confusion between real classes
on held-out validation (results/exp003/confusion_matrix_normalized.png) - but
live testing showed reflective/glossy materials (aluminium, ceramic, plastic)
occasionally misclassified as glass. This pipeline hands the actual class
decision to a dedicated ResNet trained specifically on cropped object patches
with heavy lighting/color-jitter augmentation (src/exp004_resnet50.py, 99.4% val
accuracy), while YOLO+SAHI is used purely to localize (its own predicted class
is discarded for the final label - only used internally to help SAHI's
tile-overlap NMS merge).

WHY BOT-SORT+REID INSTEAD OF PLAIN BYTETRACK:
Plain ByteTrack associates detections across frames using only motion/IoU -
no visual appearance. With 35-50 similar-looking objects per frame (verified
on this project's own video), objects passing close to each other are prone
to ID switches under pure motion matching. BoT-SORT with with_reid=True adds
an appearance-embedding model (yolo26n-reid.onnx, Ultralytics' own purpose-
built ReID network) into the association cost, so visually distinct objects
keep their identity even when their boxes get close/cross paths - directly
serving the counting goal below (one physical object = one ID = one count,
not inflated by fragmented/switched IDs).

Ultralytics' own BOTSORT class is used directly (not going through
model.track()) - fed our own SAHI-merged detections. The final report counts
UNIQUE tracked objects (majority vote of that object's ResNet predictions
across every frame it was seen in), not one count per frame it happened to be
visible - essential for a conveyor/pour scenario where the same object
appears in many consecutive frames.

Press 'q' (video window focused) to stop; report + object counts print either way.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import nms
from ultralytics import YOLO
from ultralytics.trackers.bot_sort import BOTSORT
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
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp001_yolov8l/weights/best.pt"
YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - RESNET_WEIGHTS below (exp004) is a 6-class classifier with a different taxonomy (aluminium/metal split, no "ferrous"), so its per-class labels won't line up cleanly with this YOLO model's own classes if you swap to it

YOLO_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp001/results.csv"
RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/agi/m/v2.mp4"   # default - overridden by a CLI argument if given

CONF = 0.7        # minimum YOLO detection confidence to keep a box (localization only)
IMGSZ = 640          # per-tile inference size (matches training imgsz)
DEVICE = 0           # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
RESNET_IMG_SIZE = 224   # matches exp004_resnet50.py's transform

TILE_COLS = 3          # 3x2 = 6 tiles per frame (same SAHI scheme as video_predict_sahi.py)
TILE_ROWS = 2
TILE_OVERLAP = 0.4
NMS_IOU = 0.5

TRACKER_YAML = "botsort.yaml"   # ships with Ultralytics; loaded the same way model.track() does
REID_MODEL = "yolo26n-reid.onnx"   # Ultralytics' own purpose-built appearance-embedding model -
                                     # auto-downloaded on first use, runs via onnxruntime-gpu.
                                     # CPU execution was tried first (avoiding a CUDA/onnxruntime
                                     # version-compat risk) but measured 30-340ms/frame, wildly
                                     # variable - GPU is both faster and consistent for this net.

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
    parser = argparse.ArgumentParser(description="YOLO(+SAHI) detect -> BoT-SORT+ReID track -> ResNet classify pipeline")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def compute_tile_boxes(width, height, cols, rows, overlap_ratio):
    """Returns exactly cols*rows (x1, y1, x2, y2) tile boxes covering the full frame edge-to-edge,
    with each neighboring tile overlapping by overlap_ratio of the tile's own width/height."""
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
    """Runs one batched forward pass over all tiles, translates every box back to full-frame
    coordinates, and merges duplicates from overlapping tile regions with CLASS-AGNOSTIC NMS.
    Must be class-agnostic, not per-class: YOLO's own class guess is discarded for the final
    label (ResNet decides that), so two overlapping boxes of the same physical object could
    carry different (unused) YOLO class guesses - per-class NMS would let both survive since
    it only suppresses same-class overlaps, leaving visible duplicate/overlapping boxes."""
    tiles = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in tile_boxes]

    t0 = time.time()
    results = model.predict(tiles, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    yolo_ms = (time.time() - t0) * 1000

    all_boxes, all_scores, all_cls = [], [], []
    for (x1, y1, x2, y2), res in zip(tile_boxes, results):
        if res.boxes is None or len(res.boxes) == 0:
            continue
        xyxy = res.boxes.xyxy.clone()
        xyxy[:, [0, 2]] += x1
        xyxy[:, [1, 3]] += y1
        all_boxes.append(xyxy)
        all_scores.append(res.boxes.conf)
        all_cls.append(res.boxes.cls)

    if not all_boxes:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0), yolo_ms

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    cls = torch.cat(all_cls)
    keep = nms(boxes, scores, NMS_IOU)
    return boxes[keep], scores[keep], cls[keep], yolo_ms


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
    # botsort.yaml defaults to with_reid=False, model="auto" (which needs Ultralytics' own
    # predictor-hook machinery we're bypassing). Overriding both explicitly gives BOTSORT a
    # concrete, self-contained ReID model - it crops+embeds detections internally, no external
    # wiring needed, as long as `img` is passed to update() (already done in update_tracker()).
    cfg.with_reid = True
    cfg.model = REID_MODEL
    cfg.device = TORCH_DEVICE   # onnxruntime-gpu installed - CPU execution measured 30-340ms/frame
                                 # (highly variable, likely CPU/AV contention), GPU is both faster
                                 # and consistent for this small embedding network
    return BOTSORT(args=cfg)


def update_tracker(tracker, boxes, scores, cls, frame):
    """Feeds our own SAHI-merged detections into Ultralytics' real BOTSORT tracker (the same class
    model.track() uses internally) by wrapping them in the exact Boxes format it expects:
    (N,6) = [x1,y1,x2,y2,conf,cls]. `frame` is passed through as the ReID encoder's image input.
    Returns tracked (xyxy, track_id, conf, cls) arrays - empty arrays if nothing is currently
    tracked."""
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
        width, height = roi[2], roi[3]   # everything downstream (tiling, tracking, display) now
                                           # operates on the ROI-cropped frame, not the full frame
    frame_budget = 1.0 / src_fps

    tile_boxes = compute_tile_boxes(width, height, TILE_COLS, TILE_ROWS, TILE_OVERLAP)
    tile_w = tile_boxes[0][2] - tile_boxes[0][0]
    tile_h = tile_boxes[0][3] - tile_boxes[0][1]
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}")
    print(f"SAHI grid: {TILE_COLS}x{TILE_ROWS} = {len(tile_boxes)} tiles, {int(TILE_OVERLAP * 100)}% overlap "
          f"|  Per-tile resolution: {tile_w}x{tile_h}\n")

    # WARM-UP: first-ever calls to YOLO/CUDA, the ReID ONNX session, and the ResNet GPU kernels
    # each carry a one-time JIT/compilation cost (measured: 280-870ms on the first few real
    # frames, vs. ~100-120ms steady-state) - running the full pipeline here, before the
    # timed/displayed loop starts, keeps that cost out of both the benchmark stats and the
    # visible playback (otherwise the video visibly crawls for its first few seconds).
    #
    # Must use a REAL frame with real detections, not a blank one: BOTSORT's ReID encoder is
    # only called when there ARE detections (empty-detection frames short-circuit before ever
    # touching the ONNX session), so a blank frame never actually warms up the ReID model at all
    # - tried that first, and the real warmup cost just reappeared spread across the next 10+
    # live frames instead of a few.
    #
    # Also deliberately does NOT call tracker.reset() afterward: that was tried too, but
    # BOTSORT.reset() also clears the GMC (motion-compensation) submodule's internal state,
    # which reintroduced a ~280-300ms cost on the next couple of real frames anyway - the reset
    # just moved the warmup cost, it didn't remove it. Letting this frame's real detections seed
    # real tracks is harmless (those objects get counted normally); the one tradeoff is this
    # exact frame isn't separately displayed/benchmarked - negligible for any video longer than
    # a handful of frames.
    print("Warming up (first-call JIT/compile costs)...")
    ok, warm_frame = cap.read()
    if ok:
        warm_frame = crop_to_roi(warm_frame, roi)
        for _ in range(2):   # BOTSORT's Kalman filter/GMC/ONNX session settle after 2 calls, not just 1
            wb, ws, wc, _ = sahi_infer(yolo_model, warm_frame, tile_boxes)
            wt_xyxy, wt_ids, _, _ = update_tracker(tracker, wb, ws, wc, warm_frame)
            classify_crops(resnet_model, warm_frame, wt_xyxy)
    print("Warm-up done.\n")

    window = f"YOLO+BoTSORT(ReID)+ResNet - {os.path.basename(video_path)}"
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

        boxes, scores, cls, yolo_ms = sahi_infer(yolo_model, frame, tile_boxes)
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
    print("YOLO + BOT-SORT(REID) + RESNET PIPELINE - BENCHMARK & COUNT REPORT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}  |  Source FPS: {src_fps:.2f}  |  Frames processed: {frame_count}")
    print(f"SAHI grid: {TILE_COLS}x{TILE_ROWS} = {len(tile_boxes)} tiles ({int(TILE_OVERLAP * 100)}% overlap)  "
          f"|  Per-tile resolution: {tile_w}x{tile_h}")

    print(f"\nYOLO detector accuracy (validation set, epoch {yolo_acc['epoch']}) - localization reference only:")
    print(f"  Precision: {yolo_acc['precision']:.4f}  Recall: {yolo_acc['recall']:.4f}  "
          f"mAP50: {yolo_acc['mAP50']:.4f}  mAP50-95: {yolo_acc['mAP50-95']:.4f}")
    print(f"ResNet classifier accuracy (validation set): 0.9937 (see results/exp004/results.csv)")

    print("\n--- LATENCY (separate) ---")
    summarize(f"YOLO+SAHI detection ({len(tile_boxes)} tiles/frame, 1 batched forward pass)", yolo_times)
    summarize("BoT-SORT+ReID tracking update", track_times)
    if resnet_times:
        print(f"  ResNet input size: {RESNET_IMG_SIZE}x{RESNET_IMG_SIZE}")
        print(f"  ResNet batch inference size (objects/frame): avg {stats.mean(resnet_batch_sizes):.1f}  "
              f"min {min(resnet_batch_sizes)}  max {max(resnet_batch_sizes)}")
        summarize("ResNet classification (1 batched forward pass per frame's tracked objects)", resnet_times)
    else:
        print("  ResNet classification: no objects were ever tracked - nothing to report.")

    print("\n--- END-TO-END (combined) ---")
    summarize("Full per-frame pipeline (capture + YOLO/SAHI + track + crop + ResNet + draw + display)", e2e_times)
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
