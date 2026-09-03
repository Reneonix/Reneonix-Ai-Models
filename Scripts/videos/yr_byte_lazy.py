"""
Two-model pipeline: plain YOLO (single-pass, no SAHI) for detection/localization + ByteTrack for
persistent per-object tracking + ResNet-50 for classification - LAZY (not per-frame), with
QUALITY-GATED CROPS and PROBABILITY-ACCUMULATION evidence instead of simple vote-counting.

THE PROBLEM THIS SOLVES (same as before): reclassifying every tracked object every frame wastes
GPU time - an object's material class doesn't change frame to frame, so classifying it once and
caching the result is enough. See yr_byte.py for the "classify every frame" baseline this
improves on.

THIS VERSION'S TWO ARCHITECTURAL UPGRADES over the original yr_byte_lazy.py:

  1. GOOD CROP SELECTION - not every crop is worth classifying. Before spending a ResNet call on
     a track, check the crop is actually trustworthy: big enough (not a tiny sliver that has to
     be upsampled into blur), sharp (not motion-blurred), not significantly overlapping another
     tracked box (a cheap proxy for occlusion), and from a confident-enough detection. A crop
     that fails this gate is skipped THIS frame - the track stays unlocked and gets a fresh shot
     next frame, instead of "wasting" a vote on a low-quality view that could poison the decision.

  2. PROBABILITY ACCUMULATION instead of hard vote-counting. The old version treated every
     classification as one equal "vote" for its argmax class, regardless of how confident that
     vote actually was. This version sums the full softmax probability vector across every good
     crop seen for a track - so a crop the model was 81% sure about contributes more evidence
     than one it was only 40% sure about, instead of both just counting as "1". A track locks
     once its leading class's ACCUMULATED probability mass is clearly ahead (a high single-shot
     confidence, a high accumulated total, or a wide enough margin over the runner-up), or once
     RESNET_MAX_ATTEMPTS good crops have been spent regardless.

  Fallback: a track that's still alive but has gone FORCE_CLASSIFY_AFTER_FRAMES consecutive
  frames with zero evidence (every crop it's offered has failed the gate) gets classified anyway
  on its next frame, gate bypassed - guarantees a long-lived track never disappears from tracking
  with no label at all just because its crops were persistently borderline. This happens live, in
  the main loop, while real frame data still exists - a track that disappears (occlusion, leaves
  frame, video ends) before reaching that threshold is rare and is reported honestly as excluded
  from the final count rather than guessed at after the fact.

One structural note (not a bug, same as before): this keys off ByteTrack's track_id. If the
tracker ever loses and reassigns a new ID to the same physical object (occlusion, leaving/
re-entering frame), that "new" ID starts fresh evidence-gathering - one extra ResNet call, same
as any new object. See yr_botsort_lazy.py for a tracker (BoT-SORT+ReID) that resists this better.

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
from collections import Counter
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
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - pair with exp008 (RESNET_WEIGHTS below), NOT exp004, which is a 6-class classifier with a different taxonomy

YOLO_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp007_AGI/results.csv"   # must match whichever YOLO_WEIGHTS is active above - exp007's own results, not exp001's
# RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"
RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp008_resnet50_AGI/weights/best.pt"   # 5-class AGI classifier - pairs with exp007 above, NOT the 6-class exp001/exp003/exp005/exp006 YOLO models
RESNET_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp008_AGI/results.csv"   # must match whichever RESNET_WEIGHTS is active above
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/agi/m/v2.mp4"   # default - overridden by a CLI argument if given

CONF = 0.5    # minimum YOLO detection confidence to keep a box (localization only)
IMGSZ = 640          # YOLO inference size (matches training imgsz)
DEVICE = 0           # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
RESNET_IMG_SIZE = 224   # matches exp004_resnet50.py's/exp008_resnet50_agi.py's transform
RESNET_USE_FP16 = True   # mixed-precision inference - free speedup on the RTX 5080's tensor cores

TRACKER_YAML = "bytetrack.yaml"   # ships with Ultralytics; loaded the same way model.track() does

# ---- GOOD CROP SELECTION - a crop must pass all of these to be worth classifying ----
MIN_GOOD_CROP_SIZE = 40       # px, both width and height - anything smaller upsamples badly into
                               # the 224x224 ResNet input, mostly blur by then
MIN_GOOD_DET_CONF = 0.5       # detector confidence for THIS box specifically - separate from (and
                               # can be stricter than) CONF, which only gates whether YOLO returns
                               # the box at all
MIN_SHARPNESS = 30.0          # Laplacian-variance blur measure - LOWER means blurrier. Genuinely
                               # footage-dependent (lighting, camera, motion blur amount differ per
                               # setup) - watch the printed "crops rejected" stats and adjust if
                               # this is rejecting almost everything or almost nothing
OCCLUSION_IOU_THRESH = 0.3    # skip a crop whose box overlaps another currently-tracked box by
                               # more than this IoU - a cheap proxy for "probably partially hidden
                               # behind a neighboring object", not a real occlusion detector

# ---- probability-accumulation evidence + lock rule ----
RESNET_LOCK_CONF = 0.90            # a single good crop this confident locks the track immediately
PROB_ACCUM_LOCK_THRESHOLD = 1.5    # or lock once the leading class's SUMMED probability reaches this
PROB_ACCUM_MARGIN = 0.75           # or lock once the leading class's summed probability beats the
                                     # runner-up class by at least this much
RESNET_MAX_ATTEMPTS = 5            # hard cap on GOOD-crop classification attempts per track,
                                     # regardless of confidence - bounds worst-case cost
FORCE_CLASSIFY_AFTER_FRAMES = 10   # a track with STILL zero evidence after this many consecutive
                                     # gate-rejections gets classified anyway, gate bypassed, on
                                     # whatever crop is available that frame - guarantees a track
                                     # doesn't disappear from tracking with no label at all just
                                     # because every crop it ever offered was borderline

# DEVICE (bare int) is what Ultralytics' own model.predict(device=...) expects, matching every
# other script in this project - but raw PyTorch calls (torch.load's map_location, tensor.to())
# need an actual device string, not a bare int, so this is used for those instead.
TORCH_DEVICE = f"cuda:{DEVICE}" if DEVICE != "cpu" else "cpu"
UNCLASSIFIED = -1   # sentinel: track has zero evidence yet (no good crop seen so far)


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


def load_resnet_val_acc(results_csv):
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return float(last["val_acc"])


def summarize(label, values_ms):
    print(f"  {label}:")
    print(f"    avg: {stats.mean(values_ms):.2f} ms  ({1000 / stats.mean(values_ms):.1f} FPS)")
    print(f"    median: {stats.median(values_ms):.2f} ms")
    print(f"    min: {min(values_ms):.2f} ms")
    print(f"    max: {max(values_ms):.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO (plain) detect -> ByteTrack -> quality-gated lazy ResNet classify pipeline")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def yolo_infer(model, frame):
    """Single plain forward pass over the full frame (no tiling)."""
    t0 = time.time()
    results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    yolo_ms = (time.time() - t0) * 1000

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return torch.empty((0, 4)), torch.empty(0), torch.empty(0), yolo_ms
    return res.boxes.xyxy, res.boxes.conf, res.boxes.cls, yolo_ms


def load_resnet(weights_path):
    ckpt = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    class_names = ckpt["class_names"]   # ImageFolder's own (alphabetical) order - never
                                          # assume/hardcode an order here
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(TORCH_DEVICE).eval()
    return model, class_names


RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(TORCH_DEVICE)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(TORCH_DEVICE)


def classify_crops(resnet_model, frame, boxes_xyxy):
    """Crops the given boxes out of the frame, batches them into ONE ResNet forward pass. Called
    only on the subset of tracked boxes that both need classifying AND passed the good-crop gate
    this frame (see main loop). Returns (pred_class_idx, top_conf, full_prob_matrix, ms) - the
    full probability vector per box is what drives the accumulation logic, not just the argmax."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), np.zeros((0, 0)), 0.0

    t0 = time.time()
    h, w = frame.shape[:2]
    boxes_np = boxes_xyxy.astype(int) if isinstance(boxes_xyxy, np.ndarray) else boxes_xyxy.cpu().numpy().astype(int)
    crops = np.zeros((len(boxes_np), RESNET_IMG_SIZE, RESNET_IMG_SIZE, 3), dtype=np.uint8)
    for i, (x1, y1, x2, y2) in enumerate(boxes_np):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crops[i] = cv2.resize(crop_rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    batch_tensor = torch.from_numpy(crops).to(TORCH_DEVICE).permute(0, 3, 1, 2).float() / 255.0
    batch_tensor = (batch_tensor - RESNET_MEAN) / RESNET_STD

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=RESNET_USE_FP16):
        outputs = resnet_model(batch_tensor)
        probs = torch.softmax(outputs.float(), dim=1)
        confs, preds = probs.max(dim=1)
    total_ms = (time.time() - t0) * 1000
    return preds.cpu().numpy(), confs.cpu().numpy(), probs.cpu().numpy(), total_ms


def build_tracker():
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_YAML)))
    return BYTETracker(args=cfg)


def update_tracker(tracker, boxes, scores, cls, frame):
    """Feeds our own YOLO detections into Ultralytics' real BYTETracker by wrapping them in the
    exact Boxes format it expects: (N,6) = [x1,y1,x2,y2,conf,cls]. Returns tracked
    (xyxy, track_id, conf, cls) arrays - empty arrays if nothing is currently tracked."""
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


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def crop_sharpness(frame, box):
    """Laplacian-variance blur measure - higher is sharper. Cheap, no extra model needed."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_good_crop(box, det_conf, all_boxes, this_idx, frame):
    """GOOD CROP SELECTION gate - see module docstring. Returns False the moment any criterion
    fails (cheapest checks first, so a bad crop is rejected without doing the sharpness pass
    when possible)."""
    x1, y1, x2, y2 = box
    if (x2 - x1) < MIN_GOOD_CROP_SIZE or (y2 - y1) < MIN_GOOD_CROP_SIZE:
        return False
    if det_conf < MIN_GOOD_DET_CONF:
        return False
    for j in range(len(all_boxes)):
        if j == this_idx:
            continue
        if iou(box, all_boxes[j]) >= OCCLUSION_IOU_THRESH:
            return False
    if crop_sharpness(frame, box) < MIN_SHARPNESS:
        return False
    return True


def draw_tracked_detections(frame, boxes, track_ids, classes, locked_ids, class_names, class_colors):
    """`classes` is each track's current best-known class (locked, best-accumulated-so-far, or
    UNCLASSIFIED if no good crop has landed yet). A small padlock glyph marks locked tracks."""
    annotated = frame.copy()
    for box, tid, cls_idx in zip(boxes, track_ids, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        if cls_idx == UNCLASSIFIED:
            color = (160, 160, 160)
            label = f"#{tid} ..."
        else:
            color = class_colors[int(cls_idx) % len(class_colors)]
            lock_mark = "*" if tid in locked_ids else ""
            label = f"#{tid}{lock_mark} {class_names[int(cls_idx)]}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


CLASS_COLORS = [   # BGR - indexed by ResNet's own class order, not YOLO's
    (233, 180, 86), (167, 121, 204), (0, 94, 213), (115, 158, 0), (0, 159, 230), (66, 228, 240),
]


def evaluate_lock(accum, top_conf, evidence_n):
    """Given a track's accumulated probability vector, decide (a) its current best class and
    (b) whether that decision is reliable enough to lock. Mirrors the diagram: accumulate
    probability evidence across good crops, lock once the leading class is clearly ahead."""
    top_idx = int(np.argmax(accum))
    sorted_vals = np.sort(accum)[::-1]
    top_val = sorted_vals[0]
    second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0.0
    reliable = (
        top_conf >= RESNET_LOCK_CONF
        or top_val >= PROB_ACCUM_LOCK_THRESHOLD
        or (top_val - second_val) >= PROB_ACCUM_MARGIN
        or evidence_n >= RESNET_MAX_ATTEMPTS
    )
    return top_idx, reliable


def main():
    video_path = os.path.abspath(parse_args().video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    print(f"Using video: {video_path}")

    yolo_model = YOLO(YOLO_WEIGHTS)
    resnet_model, class_names = load_resnet(RESNET_WEIGHTS)
    tracker = build_tracker()
    print(f"ResNet class order: {class_names}")
    print(f"Good-crop gate: >= {MIN_GOOD_CROP_SIZE}px, det conf >= {MIN_GOOD_DET_CONF}, "
          f"sharpness >= {MIN_SHARPNESS}, occlusion IoU < {OCCLUSION_IOU_THRESH}")
    print(f"Lock rule: single-shot conf>={RESNET_LOCK_CONF}, OR accumulated prob>={PROB_ACCUM_LOCK_THRESHOLD}, "
          f"OR margin over runner-up>={PROB_ACCUM_MARGIN}, OR {RESNET_MAX_ATTEMPTS} good crops spent\n")

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
        width, height = roi[2], roi[3]
    frame_budget = 1.0 / src_fps
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}\n")

    print("Warming up (first-call JIT/compile costs)...")
    ok, warm_frame = cap.read()
    if ok:
        warm_frame = crop_to_roi(warm_frame, roi)
        for _ in range(2):
            wb, ws, wc, _ = yolo_infer(yolo_model, warm_frame)
            wt_xyxy, wt_ids, _, _ = update_tracker(tracker, wb, ws, wc, warm_frame)
            classify_crops(resnet_model, warm_frame, wt_xyxy)
    print("Warm-up done.\n")

    window = f"YOLO+ByteTrack+ResNet (quality-gated lazy) - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    yolo_times, track_times, resnet_times, e2e_times = [], [], [], []
    resnet_batch_sizes = []   # how many objects actually got classified each frame
    track_prob_accum = {}      # track_id -> np.array of summed softmax probabilities so far
    evidence_count = {}        # track_id -> how many GOOD crops actually classified so far
    locked_class = {}          # track_id -> locked class idx, once evidence is reliable
    seen_track_ids = set()     # every track_id ever observed (for the never-classified check)
    unclassified_streak = {}   # track_id -> consecutive frames alive with zero evidence so far

    total_track_occurrences = 0    # sum of len(track_ids) every frame - what yr_byte.py would have classified
    total_resnet_calls = 0          # sum of actual ResNet calls made - what this script classified
    total_crops_rejected = 0        # good-crop-gate rejections - visibility into how often it fires
    total_forced_classifications = 0   # gate-bypassed classifications - see FORCE_CLASSIFY_AFTER_FRAMES

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
        total_track_occurrences += len(track_ids)
        seen_track_ids.update(int(tid) for tid in track_ids)

        # ---- lazy + quality-gated classification: not-yet-locked AND (a good crop OR the
        # force-classify safety net has kicked in for a track that's had zero evidence too long) ----
        pending_idx = [i for i, tid in enumerate(track_ids) if tid not in locked_class]
        to_classify_idx = []
        for i in pending_idx:
            tid = int(track_ids[i])
            has_evidence = tid in track_prob_accum
            if is_good_crop(tracked_xyxy[i], track_conf[i], tracked_xyxy, i, frame):
                to_classify_idx.append(i)
                unclassified_streak[tid] = 0
            elif not has_evidence:
                unclassified_streak[tid] = unclassified_streak.get(tid, 0) + 1
                if unclassified_streak[tid] >= FORCE_CLASSIFY_AFTER_FRAMES:
                    to_classify_idx.append(i)   # gate bypassed - see FORCE_CLASSIFY_AFTER_FRAMES
                    total_forced_classifications += 1
                else:
                    total_crops_rejected += 1
            else:
                total_crops_rejected += 1
        pending_boxes = tracked_xyxy[to_classify_idx] if to_classify_idx else np.zeros((0, 4))
        pending_ids = track_ids[to_classify_idx] if to_classify_idx else np.zeros(0, dtype=int)

        _, resnet_confs, resnet_probs, resnet_ms = classify_crops(resnet_model, frame, pending_boxes)
        if resnet_ms:
            resnet_times.append(resnet_ms)
            resnet_batch_sizes.append(len(pending_boxes))
        total_resnet_calls += len(pending_boxes)

        for tid, conf, prob_vec in zip(pending_ids, resnet_confs, resnet_probs):
            accum = track_prob_accum.setdefault(tid, np.zeros(len(class_names)))
            accum += prob_vec
            evidence_count[tid] = evidence_count.get(tid, 0) + 1
            top_idx, reliable = evaluate_lock(accum, conf, evidence_count[tid])
            if reliable:
                locked_class[tid] = top_idx

        current_classes = []
        for i, tid in enumerate(track_ids):
            if tid in locked_class:
                current_classes.append(locked_class[tid])
            elif tid in track_prob_accum:
                current_classes.append(int(np.argmax(track_prob_accum[tid])))
            else:
                current_classes.append(UNCLASSIFIED)   # no good crop has landed yet - waiting

        annotated = draw_tracked_detections(frame, tracked_xyxy, track_ids, current_classes,
                                             locked_class.keys(), class_names, CLASS_COLORS)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}  Tracked: {len(track_ids)}  "
                                f"Classifying: {len(pending_boxes)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
    resnet_acc = load_resnet_val_acc(RESNET_RESULTS_CSV)

    print("\n" + "=" * 70)
    print("YOLO (PLAIN) + BYTETRACK + QUALITY-GATED LAZY RESNET - BENCHMARK & COUNT REPORT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}  |  Source FPS: {src_fps:.2f}  |  Frames processed: {frame_count}")

    print(f"\nYOLO detector accuracy (validation set, epoch {yolo_acc['epoch']}) - localization reference only:")
    print(f"  Precision: {yolo_acc['precision']:.4f}  Recall: {yolo_acc['recall']:.4f}  "
          f"mAP50: {yolo_acc['mAP50']:.4f}  mAP50-95: {yolo_acc['mAP50-95']:.4f}")
    print(f"ResNet classifier accuracy (validation set): {resnet_acc:.4f} (see {RESNET_RESULTS_CSV})")

    print("\n--- LAZY + QUALITY-GATE EFFECT ---")
    print(f"  Track-frame occurrences (what yr_byte.py would classify): {total_track_occurrences}")
    print(f"  Crops rejected by the good-crop gate:                     {total_crops_rejected}")
    print(f"  Gate-bypassed (forced after {FORCE_CLASSIFY_AFTER_FRAMES} zero-evidence frames):    {total_forced_classifications}")
    print(f"  Actual ResNet calls made (this script):                   {total_resnet_calls}")
    if total_track_occurrences:
        reduction = 100 * (1 - total_resnet_calls / total_track_occurrences)
        print(f"  Reduction: {reduction:.1f}% fewer ResNet calls")

    print("\n--- LATENCY (separate) ---")
    summarize("YOLO detection (plain, single forward pass)", yolo_times)
    summarize("ByteTrack update", track_times)
    if resnet_times:
        print(f"  ResNet input size: {RESNET_IMG_SIZE}x{RESNET_IMG_SIZE}  (FP16: {RESNET_USE_FP16})")
        print(f"  ResNet batch size (objects actually classified/frame): avg {stats.mean(resnet_batch_sizes):.1f}  "
              f"min {min(resnet_batch_sizes)}  max {max(resnet_batch_sizes)}")
        summarize("ResNet classification (1 batched forward pass per frame's PENDING objects only)", resnet_times)
    else:
        print("  ResNet classification: no objects were ever tracked - nothing to report.")

    print("\n--- END-TO-END (combined) ---")
    summarize("Full per-frame pipeline (capture + YOLO + track + crop + ResNet + draw + display)", e2e_times)
    achievable_fps = 1000 / stats.mean(e2e_times)
    print(f"\nMax achievable FPS (unthrottled, back-to-back): {achievable_fps:.1f}")
    print(f"Source video FPS: {src_fps:.2f}  ->  "
          f"{'pipeline kept up with real time' if achievable_fps >= src_fps else 'pipeline is SLOWER than source FPS - live playback lagged behind real time'}.")

    # Every track that reached FORCE_CLASSIFY_AFTER_FRAMES with zero evidence was already
    # force-classified live, in the main loop, while its frame data still existed (see
    # FORCE_CLASSIFY_AFTER_FRAMES above) - so the only tracks that can still have zero evidence
    # here are ones that disappeared from tracking (occlusion, left frame, video ended) BEFORE
    # reaching that threshold. Genuinely rare, and there's no frame data left to classify from at
    # this point (the video is already closed) - reported honestly rather than guessed at.
    never_classified = seen_track_ids - set(track_prob_accum.keys())
    if never_classified:
        print(f"\n{len(never_classified)} track(s) disappeared before ever getting a good crop OR "
              f"reaching the {FORCE_CLASSIFY_AFTER_FRAMES}-frame force-classify threshold (very "
              f"short-lived tracks) - excluded from the count below rather than guessed at.")

    print("\n--- OBJECT COUNTS (unique tracked objects, locked/accumulated-probability class) ---")
    final_counts = Counter()
    for tid in seen_track_ids:
        if tid in locked_class:
            best_class_idx = locked_class[tid]
        elif tid in track_prob_accum:
            best_class_idx = int(np.argmax(track_prob_accum[tid]))
        else:
            continue   # see the never_classified note above
        final_counts[class_names[best_class_idx]] += 1

    total_objects = sum(final_counts.values())
    for name in sorted(class_names):
        print(f"  {name:12s}: {final_counts.get(name, 0)}")
    print(f"  {'TOTAL':12s}: {total_objects}")
    if never_classified:
        print(f"  ({len(never_classified)} track(s) with zero good crops are excluded from the "
              f"counts above - see the note before this section)")
    print("=" * 70)


if __name__ == "__main__":
    main()
