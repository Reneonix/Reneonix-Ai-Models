"""
Recommended architecture: plain YOLO detection + BoT-SORT+ReID tracking + LAZY ResNet
classification + COUNT-ONCE-AT-A-LINE. Built to directly fix the over-counting problem
yr_byte_lazy.py hit in practice (238 counted vs ~41 real objects on the conveyor).

WHY THE OVER-COUNT HAPPENED (and why this script is architecturally different, not just tuned):
yr_byte_lazy.py uses ByteTrack - motion/IoU-only association. On a moving conveyor, fast object
motion, brief occlusion, or a single missed detection is enough for ByteTrack to lose a track and
hand the same physical object a NEW id when it reappears - one real object becomes several
"unique" counted objects. This script targets that at the source with BoT-SORT+ReID (same
tracker as yr_sahi_botsort.py): it matches objects by VISUAL APPEARANCE as well as position,
which is the INTENDED fix for exactly the occlusion/fast-motion/crossing-paths cases that broke
ByteTrack here - not a guarantee. How much it actually helps depends on BoT-SORT/ReID's own
tuning (TRACK_BUFFER, match thresholds), the scene, how visually distinct the objects are from
each other, and detection quality - verify it against your own footage (see the tracking debug
log below) rather than assuming it fully solved the problem.

ON TOP OF the better tracker, three more changes vs. yr_sahi_botsort.py:
  1. GOOD CROP SELECTION - before spending a ResNet call on a track, check the crop is actually
     trustworthy: big enough, sharp (not motion-blurred), not significantly overlapping another
     tracked box (a cheap occlusion proxy), and from a confident-enough detection. A crop that
     fails is skipped this frame rather than wasting a vote on a bad view.
  2. LAZY classification with PROBABILITY ACCUMULATION (not simple vote-counting) - every good
     crop's full softmax probability vector is summed into that track's running evidence, so a
     crop the model was 81% sure about counts for more than one it was only 40% sure about. A
     track locks once its leading class is clearly ahead (high single-shot confidence, high
     accumulated total, or a wide margin over the runner-up), or once a hard attempt cap is hit.
     A track that goes many consecutive frames with zero evidence (every crop rejected by the
     gate) is force-classified once, gate bypassed, so it's never left permanently unlabeled.
  3. COUNT-ONCE-AT-A-LINE, not "count every ever-seen track at the end" - a track is only added
     to the final count the moment its centroid crosses COUNTING_LINE_POSITION, and only once,
     ever, even if it lingers near the line or the box jitters back and forth. This is a second,
     independent safeguard against over-counting: even if a track ID were somehow reused, it
     still can't be counted twice. It also means an object that never actually reaches/crosses
     the line is NOT counted - place the line somewhere every real object is guaranteed to pass,
     and where objects are reasonably well-separated (per Ultralytics' own tracker tuning advice).
     A track that crosses with zero evidence (rare - only if it crosses faster than the
     force-classify threshold catches it) is classified on the spot at crossing time instead.

PLAIN YOLO, NOT SAHI-TILED: this recommended architecture prioritizes speed + reliable counting
over SAHI's small-object recall - see yr_sahi_botsort.py if you specifically need SAHI's tiling
on top of this (would need merging the two, not done here).

Press 'q' (video window focused) to stop; report + object counts print either way.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50
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
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp001_yolov8l/weights/best.pt"
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - pair with exp008 (RESNET_WEIGHTS below), NOT exp004, which is a 6-class classifier with a different taxonomy

YOLO_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp007_AGI/results.csv"   # must match whichever YOLO_WEIGHTS is active above - exp007's own results, not exp001's
# RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"
RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp008_resnet50_AGI/weights/best.pt"   # 5-class AGI classifier - pairs with exp007 above, NOT the 6-class exp001/exp003/exp005/exp006 YOLO models
RESNET_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp008_AGI/results.csv"   # must match whichever RESNET_WEIGHTS is active above
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/agi/m/v1.mp4"   # default - overridden by a CLI argument if given
# VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/low_exposure.mp4"   # default - overridden by a CLI argument if given


CONF = 0.35   # lowered from the usual 0.5 - reduces detection flicker/missed-frame gaps (a
               # contributing cause of track loss), per the diagram's own tuning notes (0.3-0.4)
IMGSZ = 640          # YOLO inference size (matches training imgsz)
DEVICE = 0           # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
RESNET_IMG_SIZE = 224   # matches exp004_resnet50.py's/exp008_resnet50_agi.py's transform
RESNET_USE_FP16 = True   # mixed-precision inference - free speedup on the RTX 5080's tensor cores

TRACKER_YAML = "botsort.yaml"   # ships with Ultralytics; loaded the same way model.track() does
REID_MODEL = "yolo26n-reid.onnx"   # Ultralytics' own purpose-built appearance-embedding model -
                                     # auto-downloaded on first use, runs via onnxruntime-gpu.
TRACK_BUFFER = 60        # frames a lost track is kept "alive" waiting for a re-match, before
                          # giving up and starting a fresh id - raised from botsort.yaml's
                          # default 30, per the diagram's tuning notes (50-100), so a longer
                          # occlusion/missed-detection gap still re-links to the same track

# ---- GOOD CROP SELECTION - a crop must pass all of these to be worth classifying ----
MIN_GOOD_CROP_SIZE = 40       # px, both width and height - anything smaller upsamples badly into
                               # the 224x224 ResNet input, mostly blur by then
MIN_GOOD_DET_CONF = 0.5       # detector confidence for THIS box specifically - separate from (and
                               # can be stricter than) CONF, which only gates whether YOLO returns
                               # the box at all
MIN_SHARPNESS = 30.0          # Laplacian-variance blur measure - genuinely footage-dependent,
                               # watch the printed "crops rejected" stats and adjust if this is
                               # rejecting almost everything or almost nothing
OCCLUSION_IOU_THRESH = 0.3    # skip a crop whose box overlaps another currently-tracked box by
                               # more than this IoU - a cheap proxy for "probably partially hidden"

# ---- probability-accumulation evidence + lock rule (same idea as yr_byte_lazy.py) ----
RESNET_LOCK_CONF = 0.85            # a single good crop this confident locks the track immediately
PROB_ACCUM_LOCK_THRESHOLD = 1.5    # or lock once the leading class's SUMMED probability reaches this
PROB_ACCUM_MARGIN = 0.75           # or lock once the leading class's summed probability beats the
                                     # runner-up class by at least this much
RESNET_MAX_ATTEMPTS = 5            # hard cap on GOOD-crop classification attempts per track
FORCE_CLASSIFY_AFTER_FRAMES = 10   # a track with STILL zero evidence after this many consecutive
                                     # gate-rejections gets classified anyway, gate bypassed

# ---- counting line (see module docstring for why this exists) ----
COUNTING_LINE_AXIS = "vertical"     # "vertical" line -> objects cross it moving left/right (checks x);
                                      # "horizontal" line -> objects cross moving up/down (checks y).
                                      # Set to match your actual camera framing/belt direction.
COUNTING_LINE_POSITION = 0.5        # fraction (0-1) of frame width (vertical axis) or height
                                      # (horizontal axis) where the line sits - place it somewhere
                                      # every object is guaranteed to cross, ideally where objects
                                      # are well-separated (fewer simultaneous crossings to resolve)

# ---- TRACK DEBUGGING (diagnostic only; does not change tracking/counting behavior) ----
TRACK_DEBUG = True
TRACK_DEBUG_CSV = "botsort_track_debug.csv"

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
    parser = argparse.ArgumentParser(
        description="YOLO detect -> BoT-SORT+ReID track -> lazy ResNet classify -> count-once-at-line pipeline")
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
    this frame (or were force-classified - see main loop). Returns (pred_class_idx, top_conf,
    full_prob_matrix, ms) - the full probability vector per box drives the accumulation logic,
    not just the argmax."""
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
    """GOOD CROP SELECTION gate - see module docstring."""
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


def evaluate_lock(accum, top_conf, evidence_n):
    """Given a track's accumulated probability vector, decide its current best class and whether
    that decision is reliable enough to lock."""
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


def build_tracker():
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(TRACKER_YAML)))
    # botsort.yaml defaults to with_reid=False, model="auto" (which needs Ultralytics' own
    # predictor-hook machinery we're bypassing). Overriding both explicitly gives BOTSORT a
    # concrete, self-contained ReID model - see yr_sahi_botsort.py's own comment for the full
    # reasoning (this pipeline reuses that exact tracker setup).
    cfg.with_reid = True
    cfg.model = REID_MODEL
    cfg.device = TORCH_DEVICE
    cfg.track_buffer = TRACK_BUFFER
    return BOTSORT(args=cfg)


def update_tracker(tracker, boxes, scores, cls, frame):
    """Feeds our own YOLO detections into Ultralytics' real BOTSORT tracker by wrapping them in
    the exact Boxes format it expects: (N,6) = [x1,y1,x2,y2,conf,cls]. `frame` is passed through
    as the ReID encoder's image input. Returns tracked (xyxy, track_id, conf, cls) arrays - empty
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


def side_of_line(box_xyxy, line_pos_px):
    """Which side of the counting line a box's centroid is currently on - True/False, arbitrary
    but consistent, so a crossing is just "this changed since last frame"."""
    x1, y1, x2, y2 = box_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    coord = cx if COUNTING_LINE_AXIS == "vertical" else cy
    return coord >= line_pos_px


def draw_tracked_detections(frame, boxes, track_ids, classes, locked_ids, counted_ids, class_names, class_colors):
    annotated = frame.copy()
    for box, tid, cls_idx in zip(boxes, track_ids, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        if cls_idx == UNCLASSIFIED:
            color = (160, 160, 160)
            label = f"#{tid} ..."
        else:
            color = class_colors[int(cls_idx) % len(class_colors)]
            lock_mark = "*" if tid in locked_ids else ""
            count_mark = " [counted]" if tid in counted_ids else ""
            label = f"#{tid}{lock_mark} {class_names[int(cls_idx)]}{count_mark}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def draw_counting_line(frame, line_pos_px):
    h, w = frame.shape[:2]
    p1, p2 = ((int(line_pos_px), 0), (int(line_pos_px), h)) if COUNTING_LINE_AXIS == "vertical" \
        else ((0, int(line_pos_px)), (w, int(line_pos_px)))
    cv2.line(frame, p1, p2, (0, 0, 255), 2, lineType=cv2.LINE_AA)


CLASS_COLORS = [   # BGR - indexed by ResNet's own class order, not YOLO's
    (233, 180, 86), (167, 121, 204), (0, 94, 213), (115, 158, 0), (0, 159, 230), (66, 228, 240),
]


def write_track_debug_header(path):
    """Create the diagnostic CSV once. This file records first-seen and lost-track events."""
    if not TRACK_DEBUG:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event", "frame", "time_s", "track_id", "lifetime_frames",
            "x1", "y1", "x2", "y2", "track_conf",
            "class", "locked", "crossed_line"
        ])


def append_track_debug(path, event, frame_no, src_fps, tid, info, class_label, locked, crossed):
    if not TRACK_DEBUG:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        box = info.get("last_box", [np.nan] * 4)
        writer.writerow([
            event,
            frame_no,
            f"{frame_no / src_fps:.3f}",
            int(tid),
            int(info.get("last_frame", frame_no) - info.get("first_frame", frame_no) + 1),
            *[f"{float(v):.1f}" for v in box],
            f"{float(info.get('last_conf', 0.0)):.4f}",
            class_label,
            bool(locked),
            bool(crossed),
        ])


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
          f"OR margin over runner-up>={PROB_ACCUM_MARGIN}, OR {RESNET_MAX_ATTEMPTS} good crops spent")
    print(f"Counting line: {COUNTING_LINE_AXIS} at {COUNTING_LINE_POSITION:.0%} of frame "
          f"{'width' if COUNTING_LINE_AXIS == 'vertical' else 'height'}\n")

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
    line_pos_px = COUNTING_LINE_POSITION * (width if COUNTING_LINE_AXIS == "vertical" else height)
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}\n")

    print("Warming up (first-call JIT/compile costs)...")
    ok, warm_frame = cap.read()
    if ok:
        warm_frame = crop_to_roi(warm_frame, roi)
        for _ in range(2):   # BOTSORT's Kalman filter/GMC/ONNX session settle after 2 calls, not just 1
            wb, ws, wc, _ = yolo_infer(yolo_model, warm_frame)
            wt_xyxy, wt_ids, _, _ = update_tracker(tracker, wb, ws, wc, warm_frame)
            classify_crops(resnet_model, warm_frame, wt_xyxy)
    print("Warm-up done.\n")

    window = f"YOLO+BoTSORT(ReID)+LazyResNet+CountLine - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    yolo_times, track_times, resnet_times, e2e_times = [], [], [], []
    resnet_batch_sizes = []
    track_prob_accum = {}                  # track_id -> np.array of summed softmax probabilities so far
    locked_class = {}                      # track_id -> locked class idx
    evidence_count = defaultdict(int)      # track_id -> GOOD-crop ResNet calls made so far
    unclassified_streak = {}               # track_id -> consecutive frames alive with zero evidence
    track_last_side = {}                   # track_id -> which side of the line it was on last frame
    counted_ids = set()                    # track_ids already counted - never counted twice
    final_counts = Counter()               # class_name -> count, filled in AT CROSSING TIME
    total_crops_rejected = 0               # good-crop-gate rejections
    total_forced_classifications = 0       # gate-bypassed classifications (streak or line-crossing)

    # ---- track lifecycle diagnostics ----
    seen_track_ids = set()                 # every Track ID ever observed in this run
    active_track_ids_prev = set()          # Track IDs visible on the previous frame
    track_info = {}                        # track_id -> first/last frame, box/conf, crossed flag
    track_lifetimes = {}                   # completed track_id -> lifetime in frames
    track_debug_csv = os.path.join(os.path.dirname(video_path), TRACK_DEBUG_CSV)
    if TRACK_DEBUG:
        write_track_debug_header(track_debug_csv)
        print(f"Track debug CSV: {track_debug_csv}")

    total_track_occurrences = 0
    total_resnet_calls = 0

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

        # ---- track lifecycle diagnostics: first-seen IDs + IDs that disappeared ----
        current_track_ids = set(int(tid) for tid in track_ids)
        new_track_ids = current_track_ids - seen_track_ids
        lost_track_ids = active_track_ids_prev - current_track_ids

        for tid in new_track_ids:
            idx = int(np.where(track_ids == tid)[0][0])
            track_info[tid] = {
                "first_frame": frame_count,
                "last_frame": frame_count,
                "last_box": tracked_xyxy[idx].copy(),
                "last_conf": float(track_conf[idx]),
                "crossed": False,
            }
            seen_track_ids.add(tid)
            print(f"[NEW TRACK] frame={frame_count:5d}  id=#{tid:<4d}  "
                  f"conf={float(track_conf[idx]):.3f}  "
                  f"box={[round(float(v), 1) for v in tracked_xyxy[idx]]}")

        # Update lifecycle information for all currently visible tracks.
        for idx, tid_raw in enumerate(track_ids):
            tid = int(tid_raw)
            info = track_info.setdefault(tid, {
                "first_frame": frame_count,
                "last_frame": frame_count,
                "last_box": tracked_xyxy[idx].copy(),
                "last_conf": float(track_conf[idx]),
                "crossed": False,
            })
            info["last_frame"] = frame_count
            info["last_box"] = tracked_xyxy[idx].copy()
            info["last_conf"] = float(track_conf[idx])

        # Log tracks that were visible on the previous frame but disappeared now.
        for tid in sorted(lost_track_ids):
            info = track_info[tid]
            lifetime = info["last_frame"] - info["first_frame"] + 1
            track_lifetimes[tid] = lifetime
            if tid in locked_class:
                label = class_names[locked_class[tid]]
                locked = True
            elif tid in track_prob_accum:
                label = class_names[int(np.argmax(track_prob_accum[tid]))]
                locked = False
            else:
                label = "unclassified"
                locked = False
            print(f"[TRACK LOST] frame={frame_count:5d}  id=#{tid:<4d}  "
                  f"lifetime={lifetime:4d} frames  class={label:<10s}  "
                  f"crossed={info['crossed']}")
            append_track_debug(
                track_debug_csv, "LOST", frame_count, src_fps, tid, info,
                label, locked, info["crossed"]
            )

        active_track_ids_prev = current_track_ids

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
            evidence_count[tid] += 1
            top_idx, reliable = evaluate_lock(accum, conf, evidence_count[tid])
            if reliable:
                locked_class[tid] = top_idx

        current_classes = []
        for tid in track_ids:
            if tid in locked_class:
                current_classes.append(locked_class[tid])
            elif tid in track_prob_accum:
                current_classes.append(int(np.argmax(track_prob_accum[tid])))
            else:
                current_classes.append(UNCLASSIFIED)   # no good crop has landed yet - waiting

        # ---- count-once-at-line: check every currently-visible track for a crossing ----
        for i, (box, tid, cls_idx) in enumerate(zip(tracked_xyxy, track_ids, current_classes)):
            side = side_of_line(box, line_pos_px)
            prev_side = track_last_side.get(tid)
            if prev_side is not None and side != prev_side and tid not in counted_ids:
                if cls_idx == UNCLASSIFIED:
                    # crossed before ever getting evidence (rare - faster than the force-classify
                    # streak threshold) - classify it right now, gate bypassed, real frame data
                    # still available, so it's never counted with a guessed/missing label
                    _, _, fb_probs, fb_ms = classify_crops(resnet_model, frame, tracked_xyxy[i:i + 1])
                    if fb_ms:
                        resnet_times.append(fb_ms)
                        resnet_batch_sizes.append(1)
                    total_resnet_calls += 1
                    total_forced_classifications += 1
                    accum = track_prob_accum.setdefault(tid, np.zeros(len(class_names)))
                    accum += fb_probs[0]
                    cls_idx = int(np.argmax(accum))
                    current_classes[i] = cls_idx
                final_counts[class_names[cls_idx]] += 1
                counted_ids.add(tid)
                if tid in track_info:
                    track_info[tid]["crossed"] = True
            track_last_side[tid] = side

        annotated = draw_tracked_detections(frame, tracked_xyxy, track_ids, current_classes,
                                             locked_class.keys(), counted_ids, class_names, CLASS_COLORS)
        draw_counting_line(annotated, line_pos_px)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}  Tracked: {len(track_ids)}  "
                                f"Classifying: {len(pending_boxes)}  Counted: {len(counted_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

    # Any tracks still active when the video ended never generated a LOST event.
    for tid in sorted(active_track_ids_prev):
        if tid not in track_info:
            continue
        info = track_info[tid]
        lifetime = info["last_frame"] - info["first_frame"] + 1
        track_lifetimes[tid] = lifetime
        if tid in locked_class:
            label = class_names[locked_class[tid]]
            locked = True
        elif tid in track_prob_accum:
            label = class_names[int(np.argmax(track_prob_accum[tid]))]
            locked = False
        else:
            label = "unclassified"
            locked = False
        print(f"[VIDEO END]  id=#{tid:<4d}  lifetime={lifetime:4d} frames  "
              f"class={label:<10s}  crossed={info['crossed']}")
        append_track_debug(
            track_debug_csv, "VIDEO_END", frame_count, src_fps, tid, info,
            label, locked, info["crossed"]
        )

    cap.release()
    cv2.destroyAllWindows()

    if not yolo_times:
        print("No frames were processed.")
        return

    yolo_acc = load_validation_accuracy(YOLO_RESULTS_CSV)
    resnet_acc = load_resnet_val_acc(RESNET_RESULTS_CSV)

    print("\n" + "=" * 70)
    print("YOLO + BOTSORT(REID) + LAZY RESNET + COUNT-ONCE-AT-LINE - BENCHMARK & COUNT REPORT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}  |  Source FPS: {src_fps:.2f}  |  Frames processed: {frame_count}")

    print(f"\nYOLO detector accuracy (validation set, epoch {yolo_acc['epoch']}) - localization reference only:")
    print(f"  Precision: {yolo_acc['precision']:.4f}  Recall: {yolo_acc['recall']:.4f}  "
          f"mAP50: {yolo_acc['mAP50']:.4f}  mAP50-95: {yolo_acc['mAP50-95']:.4f}")
    print(f"ResNet classifier accuracy (validation set): {resnet_acc:.4f} (see {RESNET_RESULTS_CSV})")

    print("\n--- LAZY + QUALITY-GATE EFFECT ---")
    print(f"  Track-frame occurrences (what a per-frame classifier would run): {total_track_occurrences}")
    print(f"  Crops rejected by the good-crop gate:                            {total_crops_rejected}")
    print(f"  Gate-bypassed (forced after zero evidence, or at line-crossing):  {total_forced_classifications}")
    print(f"  Actual ResNet calls made (this script):                          {total_resnet_calls}")
    if total_track_occurrences:
        reduction = 100 * (1 - total_resnet_calls / total_track_occurrences)
        print(f"  Reduction: {reduction:.1f}% fewer ResNet calls")

    print("\n--- LATENCY (separate) ---")
    summarize("YOLO detection (plain, single forward pass)", yolo_times)
    summarize("BoT-SORT+ReID update", track_times)
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

    if TRACK_DEBUG:
        lifetimes = list(track_lifetimes.values())
        short_5 = sum(1 for x in lifetimes if x <= 5)
        short_15 = sum(1 for x in lifetimes if x <= 15)
        print("\n--- TRACK ID DIAGNOSTICS ---")
        print(f"  Unique Track IDs ever observed: {len(seen_track_ids)}")
        if lifetimes:
            print(f"  Completed track lifetimes: avg={stats.mean(lifetimes):.1f} frames, "
                  f"median={stats.median(lifetimes):.1f}, min={min(lifetimes)}, max={max(lifetimes)}")
            print(f"  Very short tracks (<= 5 frames):  {short_5}")
            print(f"  Short tracks (<= 15 frames):     {short_15}")
            print(f"  Longer tracks (> 15 frames):     {len(lifetimes) - short_15}")
        print(f"  Detailed events saved to: {track_debug_csv}")

    print("\n--- OBJECT COUNTS (counted once, at the moment each track crossed the counting line) ---")
    total_objects = sum(final_counts.values())
    tracks_never_crossed = len(seen_track_ids) - len(counted_ids)
    for name in sorted(class_names):
        print(f"  {name:12s}: {final_counts.get(name, 0)}")
    print(f"  {'TOTAL':12s}: {total_objects}")
    if tracks_never_crossed:
        print(f"  ({tracks_never_crossed} tracked object(s) never crossed the counting line - not counted. "
              f"If that's unexpected, check COUNTING_LINE_AXIS/COUNTING_LINE_POSITION against your actual footage.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
