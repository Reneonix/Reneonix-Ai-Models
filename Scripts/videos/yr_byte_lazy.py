"""
Two-model pipeline: plain YOLO (single-pass, no SAHI) for detection/localization + ByteTrack for
persistent per-object tracking + ResNet-50 for classification - but LAZY, not per-frame.

WHY THIS EXISTS (the problem it solves):
yr_byte.py classifies every tracked object EVERY frame it's visible - fine when the conveyor is
sparse, but FPS drops as object count rises, because the ResNet batch grows with however many
objects happen to be on screen that frame. Most of that is wasted work: an object's material
class doesn't change frame to frame, so reclassifying it 20-30 times over its life on the belt
adds no real information, just cost.

THE FIX - classify once per object, then lock:
  1. Already locked (see below)?  -> reuse the cached class, skip ResNet entirely.
  2. Not locked yet               -> run ResNet on it this frame, record the prediction as one
                                       vote for that track.
  3. Enough evidence to lock?
       - a single vote with confidence >= RESNET_LOCK_CONF (most objects, most frames), OR
       - RESNET_CONSENSUS_VOTES agreeing votes accumulated, OR
       - RESNET_MAX_ATTEMPTS reached regardless (bounds worst-case cost for a stubbornly
         ambiguous object - it locks to whatever's most common so far)
     -> lock the track to its majority-vote class; it costs zero further ResNet calls for the
        rest of its life on screen.
     Otherwise -> leave unlocked, it'll be classified again next frame it's seen.

In steady state this means most objects cost ONE ResNet call over their entire tracked
lifetime, not one per frame - so a busier conveyor (more simultaneous tracks) no longer means
a proportionally bigger ResNet batch every single frame. The final report below prints exactly
how many ResNet calls were actually made vs how many track-frame occurrences there were, so the
effect is visible, not just assumed.

One structural note (not a bug): this keys off ByteTrack's track_id. If the tracker ever loses
and reassigns a new ID to the same physical object (occlusion, leaving/re-entering frame), that
"new" ID starts fresh evidence-gathering - one extra ResNet call, same as any new object.

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
# YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
YOLO_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset - pair with exp008 (RESNET_WEIGHTS below), NOT exp004, which is a 6-class classifier with a different taxonomy

YOLO_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp007_AGI/results.csv"   # must match whichever YOLO_WEIGHTS is active above - exp007's own results, not exp001's
# RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp004_resnet50/weights/best.pt"
RESNET_WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp008_resnet50_AGI/weights/best.pt"   # 5-class AGI classifier - pairs with exp007 above, NOT the 6-class exp001/exp003/exp005/exp006 YOLO models
RESNET_RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp008_AGI/results.csv"   # must match whichever RESNET_WEIGHTS is active above
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/agi/m/v1.mp4"   # default - overridden by a CLI argument if given

CONF = 0.5    # minimum YOLO detection confidence to keep a box (localization only)
IMGSZ = 640          # YOLO inference size (matches training imgsz)
DEVICE = 0           # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2
RESNET_IMG_SIZE = 224   # matches exp004_resnet50.py's/exp008_resnet50_agi.py's transform

TRACKER_YAML = "bytetrack.yaml"   # ships with Ultralytics; loaded the same way model.track() does

# ---- lazy-classification lock rule (see module docstring) ----
RESNET_LOCK_CONF = 0.90         # a single vote this confident locks the track immediately
RESNET_CONSENSUS_VOTES = 3      # or lock once this many votes agree (doesn't need to be consecutive)
RESNET_MAX_ATTEMPTS = 5         # hard cap - locks to the majority-so-far regardless past this,
                                  # so a persistently ambiguous object doesn't get reclassified forever

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
    parser = argparse.ArgumentParser(description="YOLO (plain) detect -> ByteTrack -> lazy ResNet classify pipeline")
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
    only on the subset of tracked boxes that still need classifying this frame (see main loop) -
    NOT every tracked box, which is the whole point of this script."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int), np.array([], dtype=float), 0.0

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


def draw_tracked_detections(frame, boxes, track_ids, classes, locked_ids, class_names, class_colors):
    """`classes` is each track's current best-known class (locked, or best-vote-so-far if not
    locked yet) - every visible track always has at least one vote by the time this is called,
    since a never-before-seen track gets classified the very first frame it appears. A small
    padlock glyph marks locked (no-more-ResNet-needed) tracks, just for visual sanity-checking."""
    annotated = frame.copy()
    for box, tid, cls_idx in zip(boxes, track_ids, classes):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = class_colors[int(cls_idx) % len(class_colors)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        lock_mark = "*" if tid in locked_ids else ""
        label = f"#{tid}{lock_mark} {class_names[int(cls_idx)]}"
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
    print(f"ResNet class order: {class_names}")
    print(f"Lazy classification: lock at conf>={RESNET_LOCK_CONF} or {RESNET_CONSENSUS_VOTES} "
          f"agreeing votes, hard cap {RESNET_MAX_ATTEMPTS} attempts/track\n")

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

    window = f"YOLO+ByteTrack+ResNet (lazy) - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    yolo_times, track_times, resnet_times, e2e_times = [], [], [], []
    resnet_batch_sizes = []   # how many objects actually needed classifying each frame (NOT how
                                # many were tracked - that's the whole point of this script)
    track_votes = defaultdict(Counter)   # track_id -> Counter of ResNet predictions so far
    locked_class = {}                     # track_id -> locked class idx, once evidence is sufficient
    evidence_count = defaultdict(int)     # track_id -> how many ResNet calls made so far

    total_track_occurrences = 0    # sum of len(track_ids) every frame - what yr_byte.py would have classified
    total_resnet_calls = 0          # sum of actual ResNet calls made - what this script classified

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

        # ---- lazy classification: only the not-yet-locked tracks need a ResNet call ----
        pending_mask = np.array([tid not in locked_class for tid in track_ids], dtype=bool) \
            if len(track_ids) else np.zeros(0, dtype=bool)
        pending_boxes = tracked_xyxy[pending_mask]
        pending_ids = track_ids[pending_mask]

        resnet_preds, resnet_confs, resnet_ms = classify_crops(resnet_model, frame, pending_boxes)
        if resnet_ms:
            resnet_times.append(resnet_ms)
            resnet_batch_sizes.append(len(pending_boxes))
        total_resnet_calls += len(pending_boxes)

        for tid, pred, conf in zip(pending_ids, resnet_preds, resnet_confs):
            track_votes[tid][int(pred)] += 1
            evidence_count[tid] += 1
            top_class, top_count = track_votes[tid].most_common(1)[0]
            if (conf >= RESNET_LOCK_CONF or top_count >= RESNET_CONSENSUS_VOTES
                    or evidence_count[tid] >= RESNET_MAX_ATTEMPTS):
                locked_class[tid] = top_class

        # every visible track has at least one vote by now (locked tracks were classified in an
        # earlier frame; pending tracks were just classified above) - resolve current best class
        current_classes = [
            locked_class[tid] if tid in locked_class else track_votes[tid].most_common(1)[0][0]
            for tid in track_ids
        ]

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
    print("YOLO (PLAIN) + BYTETRACK + LAZY RESNET PIPELINE - BENCHMARK & COUNT REPORT")
    print("=" * 70)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}  |  Source FPS: {src_fps:.2f}  |  Frames processed: {frame_count}")

    print(f"\nYOLO detector accuracy (validation set, epoch {yolo_acc['epoch']}) - localization reference only:")
    print(f"  Precision: {yolo_acc['precision']:.4f}  Recall: {yolo_acc['recall']:.4f}  "
          f"mAP50: {yolo_acc['mAP50']:.4f}  mAP50-95: {yolo_acc['mAP50-95']:.4f}")
    print(f"ResNet classifier accuracy (validation set): {resnet_acc:.4f} (see {RESNET_RESULTS_CSV})")

    print("\n--- LAZY CLASSIFICATION EFFECT ---")
    print(f"  Track-frame occurrences (what yr_byte.py would classify): {total_track_occurrences}")
    print(f"  Actual ResNet calls made (this script):                   {total_resnet_calls}")
    if total_track_occurrences:
        reduction = 100 * (1 - total_resnet_calls / total_track_occurrences)
        print(f"  Reduction: {reduction:.1f}% fewer ResNet calls")

    print("\n--- LATENCY (separate) ---")
    summarize("YOLO detection (plain, single forward pass)", yolo_times)
    summarize("ByteTrack update", track_times)
    if resnet_times:
        print(f"  ResNet input size: {RESNET_IMG_SIZE}x{RESNET_IMG_SIZE}")
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

    print("\n--- OBJECT COUNTS (unique tracked objects, locked/majority-vote class) ---")
    final_counts = Counter()
    for tid, votes in track_votes.items():
        best_class_idx = locked_class[tid] if tid in locked_class else votes.most_common(1)[0][0]
        final_counts[class_names[best_class_idx]] += 1

    total_objects = sum(final_counts.values())
    for name in sorted(class_names):
        print(f"  {name:12s}: {final_counts.get(name, 0)}")
    print(f"  {'TOTAL':12s}: {total_objects}")
    print("=" * 70)


if __name__ == "__main__":
    main()
