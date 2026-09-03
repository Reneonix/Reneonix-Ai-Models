"""
Live video inference + full benchmark report using SAHI (Slicing Aided Hyper
Inference) - https://roboflow.com/how-to-use-sahi/yolov8 /
https://learnopencv.com/slicing-aided-hyper-inference/

Same benchmarking approach as video_predict.py (live OpenCV playback paced to
the video's own FPS, model accuracy from training's results.csv, per-frame
inference time, end-to-end latency, achievable FPS), but each frame is:

  1. Sliced into a TILE_ROWS x TILE_COLS = 6-tile overlapping grid
  2. All 6 tiles run through the model together as ONE batched forward pass
     (model.predict() given a list of 6 tile crops -> batch size 6)
  3. Each tile's local box coordinates are translated back to full-frame
     coordinates using that tile's offset
  4. Duplicate detections in the overlapping tile regions are merged with
     per-class NMS

Press 'q' (video window focused) to stop early; report prints either way.
"""

from ultralytics import YOLO
import torch
from torchvision.ops import batched_nms
import cv2
import time
import csv
import os
import argparse
import statistics as stats

# ---------------- CONFIG ----------------
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset (ferrous, plastic, stone, ceramic, glass) - not the same taxonomy as the 6-class models above
# VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/AGI_VIDEOS/MIXED/V1.mp4"   # default - overridden by a CLI argument if given
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/AGI/m/v1.mp4"  # default - overridden by a CLI argument if given

RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp003/results.csv"

CONF = 0.5      # minimum detection confidence to keep a box
IMGSZ = 640      # per-tile inference size (matches training imgsz)
DEVICE = 0       # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2

TILE_COLS = 3          # 3x2 = 6 tiles per frame
TILE_ROWS = 2
TILE_OVERLAP = 0.2      # 20% overlap between neighboring tiles, so objects
                         # straddling a tile border still get detected whole
                         # in at least one tile (core of the SAHI technique)
NMS_IOU = 0.5            # IoU threshold for merging duplicate detections
                         # that fall inside the overlapping tile regions

CLASS_COLORS = [   # BGR, one per waste class (aluminium, plastic, metal, stone, ceramic, glass)
    (233, 180, 86),
    (0, 159, 230),
    (115, 158, 0),
    (66, 228, 240),
    (167, 121, 204),
    (0, 94, 213),
]


def load_validation_accuracy():
    """Pull the trained model's final validation metrics straight from training's own
    results.csv (last row). This reflects the model's accuracy under standard
    single-pass validation - not a live measurement of SAHI-tiled accuracy on this
    video (no ground truth exists for the video to measure that directly)."""
    with open(RESULTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return {
        "precision": float(last["metrics/precision(B)"]),
        "recall": float(last["metrics/recall(B)"]),
        "mAP50": float(last["metrics/mAP50(B)"]),
        "mAP50-95": float(last["metrics/mAP50-95(B)"]),
        "epoch": int(last["epoch"]),
    }


def summarize(label, values_ms):
    print(f"  {label}:")
    print(f"    avg: {stats.mean(values_ms):.2f} ms  ({1000 / stats.mean(values_ms):.1f} FPS)")
    print(f"    median: {stats.median(values_ms):.2f} ms")
    print(f"    min: {min(values_ms):.2f} ms")
    print(f"    max: {max(values_ms):.2f} ms")


def parse_args():
    parser = argparse.ArgumentParser(description="Live YOLO + SAHI video inference and benchmark report")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def compute_tile_boxes(width, height, cols, rows, overlap_ratio):
    """Returns exactly cols*rows (x1, y1, x2, y2) tile boxes covering the full
    frame edge-to-edge, with each neighboring tile overlapping by overlap_ratio
    of the tile's own width/height (the standard SAHI slicing scheme)."""
    tile_w = int(round((width / cols) * (1 + overlap_ratio)))
    tile_h = int(round((height / rows) * (1 + overlap_ratio)))
    tile_w = min(tile_w, width)
    tile_h = min(tile_h, height)

    x_step = (width - tile_w) / (cols - 1) if cols > 1 else 0
    y_step = (height - tile_h) / (rows - 1) if rows > 1 else 0

    boxes = []
    for r in range(rows):
        y1 = int(round(r * y_step))
        y1 = min(y1, height - tile_h)
        for c in range(cols):
            x1 = int(round(c * x_step))
            x1 = min(x1, width - tile_w)
            boxes.append((x1, y1, x1 + tile_w, y1 + tile_h))
    return boxes


def sahi_infer(model, frame, tile_boxes):
    """Runs one batched forward pass over all tiles, translates every box back to
    full-frame coordinates, and merges duplicates from overlapping regions with
    per-class NMS. Returns (merged_boxes_xyxy, merged_scores, merged_cls,
    raw_detection_count) plus the wall-clock time of the batched model call."""
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
        xyxy[:, [0, 2]] += x1   # translate tile-local -> full-frame x
        xyxy[:, [1, 3]] += y1   # translate tile-local -> full-frame y
        all_boxes.append(xyxy)
        all_scores.append(res.boxes.conf)
        all_cls.append(res.boxes.cls)

    if not all_boxes:
        return (torch.empty((0, 4)), torch.empty(0), torch.empty(0), 0, batched_inference_ms)

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    cls = torch.cat(all_cls)

    keep = batched_nms(boxes, scores, cls, NMS_IOU)  # per-class NMS merges tile-overlap duplicates
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
        cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def draw_tile_grid(frame, tile_boxes):
    """Faint overlay showing the current 6-tile slicing grid, for visual sanity-checking."""
    for (x1, y1, x2, y2) in tile_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)


def main():
    video_path = os.path.abspath(parse_args().video)

    # Fail loudly and immediately if the path is wrong, BEFORE loading the model -
    # so there is never any doubt about which file is actually being processed.
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    print(f"Using video: {video_path}\n")

    model = YOLO(WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # DESIGN RULE: always read the given video's own FPS/resolution from the file -
    # never assume/hardcode a value. Live playback is paced to match that detected
    # FPS exactly, whatever video is passed in.
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_budget = 1.0 / src_fps

    tile_boxes = compute_tile_boxes(width, height, TILE_COLS, TILE_ROWS, TILE_OVERLAP)
    tile_w = tile_boxes[0][2] - tile_boxes[0][0]   # uniform across all tiles in this grid scheme
    tile_h = tile_boxes[0][3] - tile_boxes[0][1]
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}")
    print(f"SAHI grid: {TILE_COLS}x{TILE_ROWS} = {len(tile_boxes)} tiles, "
          f"{int(TILE_OVERLAP * 100)}% overlap")
    print(f"Per-tile resolution: {tile_w}x{tile_h}  |  Batch inference size: {len(tile_boxes)}")
    print(f"Live playback will be paced to {src_fps:.2f} FPS to match the source.\n")

    window = f"Waste Detection (SAHI) - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    inference_times = []    # the single batched model.predict() call over all 6 tiles
    e2e_times = []            # slice -> batched inference -> merge/NMS -> draw -> display
    raw_detection_counts = []
    merged_detection_counts = []

    prev_time = time.time()
    frame_count = 0
    while True:
        loop_start = time.time()

        ok, frame = cap.read()
        if not ok:
            break  # end of video
        frame_count += 1

        boxes, scores, cls, raw_count, inference_ms = sahi_infer(model, frame, tile_boxes)
        inference_times.append(inference_ms)
        raw_detection_counts.append(raw_count)
        merged_detection_counts.append(len(boxes))

        annotated = draw_detections(frame, boxes, scores, cls, model.names)
        draw_tile_grid(annotated, tile_boxes)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window, annotated)
        display_end = time.time()

        e2e_times.append((display_end - loop_start) * 1000)  # true per-frame processing latency

        # hold live playback at the video's own native FPS
        remaining_ms = max(1, int((frame_budget - (display_end - loop_start)) * 1000))
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not inference_times:
        print("No frames were processed.")
        return

    acc = load_validation_accuracy()

    print("\n" + "=" * 60)
    print("SAHI BENCHMARK REPORT")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Input resolution: {width}x{height}")
    print(f"Source FPS: {src_fps:.2f}")
    print(f"Frames processed: {frame_count}")
    print(f"Tiles per frame: {TILE_COLS}x{TILE_ROWS} = {len(tile_boxes)}  ({int(TILE_OVERLAP * 100)}% overlap)")
    print(f"Per-tile resolution: {tile_w}x{tile_h}")
    print(f"Batch inference size: {len(tile_boxes)}")
    print(f"Avg raw detections/frame (pre-merge, across all tiles): {stats.mean(raw_detection_counts):.1f}")
    print(f"Avg merged detections/frame (after per-class NMS):      {stats.mean(merged_detection_counts):.1f}")

    print(f"\nModel accuracy (single-pass validation set, epoch {acc['epoch']}) - reference only,")
    print(f"not a direct measurement of SAHI-tiled accuracy on this unlabeled video:")
    print(f"  Precision:  {acc['precision']:.4f}")
    print(f"  Recall:     {acc['recall']:.4f}")
    print(f"  mAP50:      {acc['mAP50']:.4f}")
    print(f"  mAP50-95:   {acc['mAP50-95']:.4f}")

    print()
    summarize(f"Batched model inference time ({len(tile_boxes)} tiles/frame, 1 batched forward pass)", inference_times)

    print()
    summarize("End-to-end latency per frame (slice + batched inference + merge/NMS + draw + display)", e2e_times)

    achievable_fps = 1000 / stats.mean(e2e_times)
    print(f"\nMax achievable FPS (unthrottled, back-to-back): {achievable_fps:.1f}")
    print(f"Live playback was paced to source FPS ({src_fps:.2f}); "
          f"{'model keeps up in real time' if achievable_fps >= src_fps else 'SAHI pipeline is SLOWER than source FPS - live playback lagged behind real time'}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
