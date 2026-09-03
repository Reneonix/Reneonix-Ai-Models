"""
Live video inference + full benchmark report for the trained YOLOv8-L waste
detector, run over videos/mixed_glass.mp4 via OpenCV.

- Displays annotated detections live, paced to the video's OWN native FPS
  (read from the file itself, not assumed).
- Press 'q' (video window focused) to stop early; report prints either way.
- On exit, prints: model accuracy (from training validation), per-frame
  inference time, per-frame end-to-end latency, and achieved live FPS -
  all at the video's actual frame resolution.
"""

from ultralytics import YOLO
import cv2
import time
import csv
import os
import argparse
import statistics as stats

# ---------------- CONFIG ----------------
# WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp003_yolo26l_p2/weights/best.pt"
WEIGHTS = "d:/Reneonix/yolo_projects/Wastes_identification/experiments/exp007_yolov8l_AGI/weights/best.pt"   # 5-class AGI dataset (ferrous, plastic, stone, ceramic, glass) - not the same taxonomy as the 6-class models above
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/AGI/m/v1.mp4"  # default - overridden by a CLI argument if given
RESULTS_CSV = "d:/Reneonix/yolo_projects/Wastes_identification/results/exp003/results.csv"

CONF = 0.5   # minimum detection confidence to draw a box
IMGSZ = 640      # matches training imgsz
DEVICE = 0       # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2


def load_validation_accuracy():
    """Pull the trained model's final validation metrics straight from training's own
    results.csv (last row), so the benchmark report always reflects the actual model
    being loaded - not a hand-typed, possibly-stale number."""
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
    parser = argparse.ArgumentParser(description="Live YOLO video inference + benchmark report")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


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
    print(f"Video: {video_path}")
    print(f"Detected source FPS: {src_fps:.2f}  |  Resolution: {width}x{height}")
    print(f"Live playback will be paced to {src_fps:.2f} FPS to match the source.\n")

    window = f"Waste Detection - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # per-frame timing buckets (all in ms)
    capture_times = []
    inference_times = []   # pure model forward pass, from Ultralytics' own speed report
    e2e_times = []          # capture -> inference -> draw -> imshow, excludes only the FPS-pacing wait

    prev_time = time.time()
    frame_count = 0
    while True:
        loop_start = time.time()

        ok, frame = cap.read()
        capture_end = time.time()
        if not ok:
            break  # end of video
        frame_count += 1

        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        inference_times.append(results[0].speed["inference"])

        annotated = results[0].plot(line_width=BOX_THICKNESS)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window, annotated)
        display_end = time.time()

        capture_times.append((capture_end - loop_start) * 1000)
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
    print("BENCHMARK REPORT")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Source FPS: {src_fps:.2f}")
    print(f"Frames processed: {frame_count}")

    print(f"\nModel accuracy (validation set, epoch {acc['epoch']}):")
    print(f"  Precision:  {acc['precision']:.4f}")
    print(f"  Recall:     {acc['recall']:.4f}")
    print(f"  mAP50:      {acc['mAP50']:.4f}")
    print(f"  mAP50-95:   {acc['mAP50-95']:.4f}")

    print()
    summarize("Model inference time (forward pass only)", inference_times)

    print()
    summarize("End-to-end latency per frame (capture + inference + draw + display)", e2e_times)

    achievable_fps = 1000 / stats.mean(e2e_times)
    print(f"\nMax achievable FPS (unthrottled, back-to-back): {achievable_fps:.1f}")
    print(f"Live playback was paced to source FPS ({src_fps:.2f}); "
          f"{'model keeps up in real time' if achievable_fps >= src_fps else 'model is SLOWER than source FPS - live playback lagged behind real time'}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
