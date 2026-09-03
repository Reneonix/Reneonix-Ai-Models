"""
Quick sanity-test for the stock RT-DETR-L checkpoint sitting in this folder (rtdetr-l.pt) -
Baidu's architecture ("DETRs Beat YOLOs on Real-Time Object Detection", 2023), loaded via
Ultralytics' own `RTDETR` class (confirmed to share YOLO's `.train()`/`.predict()`/callback
API - same `Model` base class - verified against the installed ultralytics package before
src/exp005_rtdetr.py was written).

IMPORTANT: this is the RAW COCO-pretrained checkpoint, NOT fine-tuned on our 6 waste classes
yet (that's what src/exp005_rtdetr.py trains). Detections here will be labeled with COCO's 80
generic classes (person, car, bottle, ...) - this script exists purely to confirm the model
loads and runs correctly on real video before spending time on a full training run, not to
evaluate waste-detection accuracy.

Usage:
    python testing.py [path/to/video.mp4]

Press 'q' (video window focused) to stop.
"""

from ultralytics import RTDETR
import cv2
import time
import os
import argparse

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtdetr-l.pt")
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/low_exposure.mp4"   # default - overridden by a CLI argument if given

CONF = 0.5
IMGSZ = 640
DEVICE = 0   # RTX 5080 (cuda:0); set to "cpu" if no GPU available
BOX_THICKNESS = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-test the stock RT-DETR-L checkpoint on a video")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def main():
    video_path = os.path.abspath(parse_args().video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(WEIGHTS):
        raise FileNotFoundError(f"Weights not found: {WEIGHTS}")

    print(f"Using video: {video_path}")
    print(f"Weights: {WEIGHTS} (stock COCO-pretrained - NOT fine-tuned on our waste classes)")

    model = RTDETR(WEIGHTS)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_budget = 1.0 / src_fps
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}\n")

    # Warm up on the real first frame so the first-call CUDA/JIT cost doesn't bleed into the
    # displayed/benchmarked loop (same pattern used by every other live-test script in this project).
    print("Warming up...")
    ok, warm_frame = cap.read()
    if ok:
        model.predict(warm_frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
    print("Warm-up done.\n")

    window = f"RT-DETR-L (stock, COCO classes) - {os.path.basename(video_path)}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    infer_times = []
    prev_time = time.time()
    frame_count = 0
    while True:
        loop_start = time.time()

        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        t0 = time.time()
        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        infer_ms = (time.time() - t0) * 1000
        infer_times.append(infer_ms)

        annotated = results[0].plot(line_width=BOX_THICKNESS)   # Ultralytics' own built-in
                                                                    # box+label+confidence renderer

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        n_det = len(results[0].boxes) if results[0].boxes is not None else 0
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}  Detections: {n_det}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(window, annotated)
        display_end = time.time()

        remaining_ms = max(1, int((frame_budget - (display_end - loop_start)) * 1000))
        if cv2.waitKey(remaining_ms) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not infer_times:
        print("No frames were processed.")
        return

    import statistics as stats
    print("\n" + "=" * 60)
    print("RT-DETR-L (STOCK, COCO CLASSES) - SANITY TEST REPORT")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Frames processed: {frame_count}")
    print(f"Inference latency: avg {stats.mean(infer_times):.2f} ms "
          f"({1000 / stats.mean(infer_times):.1f} FPS)  |  "
          f"median {stats.median(infer_times):.2f} ms  |  "
          f"min {min(infer_times):.2f} ms  |  max {max(infer_times):.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
