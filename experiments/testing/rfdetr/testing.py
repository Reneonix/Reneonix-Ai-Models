"""
Quick sanity-test for Google's real CircularNet segmentation model (RF-DETR architecture),
downloaded straight from the tensorflow/models repo (model_repository/, labels50.csv,
triton_server_inference.py, utils.py in this same folder - the last two are Google's own
reference source, kept here for provenance/comparison, not imported by this script).

WHY THIS SCRIPT DOESN'T JUST IMPORT triton_server_inference.py's TritonObjectDetector:
its own .predict() method is hard-wired to call a REAL Triton Inference Server over HTTP
(`tritonclient.http.InferenceServerClient(...).infer(...)`) - not usable without actually
running a Triton server. Its individual pre/post-processing methods ARE pure functions with
no server dependency though (verified by reading the real downloaded source) - reproduced
verbatim below (sigmoid, box_cxcywh_to_xyxyn, preprocess, reformat_outputs, resize_mask_batch,
scale_bbox_and_masks all match TritonObjectDetector's private methods exactly), so this script
can run real local inference via onnxruntime directly, no Triton server or `tritonclient`
package needed.

Verified against the actual model.onnx file (not assumed - checked via
onnxruntime.InferenceSession(...).get_inputs()/.get_outputs()):
    input:  "input"   (1, 3, 432, 432)  float32
    output: "dets"    (1, 200, 4)        - box coords, cxcywh, normalized [0,1]
            "labels"  (1, 200, 51)       - per-box per-class sigmoid logits (labels50.csv ids
                                            are 1-indexed; model's argmax class index needs +1)
            "masks"   (1, 200, 108, 108) - per-box low-res mask logits, upscaled after

IMPORTANT: this is Google's actual production CircularNet model, not fine-tuned by us - its
50 classes (labels50.csv) are Google's own material/form taxonomy, unrelated to this
project's 6-class scheme. This script exists to confirm the model loads and runs correctly,
and to see how it performs on our own footage as a reference point - not to integrate it into
our own pipeline.

Usage:
    python testing.py [path/to/video.mp4]

Press 'q' (video window focused) to stop.
"""

import os
import argparse
import time
import statistics as stats

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(SCRIPT_DIR, "model_repository", "cn_segmentation_onnx_model", "1", "model.onnx")
LABELS_CSV = os.path.join(SCRIPT_DIR, "labels50.csv")
VIDEO = "d:/Reneonix/yolo_projects/Wastes_identification/videos/high_exposure.mp4"   # default - overridden by a CLI argument if given

INPUT_SIZE = (432, 432)   # verified against the real model.onnx input shape
MEANS = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # same ImageNet normalization this
STDS = np.array([0.229, 0.224, 0.225], dtype=np.float32)     # project's own ResNet already uses

CONF_THRESHOLD = 0.5
MAX_BOXES = 100
BOX_THICKNESS = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity-test Google's real CircularNet ONNX model on a video")
    parser.add_argument("video", nargs="?", default=VIDEO,
                         help=f"Path to a video file (default: {VIDEO})")
    return parser.parse_args()


def load_class_names():
    """id -> name mapping from labels50.csv (ids are 1-indexed)."""
    df = pd.read_csv(LABELS_CSV)
    return df.set_index("id").to_dict()["names"]


def preprocess(frame_bgr):
    """Reimplementation of TritonObjectDetector._get_input_batch_for_inference (verified
    against the real triton_server_inference.py source), adapted to take an in-memory frame
    instead of an image file path (avoids per-frame disk I/O for video)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    float_img = resized.astype(np.float32) / 255.0
    normalized = (float_img - MEANS) / STDS
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).astype(np.float32)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin, ymin = cx - w / 2, cy - h / 2
    xmax, ymax = cx + w / 2, cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


def reformat_outputs(outputs, confidence_threshold, max_boxes):
    """Exact reimplementation of TritonObjectDetector._reformat_triton_output_to_dict -
    outputs = [dets, labels, masks] raw ONNX arrays, matching the model's verified output order."""
    raw_boxes = outputs[0].squeeze()
    raw_probs = sigmoid(outputs[1])
    masks = outputs[2].squeeze() if len(outputs) == 3 else None

    scores = np.max(raw_probs, axis=2).squeeze()
    labels = np.argmax(raw_probs, axis=2).squeeze()

    sorted_idx = np.argsort(scores)[::-1][:max_boxes]
    confidence_mask = scores[sorted_idx] > confidence_threshold
    final_idx = sorted_idx[confidence_mask]

    return {
        "confidence": scores[final_idx],
        "labels": labels[final_idx],
        "xyxy": box_cxcywh_to_xyxyn(raw_boxes[final_idx]),
        "masks": masks[final_idx] if masks is not None else None,
    }


def resize_mask_batch(masks, target_dims):
    target_w, target_h = target_dims
    transposed = np.transpose(masks, (1, 2, 0))
    resized = cv2.resize(transposed, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    if resized.ndim == 2:
        return resized[np.newaxis, ...]
    return np.transpose(resized, (2, 0, 1))


def scale_bbox_and_masks(results, target_dims):
    """Exact reimplementation of TritonObjectDetector._scale_bbox_and_masks."""
    target_w, target_h = target_dims
    results["xyxy"][..., [0, 2]] *= target_w
    results["xyxy"][..., [1, 3]] *= target_h
    if results["masks"] is not None and len(results["masks"]) > 0:
        rescaled = resize_mask_batch(results["masks"], target_dims)
        results["masks"] = (rescaled > 0).astype(bool)
    return results


CLASS_COLORS = [   # BGR, cycled per class id
    (0, 255, 255), (0, 155, 255), (255, 102, 255), (255, 153, 51), (178, 102, 255),
    (128, 128, 255), (255, 102, 178), (255, 153, 153), (255, 255, 102), (153, 255, 51),
]


def draw_detections(frame, results, class_names):
    annotated = frame.copy()
    for box, conf, label in zip(results["xyxy"], results["confidence"], results["labels"]):
        x1, y1, x2, y2 = [int(v) for v in box]
        name = class_names.get(int(label) + 1, "unknown")   # CSV ids are 1-indexed
        color = CLASS_COLORS[int(label) % len(CLASS_COLORS)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        text = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return annotated


def main():
    video_path = os.path.abspath(parse_args().video)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.isfile(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"ONNX model not found: {ONNX_MODEL_PATH}")

    print(f"Using video: {video_path}")
    print(f"Model: {ONNX_MODEL_PATH}")
    print("This is Google's real CircularNet model - 50 classes (labels50.csv), Google's own "
          "material/form taxonomy, unrelated to this project's 6-class scheme.\n")

    class_names = load_class_names()

    # CPU-only, deliberately: CUDAExecutionProvider needs a separately-installed cuDNN DLL
    # matching onnxruntime-gpu's expected version (cudnn64_9.dll) that isn't present on this
    # machine - PyTorch's own bundled cuDNN doesn't cover onnxruntime's separate requirement.
    # Confirmed via a real error (NOT_IMPLEMENTED: cuDNN unavailable) when CUDA was tried first.
    # Fine for a one-off sanity test; install a matching cuDNN separately if GPU speed matters.
    providers = ["CPUExecutionProvider"]
    print(f"onnxruntime providers: {providers}")
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_budget = 1.0 / src_fps
    print(f"Detected source FPS: {src_fps:.2f}  |  Input resolution: {width}x{height}\n")

    print("Warming up...")
    ok, warm_frame = cap.read()
    if ok:
        session.run(None, {"input": preprocess(warm_frame)})
    print("Warm-up done.\n")

    window = f"CircularNet (Google, stock, 50 classes) - {os.path.basename(video_path)}"
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
        outputs = session.run(None, {"input": preprocess(frame)})
        results = reformat_outputs(outputs, CONF_THRESHOLD, MAX_BOXES)
        if results["labels"].size != 0:
            results = scale_bbox_and_masks(results, (width, height))
        infer_ms = (time.time() - t0) * 1000
        infer_times.append(infer_ms)

        annotated = draw_detections(frame, results, class_names)

        now = time.time()
        disp_fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {disp_fps:.1f}  Detections: {len(results['labels'])}", (10, 30),
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

    print("\n" + "=" * 60)
    print("CIRCULARNET (GOOGLE, STOCK, 50 CLASSES) - SANITY TEST REPORT")
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
