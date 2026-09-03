"""
One-off consolidated benchmark for the 'notebook' training report - runs every trained model
(exp001-exp006) against the same testing image, both plain and SAHI (6-tile), with a proper
warmup, and dumps a JSON summary. Does NOT touch Scripts/images/*.py's MODEL SELECTION state.
"""
import json
import time
import statistics as stats

import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.ops import batched_nms
from ultralytics import YOLO, RTDETR

PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
IMG_PATH = f"{PROJECT_ROOT}/results/testing_images/testing.png.png"
DEVICE = 0
TORCH_DEVICE = "cuda:0"
IMGSZ = 640
CONF = 0.5
N_ITERS = 30
N_WARMUP = 5
TILE_COLS, TILE_ROWS, TILE_OVERLAP, NMS_IOU = 3, 2, 0.2, 0.5
RESNET_IMG_SIZE = 224
RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(TORCH_DEVICE)
RESNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(TORCH_DEVICE)

MODELS = [
    ("exp001", "yolo", f"{PROJECT_ROOT}/experiments/exp001_yolov8l/weights/best.pt"),
    ("exp002", "yolo", f"{PROJECT_ROOT}/experiments/exp002_yolo26s/weights/best.pt"),
    ("exp003", "yolo", f"{PROJECT_ROOT}/experiments/exp003_yolo26l_p2/weights/best.pt"),
    ("exp004", "resnet", f"{PROJECT_ROOT}/experiments/exp004_resnet50/weights/best.pt"),
    ("exp005", "rtdetr", f"{PROJECT_ROOT}/experiments/exp005_rtdetr_l/weights/best.pt"),
    ("exp006", "yolo", f"{PROJECT_ROOT}/experiments/exp006_yolo26s_finetuned/weights/best.pt"),
]


def compute_tile_boxes(width, height, cols, rows, overlap_ratio):
    tile_w = min(int(round((width / cols) * (1 + overlap_ratio))), width)
    tile_h = min(int(round((height / rows) * (1 + overlap_ratio))), height)
    x_step = (width - tile_w) / (cols - 1) if cols > 1 else 0
    y_step = (height - tile_h) / (rows - 1) if rows > 1 else 0
    boxes = []
    for r in range(rows):
        y1 = min(int(round(r * y_step)), height - tile_h)
        for c in range(cols):
            x1 = min(int(round(c * x_step)), width - tile_w)
            boxes.append((x1, y1, x1 + tile_w, y1 + tile_h))
    return boxes


def load_resnet(weights_path):
    ckpt = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    class_names = ckpt["class_names"]
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(TORCH_DEVICE).eval()
    return model, class_names


def bench_yolo_plain(model, frame):
    times = []
    for i in range(N_WARMUP + N_ITERS):
        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        if i >= N_WARMUP:
            times.append(results[0].speed["inference"])
    return times


def bench_yolo_sahi(model, frame, tile_boxes):
    tiles = [frame[y1:y2, x1:x2] for (x1, y1, x2, y2) in tile_boxes]
    times = []
    for i in range(N_WARMUP + N_ITERS):
        t0 = time.time()
        model.predict(tiles, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        dt = (time.time() - t0) * 1000
        if i >= N_WARMUP:
            times.append(dt)
    return times


def bench_resnet_whole(model, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized).to(TORCH_DEVICE).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    tensor = (tensor - RESNET_MEAN) / RESNET_STD
    times = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_ITERS):
            t0 = time.time()
            model(tensor)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            if i >= N_WARMUP:
                times.append(dt)
    return times


def bench_resnet_tiled(model, frame, tile_boxes):
    crops = torch.zeros(len(tile_boxes), 3, RESNET_IMG_SIZE, RESNET_IMG_SIZE)
    for i, (x1, y1, x2, y2) in enumerate(tile_boxes):
        tile_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        resized = cv2.resize(tile_rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        crops[i] = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    crops = (crops.to(TORCH_DEVICE) - RESNET_MEAN) / RESNET_STD
    times = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_ITERS):
            t0 = time.time()
            model(crops)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            if i >= N_WARMUP:
                times.append(dt)
    return times


def bench_e2e_yolo(weights_path, mtype, img_path):
    """End-to-end latency: disk read -> model forward pass -> draw annotations, matching what
    a live capture-to-display loop actually pays per frame (excludes disk write)."""
    model = YOLO(weights_path) if mtype == "yolo" else RTDETR(weights_path)
    times = []
    for i in range(N_WARMUP + N_ITERS):
        t0 = time.time()
        frame = cv2.imread(img_path)
        results = model.predict(frame, imgsz=IMGSZ, conf=CONF, device=DEVICE, verbose=False)
        results[0].plot(line_width=2)
        dt = (time.time() - t0) * 1000
        if i >= N_WARMUP:
            times.append(dt)
    del model
    return times


def bench_e2e_resnet(weights_path, img_path):
    model, class_names = load_resnet(weights_path)
    times = []
    with torch.no_grad():
        for i in range(N_WARMUP + N_ITERS):
            t0 = time.time()
            frame = cv2.imread(img_path)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (RESNET_IMG_SIZE, RESNET_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            tensor = torch.from_numpy(resized).to(TORCH_DEVICE).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = (tensor - RESNET_MEAN) / RESNET_STD
            probs = torch.softmax(model(tensor), dim=1)
            conf, pred = probs.max(dim=1)
            annotated = frame.copy()
            cv2.putText(annotated, f"{class_names[int(pred)]} {float(conf):.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000
            if i >= N_WARMUP:
                times.append(dt)
    del model
    return times


def summary(times):
    return {
        "avg_ms": round(stats.mean(times), 2),
        "median_ms": round(stats.median(times), 2),
        "min_ms": round(min(times), 2),
        "max_ms": round(max(times), 2),
        "fps": round(1000 / stats.mean(times), 1),
    }


def main():
    frame = cv2.imread(IMG_PATH)
    h, w = frame.shape[:2]
    tile_boxes = compute_tile_boxes(w, h, TILE_COLS, TILE_ROWS, TILE_OVERLAP)
    print(f"Image: {IMG_PATH} ({w}x{h}), {len(tile_boxes)} tiles for SAHI")

    out = {"image_resolution": f"{w}x{h}", "tiles": len(tile_boxes), "models": {}}

    for exp_id, mtype, weights in MODELS:
        print(f"\n=== {exp_id} ({mtype}) ===")
        try:
            if mtype == "yolo":
                model = YOLO(weights)
                plain = bench_yolo_plain(model, frame)
                sahi = bench_yolo_sahi(model, frame, tile_boxes)
                nparams = sum(p.numel() for p in model.model.parameters())
            elif mtype == "rtdetr":
                model = RTDETR(weights)
                plain = bench_yolo_plain(model, frame)
                sahi = bench_yolo_sahi(model, frame, tile_boxes)
                nparams = sum(p.numel() for p in model.model.parameters())
            elif mtype == "resnet":
                model, class_names = load_resnet(weights)
                plain = bench_resnet_whole(model, frame)
                sahi = bench_resnet_tiled(model, frame, tile_boxes)
                nparams = sum(p.numel() for p in model.parameters())
            else:
                continue

            if mtype == "resnet":
                e2e = bench_e2e_resnet(weights, IMG_PATH)
            else:
                e2e = bench_e2e_yolo(weights, mtype, IMG_PATH)

            out["models"][exp_id] = {
                "params_millions": round(nparams / 1e6, 2),
                "plain": summary(plain),
                "sahi": summary(sahi),
                "e2e_plain": summary(e2e),
            }
            print(f"  plain: {out['models'][exp_id]['plain']}")
            print(f"  sahi:  {out['models'][exp_id]['sahi']}")
            print(f"  e2e:   {out['models'][exp_id]['e2e_plain']}")
            print(f"  params: {out['models'][exp_id]['params_millions']}M")
        except Exception as e:
            print(f"  FAILED: {e}")
            out["models"][exp_id] = {"error": str(e)}

        try:
            del model
        except Exception:
            pass
        torch.cuda.empty_cache()

    out_path = f"{PROJECT_ROOT}/notebook/build/benchmark_all_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
