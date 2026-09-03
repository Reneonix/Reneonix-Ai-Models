"""
Builds a classification dataset (data/versions/exp008_AGI) by cropping every labeled bounding
box out of data/versions/exp007_AGI's images - one crop per labeled object, saved into
class-named subfolders (torchvision ImageFolder layout), for training the AGI-dataset ResNet
classifier (src/exp008_resnet50_agi.py). Same cropping methodology as
data/scripts/build_resnet_dataset.py (exp004's own dataset builder) - 10% padding, boxes under
10px skipped - just pointed at the AGI dataset/taxonomy instead.

No new labeling needed: exp007_AGI's existing YOLO-format boxes ARE the ground truth for each
crop - this just re-slices data that already exists. "AGI" kept in the dataset folder name
deliberately, per this experiment's own naming exception (see README.md).
"""

import cv2
import os
import glob

SRC_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp007_AGI"
DST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp008_AGI"

# Same 5-class taxonomy/indices as exp007_AGI's own data.yaml - see that dataset's README.
CLASS_NAMES = {0: "ferrous", 1: "plastic", 2: "stone", 3: "ceramic", 4: "glass"}

PADDING = 0.10       # extra context around each box, as a fraction of box width/height -
                      # gives the classifier a little surrounding texture, not just a bare crop
MIN_CROP_SIZE = 10    # skip degenerate/too-small boxes (pixels)

# FRAME_STRIDE: only crop every Nth source frame. AGI's source frames are consecutive video
# frames (video1_frame_00001, _00002, ...) - adjacent frames show the same physical object
# barely moved, so cropping every single frame produced 2.29M/569K near-duplicate crops (8x
# denser than exp004's dataset) with very little real added signal, just far more disk I/O
# (millions of small JPEGs) than the GPU compute can keep up with. Stride 5 cuts frame count
# (and therefore crop count and disk reads) by ~5x while still covering the same objects across
# enough distinct moments to stay diverse.
FRAME_STRIDE = 5


def crop_split(split):
    img_dir = f"{SRC_ROOT}/images/{split}"
    label_dir = f"{SRC_ROOT}/labels/{split}"
    counts = {name: 0 for name in CLASS_NAMES.values()}

    for cls_name in CLASS_NAMES.values():
        os.makedirs(f"{DST_ROOT}/{split}/{cls_name}", exist_ok=True)

    image_paths = sorted(glob.glob(f"{img_dir}/*.png"))[::FRAME_STRIDE]
    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = f"{label_dir}/{stem}.txt"
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            lines = [line.split() for line in f if line.strip()]

        for i, parts in enumerate(lines):
            cls_id = int(parts[0])
            cls_name = CLASS_NAMES.get(cls_id)
            if cls_name is None:
                continue

            cx, cy, bw, bh = map(float, parts[1:5])
            bw_pad = bw * (1 + PADDING)
            bh_pad = bh * (1 + PADDING)

            x1 = max(0, int((cx - bw_pad / 2) * w))
            y1 = max(0, int((cy - bh_pad / 2) * h))
            x2 = min(w, int((cx + bw_pad / 2) * w))
            y2 = min(h, int((cy + bh_pad / 2) * h))

            if x2 - x1 < MIN_CROP_SIZE or y2 - y1 < MIN_CROP_SIZE:
                continue

            crop = img[y1:y2, x1:x2]
            out_path = f"{DST_ROOT}/{split}/{cls_name}/{stem}_{i}.jpg"
            cv2.imwrite(out_path, crop)
            counts[cls_name] += 1

    return counts


if __name__ == "__main__":
    print("Building classification dataset from data/versions/exp007_AGI bounding boxes...\n")
    total_summary = {}
    for split in ("train", "val"):
        counts = crop_split(split)
        total_summary[split] = counts
        print(f"{split}: {counts}  (total: {sum(counts.values())})")

    print("\nDone. Dataset written to:", DST_ROOT)
