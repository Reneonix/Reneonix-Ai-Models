"""
Step 2 of the exp005 (RT-DETR) dataset build. Splits data/raw/ (built by
consolidate_raw_dataset.py) back into a train/val dataset at data/versions/exp005, for
src/exp005_rtdetr.py.

RT-DETR is trained here via Ultralytics (`from ultralytics import RTDETR`) - the SAME
dataset format as every other Ultralytics script in this project (images/{train,val},
labels/{train,val}, data.yaml with train:/val: keys), unlike RF-DETR (Roboflow's separate
package, removed from this project) which needed a different folder layout entirely.

The train/val SPLIT ITSELF is not re-randomized - it's read directly off
data/versions/exp001_exp003_exp006's own images/train vs images/val folders (used here purely as a
manifest of which filename belongs to which split, nothing is copied FROM there - all actual
file copies come from data/raw/). This keeps exp005 a fair like-for-like comparison against
exp001/exp003 (same train/val split as both), the same principle exp003's own dataset choice
already followed.
"""

import os
import shutil
import glob

RAW_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/raw"
SPLIT_MANIFEST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp001_exp003_exp006"
DST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp005"

CLASS_NAMES = {0: "aluminium", 1: "plastic", 2: "metal", 3: "stone", 4: "ceramic", 5: "glass"}


def build_split(split):
    manifest_dir = f"{SPLIT_MANIFEST_ROOT}/images/{split}"
    dst_img_dir = f"{DST_ROOT}/images/{split}"
    dst_lbl_dir = f"{DST_ROOT}/labels/{split}"
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    n_images, n_labels = 0, 0
    for manifest_path in sorted(glob.glob(f"{manifest_dir}/*.png")):
        fname = os.path.basename(manifest_path)
        stem = os.path.splitext(fname)[0]

        raw_img = f"{RAW_ROOT}/images/{fname}"
        if not os.path.exists(raw_img):
            print(f"WARNING: {fname} listed in exp001_exp003_exp006/{split} manifest but missing from data/raw/images/")
            continue
        shutil.copy2(raw_img, f"{dst_img_dir}/{fname}")
        n_images += 1

        raw_lbl = f"{RAW_ROOT}/labels/{stem}.txt"
        if os.path.exists(raw_lbl):
            shutil.copy2(raw_lbl, f"{dst_lbl_dir}/{stem}.txt")
            n_labels += 1

    return n_images, n_labels


def write_data_yaml():
    lines = [
        f"path: {DST_ROOT}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for idx in sorted(CLASS_NAMES):
        lines.append(f"  {idx}: {CLASS_NAMES[idx]}")
    with open(f"{DST_ROOT}/data.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(f"{RAW_ROOT} not found - run consolidate_raw_dataset.py first.")

    print("Building exp005 (RT-DETR) dataset from data/raw/, using exp001_exp003_exp006's own train/val split...\n")
    summary = {}
    for split in ("train", "val"):
        n_images, n_labels = build_split(split)
        summary[split] = (n_images, n_labels)
        print(f"{split}: {n_images} images, {n_labels} labels")

    write_data_yaml()

    for split, (n_images, n_labels) in summary.items():
        if n_images != n_labels:
            print(f"WARNING: {split} has {n_images} images but only {n_labels} labels.")

    print(f"\nDone. Dataset written to: {DST_ROOT}")
