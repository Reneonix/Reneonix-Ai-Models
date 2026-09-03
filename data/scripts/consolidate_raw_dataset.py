"""
Step 1 of the exp005 (RT-DETR) dataset build. Consolidates the 6-class detection dataset's
train+val images and labels back into ONE flat, unsplit copy at data/raw/ - fulfilling that
folder's long-documented-but-never-filled purpose ("Original untouched captures, before
splitting"). Source: data/versions/exp001_exp003_exp006 (the only 6-class detection dataset in this
project); its own train/val split is preserved unmodified on disk, this only ADDS a merged copy.

    data/versions/exp001_exp003_exp006/          data/raw/
    images/train/*.png  ---+               images/*.png  (train+val merged, flat)
    images/val/*.png    ---+
    labels/train/*.txt  ---+               labels/*.txt  (train+val merged, flat)
    labels/val/*.txt    ---+
    data.yaml (nc/names only, copied)      data.yaml (nc/names only - no train/val split here,
                                             this is the pre-split source)

Run data/scripts/build_exp005_rtdetr_dataset.py next to re-split this into
data/versions/exp005 (Ultralytics/RT-DETR format) for training.
"""

import os
import shutil
import glob

SRC_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp001_exp003_exp006"
DST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/raw"

CLASS_NAMES = {0: "aluminium", 1: "plastic", 2: "metal", 3: "stone", 4: "ceramic", 5: "glass"}


def consolidate():
    dst_img_dir = f"{DST_ROOT}/images"
    dst_lbl_dir = f"{DST_ROOT}/labels"
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    n_images, n_labels = 0, 0
    for split in ("train", "val"):
        img_dir = f"{SRC_ROOT}/images/{split}"
        lbl_dir = f"{SRC_ROOT}/labels/{split}"
        for img_path in sorted(glob.glob(f"{img_dir}/*.png")):
            fname = os.path.basename(img_path)
            stem = os.path.splitext(fname)[0]
            shutil.copy2(img_path, f"{dst_img_dir}/{fname}")
            n_images += 1

            lbl_path = f"{lbl_dir}/{stem}.txt"
            if os.path.exists(lbl_path):
                shutil.copy2(lbl_path, f"{dst_lbl_dir}/{stem}.txt")
                n_labels += 1

    return n_images, n_labels


def write_data_yaml():
    lines = ["names:"]
    for idx in sorted(CLASS_NAMES):
        lines.append(f"  {idx}: {CLASS_NAMES[idx]}")
    lines.append(f"nc: {len(CLASS_NAMES)}")
    with open(f"{DST_ROOT}/data.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("Consolidating exp001_exp003_exp006's train+val into a flat data/raw/ copy...\n")
    n_images, n_labels = consolidate()
    write_data_yaml()
    print(f"images: {n_images}  labels: {n_labels}")
    if n_images != n_labels:
        print(f"WARNING: {n_images - n_labels} image(s) have no matching label file.")
    print(f"\nDone. Raw dataset written to: {DST_ROOT}")
