"""
One-time import of the single-class "glass" detection dataset from the laptop-trained
YOLO26s run (originally in `new model/dataset/`) into this project's versioned dataset
layout, as data/versions/exp002.

This is NOT derived from exp001_exp003_exp006 - it's a completely separate dataset (different
images, single-class "glass" only vs. the main 6-class set) that happened to arrive as its
own standalone folder. This script only re-arranges directory SHAPE to match this project's
convention (the source uses the Roboflow-style split-then-type layout):

    new model/dataset/ (split-then-type,           data/versions/exp002/ (type-then-split,
    Roboflow convention):                            this project's Ultralytics convention):
    train/images/*.jpg,png                           images/train/*.jpg,png
    train/labels/*.txt                               labels/train/*.txt
    val/images/*.jpg,png                             images/val/*.jpg,png
    val/labels/*.txt                                 labels/val/*.txt

No re-labeling, no re-encoding - same images, same YOLO-format .txt boxes, copied as-is.
"""

import os
import shutil
import glob

SRC_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/new model/dataset"
DST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp002"

SPLIT_MAP = {"train": "train", "val": "val"}   # source's own split names, unchanged

CLASS_NAMES = {0: "glass"}


def convert_split(split):
    src_img_dir = f"{SRC_ROOT}/{split}/images"
    src_lbl_dir = f"{SRC_ROOT}/{split}/labels"
    dst_img_dir = f"{DST_ROOT}/images/{split}"
    dst_lbl_dir = f"{DST_ROOT}/labels/{split}"
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    image_paths = sorted(
        glob.glob(f"{src_img_dir}/*.jpg") + glob.glob(f"{src_img_dir}/*.jpeg") + glob.glob(f"{src_img_dir}/*.png")
    )
    n_images, n_labels = 0, 0
    for img_path in image_paths:
        fname = os.path.basename(img_path)
        stem = os.path.splitext(fname)[0]
        shutil.copy2(img_path, f"{dst_img_dir}/{fname}")
        n_images += 1

        lbl_path = f"{src_lbl_dir}/{stem}.txt"
        if os.path.exists(lbl_path):
            shutil.copy2(lbl_path, f"{dst_lbl_dir}/{stem}.txt")
            n_labels += 1

    return n_images, n_labels


def write_data_yaml():
    lines = [
        f"path: {DST_ROOT}",
        "train: images/train",
        "val: images/val",
        "",
        "nc: 1",
        "names:",
    ]
    for idx in sorted(CLASS_NAMES):
        lines.append(f"  {idx}: {CLASS_NAMES[idx]}")
    with open(f"{DST_ROOT}/data.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("Importing glass-only dataset from 'new model/dataset/' into data/versions/exp002...\n")
    summary = {}
    for split in SPLIT_MAP:
        n_images, n_labels = convert_split(split)
        summary[split] = (n_images, n_labels)
        print(f"{split}: {n_images} images, {n_labels} labels")

    write_data_yaml()

    for split, (n_images, n_labels) in summary.items():
        if n_images != n_labels:
            print(f"WARNING: {split} has {n_images} images but only {n_labels} labels - "
                  f"{n_images - n_labels} image(s) have no matching .txt label file.")

    print(f"\nDone. Dataset written to: {DST_ROOT}")
