"""
One-time import of the 6-class "materials" detection dataset from the laptop-trained
YOLO26s run in KAVIYA/dataset/ (base-training stage of a two-stage pipeline - see
src/exp006_yolo26s_finetuned.py) into this project's versioned dataset layout, as
data/versions/exp006.

Same 6 classes as this project's main dataset (aluminium, plastic, metal, stone, ceramic,
glass) but a DIFFERENT, smaller image set (1,631 train / 408 val) - this is NOT derived from
exp001_exp003_exp006. The pipeline's second (fine-tuning) stage separately reuses exp001_exp003_exp006
directly (verified byte-identical to KAVIYA/fine_tune_dataset/, no import needed for that part).

Only re-arranges directory SHAPE (KAVIYA/dataset/ already uses this project's own
images/{train,val}, labels/{train,val} convention, so this is mostly a straight copy):
no re-labeling, no re-encoding - same images (mixed .jpg/.png), same YOLO-format .txt boxes.
"""

import os
import shutil
import glob

SRC_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/KAVIYA/dataset"
DST_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification/data/versions/exp006"

CLASS_NAMES = {0: "aluminium", 1: "plastic", 2: "metal", 3: "stone", 4: "ceramic", 5: "glass"}


def copy_split(split):
    src_img_dir = f"{SRC_ROOT}/images/{split}"
    src_lbl_dir = f"{SRC_ROOT}/labels/{split}"
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
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ]
    for idx in sorted(CLASS_NAMES):
        lines.append(f"  {idx}: {CLASS_NAMES[idx]}")
    with open(f"{DST_ROOT}/data.yaml", "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    print("Importing KAVIYA/dataset (base-training stage) into data/versions/exp006...\n")
    summary = {}
    for split in ("train", "val"):
        n_images, n_labels = copy_split(split)
        summary[split] = (n_images, n_labels)
        print(f"{split}: {n_images} images, {n_labels} labels")

    write_data_yaml()

    for split, (n_images, n_labels) in summary.items():
        if n_images != n_labels:
            print(f"WARNING: {split} has {n_images} images but only {n_labels} labels.")

    print(f"\nDone. Dataset written to: {DST_ROOT}")
