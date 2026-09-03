"""Builds data/versions/exp007_AGI from data/raw/AGI/ - 5 pre-split-by-class folders
(Ceramic, Ferrous, glass, plastic, Stone), each already YOLO-labeled with one consistent
class index (Ceramic=3, Ferrous=0, glass=4, plastic=1, Stone=2 - verified against a sample
of every folder). 80/20 train/val split, stratified per class, seeded for reproducibility.

Immutable once used by exp007 - per this project's dataset convention, re-run only into a new
data/versions/ folder if the source data changes.
"""
import os
import random
import shutil

ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
SRC = f"{ROOT}/data/raw/AGI"
DST = f"{ROOT}/data/versions/exp007_AGI"
SEED = 42
VAL_FRACTION = 0.20

# folder name -> (class index already baked into its label files, output class name)
CLASSES = {
    "Ferrous": (0, "ferrous"),
    "plastic": (1, "plastic"),
    "Stone": (2, "stone"),
    "Ceramic": (3, "ceramic"),
    "glass": (4, "glass"),
}

random.seed(SEED)

for split in ("train", "val"):
    os.makedirs(f"{DST}/images/{split}", exist_ok=True)
    os.makedirs(f"{DST}/labels/{split}", exist_ok=True)

counts = {}
for folder, (cls_id, cls_name) in CLASSES.items():
    img_dir = f"{SRC}/{folder}/images"
    lbl_dir = f"{SRC}/{folder}/labels"
    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(img_dir))
    random.shuffle(stems)

    n_val = int(round(len(stems) * VAL_FRACTION))
    val_stems = set(stems[:n_val])

    for stem in stems:
        split = "val" if stem in val_stems else "train"
        # image extension varies file-to-file - find it
        img_src = next(
            (f"{img_dir}/{stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".bmp")
             if os.path.exists(f"{img_dir}/{stem}{ext}")), None
        )
        lbl_src = f"{lbl_dir}/{stem}.txt"
        if img_src is None or not os.path.exists(lbl_src):
            continue
        ext = os.path.splitext(img_src)[1]
        # prefix with class name to avoid filename collisions across the 5 source folders
        out_stem = f"{cls_name}_{stem}"
        shutil.copy2(img_src, f"{DST}/images/{split}/{out_stem}{ext}")
        shutil.copy2(lbl_src, f"{DST}/labels/{split}/{out_stem}.txt")

    counts[cls_name] = {"total": len(stems), "val": len(val_stems), "train": len(stems) - len(val_stems)}

# data.yaml - class indices unchanged from the source label files (0..4)
names_block = "\n".join(f"  {cls_id}: {cls_name}" for _, (cls_id, cls_name) in
                         sorted(CLASSES.items(), key=lambda kv: kv[1][0]))
with open(f"{DST}/data.yaml", "w") as f:
    f.write(f"""path: {DST}
train: images/train
val: images/val

nc: 5
names:
{names_block}
""")

train_total = sum(c["train"] for c in counts.values())
val_total = sum(c["val"] for c in counts.values())
print("Per-class counts:")
for cls_name, c in counts.items():
    print(f"  {cls_name}: train={c['train']} val={c['val']} total={c['total']}")
print(f"\nTOTAL: train={train_total} val={val_total}")
print(f"Written to {DST}")
