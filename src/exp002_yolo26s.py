"""
YOLO26s Training - single-class "glass" detector, 640x640
-----------------------------------------------------------
Reproduces the exact recipe from the laptop-trained run that was brought into this project
as exp002 (originally `new model/train.py`, run folder
`glass_cullet_runs/exp5_yolo26_glass_only-2`, imported into
experiments/exp002_yolo26s + results/exp002 + data/versions/exp002). That import was
a one-time copy of an already-completed run, not produced by this script - this script exists
so a FUTURE retrain (more epochs, a dataset update, hyperparameter tuning) follows this
project's normal conventions (wandb, exp-folder structure, resume support, models/ copy)
instead of a one-off unwired script.

No differential learning rates here (unlike src/exp001_yolov8l.py) - the original recipe used a single
optimizer group via Ultralytics' own optimizer='auto', which this reproduces as-is.

WHY SINGLE-CLASS, NOT PART OF THE MAIN 6-CLASS PIPELINE:
This is a narrower, glass-only detector trained on its own dataset (data/versions/exp002,
2,350 train / 587 val images) - not a drop-in replacement for exp001/exp003, which
all detect all 6 material classes. Per the original recipe's own docstring, an earlier 2-class
"glass vs. other-materials-lumped-together" attempt was abandoned because the "others" bucket
was too broad/inconsistent to learn - single-class "glass vs. background" was more consistent.
"""

from ultralytics import YOLO
import os
import shutil
import wandb

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new training run. RUN_NAME is built from them automatically,
# enforcing the exp{id}_{model}_{dataset-version} convention project-wide. exp002 itself
# (the imported laptop run this script reproduces) is already taken - this targets the next
# free number for an actual from-scratch run through this script. Was originally 7, bumped to
# 10 once exp007 was claimed by the separate AGI-dataset run (src/exp007_yolov8l_agi.py).
EXP_ID = 10
MODEL_NAME = "yolo26s"
DATASET_VERSION = "exp002"   # single-class glass dataset, see data/versions/exp002/README.md
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"    # wandb run name only - keeps full provenance there

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
DATA = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
LAST_PT = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}/weights/last.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

WANDB_PROJECT = "pour-defect-yolov8l"   # same wandb project as every other experiment in this project
WANDB_RUN_NAME = RUN_NAME

# Exact values from the original laptop recipe (new model/train.py) - unchanged.
EPOCHS = 120          # original comment: "100+ epochs already plateaued" in an earlier run
IMGSZ = 640            # source images are native 640x640, no upscaling needed
BATCH = 8
PATIENCE = 30
LR0 = 0.01
SEED = 42


def wandb_log_train_epoch(trainer):
    """Runs after every training epoch: losses + learning rate."""
    loss_dict = trainer.label_loss_items(trainer.tloss, prefix="train")
    lr_dict = {f"lr/{k}": v for k, v in trainer.lr.items()}
    wandb.log({**loss_dict, **lr_dict}, step=trainer.epoch + 1)


def wandb_log_fit_epoch(trainer):
    """Runs after every validation pass: mAP/precision/recall, plus plot images if enabled."""
    wandb.log(trainer.metrics, step=trainer.epoch + 1)
    if trainer.args.plots:
        for img_path in trainer.save_dir.glob("*.png"):
            wandb.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))}, step=trainer.epoch + 1)


def wandb_log_train_end(trainer):
    """Runs once, after training finishes: upload best.pt as a versioned wandb Artifact,
    then split the run folder to match this project's layout - experiments/<run>/ keeps
    only weights/ + config.yaml, everything else moves into results/exp<id>/, and best.pt
    also gets a flat copy in models/."""
    best_path = trainer.best
    if best_path.exists():
        artifact = wandb.Artifact(name=f"{WANDB_RUN_NAME}-best", type="model")
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

        os.makedirs(MODELS_DIR, exist_ok=True)
        models_copy = os.path.join(MODELS_DIR, f"{EXP_FOLDER}_best.pt")
        shutil.copy2(str(best_path), models_copy)
        print(f"Best weights also copied to {models_copy}")
    wandb.finish()

    save_dir = trainer.save_dir
    args_path = save_dir / "args.yaml"
    if args_path.exists():
        args_path.rename(save_dir / "config.yaml")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for item in save_dir.iterdir():
        if item.name in ("weights", "config.yaml", "wandb"):
            continue
        shutil.move(str(item), os.path.join(RESULTS_DIR, item.name))


if __name__ == "__main__":
    if not os.path.isfile(DATA):
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            f"Run data/scripts/import_glass_dataset.py first to generate it."
        )

    resuming = os.path.exists(LAST_PT)

    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,
        config={
            "model": "yolo26s.pt", "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH,
            "patience": PATIENCE, "lr0": LR0, "seed": SEED, "data": DATA,
            "resumed_from": LAST_PT if resuming else None,
        },
    )

    model = YOLO(LAST_PT if resuming else "yolo26s.pt")

    model.add_callback("on_train_epoch_end", wandb_log_train_epoch)
    model.add_callback("on_fit_epoch_end", wandb_log_fit_epoch)
    model.add_callback("on_train_end", wandb_log_train_end)

    if resuming:
        print(f"[RESUME] Continuing from {LAST_PT}")
        model.train(resume=True)
    else:
        model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            device=0,           # force the RTX 5080 (cuda:0); fail loudly instead of silently falling back to CPU
            optimizer="auto",
            lr0=LR0,
            augment=True,
            seed=SEED,
            val=True,
            project=EXPERIMENTS_DIR,
            name=EXP_FOLDER,
            exist_ok=True,
        )
