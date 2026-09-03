"""
exp005 - RT-DETR-L, 640x640

Purpose: a third detection architecture on the exact same underlying data as exp001
(YOLOv8L) and exp003 (YOLO26L+P2), for comparison. RT-DETR ("DETRs Beat YOLOs on Real-Time
Object Detection", Baidu, 2023) is a transformer-family detector - no anchor boxes, no NMS
post-processing (unlike the CNN-based YOLO variants) - but trained here via **Ultralytics'
own `RTDETR` class**, the same framework/conventions as every other script in this project.
Confirmed real: Ultralytics ships RT-DETR as a first-class citizen (same `Model` base class
as `YOLO`, same `.train()`/callback API) - verified against the installed ultralytics package
before writing this script, not assumed.

NOT to be confused with RF-DETR (Roboflow's unrelated, separate-package architecture with its
own training API - tried as an earlier exp005 attempt and removed from this project after
repeated training crashes; see git history / prior README versions).

Dataset: data/versions/exp005 - same images and train/val split as exp001_exp003_exp006 (built via
data/scripts/consolidate_raw_dataset.py + build_exp005_rtdetr_dataset.py), kept under its own
experiment-named folder per this project's dataset-naming convention.
"""

from ultralytics import RTDETR
import wandb
import os
import shutil

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new training run. RUN_NAME is built from them automatically,
# enforcing the exp{id}_{model}_{dataset-version} convention project-wide. exp001
# (yolov8l), exp002 (yolo26s, glass-only), exp003 (yolo26l+p2), exp004 (resnet50) are all
# taken; exp005 was RF-DETR/"CircleNet" (removed after repeated crashes) and is reclaimed
# here for RT-DETR.
EXP_ID = 5
MODEL_NAME = "rtdetr_l"        # RT-DETR, 'l' scale (rtdetr-l.pt)
DATASET_VERSION = "exp005"     # same images/split as exp001_exp003_exp006, own experiment-named copy -
                                 # see data/versions/exp005/README.md
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"    # wandb run name only - keeps full provenance there

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
PRETRAINED_WEIGHTS = "rtdetr-l.pt"   # Ultralytics' own COCO-pretrained checkpoint - auto-downloaded
                                       # on first run if not cached; this IS the transfer-learning
                                       # starting point (fine-tuning onto our 6 classes)
DATA = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
LAST_PT = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}/weights/last.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

WANDB_PROJECT = "pour-defect-yolov8l"   # same wandb project as every other experiment in this project
WANDB_RUN_NAME = RUN_NAME

EPOCHS = 100    # matches exp001/exp003, for a fair comparison
IMGSZ = 640
BATCH = 16      # same batch size exp003 (a comparably-sized model) already trains at cleanly
                 # on this exact RTX 5080 16GB - lower this first if you hit CUDA OOM
DEVICE = 0      # RTX 5080 (cuda:0); fail loudly instead of silently falling back to CPU


def wandb_log_train_epoch(trainer):
    loss_dict = trainer.label_loss_items(trainer.tloss, prefix="train")
    lr_dict = {f"lr/{k}": v for k, v in trainer.lr.items()}
    wandb.log({**loss_dict, **lr_dict}, step=trainer.epoch + 1)


def wandb_log_fit_epoch(trainer):
    wandb.log(trainer.metrics, step=trainer.epoch + 1)
    if trainer.args.plots:
        for img_path in trainer.save_dir.glob("*.png"):
            wandb.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))}, step=trainer.epoch + 1)


def wandb_log_train_end(trainer):
    """Uploads best.pt to wandb, then splits the run folder to match this project's
    layout - experiments/<run>/ keeps only weights/ + config.yaml, everything else
    (plots, results.csv, batch preview jpgs) moves into results/exp<id>/."""
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
        # "wandb" must stay put: it's wandb's own local sync folder, and wandb.finish() above
        # doesn't synchronously release its background service's file handles, so trying to
        # move/delete it right here can crash with a Windows file-in-use error.
        if item.name in ("weights", "config.yaml", "wandb"):
            continue
        shutil.move(str(item), os.path.join(RESULTS_DIR, item.name))


if __name__ == "__main__":
    if not os.path.isfile(DATA):
        raise FileNotFoundError(
            f"Dataset yaml not found: {DATA}\n"
            f"Run data/scripts/consolidate_raw_dataset.py then "
            f"data/scripts/build_exp005_rtdetr_dataset.py first."
        )

    resuming = os.path.exists(LAST_PT)

    # Without this, wandb's local sync cache lands in whatever directory you happen to run
    # the script from (e.g. src/wandb/) instead of staying with this experiment's own files.
    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,
        config={
            "model": PRETRAINED_WEIGHTS,
            "data": DATA,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "resumed_from": LAST_PT if resuming else None,
        },
    )

    # RESUME: if the crash left a checkpoint, load it (weights + optimizer/epoch state)
    # instead of starting over from COCO-pretrained weights.
    model = RTDETR(LAST_PT if resuming else PRETRAINED_WEIGHTS)

    model.add_callback("on_train_epoch_end", wandb_log_train_epoch)
    model.add_callback("on_fit_epoch_end", wandb_log_fit_epoch)
    model.add_callback("on_train_end", wandb_log_train_end)

    if resuming:
        # resume=True pulls data/epochs/imgsz/batch/device/project/name etc. straight from
        # the args.yaml saved next to the checkpoint - passing them again here would fight
        # Ultralytics' own resume validation.
        print(f"[RESUME] Continuing from {LAST_PT}")
        model.train(resume=True)
    else:
        model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,

            project=EXPERIMENTS_DIR,
            name=EXP_FOLDER,
            exist_ok=True,   # wandb's dir= already created this exact folder above (for its local
                              # sync cache) - without this, Ultralytics sees it "already exists" and
                              # auto-increments to a different name instead of reusing it
        )
