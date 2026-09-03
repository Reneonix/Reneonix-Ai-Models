"""
exp003 - YOLO26L + P2, 640x640

Purpose: test whether an additional high-resolution P2 detection level
improves small-object detection versus the plain P3/P4/P5 yolov8l baseline
(exp001).

Architecture (yolo26l-p2.yaml, 'l' scale - see note below):
    Input: 640x640
    P2 -> 160x160  (extra head vs. a standard P3-P5 model)
    P3 -> 80x80
    P4 -> 40x40
    P5 -> 20x20
         |
        Neck
         |
        Head (Detect on P2, P3, P4, P5)

NOTE on model name: "yolo26-p2.yaml" alone (no scale letter) builds the
smallest default scale ('n'), which would then mismatch when loading
yolo26l.pt (large) pretrained weights into it - most tensors would silently
fail to transfer. "yolo26l-p2.yaml" is the correct name for the 'l' scale
(verified: 25,815,520 params, matching Ultralytics' own yaml comment for
that scale) - that's what's used below.
"""

from ultralytics import YOLO
import wandb
import os
import shutil

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new training run. RUN_NAME is built from them automatically,
# enforcing the exp{id}_{model}_{dataset-version} convention project-wide. This is exp003's
# own actual completed run - EXP_ID stays 3, matching experiments/exp003_yolo26l_p2/ on disk.
EXP_ID = 3
MODEL_NAME = "yolo26l_p2"     # yolo26, 'l' scale, P2 head variant
DATASET_VERSION = "exp001_exp003_exp006"   # same dataset as exp001, for a like-for-like architecture
                                      # comparison - named for every experiment sharing it
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"    # wandb run name only - keeps full provenance there

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
MODEL_YAML = "yolo26l-p2.yaml"     # architecture definition (ships with Ultralytics)
PRETRAINED_WEIGHTS = "yolo26l.pt"    # transferred into the P2 architecture where shapes match;
                                      # auto-downloaded by Ultralytics on first run if not cached
DATA = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
LAST_PT = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}/weights/last.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

WANDB_PROJECT = "pour-defect-yolov8l"
WANDB_RUN_NAME = RUN_NAME

EPOCHS = 100
IMGSZ = 640
BATCH = 16
DEVICE = 0   # RTX 5080 (cuda:0); fail loudly instead of silently falling back to CPU


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

        # Standing project convention: every finished run's best weights also get a flat copy
        # in models/, named the same as its experiments/ folder (EXP_FOLDER, no dataset suffix).
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
        raise FileNotFoundError(f"Dataset yaml not found: {DATA}")

    resuming = os.path.exists(LAST_PT)

    # Without this, wandb's local sync cache lands in whatever directory you happen to run
    # the script from (e.g. src/wandb/) instead of staying with this experiment's own files.
    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,   # creates {wandb_dir}/wandb/run-.../ - kept alongside this experiment's weights/config
        config={
            "model_yaml": MODEL_YAML,
            "pretrained_weights": PRETRAINED_WEIGHTS,
            "data": DATA,
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "resumed_from": LAST_PT if resuming else None,
        },
    )

    if resuming:
        # RESUME: continue from the checkpoint left by an interrupted run. No custom
        # optimizer is installed in this script (plain Ultralytics default throughout),
        # so a plain resume=True works cleanly - no param-group workaround needed.
        model = YOLO(LAST_PT)
    else:
        # Build the YOLO26 P2 architecture, then transfer compatible pretrained weights
        # from yolo26l.pt (matching-shaped tensors only; the extra P2 head starts fresh).
        model = YOLO(MODEL_YAML).load(PRETRAINED_WEIGHTS)

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
            device=DEVICE,

            project=EXPERIMENTS_DIR,
            name=EXP_FOLDER,
            exist_ok=True,   # wandb's dir= already created this exact folder above (for its local
                              # sync cache) - without this, Ultralytics sees it "already exists" and
                              # auto-increments to a different name instead of reusing it
        )
