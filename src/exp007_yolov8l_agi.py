"""
YOLOv8-L Training on the AGI dataset - Single Run with Differential Learning Rates
------------------------------------------------------------------------------------
Same methodology as exp001_yolov8l.py (see that file's own header for the full explanation):
  - First FREEZE_LAYERS layers      -> FROZEN (no gradient update at all)
  - Remaining unfrozen backbone+neck -> LOW learning rate
  - Detect head (layer 22)           -> HIGH learning rate
  - Cosine annealing LR schedule applied to both groups
  - Everything happens in ONE single training run (model.train() called once)

Trains on data/versions/exp007_AGI - a 5-class dataset (ferrous, plastic, stone, ceramic,
glass; no separate aluminium class) built from data/raw/AGI/ (5 pre-split-by-class folders,
80/20 train/val split - see data/scripts/build_exp007_agi_dataset.py). This is a DIFFERENT
class taxonomy from the project's main 6-class dataset (exp001/exp003/exp005/exp006) - do not
directly compare mAP/precision/recall numbers between exp007 and those runs.
"""

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import torch
import math
import os
import shutil
import wandb

# RESUME COMPATIBILITY: see exp001_yolov8l.py's own comment on this - identical reasoning.
_orig_load_checkpoint_state = BaseTrainer._load_checkpoint_state


def _load_checkpoint_state_skip_optimizer(self, ckpt):
    patched_ckpt = dict(ckpt)
    patched_ckpt["optimizer"] = None
    return _orig_load_checkpoint_state(self, patched_ckpt)


BaseTrainer._load_checkpoint_state = _load_checkpoint_state_skip_optimizer

# ---------------- EXPERIMENT IDENTITY ----------------
EXP_ID = 7
MODEL_NAME = "yolov8l_AGI"        # AGI kept in the name deliberately - every folder this
                                    # experiment touches (experiments/, results/, models/) must
                                    # say AGI, per this experiment's own explicit naming rule
                                    # (an intentional exception to the usual no-dataset-suffix
                                    # convention - see README.md's naming-convention section).
DATASET_VERSION = "exp007_AGI"     # data/versions/exp007_AGI - 5-class AGI dataset
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"     # -> exp007_yolov8l_AGI
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
DATA = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}_AGI"     # AGI in the results folder name too
LAST_PT = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}/weights/last.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

WANDB_PROJECT = "pour-defect-yolov8l"
WANDB_RUN_NAME = RUN_NAME

# Same differential-LR setup as exp001 (identical architecture, yolov8l.pt) - see that
# script's own comments for the full layer-map/rationale.
FREEZE_LAYERS = 5
HEAD_LAYER_IDX = 22

BACKBONE_LR = 0.0005      # restored from exp001's original batch=16 values (0.0005/0.01) - those
HEAD_LR = 0.01             # were scaled DOWN to 0.00035/0.007 specifically for exp001's batch=8;
                            # since exp007 now trains at batch=16 too, the un-scaled values are the
                            # ones that actually match this batch size (linear LR-scaling rule).
FINAL_LR_FRACTION = 0.01

EPOCHS = 100
IMGSZ = 640
BATCH = 16                # bumped from exp001's 8 - that value predates this project's move to
                           # the RTX 5080 16GB and was overly conservative (its own comment says
                           # "avoid OOM on an 8GB GPU"). 16 is proven safe on this exact card by
                           # exp003, which trained a HEAVIER model (yolo26l + a P2 head) at
                           # batch=16 with no OOM - plain yolov8l has more headroom than that.
PATIENCE = 20


# ---------------- CALLBACK: differential LR setup ----------------
def setup_differential_lr(trainer):
    model = trainer.model

    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        layer_idx = int(name.split(".")[1])

        if layer_idx >= HEAD_LAYER_IDX:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": BACKBONE_LR, "initial_lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR, "initial_lr": HEAD_LR},
        ],
        momentum=0.937,
        nesterov=True,
    )

    trainer.optimizer = optimizer

    def cosine_lambda(epoch):
        progress = epoch / max(1, trainer.epochs)
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2
        return FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * cosine_factor

    start_epoch = getattr(trainer, "start_epoch", 0)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_lambda, last_epoch=start_epoch - 1
    )

    print("[Differential LR ACTIVE - AGI dataset]")
    print(f"  Backbone/neck params: {len(backbone_params)} tensors  -> LR = {BACKBONE_LR}")
    print(f"  Head params:          {len(head_params)} tensors  -> LR = {HEAD_LR}")
    print(f"  Frozen layers:        0 to {FREEZE_LAYERS - 1}")

    wandb.config.update({
        "freeze_layers": FREEZE_LAYERS,
        "head_layer_idx": HEAD_LAYER_IDX,
        "backbone_lr": BACKBONE_LR,
        "head_lr": HEAD_LR,
        "final_lr_fraction": FINAL_LR_FRACTION,
        "backbone_trainable_tensors": len(backbone_params),
        "head_trainable_tensors": len(head_params),
    })


# ---------------- CALLBACK: per-epoch wandb logging ----------------
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
    """Same split as exp001: experiments/<run>/ keeps only weights/+config.yaml+wandb/,
    everything else moves into results/exp007_AGI/."""
    best_path = trainer.best
    if best_path.exists():
        artifact = wandb.Artifact(name=f"{WANDB_RUN_NAME}-best", type="model")
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

        os.makedirs(MODELS_DIR, exist_ok=True)
        models_copy = os.path.join(MODELS_DIR, f"{EXP_FOLDER}_best.pt")   # -> exp007_yolov8l_AGI_best.pt
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


# ---------------- MAIN: single training run ----------------
if __name__ == "__main__":
    resuming = os.path.exists(LAST_PT)

    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,
        config={
            "model": "yolov8l.pt",
            "epochs": EPOCHS,
            "imgsz": IMGSZ,
            "batch": BATCH,
            "patience": PATIENCE,
            "data": DATA,
            "resumed_from": LAST_PT if resuming else None,
        },
    )

    model = YOLO(LAST_PT if resuming else "yolov8l.pt")

    model.add_callback("on_train_start", setup_differential_lr)
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
            freeze=FREEZE_LAYERS,
            patience=PATIENCE,
            device=0,               # RTX 5080 (cuda:0); fail loudly instead of silently falling back to CPU

            warmup_epochs=0,
            cos_lr=False,

            project=EXPERIMENTS_DIR,
            name=EXP_FOLDER,
            exist_ok=True,
        )
