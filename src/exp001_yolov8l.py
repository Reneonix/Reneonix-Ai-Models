"""
YOLOv8-L Training - Single Run with Differential Learning Rates
----------------------------------------------------------------
- First FREEZE_LAYERS layers      -> FROZEN (no gradient update at all)
- Remaining unfrozen backbone+neck -> LOW learning rate
- Detect head (layer 22)           -> HIGH learning rate
- Cosine annealing LR schedule applied to both groups
- Everything happens in ONE single training run (model.train() called once)

Verified layer map for yolov8l.pt (confirmed earlier by printing model.model):
  0 Conv, 1 Conv, 2 C2f, 3 Conv, 4 C2f, 5 Conv, 6 C2f, 7 Conv, 8 C2f, 9 SPPF   <- backbone (0-9)
  10-21: Upsample/Concat/C2f/Conv                                             <- neck
  22: Detect                                                                  <- head
"""

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer
import torch
import math
import os
import shutil
import wandb   # NEW: pip install wandb --break-system-packages, then `wandb login` once (needs an API key from wandb.ai)

# RESUME COMPATIBILITY: our differential-LR optimizer has 2 param groups (backbone/head),
# but Ultralytics' own default-built optimizer (created earlier in _setup_train, before our
# on_train_start callback runs) has 3. Restoring a 2-group state dict into a 3-group
# optimizer raises "different number of parameter groups" and crashes _setup_train.
# We DON'T want to null out ckpt["optimizer"] on disk - Ultralytics' model.train(resume=True)
# checks `self.ckpt.get("optimizer") is not None` to even decide the checkpoint is
# resumable at all; if it's None there it silently falls back to training on default
# coco8/settings instead of raising. So instead we patch only the internal restore call:
# the checkpoint keeps a real optimizer, but the state dict never actually gets applied to
# Ultralytics' default optimizer (our callback replaces trainer.optimizer immediately after
# anyway, so the saved momentum buffers wouldn't be used even if the restore succeeded).
_orig_load_checkpoint_state = BaseTrainer._load_checkpoint_state


def _load_checkpoint_state_skip_optimizer(self, ckpt):
    patched_ckpt = dict(ckpt)
    patched_ckpt["optimizer"] = None
    return _orig_load_checkpoint_state(self, patched_ckpt)


BaseTrainer._load_checkpoint_state = _load_checkpoint_state_skip_optimizer

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new training run. RUN_NAME is built from them automatically,
# enforcing the exp{id}_{model}_{dataset-version} convention project-wide so any
# experiment folder name alone tells you exactly what model + what dataset made it.
# exp001 (this script's own actual completed run), exp002 (yolo26s, glass-only, brought in
# from a laptop run), exp003 (yolo26l+p2), exp004 (resnet50), exp005 (RT-DETR-L, via
# Ultralytics - a prior RF-DETR/"CircleNet" attempt at exp005 was removed 2026-08-28 after
# repeated unrecoverable training crashes; the number was reclaimed for RT-DETR), exp006
# (yolo26s_finetuned, two-stage, brought in from a laptop run), exp007 (yolov8l on the separate
# AGI dataset, same differential-LR methodology as this script - a deliberate naming exception,
# see README.md) are all taken - exp008 is also already reserved
# (exp006_yolo26s_finetuned.py's own future-rerun target). 9 is the next free number if this
# script is run again.
EXP_ID = 9                    # next free experiment number
MODEL_NAME = "yolov8l"        # change when training a different architecture (yolov9, yolo11, ...)
DATASET_VERSION = "exp001_exp003_exp006"   # which data/versions/ folder this experiment trains on -
                                      # named for every experiment sharing this exact dataset
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix (that
                                                 # provenance lives in the folder's own config.yaml
                                                 # and in RUN_NAME below, not the folder name itself)
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"    # wandb run name only - keeps full provenance there

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
DATA = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
LAST_PT = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}/weights/last.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

# NEW: wandb project/run naming - shows up in your wandb dashboard
WANDB_PROJECT = "pour-defect-yolov8l"
WANDB_RUN_NAME = RUN_NAME

# CHANGED: 10 -> 5.
# freeze=10 locks the ENTIRE backbone, including layers 6 and 8 which encode
# semantic ("COCO-object-like") features. Your domain (conveyor/pour, ~10k
# frames, 6 classes) is visually very different from COCO, so those layers
# need to adapt. freeze=5 locks only the generic low-level filters
# (Conv, Conv, C2f, Conv, C2f = layers 0-4: edges/textures), and lets
# layers 5-21 (including the domain-specific C2f blocks) train.
FREEZE_LAYERS = 5

HEAD_LAYER_IDX = 22       # Detect head layer index (confirmed above, not just assumed)

BACKBONE_LR = 0.00035     # low LR for unfrozen backbone/neck layers (FREEZE_LAYERS to HEAD_LAYER_IDX-1)
                          # scaled down from 0.0005 - batch dropped 16->8 for yolov8l (linear scaling rule)
HEAD_LR = 0.007           # high LR for the Detect head
                          # scaled down from 0.01 - same batch-size reasoning

FINAL_LR_FRACTION = 0.01  # cosine anneal target: LR shrinks down to (start_lr * this fraction)

EPOCHS = 100
IMGSZ = 640
BATCH = 8                 # yolov8l is much bigger than yolov8n - lowered from 16 to avoid CUDA OOM on an 8GB GPU
PATIENCE = 20


# ---------------- CALLBACK: differential LR setup ----------------
def setup_differential_lr(trainer):
    """
    Runs once, right when training starts (after Ultralytics has already
    built its default optimizer and applied `freeze=` inside _setup_train).
    We simply REPLACE trainer.optimizer and trainer.scheduler with our own
    2-group version. This is safe: on_train_start fires after freezing and
    after the default optimizer/scheduler are constructed, but before the
    epoch loop begins, so nothing has trained yet with the wrong optimizer.
    """
    model = trainer.model

    backbone_params = []   # unfrozen backbone + neck (layers FREEZE_LAYERS..21) -> low LR
    head_params = []       # Detect head (layer 22)                              -> high LR

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen layers (0 to FREEZE_LAYERS-1) -> skip, they never train

        # parameter names look like "model.15.conv.weight" -> extract the "15"
        layer_idx = int(name.split(".")[1])

        if layer_idx >= HEAD_LAYER_IDX:
            head_params.append(param)       # Detect head -> high LR group
        else:
            backbone_params.append(param)   # unfrozen backbone/neck -> low LR group

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": BACKBONE_LR, "initial_lr": BACKBONE_LR},
            {"params": head_params, "lr": HEAD_LR, "initial_lr": HEAD_LR},
        ],
        momentum=0.937,
        nesterov=True,
    )

    trainer.optimizer = optimizer

    # cosine annealing: both groups decay smoothly from their own initial_lr
    # down to (initial_lr * FINAL_LR_FRACTION) by the end of training
    def cosine_lambda(epoch):
        progress = epoch / max(1, trainer.epochs)
        cosine_factor = (1 + math.cos(math.pi * progress)) / 2
        return FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * cosine_factor

    # RESUME-SAFE: on a fresh run trainer.start_epoch is 0 -> last_epoch=-1 (unchanged
    # default behavior). On a resumed run trainer.start_epoch is the checkpoint's next
    # epoch (e.g. 91) -> last_epoch=90, so this freshly-built LambdaLR picks the cosine
    # curve back up where it left off instead of restarting the anneal from epoch 0.
    start_epoch = getattr(trainer, "start_epoch", 0)
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=cosine_lambda, last_epoch=start_epoch - 1
    )

    print("[Differential LR ACTIVE]")
    print(f"  Backbone/neck params: {len(backbone_params)} tensors  -> LR = {BACKBONE_LR}")
    print(f"  Head params:          {len(head_params)} tensors  -> LR = {HEAD_LR}")
    print(f"  Frozen layers:        0 to {FREEZE_LAYERS - 1}")

    # NEW: log the run config to wandb once, right when the real optimizer/scheduler exist
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
# NEW: three hooks covering everything you'd want to "track everything":
#   - on_train_epoch_end -> training losses (box/cls/dfl) + current LR per group
#   - on_fit_epoch_end   -> validation metrics (mAP50, mAP50-95, precision, recall)
#                           + Ultralytics' own auto-generated plots/curves as images
#   - on_train_end        -> final best.pt weights uploaded as a wandb Artifact
def wandb_log_train_epoch(trainer):
    """Runs after every training epoch: losses + learning rates."""
    loss_dict = trainer.label_loss_items(trainer.tloss, prefix="train")
    lr_dict = {f"lr/{k}": v for k, v in trainer.lr.items()}  # trainer.lr = {'lr/pg0': ..., 'lr/pg1': ...}
    wandb.log({**loss_dict, **lr_dict}, step=trainer.epoch + 1)


def wandb_log_fit_epoch(trainer):
    """Runs after every validation pass: mAP/precision/recall, plus plot images if enabled."""
    wandb.log(trainer.metrics, step=trainer.epoch + 1)

    # Ultralytics saves confusion matrix / PR curve / label distribution plots to save_dir
    # during training when plots=True. Push the latest ones into wandb as images too.
    if trainer.args.plots:
        for img_path in trainer.save_dir.glob("*.png"):
            wandb.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))}, step=trainer.epoch + 1)


def wandb_log_train_end(trainer):
    """Runs once, after training finishes: upload best.pt as a versioned wandb Artifact,
    then split the run folder to match this project's layout - experiments/<run>/ keeps
    only weights/ + config.yaml, everything else (plots, results.csv, batch preview jpgs)
    moves into results/exp<id>/. Keeps every future run consistent automatically."""
    best_path = trainer.best  # path to best.pt, set internally by Ultralytics
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


# ---------------- MAIN: single training run ----------------
if __name__ == "__main__":   # FIXED: original had "_name_"/"_main_" (single underscores),
                              # which is NOT the Python dunder — that block would never
                              # have executed via `python script.py`, only if imported oddly.
    resuming = os.path.exists(LAST_PT)

    # NEW: start the wandb run BEFORE model.train() so wandb.config/wandb.log
    # calls inside the callbacks below have an active run to write to.
    # Without this, wandb's local sync cache lands in whatever directory you happen to run
    # the script from (e.g. src/wandb/) instead of staying with this experiment's own files.
    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,   # creates {wandb_dir}/wandb/run-.../ - kept alongside this experiment's weights/config
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

    # RESUME: if the crash left a checkpoint, load it (weights + optimizer/epoch state)
    # instead of starting over from COCO-pretrained weights.
    model = YOLO(LAST_PT if resuming else "yolov8l.pt")

    model.add_callback("on_train_start", setup_differential_lr)
    model.add_callback("on_train_epoch_end", wandb_log_train_epoch)   # NEW
    model.add_callback("on_fit_epoch_end", wandb_log_fit_epoch)       # NEW
    model.add_callback("on_train_end", wandb_log_train_end)           # NEW

    if resuming:
        # resume=True pulls data/epochs/imgsz/batch/freeze/patience/device/project/name
        # etc. straight from the args.yaml saved next to the checkpoint - passing them
        # again here would fight Ultralytics' own resume validation.
        print(f"[RESUME] Continuing from {LAST_PT}")
        model.train(resume=True)
    else:
        model.train(
            data=DATA,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            freeze=FREEZE_LAYERS,   # layers 0 to FREEZE_LAYERS-1 fully frozen
            patience=PATIENCE,
            device=0,               # force the RTX 5080 (cuda:0); fail loudly instead of silently falling back to CPU

            # NOTE: Ultralytics' own warmup logic (first `warmup_epochs`) interpolates
            # each param group's LR using ITS OWN built-in cosine/linear schedule
            # (self.lf), not our custom cosine_lambda above, for the first few epochs.
            # Setting warmup_epochs=0 avoids this schedule clash so our differential
            # cosine schedule governs LR from epoch 0 onward, cleanly.
            warmup_epochs=0,

            cos_lr=False,  # our custom LambdaLR scheduler already IS the cosine schedule;
                            # leave Ultralytics' built-in cos_lr off to avoid double-scheduling

            project=EXPERIMENTS_DIR,
            name=EXP_FOLDER,
            exist_ok=True,   # wandb's dir= already created this exact folder above (for its local
                              # sync cache) - without this, Ultralytics sees it "already exists" and
                              # auto-increments to a different name instead of reusing it
        )