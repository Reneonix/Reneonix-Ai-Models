"""
YOLO26s Training - 6-class "materials" detector, TWO-STAGE, 640x640
----------------------------------------------------------------------
Reproduces the exact two-stage recipe from the laptop-trained run that was brought into this
project as exp006 (originally `KAVIYA/train.py` + `KAVIYA/fine_tune.py`, run folders
`runs/detect/runs/train/yolo26s_materials{,_finetune}`, imported into
experiments/exp006_yolo26s_finetuned + results/exp006 + data/versions/exp006). That import
was a one-time copy of an already-completed two-stage run, not produced by this script - this
script exists so a FUTURE retrain follows this project's normal conventions (wandb,
exp-folder structure, resume support, models/ copy) instead of two disconnected one-off scripts.

TWO STAGES, run in sequence:
  1. Base training on data/versions/exp006 (KAVIYA's own smaller 6-class set, 1,631/408) -
     transfer learning from yolo26s.pt (COCO-pretrained).
  2. Fine-tuning on data/versions/exp001_exp003_exp006 (this project's main 6-class dataset,
     verified byte-identical to what KAVIYA/fine_tune_dataset/ was) - continues from stage 1's
     best.pt, lower LR.

Stage 2's validation set is the EXACT SAME 1,847-image split used by exp001/exp003 (verified
byte-identical file lists during import) - its mAP50-95 is a genuinely valid apples-to-apples
comparison against those two experiments, not just a similar-sounding number. Stage 1's own
val metrics (on its own 408-image split) are NOT comparable to that - see
results/exp006/config.json for which metric came from which stage.
"""

from ultralytics import YOLO
import wandb
import os
import shutil

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new training run. RUN_NAME is built from them automatically,
# enforcing the exp{id}_{model}_{dataset-version} convention project-wide. exp006 itself
# (the imported laptop run this script reproduces) is already taken - this targets the next
# free number for an actual from-scratch run through this script. Was originally 8, bumped to
# 11 once exp007 (AGI YOLOv8L) and exp008 (AGI ResNet-50) claimed 7 and 8 instead.
EXP_ID = 11
MODEL_NAME = "yolo26s_finetuned"
STAGE1_DATASET = "exp006"                    # base-training dataset, see data/versions/exp006/README.md
STAGE2_DATASET = "exp001_exp003_exp006"      # fine-tuning dataset - this project's main dataset
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix
RUN_NAME = f"{EXP_FOLDER}_{STAGE1_DATASET}_{STAGE2_DATASET}"   # wandb run name only - full provenance

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
STAGE1_DATA = f"{PROJECT_ROOT}/data/versions/{STAGE1_DATASET}/data.yaml"
STAGE2_DATA = f"{PROJECT_ROOT}/data/versions/{STAGE2_DATASET}/data.yaml"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
MODELS_DIR = f"{PROJECT_ROOT}/models"

EXP_DIR = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}"
STAGE1_LAST_PT = f"{EXP_DIR}/base_stage/weights/last.pt"
STAGE1_BEST_PT = f"{EXP_DIR}/base_stage/weights/best.pt"
STAGE2_LAST_PT = f"{EXP_DIR}/weights/last.pt"

WANDB_PROJECT = "pour-defect-yolov8l"   # same wandb project as every other experiment in this project

# Exact values from the original laptop recipe (KAVIYA/train.py + KAVIYA/fine_tune.py) - unchanged.
STAGE1_EPOCHS = 100
STAGE1_PATIENCE = 20
STAGE2_EPOCHS = 100
STAGE2_PATIENCE = 15
STAGE2_LR0 = 0.001    # lower initial LR for fine-tuning
IMGSZ = 640
BATCH = 8
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


def run_stage(model, data, epochs, patience, project, name, lr0=None, resume=False):
    if resume:
        print(f"[RESUME] Continuing {name} from its last checkpoint")
        model.train(resume=True)
        return

    kwargs = dict(
        data=data, epochs=epochs, imgsz=IMGSZ, batch=BATCH, device=DEVICE,
        patience=patience, workers=8, pretrained=True, optimizer="auto", seed=0,
        project=project, name=name, exist_ok=True,
    )
    if lr0 is not None:
        kwargs["lr0"] = lr0
    model.train(**kwargs)


if __name__ == "__main__":
    if not os.path.isfile(STAGE1_DATA):
        raise FileNotFoundError(
            f"Stage 1 dataset yaml not found: {STAGE1_DATA}\n"
            f"Run data/scripts/import_kaviya_dataset.py first."
        )
    if not os.path.isfile(STAGE2_DATA):
        raise FileNotFoundError(f"Stage 2 dataset yaml not found: {STAGE2_DATA}")

    stage2_resuming = os.path.exists(STAGE2_LAST_PT)
    stage1_resuming = os.path.exists(STAGE1_LAST_PT)
    stage1_done = os.path.exists(STAGE1_BEST_PT) and not stage1_resuming

    wandb_dir = EXP_DIR
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME + ("-resumed" if (stage1_resuming or stage2_resuming) else ""),
        dir=wandb_dir,
        config={
            "model": "yolo26s.pt", "stage1_data": STAGE1_DATA, "stage2_data": STAGE2_DATA,
            "stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS,
            "imgsz": IMGSZ, "batch": BATCH, "stage2_lr0": STAGE2_LR0,
        },
    )

    # ---------------- STAGE 1: base training ----------------
    if stage2_resuming:
        print("Stage 1 already complete (stage 2 checkpoint exists) - skipping to stage 2.")
    else:
        print(f"{'[RESUME]' if stage1_resuming else '[NEW]'} Stage 1: base training on {STAGE1_DATASET}")
        stage1_model = YOLO(STAGE1_LAST_PT if stage1_resuming else "yolo26s.pt")
        stage1_model.add_callback("on_train_epoch_end", wandb_log_train_epoch)
        stage1_model.add_callback("on_fit_epoch_end", wandb_log_fit_epoch)
        run_stage(stage1_model, STAGE1_DATA, STAGE1_EPOCHS, STAGE1_PATIENCE,
                  project=EXP_DIR, name="base_stage", resume=stage1_resuming)

    # ---------------- STAGE 2: fine-tuning ----------------
    print(f"{'[RESUME]' if stage2_resuming else '[NEW]'} Stage 2: fine-tuning on {STAGE2_DATASET}")
    stage2_model = YOLO(STAGE2_LAST_PT if stage2_resuming else STAGE1_BEST_PT)
    stage2_model.add_callback("on_train_epoch_end", wandb_log_train_epoch)
    stage2_model.add_callback("on_fit_epoch_end", wandb_log_fit_epoch)
    run_stage(stage2_model, STAGE2_DATA, STAGE2_EPOCHS, STAGE2_PATIENCE,
              project=EXPERIMENTS_DIR, name=EXP_FOLDER, lr0=STAGE2_LR0, resume=stage2_resuming)

    # ---------------- wrap up: wandb artifact, models/ copy, results split ----------------
    best_path = stage2_model.trainer.best
    if best_path.exists():
        artifact = wandb.Artifact(name=f"{RUN_NAME}-best", type="model")
        artifact.add_file(str(best_path))
        wandb.log_artifact(artifact)

        os.makedirs(MODELS_DIR, exist_ok=True)
        models_copy = os.path.join(MODELS_DIR, f"{EXP_FOLDER}_best.pt")
        shutil.copy2(str(best_path), models_copy)
        print(f"Best weights also copied to {models_copy}")
    wandb.finish()

    save_dir = stage2_model.trainer.save_dir
    args_path = save_dir / "args.yaml"
    if args_path.exists():
        args_path.rename(save_dir / "config.yaml")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for item in save_dir.iterdir():
        if item.name in ("weights", "config.yaml", "wandb", "base_stage"):
            continue
        shutil.move(str(item), os.path.join(RESULTS_DIR, item.name))

    print(f"\nDone. Final (fine-tuned) weights: {EXP_DIR}/weights/best.pt")
    print(f"Base-stage weights kept at: {EXP_DIR}/base_stage/weights/best.pt")
