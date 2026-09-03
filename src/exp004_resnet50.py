"""
Trains a ResNet-50 classifier on cropped waste-material images
(data/versions/exp004, built from exp001_exp003_exp006's YOLO boxes by
data/scripts/build_resnet_dataset.py).

WHY A SEPARATE CLASSIFIER:
YOLO's own classification head shows near-zero confusion between the 6 real
classes on held-out validation (results/exp003/confusion_matrix_normalized.png)
- the detector's actual weakness there is missed detections (recall), not
misclassification. This ResNet is stage two of a two-model pipeline: YOLO
finds/boxes an object, then this classifier looks at just that cropped region
and assigns the final label - useful if live-deployment conditions (glare,
lighting) cause confusions between visually similar surfaces (shiny aluminium,
glossy ceramic, clear plastic, glass) that the training/validation images
don't cover. Augmentation below leans on color/lighting jitter specifically
because of that.

Follows this project's experiment conventions (see README.md): EXP_ID-based
naming, weights+config in experiments/, plots+metrics in results/, wandb
tracking, auto-resume from a crash.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import os
import shutil
import json
import time

# ---------------- EXPERIMENT IDENTITY ----------------
# Bump these for every new run. RUN_NAME is built from them automatically, matching the
# exp{id}_{model}_{dataset-version} convention used by every other experiment in this project.
# This is exp004's own actual completed run - EXP_ID stays 4, matching
# experiments/exp004_resnet50/ on disk.
EXP_ID = 4
MODEL_NAME = "resnet50"
DATASET_VERSION = "exp004"    # cropped-classification dataset, see data/versions/exp004/README.md
EXP_FOLDER = f"exp{EXP_ID:03d}_{MODEL_NAME}"   # actual experiments/ + models/ folder/file name -
                                                 # deliberately drops the dataset suffix
RUN_NAME = f"{EXP_FOLDER}_{DATASET_VERSION}"    # wandb run name only - keeps full provenance there

# ---------------- CONFIG ----------------
PROJECT_ROOT = "d:/Reneonix/yolo_projects/Wastes_identification"
DATA_DIR = f"{PROJECT_ROOT}/data/versions/{DATASET_VERSION}"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
RESULTS_DIR = f"{PROJECT_ROOT}/results/exp{EXP_ID:03d}"
RUN_DIR = f"{EXPERIMENTS_DIR}/{EXP_FOLDER}"
LAST_PT = f"{RUN_DIR}/weights/last.pt"
BEST_PT = f"{RUN_DIR}/weights/best.pt"
MODELS_DIR = f"{PROJECT_ROOT}/models"

WANDB_PROJECT = "pour-defect-yolov8l"
WANDB_RUN_NAME = RUN_NAME

CLASS_NAMES = ["aluminium", "plastic", "metal", "stone", "ceramic", "glass"]

IMG_SIZE = 224
BATCH = 64
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7          # early stop if val accuracy doesn't improve for this many epochs
NUM_WORKERS = 8
DEVICE = "cuda:0"      # RTX 5080; fails loudly instead of silently falling back to CPU


def build_dataloaders():
    # Color/lighting jitter is deliberate, not a default augmentation choice - the whole point
    # of this classifier is robustness to glare/lighting conditions the detector struggles with.
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(f"{DATA_DIR}/train", transform=train_transform)
    val_ds = datasets.ImageFolder(f"{DATA_DIR}/val", transform=val_transform)

    # ImageFolder assigns class indices alphabetically - confirm that matches CLASS_NAMES
    # (also alphabetical) so checkpoints/labels stay consistent with the rest of the project.
    assert train_ds.classes == sorted(CLASS_NAMES), (
        f"Dataset class order {train_ds.classes} doesn't match expected {sorted(CLASS_NAMES)}"
    )

    # persistent_workers=True: keeps the same worker processes alive across epochs instead of
    # tearing them down and respawning fresh ones every epoch. Windows' spawn-based multiprocessing
    # occasionally fails to reload torch's shm.dll in a freshly spawned worker (a known Windows
    # fragility, not a code bug) - this cuts the number of spawn events from "every epoch" to
    # "once total", which removes almost all exposure to that failure mode.
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader, train_ds.classes


def build_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None, desc=""):
    """optimizer given -> training pass; None -> eval pass (no grad, no dropout/BN update)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    with torch.set_grad_enabled(is_train):
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(preds, labels, class_names, save_path):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("ResNet Classifier - Confusion Matrix (normalized)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return cm


def save_checkpoint(path, model, optimizer, epoch, best_val_acc, classes):
    # `classes` must be the ImageFolder-derived (alphabetical) order, NOT the CLASS_NAMES
    # constant above - ImageFolder assigns output indices alphabetically regardless of how
    # CLASS_NAMES is written, so saving the wrong order here would silently mislabel every
    # prediction for anyone loading this checkpoint later.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "class_names": classes,
    }, path)


if __name__ == "__main__":
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_DIR}\n"
            f"Run data/scripts/build_resnet_dataset.py first to generate it from v1."
        )

    resuming = os.path.exists(LAST_PT)

    # Same convention as the YOLO training scripts: wandb's local sync cache lives inside this
    # run's own experiment folder, not wherever the script happens to be launched from.
    wandb_dir = os.path.join(EXPERIMENTS_DIR, EXP_FOLDER)
    os.makedirs(wandb_dir, exist_ok=True)
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME + ("-resumed" if resuming else ""),
        dir=wandb_dir,
        config={
            "model": MODEL_NAME, "dataset": DATASET_VERSION, "img_size": IMG_SIZE,
            "batch": BATCH, "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE, "resumed_from": LAST_PT if resuming else None,
        },
    )

    train_loader, val_loader, classes = build_dataloaders()
    model = build_model(len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_acc = 0.0
    epochs_without_improvement = 0

    if resuming:
        print(f"[RESUME] Continuing from {LAST_PT}")
        ckpt = torch.load(LAST_PT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]

    history = []  # per-epoch metrics, written to results/exp00N/results.csv at the end

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer,
                                                  desc=f"Epoch {epoch + 1}/{EPOCHS} [train]")
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion,
                                                               desc=f"Epoch {epoch + 1}/{EPOCHS} [val]")
        elapsed = time.time() - t0

        print(f"Epoch {epoch + 1}/{EPOCHS}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  ({elapsed:.1f}s)")

        wandb.log({
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        }, step=epoch + 1)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_loss, "val_acc": val_acc})

        save_checkpoint(LAST_PT, model, optimizer, epoch, best_val_acc, classes)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(BEST_PT, model, optimizer, epoch, best_val_acc, classes)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping: no improvement for {PATIENCE} epochs.")
                break

    # ---------------- final validation report + results split ----------------
    ckpt = torch.load(BEST_PT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_val_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion,
                                                               desc="Final validation (best.pt)")
    print(f"\nBest model val accuracy: {final_val_acc:.4f}")

    if ckpt["model_state_dict"] is not None:
        artifact = wandb.Artifact(name=f"{WANDB_RUN_NAME}-best", type="model")
        artifact.add_file(BEST_PT)
        wandb.log_artifact(artifact)
    wandb.finish()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(f"{RESULTS_DIR}/results.csv", "w") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc\n")
        for row in history:
            f.write(f"{row['epoch']},{row['train_loss']},{row['train_acc']},"
                     f"{row['val_loss']},{row['val_acc']}\n")

    cm = plot_confusion_matrix(final_preds, final_labels, classes,
                                f"{RESULTS_DIR}/confusion_matrix.png")

    with open(f"{RESULTS_DIR}/config.json", "w") as f:
        json.dump({
            "model": MODEL_NAME, "dataset": DATASET_VERSION, "img_size": IMG_SIZE,
            "batch": BATCH, "epochs_run": len(history), "lr": LR,
            "weight_decay": WEIGHT_DECAY, "final_val_acc": final_val_acc,
            "class_names": classes,
        }, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}")

    # Standing project convention: every finished run's best weights also get a flat copy in
    # models/, named the same as its experiments/ folder (EXP_FOLDER, no dataset suffix).
    os.makedirs(MODELS_DIR, exist_ok=True)
    models_copy = f"{MODELS_DIR}/{EXP_FOLDER}_best.pt"
    shutil.copy2(BEST_PT, models_copy)
    print(f"Best weights also copied to {models_copy}")
