"""
train.py – Phase 1: Train the baseline ResNet-18 classifier on HAM10000.

Usage:
    python train.py [--resume]
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
from utils import (
    set_seed, build_dataframe, split_dataframe,
    get_dataloaders, get_device, save_checkpoint, load_checkpoint,
    accuracy,
)
from model import build_resnet18


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            loss    = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc  += accuracy(outputs.detach(), labels)
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        total_loss += loss.item()
        total_acc  += accuracy(outputs, labels)
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    n = len(loader)
    return total_loss / n, total_acc / n, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#DD8452")
    axes[0].set_title("Loss", fontsize=14); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#4C72B0")
    axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#DD8452")
    axes[1].set_title("Accuracy", fontsize=14); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.suptitle("Baseline ResNet-18 – Training Curves", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Saved training curves -> {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix – Baseline ResNet-18", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[train] Saved confusion matrix -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(resume: bool = False):
    set_seed()
    device = get_device()
    print(f"[train] Device: {device}")

    # Data
    df = build_dataframe()
    train_df, val_df, test_df = split_dataframe(df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # Model
    model     = build_resnet18().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 1
    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    if resume and config.BASELINE_MODEL_PATH.exists():
        ckpt = load_checkpoint(model, config.BASELINE_MODEL_PATH, device)
        start_epoch  = ckpt.get("epoch", 1) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        print(f"[train] Resuming from epoch {start_epoch}")

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"  Train – loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  Val   – loss={va_loss:.4f}  acc={va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_acc": va_acc, "history": history},
                config.BASELINE_MODEL_PATH,
            )

    # ── Final evaluation on test set ─────────────────────────────────────────
    load_checkpoint(model, config.BASELINE_MODEL_PATH, device)
    _, te_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    f1  = f1_score(labels, preds, average="weighted")
    cm  = confusion_matrix(labels, preds, labels=range(config.NUM_CLASSES))
    rpt = classification_report(
        labels, preds, 
        target_names=config.CLASS_NAMES, 
        labels=range(config.NUM_CLASSES),
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"TEST  accuracy : {te_acc:.4f}")
    print(f"TEST  F1-score : {f1:.4f}")
    print("=" * 60)
    print(rpt)

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_training_curves(history, config.PLOT_DIR / "training_curves.png")
    plot_confusion_matrix(cm, config.CLASS_NAMES, config.PLOT_DIR / "confusion_matrix.png")

    # ── Save baseline metrics ────────────────────────────────────────────────
    baseline_metrics = {
        "clean_accuracy":  round(te_acc, 4),
        "weighted_f1":     round(f1, 4),
        "classification_report": rpt,
    }
    with open(config.METRIC_DIR / "baseline_metrics.json", "w") as f:
        json.dump(baseline_metrics, f, indent=2)
    print(f"[train] Metrics saved -> {config.METRIC_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args.resume)
