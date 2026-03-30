"""
defense.py - Phase 4: Adversarial training (PGD-10) + AE denoising defense.

Usage:
    python defense.py

Outputs:
    models/adv_trained_cnn.pth
    metrics/defense_metrics.json
    plots/defense_comparison.png
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchattacks
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score

import config
from utils import (
    set_seed, build_dataframe, split_dataframe, get_dataloaders,
    get_device, load_checkpoint, save_checkpoint, accuracy, denormalize,
)
from model import build_resnet18, ConvAutoencoder
from train import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Adversarial Training (PGD-10)
# ─────────────────────────────────────────────────────────────────────────────

def adv_train_one_epoch(model, loader, optimizer, criterion, device, scaler,
                        pgd_attack):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for images, labels in tqdm(loader, desc="  AdvTrain", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Generate PGD adversarial examples
        model.eval()   # freeze BN stats for attack
        adv_images = pgd_attack(images, labels)
        model.train()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            # Mix clean + adversarial (50/50)
            mixed_imgs  = torch.cat([images, adv_images])
            mixed_lbls  = torch.cat([labels, labels])
            outputs     = model(mixed_imgs)
            loss        = criterion(outputs, mixed_lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_acc  += accuracy(outputs[:len(images)].detach(), labels)
    n = len(loader)
    return total_loss / n, total_acc / n


def train_adversarially(
    train_loader, val_loader, device: torch.device,
    n_epochs: int = 30,
) -> nn.Module:
    """Fine-tune the baseline model with PGD adversarial training."""
    model     = build_resnet18().to(device)
    load_checkpoint(model, config.BASELINE_MODEL_PATH, device)

    pgd_atk   = torchattacks.PGD(
        model,
        eps=config.ADV_TRAIN_EPS,
        alpha=config.ADV_TRAIN_ALPHA,
        steps=config.ADV_TRAIN_STEPS,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE / 2,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val  = 0.0
    history   = []

    for epoch in range(1, n_epochs + 1):
        print(f"\nAdvTrain Epoch {epoch}/{n_epochs}")
        tr_loss, tr_acc = adv_train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, pgd_atk
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "tr_acc": tr_acc, "va_acc": va_acc})
        print(f"  Train acc={tr_acc:.4f}  Val acc={va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict()},
                config.ADV_TRAIN_PATH,
            )

    load_checkpoint(model, config.ADV_TRAIN_PATH, device)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Input Preprocessing (AE denoising)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def denoise_and_classify(model, ae, images, device, batch_size=32) -> np.ndarray:
    """Pass images through autoencoder then classifier; return predictions."""
    model.eval(); ae.eval()
    all_preds = []
    for i in range(0, len(images), batch_size):
        xi = images[i:i+batch_size].to(device)
        # Denormalize → AE → reclamp
        xi_dn   = denormalize(xi)          # [0,1]
        xi_rec  = ae(xi_dn).clamp(0, 1)   # reconstructed in [0,1]
        # Re-normalize for classifier
        mean_t  = torch.tensor(config.MEAN, device=device).view(1, 3, 1, 1)
        std_t   = torch.tensor(config.STD,  device=device).view(1, 3, 1, 1)
        xi_ren  = (xi_rec - mean_t) / std_t
        preds   = model(xi_ren).argmax(1).cpu().numpy()
        all_preds.append(preds)
    return np.concatenate(all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_on_tensors(model, images, labels, device, batch_size=64) -> float:
    model.eval()
    all_preds = []
    for i in range(0, len(images), batch_size):
        xi = images[i:i+batch_size].to(device)
        p  = model(xi).argmax(1).cpu().numpy()
        all_preds.append(p)
    preds = np.concatenate(all_preds)
    return accuracy_score(labels.numpy(), preds)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_defense_comparison(results: Dict, save_path: Path):
    methods    = list(results.keys())
    clean_accs = [results[m]["clean_acc"] for m in methods]
    fgsm_accs  = [results[m].get("fgsm_acc", 0) for m in methods]
    pgd_accs   = [results[m].get("pgd_acc",  0) for m in methods]
    cw_accs    = [results[m].get("cw_acc",   0) for m in methods]

    x     = np.arange(len(methods))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*width, clean_accs, width, label="Clean",  color="#4C72B0")
    ax.bar(x - 0.5*width, fgsm_accs,  width, label="FGSM",   color="#DD8452")
    ax.bar(x + 0.5*width, pgd_accs,   width, label="PGD",    color="#55A868")
    ax.bar(x + 1.5*width, cw_accs,    width, label="CW",     color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Robustness Comparison: Baseline vs Defenses", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[defense] Saved comparison -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    device = get_device()
    print(f"[defense] Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────────
    df = build_dataframe()
    train_df, val_df, test_df = split_dataframe(df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    # ── Baseline model ────────────────────────────────────────────────────────
    baseline = build_resnet18().to(device)
    load_checkpoint(baseline, config.BASELINE_MODEL_PATH, device)
    baseline.eval()

    # ── Load adversarial examples ─────────────────────────────────────────────
    adv_data = {}
    for aname in ["fgsm", "pgd", "cw"]:
        p = config.ADV_DIR / aname / "adv_tensors.pt"
        if p.exists():
            d = torch.load(p, map_location="cpu")
            adv_data[aname] = d["adv"], d["labels"]
        else:
            print(f"[defense] Warning: {p} not found - skipping {aname}")

    # ── Collect test clean images ─────────────────────────────────────────────
    test_imgs_list, test_lbls_list = [], []
    for imgs, lbls in test_loader:
        test_imgs_list.append(imgs); test_lbls_list.append(lbls)
    test_imgs   = torch.cat(test_imgs_list)
    test_labels = torch.cat(test_lbls_list)

    def acc_on_adv(model, aname):
        if aname not in adv_data:
            return float("nan")
        adv_imgs, adv_lbls = adv_data[aname]
        n = min(len(adv_imgs), len(test_imgs))
        return evaluate_on_tensors(model, adv_imgs[:n], adv_lbls[:n], device)

    # ── 1. Baseline evaluation ────────────────────────────────────────────────
    clean_acc = evaluate_on_tensors(baseline, test_imgs, test_labels, device)
    results   = {
        "Baseline": {
            "clean_acc": round(clean_acc, 4),
            "fgsm_acc":  round(acc_on_adv(baseline, "fgsm"), 4),
            "pgd_acc":   round(acc_on_adv(baseline, "pgd"),  4),
            "cw_acc":    round(acc_on_adv(baseline, "cw"),   4),
        }
    }
    print(f"\n[defense] Baseline  clean={results['Baseline']['clean_acc']:.4f}  "
          f"fgsm={results['Baseline']['fgsm_acc']:.4f}  "
          f"pgd={results['Baseline']['pgd_acc']:.4f}  "
          f"cw={results['Baseline']['cw_acc']:.4f}")

    # ── 2. Adversarial Training ───────────────────────────────────────────────
    if config.ADV_TRAIN_PATH.exists():
        print("[defense] Loading existing adversarially trained model ...")
        adv_model = build_resnet18().to(device)
        load_checkpoint(adv_model, config.ADV_TRAIN_PATH, device)
        adv_model.eval()
    else:
        print("[defense] Training adversarially robust model ...")
        adv_model = train_adversarially(train_loader, val_loader, device)

    adv_clean = evaluate_on_tensors(adv_model, test_imgs, test_labels, device)
    results["Adv-Trained"] = {
        "clean_acc": round(adv_clean, 4),
        "fgsm_acc":  round(acc_on_adv(adv_model, "fgsm"), 4),
        "pgd_acc":   round(acc_on_adv(adv_model, "pgd"),  4),
        "cw_acc":    round(acc_on_adv(adv_model, "cw"),   4),
    }
    print(f"[defense] AdvTrained clean={results['Adv-Trained']['clean_acc']:.4f}  "
          f"fgsm={results['Adv-Trained']['fgsm_acc']:.4f}  "
          f"pgd={results['Adv-Trained']['pgd_acc']:.4f}  "
          f"cw={results['Adv-Trained']['cw_acc']:.4f}")

    # ── 3. AE-denoising defense ───────────────────────────────────────────────
    if config.AE_MODEL_PATH.exists():
        print("[defense] Loading autoencoder denoiser ...")
        ae = ConvAutoencoder().to(device)
        load_checkpoint(ae, config.AE_MODEL_PATH, device)
        ae.eval()

        def acc_ae(aname):
            if aname not in adv_data:
                return float("nan")
            adv_imgs, adv_lbls = adv_data[aname]
            n      = min(len(adv_imgs), len(test_imgs))
            adv_n  = adv_imgs[:n]
            preds  = denoise_and_classify(baseline, ae, adv_n, device)
            return accuracy_score(adv_lbls[:n].numpy(), preds)

        ae_clean_preds = denoise_and_classify(baseline, ae, test_imgs, device)
        ae_clean_acc   = accuracy_score(test_labels.numpy(), ae_clean_preds)

        results["AE-Denoising"] = {
            "clean_acc": round(ae_clean_acc, 4),
            "fgsm_acc":  round(acc_ae("fgsm"), 4),
            "pgd_acc":   round(acc_ae("pgd"),  4),
            "cw_acc":    round(acc_ae("cw"),   4),
        }
        print(f"[defense] AE-Denoise clean={results['AE-Denoising']['clean_acc']:.4f}  "
              f"fgsm={results['AE-Denoising']['fgsm_acc']:.4f}  "
              f"pgd={results['AE-Denoising']['pgd_acc']:.4f}  "
              f"cw={results['AE-Denoising']['cw_acc']:.4f}")
    else:
        print("[defense] AE model not found - skipping AE denoising defense.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_defense_comparison(results, config.PLOT_DIR / "defense_comparison.png")

    # ── Save metrics ──────────────────────────────────────────────────────────
    with open(config.METRIC_DIR / "defense_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"[defense] Saved defense metrics -> {config.METRIC_DIR / 'defense_metrics.json'}")
    return results


if __name__ == "__main__":
    main()
