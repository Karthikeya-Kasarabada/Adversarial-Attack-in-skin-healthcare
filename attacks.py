"""
attacks.py – Phase 2: FGSM, PGD, and CW adversarial attack generation.

Usage:
    python attacks.py

Outputs:
    adv_images/fgsm/   – tensors (.pt) per class
    adv_images/pgd/
    adv_images/cw/
    metrics/attack_metrics.json
    plots/attack_examples.png
    plots/perturbation_hist.png
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchattacks
import matplotlib.pyplot as plt
from pytorch_msssim import ssim as pytorch_ssim
from tqdm import tqdm

import config
from utils import (
    set_seed, build_dataframe, split_dataframe, get_dataloaders,
    get_device, load_checkpoint, denormalize, ssim_batch,
    HAM10000Dataset, get_val_transform,
)
from model import build_resnet18


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_model(device: torch.device) -> nn.Module:
    model = build_resnet18().to(device)
    load_checkpoint(model, config.BASELINE_MODEL_PATH, device)
    model.eval()
    return model


def attack_success_rate(model, adv_images, true_labels, device) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(adv_images.to(device))
        preds   = outputs.argmax(dim=1).cpu()
    asr = (preds != true_labels).float().mean().item()
    return asr


def linf_norm(clean: torch.Tensor, adv: torch.Tensor) -> float:
    return (adv - clean).abs().max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].mean().item()


def compute_ssim(clean: torch.Tensor, adv: torch.Tensor) -> float:
    c = denormalize(clean)
    a = denormalize(adv)
    return ssim_batch(c, a)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class sample collection
# ─────────────────────────────────────────────────────────────────────────────

def collect_test_batches(test_loader: DataLoader, device):
    """Return all (image, label) tensors from the test set."""
    images_list, labels_list = [], []
    for imgs, lbls in test_loader:
        images_list.append(imgs)
        labels_list.append(lbls)
    return torch.cat(images_list), torch.cat(labels_list)


# ─────────────────────────────────────────────────────────────────────────────
# Attack runners
# ─────────────────────────────────────────────────────────────────────────────

def run_attack(
    attack_name: str,
    model: nn.Module,
    clean_images: torch.Tensor,
    true_labels:  torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict]:
    """Generate adversarial examples for one attack type."""

    model.eval()

    # Build attack object
    if attack_name == "fgsm":
        atk = torchattacks.FGSM(model, eps=config.FGSM_EPS)
    elif attack_name == "fgsm_targeted":
        atk = torchattacks.FGSM(model, eps=config.FGSM_EPS)
        atk.set_mode_targeted_by_label(quiet=True)
    elif attack_name == "pgd":
        atk = torchattacks.PGD(
            model,
            eps=config.PGD_EPS,
            alpha=config.PGD_ALPHA,
            steps=config.PGD_STEPS,
        )
    elif attack_name == "cw":
        atk = torchattacks.CW(
            model,
            c=1,
            kappa=0,
            steps=config.CW_MAX_ITER,
            lr=config.CW_LR,
        )
    else:
        raise ValueError(f"Unknown attack: {attack_name}")

    print(f"\n[attacks] Running {attack_name.upper()} on {len(clean_images)} images ...")
    t0 = time.time()

    # Process in batches to avoid OOM
    batch_sz = 32
    adv_list = []
    for i in range(0, len(clean_images), batch_sz):
        xi = clean_images[i:i+batch_sz].to(device)
        yi = true_labels[i:i+batch_sz].to(device)
        if attack_name == "fgsm_targeted":
            # Target next class (circular)
            yi_tgt = (yi + 1) % config.NUM_CLASSES
            adv_i  = atk(xi, yi_tgt)
        else:
            adv_i = atk(xi, yi)
        adv_list.append(adv_i.cpu())

    adv_images = torch.cat(adv_list, dim=0)

    elapsed = time.time() - t0
    asr     = attack_success_rate(model, adv_images, true_labels, device)
    linf    = linf_norm(clean_images, adv_images)
    ssim_v  = compute_ssim(clean_images, adv_images)

    metrics = {
        "attack":       attack_name,
        "n_samples":    len(clean_images),
        "asr":          round(asr,    4),
        "linf_mean":    round(linf,   6),
        "ssim":         round(ssim_v, 4),
        "elapsed_sec":  round(elapsed, 1),
    }
    print(f"  ASR={asr:.2%}  Linf={linf:.4f}  SSIM={ssim_v:.4f}  ({elapsed:.1f}s)")
    return adv_images, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_attack_examples(
    clean: torch.Tensor,
    adv_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    n_show: int = 4,
):
    attack_names = list(adv_dict.keys())
    n_attacks    = len(attack_names)
    fig, axes    = plt.subplots(n_show, n_attacks + 1, figsize=(4 * (n_attacks + 1), 4 * n_show))

    for row in range(n_show):
        # Clean
        img_clean = denormalize(clean[row:row+1])[0].permute(1, 2, 0).numpy()
        ax = axes[row][0]
        ax.imshow(np.clip(img_clean, 0, 1))
        ax.set_title(f"Clean\n({config.CLASS_NAMES[labels[row]]})", fontsize=9)
        ax.axis("off")

        # Each attack
        for col, aname in enumerate(attack_names, start=1):
            img_adv = denormalize(adv_dict[aname][row:row+1])[0].permute(1, 2, 0).numpy()
            pert    = np.abs(img_adv - img_clean) * 5   # amplify perturbation for viz
            ax = axes[row][col]
            ax.imshow(np.clip(img_adv, 0, 1))
            ax.set_title(f"{aname.upper()}", fontsize=9)
            ax.axis("off")

    plt.suptitle("Clean vs Adversarial Examples", fontsize=14, y=1.01)
    plt.tight_layout()
    path = config.PLOT_DIR / "attack_examples.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[attacks] Saved attack examples -> {path}")


def plot_perturbation_histograms(
    clean: torch.Tensor,
    adv_dict: Dict[str, torch.Tensor],
):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (aname, adv) in enumerate(adv_dict.items()):
        perturb = (adv - clean).abs().flatten(1).max(dim=1).values.numpy()
        ax.hist(perturb, bins=60, alpha=0.6, label=aname.upper(), color=colors[i % len(colors)])

    ax.set_xlabel("L∞ perturbation magnitude", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Perturbation Magnitude Distribution per Attack", fontsize=13)
    ax.legend()
    ax.grid(alpha=0.3)
    path = config.PLOT_DIR / "perturbation_hist.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[attacks] Saved perturbation histogram -> {path}")


def plot_accuracy_vs_eps(model, clean_images, true_labels, device):
    """Sweep eps for FGSM and plot clean accuracy degradation."""
    eps_values = np.linspace(0, 0.1, 11)
    accs       = []

    model.eval()
    for eps in eps_values:
        if eps == 0:
            with torch.no_grad():
                outs  = model(clean_images.to(device))
                preds = outs.argmax(1).cpu()
            acc = (preds == true_labels).float().mean().item()
        else:
            atk   = torchattacks.FGSM(model, eps=float(eps))
            adv_b = []
            for i in range(0, len(clean_images), 64):
                xi  = clean_images[i:i+64].to(device)
                yi  = true_labels[i:i+64].to(device)
                adv_b.append(atk(xi, yi).cpu())
            adv_all = torch.cat(adv_b)
            with torch.no_grad():
                outs  = model(adv_all.to(device))
                preds = outs.argmax(1).cpu()
            acc = (preds == true_labels).float().mean().item()
        accs.append(acc)
        print(f"  eps={eps:.3f}  acc={acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(eps_values, accs, marker="o", color="#4C72B0", linewidth=2)
    ax.set_xlabel("ε (epsilon)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FGSM: Accuracy vs. ε", fontsize=13)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = config.PLOT_DIR / "accuracy_vs_eps.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[attacks] Saved accuracy-vs-eps -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# GradCAM saliency
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_comparison(model, clean_imgs, adv_imgs, labels, device, n_show=4):
    """GradCAM side-by-side on clean vs PGD adversarial."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[attacks] grad-cam not installed – skipping GradCAM plot.")
        return

    target_layer = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layer)

    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))

    for row in range(n_show):
        ci = clean_imgs[row:row+1].to(device)
        ai = adv_imgs[row:row+1].to(device)

        grayscale_cam_c = cam(input_tensor=ci)[0]
        grayscale_cam_a = cam(input_tensor=ai)[0]

        rgb_c = denormalize(ci)[0].permute(1, 2, 0).cpu().numpy()
        rgb_a = denormalize(ai)[0].permute(1, 2, 0).cpu().numpy()

        cam_c = show_cam_on_image(np.clip(rgb_c, 0, 1), grayscale_cam_c, use_rgb=True)
        cam_a = show_cam_on_image(np.clip(rgb_a, 0, 1), grayscale_cam_a, use_rgb=True)

        axes[row][0].imshow(np.clip(rgb_c, 0, 1)); axes[row][0].set_title("Clean"); axes[row][0].axis("off")
        axes[row][1].imshow(cam_c);                 axes[row][1].set_title("GradCAM (clean)"); axes[row][1].axis("off")
        axes[row][2].imshow(np.clip(rgb_a, 0, 1)); axes[row][2].set_title("PGD Adv"); axes[row][2].axis("off")
        axes[row][3].imshow(cam_a);                 axes[row][3].set_title("GradCAM (adv)"); axes[row][3].axis("off")

    plt.suptitle("GradCAM: Clean vs PGD Adversarial", fontsize=14)
    plt.tight_layout()
    path = config.PLOT_DIR / "gradcam_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[attacks] Saved GradCAM -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    device = get_device()
    print(f"[attacks] Device: {device}")

    # Load model & data
    model = load_model(device)
    df    = build_dataframe()
    _, _, test_df = split_dataframe(df)
    from torch.utils.data import DataLoader as _DL
    test_loader = _DL(
        HAM10000Dataset(test_df, get_val_transform()),
        batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    # Collect all test images (or cap to keep compute manageable)
    print("[attacks] Collecting test images ...")
    clean_images, true_labels = collect_test_batches(test_loader, device)
    MAX_SAMPLES = config.ADV_SAMPLES_PER_CLASS * config.NUM_CLASSES
    if len(clean_images) > MAX_SAMPLES:
        idx = torch.randperm(len(clean_images))[:MAX_SAMPLES]
        clean_images = clean_images[idx]
        true_labels  = true_labels[idx]
    print(f"[attacks] Using {len(clean_images)} test images.")

    # ── Run attacks ──────────────────────────────────────────────────────────
    attack_names = ["fgsm", "pgd", "cw"]
    adv_dict     = {}
    all_metrics  = {}

    for aname in attack_names:
        adv_out = config.ADV_DIR / aname
        adv_out.mkdir(parents=True, exist_ok=True)
        adv_imgs, metrics = run_attack(aname, model, clean_images, true_labels, device)
        adv_dict[aname]   = adv_imgs
        all_metrics[aname] = metrics
        # Save tensors
        torch.save({"adv": adv_imgs, "labels": true_labels},
                   adv_out / "adv_tensors.pt")
        print(f"[attacks] Saved adv tensors -> {adv_out / 'adv_tensors.pt'}")

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_attack_examples(clean_images, adv_dict, true_labels)
    plot_perturbation_histograms(clean_images, adv_dict)
    print("[attacks] Computing accuracy-vs-eps sweep ...")
    plot_accuracy_vs_eps(model, clean_images, true_labels, device)
    plot_gradcam_comparison(model, clean_images, adv_dict.get("pgd", clean_images),
                            true_labels, device)

    # ── Save metrics ─────────────────────────────────────────────────────────
    with open(config.METRIC_DIR / "attack_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[attacks] Attack metrics saved -> {config.METRIC_DIR / 'attack_metrics.json'}")


if __name__ == "__main__":
    main()
