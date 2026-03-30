"""
detect.py - Phase 3: Detection mechanisms for adversarial examples.

Detectors:
  1. LID   - Local Intrinsic Dimensionality (k-NN in feature space)
  2. Mahal - Mahalanobis distance (class-conditional Gaussians)
  3. AE    - Autoencoder reconstruction error

Usage:
    python detect.py

Outputs:
    models/autoencoder.pth
    metrics/detection_metrics.json
    plots/roc_curves.png
    plots/detection_score_dist.png
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import config
from utils import (
    set_seed, build_dataframe, split_dataframe, get_dataloaders,
    get_device, load_checkpoint, save_checkpoint, FeatureExtractor,
    denormalize,
)
from model import build_resnet18, ConvAutoencoder
from train import evaluate


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(
    extractor: nn.Module,
    loader_or_tensors,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 512-d avgpool features.
    loader_or_tensors: DataLoader  OR  (images_tensor, labels_tensor)
    Returns:  (features [N,512], labels [N])
    """
    extractor.eval()
    feats_list, labels_list = [], []

    if isinstance(loader_or_tensors, tuple):
        images, labels = loader_or_tensors
        for i in range(0, len(images), batch_size):
            xi = images[i:i+batch_size].to(device)
            fi = extractor(xi).cpu().numpy()
            feats_list.append(fi)
            labels_list.append(labels[i:i+batch_size].numpy())
    else:
        for imgs, lbls in loader_or_tensors:
            imgs = imgs.to(device)
            fi   = extractor(imgs).cpu().numpy()
            feats_list.append(fi)
            labels_list.append(lbls.numpy())

    return np.vstack(feats_list), np.concatenate(labels_list)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LID Detector
# ─────────────────────────────────────────────────────────────────────────────

def compute_lid(query: np.ndarray,
                reference: np.ndarray,
                k: int = config.LID_K) -> np.ndarray:
    """
    Compute Local Intrinsic Dimensionality for each query point.
    Reference = clean training features.
    """
    # Euclidean distances from each query to all reference points
    dists = cdist(query, reference, metric="euclidean")          # (N_q, N_ref)
    dists.sort(axis=1)
    knn_dists = dists[:, 1:k+1]                                  # exclude self (0)

    # MLE estimator:  LID_i = - k / sum_j log(r_j / r_k)
    eps   = 1e-8
    r_k   = knn_dists[:, -1:] + eps
    lid   = -k / np.sum(np.log(knn_dists / r_k + eps), axis=1)
    return lid.astype(np.float32)


class LIDDetector:
    def __init__(self, k: int = config.LID_K):
        self.k          = k
        self.ref_feats  : np.ndarray | None = None
        self.threshold  : float             = 0.0
        self.scaler     = StandardScaler()

    def fit(self, clean_feats: np.ndarray, val_clean_feats: np.ndarray,
            val_adv_feats: np.ndarray):
        """Fit reference set and tune threshold on val split."""
        self.ref_feats = self.scaler.fit_transform(clean_feats)

        val_c_s = self.scaler.transform(val_clean_feats)
        val_a_s = self.scaler.transform(val_adv_feats)

        lid_clean = compute_lid(val_c_s, self.ref_feats, self.k)
        lid_adv   = compute_lid(val_a_s, self.ref_feats, self.k)

        # Threshold: keep ≥ 95% clean correctly classified as clean
        self.threshold = float(np.percentile(lid_clean,
                               (1 - config.DETECT_TPR_TARGET) * 100))
        print(f"[LID] threshold={self.threshold:.4f}  "
              f"clean LID mean={lid_clean.mean():.4f}  "
              f"adv LID mean={lid_adv.mean():.4f}")
        return lid_clean, lid_adv

    def score(self, feats: np.ndarray) -> np.ndarray:
        feats_s = self.scaler.transform(feats)
        return compute_lid(feats_s, self.ref_feats, self.k)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """1 = adversarial, 0 = clean"""
        return (scores > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Mahalanobis Distance Detector
# ─────────────────────────────────────────────────────────────────────────────

class MahalanobisDetector:
    def __init__(self):
        self.class_means  : List[np.ndarray] = []
        self.precision    : np.ndarray | None = None
        self.threshold    : float             = 0.0

    def fit(self, train_feats: np.ndarray, train_labels: np.ndarray,
            val_clean_feats: np.ndarray, val_adv_feats: np.ndarray):
        """Fit class-conditional Gaussians using Ledoit-Wolf shrinkage."""
        classes = np.unique(train_labels)

        # Per-class means
        self.class_means = [
            train_feats[train_labels == c].mean(axis=0) for c in classes
        ]
        # Shared precision matrix (Ledoit-Wolf regularised)
        cov_est = LedoitWolf().fit(train_feats)
        self.precision = cov_est.precision_

        # Tune threshold
        s_clean = self._min_class_distance(val_clean_feats)
        self.threshold = float(np.percentile(s_clean,
                               (1 - config.DETECT_TPR_TARGET) * 100))
        print(f"[Mahal] threshold={self.threshold:.4f}  "
              f"clean mean={s_clean.mean():.4f}  "
              f"adv mean={self._min_class_distance(val_adv_feats).mean():.4f}")

    def _min_class_distance(self, feats: np.ndarray) -> np.ndarray:
        """Minimum Mahalanobis distance to any class centroid."""
        dists = []
        for mu in self.class_means:
            diff = feats - mu
            d    = np.sqrt(np.einsum("ni,ij,nj->n", diff, self.precision, diff))
            dists.append(d)
        return np.stack(dists, axis=1).min(axis=1)

    def score(self, feats: np.ndarray) -> np.ndarray:
        return self._min_class_distance(feats)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Autoencoder Detector
# ─────────────────────────────────────────────────────────────────────────────

def train_autoencoder(
    train_loader,
    device: torch.device,
) -> ConvAutoencoder:
    """Train ConvAutoencoder on clean images."""
    ae        = ConvAutoencoder().to(device)
    optimizer = optim.Adam(ae.parameters(), lr=config.AE_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = float("inf")

    for epoch in range(1, config.AE_EPOCHS + 1):
        ae.train()
        epoch_loss = 0.0
        for imgs, _ in tqdm(train_loader, desc=f"  AE Epoch {epoch:02d}", leave=False):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                recon = ae(imgs)
                loss  = criterion(recon, imgs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)
        print(f"  AE Epoch {epoch:02d}/{config.AE_EPOCHS}  loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint({"epoch": epoch, "model_state": ae.state_dict()},
                            config.AE_MODEL_PATH)

    load_checkpoint(ae, config.AE_MODEL_PATH, device)
    return ae


class AutoencoderDetector:
    def __init__(self, ae: ConvAutoencoder, device: torch.device):
        self.ae        = ae.eval()
        self.device    = device
        self.threshold = 0.0

    def _get_errors(self, images: torch.Tensor) -> np.ndarray:
        errors = []
        for i in range(0, len(images), 64):
            xi = images[i:i+64].to(self.device)
            e  = self.ae.reconstruction_error(xi).cpu().numpy()
            errors.append(e)
        return np.concatenate(errors)

    def fit_threshold(self,
                      val_clean: torch.Tensor,
                      val_adv:   torch.Tensor):
        err_clean = self._get_errors(val_clean)
        self.threshold = float(np.percentile(
            err_clean, (1 - config.DETECT_TPR_TARGET) * 100
        ))
        print(f"[AE] threshold={self.threshold:.6f}  "
              f"clean err mean={err_clean.mean():.6f}  "
              f"adv err mean={self._get_errors(val_adv).mean():.6f}")

    def score(self, images: torch.Tensor) -> np.ndarray:
        return self._get_errors(images)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self.threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Ensemble detector
# ─────────────────────────────────────────────────────────────────────────────

def ensemble_predict(lid_preds, mahal_preds, ae_preds) -> np.ndarray:
    """Majority vote of three binary detectors."""
    votes = np.stack([lid_preds, mahal_preds, ae_preds], axis=1).sum(axis=1)
    return (votes >= 2).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(results: Dict, save_path: Path):
    fig, ax   = plt.subplots(figsize=(8, 6))
    palette   = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, (name, r) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(r["y_true"], r["scores"])
        auc         = roc_auc_score(r["y_true"], r["scores"])
        ax.plot(fpr, tpr, label=f"{name}  (AUC={auc:.3f})",
                color=palette[i % len(palette)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curves - Adversarial Detection", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[detect] Saved ROC curves -> {save_path}")


def plot_score_distributions(results: Dict, save_path: Path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, r) in zip(axes, results.items()):
        y_true  = np.array(r["y_true"])
        scores  = np.array(r["scores"])
        ax.hist(scores[y_true == 0], bins=50, alpha=0.6, label="Clean", color="#4C72B0")
        ax.hist(scores[y_true == 1], bins=50, alpha=0.6, label="Adversarial", color="#DD8452")
        ax.axvline(r.get("threshold", np.nan), color="red", linestyle="--", label="Threshold")
        ax.set_title(f"{name} – Score Distribution", fontsize=12)
        ax.set_xlabel("Score"); ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[detect] Saved score distributions -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_seed()
    device = get_device()
    print(f"[detect] Device: {device}")

    # ── Load backbone + feature extractor ────────────────────────────────────
    backbone  = build_resnet18().to(device)
    load_checkpoint(backbone, config.BASELINE_MODEL_PATH, device)
    backbone.eval()
    extractor = FeatureExtractor(backbone).to(device)

    # ── Load data ────────────────────────────────────────────────────────────
    df = build_dataframe()
    train_df, val_df, test_df = split_dataframe(df)
    train_loader, val_loader, test_loader = get_dataloaders(train_df, val_df, test_df)

    print("[detect] Extracting train features ...")
    train_feats, train_labels = extract_features(extractor, train_loader, device)
    print("[detect] Extracting val features ...")
    val_feats,   val_labels   = extract_features(extractor, val_loader,   device)
    print("[detect] Extracting test features ...")
    test_feats,  test_labels  = extract_features(extractor, test_loader,  device)

    # Load saved adversarial examples (PGD used for detection evaluation)
    pgd_path = config.ADV_DIR / "pgd" / "adv_tensors.pt"
    if not pgd_path.exists():
        raise FileNotFoundError(
            f"Adversarial tensors not found at {pgd_path}. "
            "Run attacks.py first."
        )
    pgd_data = torch.load(pgd_path, map_location="cpu")
    adv_imgs  = pgd_data["adv"]
    adv_lbls  = pgd_data["labels"]

    print("[detect] Extracting adv features …")
    adv_feats, _ = extract_features(extractor, (adv_imgs, adv_lbls), device)

    # Val split from adv: use first 20% as val proxy for threshold tuning
    n_val = len(val_feats)
    val_adv_feats  = adv_feats[:n_val]
    test_adv_feats = adv_feats[n_val:]
    val_adv_imgs   = adv_imgs[:n_val]
    test_adv_imgs  = adv_imgs[n_val:]

    # ── 1. LID ───────────────────────────────────────────────────────────────
    print("\n[detect] ── LID ──")
    lid_det = LIDDetector()
    lid_c_val, lid_a_val = lid_det.fit(train_feats, val_feats, val_adv_feats)

    lid_c_test = lid_det.score(test_feats)
    lid_a_test = lid_det.score(test_adv_feats)

    lid_y  = np.concatenate([np.zeros(len(lid_c_test)), np.ones(len(lid_a_test))])
    lid_sc = np.concatenate([lid_c_test, lid_a_test])
    lid_auc = roc_auc_score(lid_y, lid_sc)
    lid_prd = lid_det.predict(lid_sc)
    lid_fpr = ((lid_prd[lid_y == 0]) == 1).mean()
    lid_tpr = ((lid_prd[lid_y == 1]) == 1).mean()
    print(f"[LID] AUC={lid_auc:.4f}  TPR={lid_tpr:.4f}  FPR={lid_fpr:.4f}")

    # ── 2. Mahalanobis ───────────────────────────────────────────────────────
    print("\n[detect] ── Mahalanobis ──")
    mah_det = MahalanobisDetector()
    mah_det.fit(train_feats, train_labels, val_feats, val_adv_feats)

    mah_c = mah_det.score(test_feats)
    mah_a = mah_det.score(test_adv_feats)
    mah_y  = np.concatenate([np.zeros(len(mah_c)), np.ones(len(mah_a))])
    mah_sc = np.concatenate([mah_c, mah_a])
    mah_auc = roc_auc_score(mah_y, mah_sc)
    mah_prd = mah_det.predict(mah_sc)
    mah_fpr = ((mah_prd[mah_y == 0]) == 1).mean()
    mah_tpr = ((mah_prd[mah_y == 1]) == 1).mean()
    print(f"[Mahal] AUC={mah_auc:.4f}  TPR={mah_tpr:.4f}  FPR={mah_fpr:.4f}")

    # ── 3. Autoencoder ───────────────────────────────────────────────────────
    print("\n[detect] ── Autoencoder ──")
    if config.AE_MODEL_PATH.exists():
        print("[detect] Loading existing autoencoder ...")
        ae = ConvAutoencoder().to(device)
        load_checkpoint(ae, config.AE_MODEL_PATH, device)
        ae.eval()
    else:
        print("[detect] Training autoencoder ...")
        ae = train_autoencoder(train_loader, device)

    ae_det = AutoencoderDetector(ae, device)

    # Collect val images as tensors
    val_imgs_list = []
    for imgs, _ in val_loader:
        val_imgs_list.append(imgs)
    val_imgs_tensor = torch.cat(val_imgs_list)

    test_imgs_list = []
    for imgs, _ in test_loader:
        test_imgs_list.append(imgs)
    test_imgs_tensor = torch.cat(test_imgs_list)

    ae_det.fit_threshold(val_imgs_tensor, val_adv_imgs)

    ae_c = ae_det.score(test_imgs_tensor)
    ae_a = ae_det.score(test_adv_imgs)
    ae_y  = np.concatenate([np.zeros(len(ae_c)), np.ones(len(ae_a))])
    ae_sc = np.concatenate([ae_c, ae_a])
    ae_auc = roc_auc_score(ae_y, ae_sc)
    ae_prd = ae_det.predict(ae_sc)
    ae_fpr = ((ae_prd[ae_y == 0]) == 1).mean()
    ae_tpr = ((ae_prd[ae_y == 1]) == 1).mean()
    print(f"[AE] AUC={ae_auc:.4f}  TPR={ae_tpr:.4f}  FPR={ae_fpr:.4f}")

    # ── 4. Ensemble ──────────────────────────────────────────────────────────
    ens_prd = ensemble_predict(
        lid_det.predict(lid_sc),
        mah_det.predict(mah_sc),
        ae_det.predict(ae_sc),
    )
    ens_fpr = ((ens_prd[lid_y == 0]) == 1).mean()
    ens_tpr = ((ens_prd[lid_y == 1]) == 1).mean()
    # Use average score for ensemble AUC approximation
    ens_sc   = (lid_sc / (lid_sc.max() + 1e-8) +
                mah_sc / (mah_sc.max() + 1e-8) +
                ae_sc  / (ae_sc.max()  + 1e-8)) / 3
    ens_auc = roc_auc_score(lid_y, ens_sc)
    print(f"[Ens] AUC={ens_auc:.4f}  TPR={ens_tpr:.4f}  FPR={ens_fpr:.4f}")

    # ── Plots ────────────────────────────────────────────────────────────────
    roc_results = {
        "LID":       {"y_true": lid_y.tolist(), "scores": lid_sc.tolist(), "threshold": lid_det.threshold},
        "Mahalanobis": {"y_true": mah_y.tolist(), "scores": mah_sc.tolist(), "threshold": mah_det.threshold},
        "Autoencoder": {"y_true": ae_y.tolist(),  "scores": ae_sc.tolist(),  "threshold": ae_det.threshold},
        "Ensemble":  {"y_true": lid_y.tolist(), "scores": ens_sc.tolist()},
    }
    plot_roc_curves(roc_results, config.PLOT_DIR / "roc_curves.png")
    plot_score_distributions(
        {k: v for k, v in roc_results.items() if k != "Ensemble"},
        config.PLOT_DIR / "detection_score_dist.png",
    )

    # ── Save metrics ─────────────────────────────────────────────────────────
    detect_metrics = {
        "LID":         {"AUC": round(lid_auc, 4), "TPR": round(lid_tpr, 4), "FPR": round(lid_fpr, 4)},
        "Mahalanobis": {"AUC": round(mah_auc, 4), "TPR": round(mah_tpr, 4), "FPR": round(mah_fpr, 4)},
        "Autoencoder": {"AUC": round(ae_auc,  4), "TPR": round(ae_tpr,  4), "FPR": round(ae_fpr,  4)},
        "Ensemble":    {"AUC": round(ens_auc, 4), "TPR": round(ens_tpr, 4), "FPR": round(ens_fpr, 4)},
    }
    with open(config.METRIC_DIR / "detection_metrics.json", "w") as f:
        json.dump(detect_metrics, f, indent=2)
    print(f"[detect] Detection metrics saved -> {config.METRIC_DIR / 'detection_metrics.json'}")
    return detect_metrics


if __name__ == "__main__":
    main()
