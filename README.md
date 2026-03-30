# Adversarial Attacks in Healthcare AI: Vulnerabilities and Detection Mechanisms

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange) ![License](https://img.shields.io/badge/License-MIT-green)

> ⚠️ **Research Use Only** – This framework is for academic study. Clinical deployment of AI-based diagnostic tools requires FDA AI/ML-SaMD regulatory clearance and multi-site validation.

---

## Overview

End-to-end pipeline for adversarial robustness evaluation on the **HAM10000 skin lesion dataset** (10,015 dermatoscopic images, 7 classes). Demonstrates how state-of-the-art CNN classifiers are vulnerable to imperceptible adversarial perturbations, and provides three complementary detection/defense mechanisms.

### Key Results (expected)

| Metric | Value |
|--------|-------|
| Baseline clean accuracy | **>90%** |
| FGSM Attack Success Rate | **>90%** |
| PGD Attack Success Rate | **>95%** |
| LID Detection AUC | **~0.92** |
| Mahalanobis AUC | **~0.88** |
| Autoencoder AUC | **~0.85** |
| Adv. Training PGD accuracy | **~60–70%** |

---

## Project Structure

```
Medical Adversarial Detection/
├── config.py              ← All hyperparameters & paths
├── utils.py               ← Dataset, DataLoader, feature extraction
├── model.py               ← ResNet-18 + ConvAutoencoder
├── train.py               ← Phase 1: Baseline training
├── attacks.py             ← Phase 2: FGSM / PGD / CW generation
├── detect.py              ← Phase 3: LID / Mahalanobis / AE detection
├── defense.py             ← Phase 4: Adversarial training + AE denoising
├── evaluate.py            ← Phase 5: Metrics CSV + report
├── main.py                ← End-to-end orchestrator
├── download_data.py       ← Kaggle / ISIC download helper
├── adversarial_healthcare.ipynb  ← Interactive Jupyter notebook
├── data/
│   └── raw/               ← HAM10000 images + metadata.csv
├── models/
│   ├── baseline_cnn.pth
│   ├── autoencoder.pth
│   └── adv_trained_cnn.pth
├── adv_images/
│   ├── fgsm/adv_tensors.pt
│   ├── pgd/adv_tensors.pt
│   └── cw/adv_tensors.pt
├── plots/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── attack_examples.png
│   ├── perturbation_hist.png
│   ├── accuracy_vs_eps.png
│   ├── gradcam_comparison.png
│   ├── roc_curves.png
│   ├── detection_score_dist.png
│   ├── defense_comparison.png
│   └── metrics_heatmap.png
├── metrics/
│   ├── baseline_metrics.json
│   ├── attack_metrics.json
│   ├── detection_metrics.json
│   ├── defense_metrics.json
│   └── metrics.csv
├── requirements.txt
└── REPORT.md
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

**Option A – Kaggle API** (full 10k images, recommended):
```bash
# Place your kaggle.json at %USERPROFILE%\.kaggle\kaggle.json
python download_data.py --source kaggle
```

**Option B – ISIC API** (no auth, ~1 000 images for quick testing):
```bash
python download_data.py --source isic
```

---

## Running the Pipeline

### Full pipeline (all 5 phases):
```bash
python main.py
```

### Individual phases:
```bash
python main.py --phase 1   # Train baseline
python main.py --phase 2   # Generate attacks
python main.py --phase 3   # Detection
python main.py --phase 4   # Defenses
python main.py --phase 5   # Evaluate & report
```

### Jupyter notebook (interactive):
```bash
jupyter notebook adversarial_healthcare.ipynb
```

---

## Phase Detail

### Phase 1 – Baseline Classifier
- **Architecture**: ResNet-18 (ImageNet pre-trained)
- **Training**: Adam (lr=1e-4), CosineAnnealingLR, 50 epochs, mixed-precision
- **Augmentation**: Flips, rotation ±20°, ColorJitter
- **Balancing**: WeightedRandomSampler per class

### Phase 2 – Adversarial Attacks

| Attack | Type | ε | Steps |
|--------|------|---|-------|
| FGSM | Untargeted | 0.03 | 1 |
| FGSM Targeted | Targeted | 0.03 | 1 |
| PGD | Untargeted | 0.03 | 40 |
| CW (L2) | Targeted | — | 100 |

**Metrics**: Attack Success Rate, L∞ norm, SSIM

### Phase 3 – Detection

| Detector | Method | Key Param |
|----------|--------|-----------|
| LID | k-NN in feature space | k=20 |
| Mahalanobis | Class-conditional Gaussians (Ledoit-Wolf) | Layer=avgpool |
| Autoencoder | Reconstruction error (MSE) | 30 epochs |
| Ensemble | Majority vote (≥2/3) | 95% clean TPR target |

### Phase 4 – Defenses
1. **Adversarial Training (PGD-10)**: Mixed clean + PGD examples per batch, 30 epochs
2. **AE Denoising**: Pass images through trained autoencoder before classification

### Phase 5 – Report
Auto-generates `REPORT.md`, `metrics/metrics.csv`, and `plots/metrics_heatmap.png`

---

## Methodology

### White-box vs. Black-box
All attacks assume white-box access (attacker knows model weights). Real-world attacks may be black-box (transfer-based). Results represent **worst-case vulnerability**.

### Threshold Tuning
Detection thresholds are tuned on the validation set to achieve ≥95% True Positive Rate (TPR) for clean images, minimising false alarms in clinical workflow.

---

## Limitations

- **White-box assumption**: Real attacks may be weaker (black-box setting)
- **Compute constraints**: CW is run for 100 iterations; full convergence requires ~1,000+
- **Dataset bias**: HAM10000 is predominantly light-skinned; model may underperform on darker skin tones
- **Single architecture**: Results may differ with ViT or EfficientNet backbones

---

## Ethical Statement

- **Research only**: Not validated for clinical use
- **Reproducibility**: Seed = 42 throughout
- **Data privacy**: HAM10000 is fully anonymised public data (ISIC 2018 challenge)
- **Regulatory**: Real deployment requires FDA 510(k) or De Novo review under AI/ML-SaMD guidelines

---

## Citation

```
@misc{adversarial_healthcare_2026,
  title  = {Adversarial Attacks in Healthcare AI: Vulnerabilities and Detection},
  year   = {2026},
  note   = {Research project, seed=42}
}
```

---

## License

MIT License – See [LICENSE](LICENSE) for details.
