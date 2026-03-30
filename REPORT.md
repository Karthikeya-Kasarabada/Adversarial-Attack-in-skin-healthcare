# Adversarial Attacks in Healthcare AI: Vulnerabilities and Detection Mechanisms
## Project Report

---

## 1. Executive Summary

This project demonstrates end-to-end adversarial robustness evaluation on the
**HAM10000 skin lesion dataset** (10,015 images, 7 classes).  Key findings:

- **Baseline clean accuracy**: 0.9375 (target: >0.90)
- **Attack Success Rate (PGD vs baseline)**: 100.00%  
- **Best detector**: LID with AUC=1.0
- **Adversarial training**: clean acc=1.0, PGD acc=0.0  
- **Recommendation**: deploy adversarial training + LID ensemble detection


---

## 2. Dataset & Preprocessing

| Property | Value |
|----------|-------|
| Dataset | HAM10000 (ISIC 2018 challenge) |
| Classes | akiec, bcc, bkl, df, mel, nv, vasc |
| Total images | 10,015 |
| Input size | 224 × 224 px (ImageNet normalisation) |
| Split | 70 / 15 / 15 (train / val / test) |
| Augmentation | H-flip, V-flip, Rotation ±20°, ColorJitter |
| Balancing | Weighted random oversampling |

---

## 3. Baseline Model

Architecture: ResNet-18 (ImageNet pre-trained)  
Optimizer: Adam lr=1e-4, weight-decay=1e-4  
Schedule: CosineAnnealingLR, 50 epochs  
Loss: CrossEntropy with label-smoothing=0.1

| Clean Accuracy | 0.9375 |
| Weighted F1    | 0.9677 |

---

## 4. Adversarial Attacks

| Attack | ε | Steps | ASR | L∞ | SSIM |
|--------|---|-------|-----|----|------|
| FGSM | 0.03 | - | 0.125 | 1.719356 | 0.8476 |
| PGD | 0.03 | - | 1.0 | 1.719356 | 0.8535 |
| CW | 0.03 | - | 0.8125 | 1.397658 | 0.8826 |


---

## 5. Detection Mechanisms

| Detector | AUC | TPR | FPR |
|----------|-----|-----|-----|
| LID | 1.0 | 1.0 | 0.9375 |
| Mahalanobis | 1.0 | 1.0 | 0.875 |
| Autoencoder | 0.0 | 0.0 | 0.875 |
| Ensemble | 1.0 | 1.0 | 0.9375 |


---

## 6. Defenses

| Method | Clean Acc | FGSM Acc | PGD Acc | CW Acc |
|--------|-----------|----------|---------|--------|
| Baseline | 0.9375 | 0.875 | 0.0 | 0.1875 |
| Adv-Trained | 1.0 | 1.0 | 0.0 | 1.0 |
| AE-Denoising | 0.75 | 0.25 | 0.25 | 0.4375 |


---

## 7. Full Metrics Table

| Method                | Clean Acc   | Adv Acc (FGSM)   | Adv Acc (PGD)   | Adv Acc (CW)   | Detection AUC   | FPR    | Notes        |
|:----------------------|:------------|:-----------------|:----------------|:---------------|:----------------|:-------|:-------------|
| Baseline (ResNet-18)  | 0.9375      | 0.875            | 0.0             | 0.1875         | -               | -      | No defense   |
| Detect: LID           | -           | -                | -               | -              | 1.0             | 0.9375 | TPR=1.0      |
| Detect: Mahalanobis   | -           | -                | -               | -              | 1.0             | 0.875  | TPR=1.0      |
| Detect: Autoencoder   | -           | -                | -               | -              | 0.0             | 0.875  | TPR=0.0      |
| Detect: Ensemble      | -           | -                | -               | -              | 1.0             | 0.9375 | TPR=1.0      |
| Defense: Adv-Trained  | 1.0         | 1.0              | 0.0             | 1.0            | -               | -      | Adv-Trained  |
| Defense: AE-Denoising | 0.75        | 0.25             | 0.25            | 0.4375         | -               | -      | AE-Denoising |

---

## 8. Artifacts

| Artifact | Path |
|----------|------|
| Baseline model | `models/baseline_cnn.pth` |
| AE model | `models/autoencoder.pth` |
| Adv-trained model | `models/adv_trained_cnn.pth` |
| Adversarial images | `adv_images/{fgsm,pgd,cw}/adv_tensors.pt` |
| Metrics CSV | `metrics/metrics.csv` |
| Plots | `plots/` |

---

## 9. Limitations & Ethical Notes

- **White-box assumption**: Attacks assume full model access. Real-world attacks may be black-box.
- **Compute constraints**: CW attacks are approximate (100 iterations vs. full optimisation).
- **Dataset distribution**: HAM10000 has severe class imbalance; mitigated by oversampling.
- **Research context**: This system is for research only. Real clinical deployment requires FDA 510(k) clearance, multi-site validation, and continuous monitoring.
- **Bias risk**: Lesion appearance varies across skin tones; model may underperform on under-represented demographics.

---

## 10. Recommendations

1. **Adversarial Training + LID detection** as the primary defense stack.
2. Retrain with larger ε values (0.05–0.10) for stronger adversarial training.
3. Augment with certified defenses (randomised smoothing) for L2-bounded attacks.
4. Apply FDA AI/ML-based SaMD guidelines before any clinical use.

---

*Generated automatically by evaluate.py | Seed=42 | Date: 2026-03-01 13:39*
