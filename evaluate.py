"""
evaluate.py - Phase 4: Generate the full metrics table and summary report.

Usage:
    python evaluate.py

Reads saved JSON metrics from all previous phases and produces:
    metrics/metrics.csv       - summary table
    plots/metrics_heatmap.png - styled heatmap
    REPORT.md                 - human-readable report
"""

import json
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import config


# ─────────────────────────────────────────────────────────────────────────────
# Load all saved metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def build_metrics_table() -> pd.DataFrame:
    baseline  = load_json(config.METRIC_DIR / "baseline_metrics.json")
    attack    = load_json(config.METRIC_DIR / "attack_metrics.json")
    detection = load_json(config.METRIC_DIR / "detection_metrics.json")
    defense   = load_json(config.METRIC_DIR / "defense_metrics.json")

    rows = []

    # ── Baseline ─────────────────────────────────────────────────────────────
    base_clean = defense.get("Baseline", {}).get("clean_acc",
                  baseline.get("clean_accuracy", "-"))
    rows.append({
        "Method":         "Baseline (ResNet-18)",
        "Clean Acc":      base_clean,
        "Adv Acc (FGSM)": defense.get("Baseline", {}).get("fgsm_acc", "-"),
        "Adv Acc (PGD)":  defense.get("Baseline", {}).get("pgd_acc",  "-"),
        "Adv Acc (CW)":   defense.get("Baseline", {}).get("cw_acc",   "-"),
        "Detection AUC":  "-",
        "FPR":            "-",
        "Notes":          "No defense",
    })

    # ── Detectors ────────────────────────────────────────────────────────────
    for det_name, det_res in detection.items():
        rows.append({
            "Method":         f"Detect: {det_name}",
            "Clean Acc":      "-",
            "Adv Acc (FGSM)": "-",
            "Adv Acc (PGD)":  "-",
            "Adv Acc (CW)":   "-",
            "Detection AUC":  det_res.get("AUC", "-"),
            "FPR":            det_res.get("FPR", "-"),
            "Notes":          f"TPR={det_res.get('TPR','-')}",
        })

    # ── Defenses ─────────────────────────────────────────────────────────────
    for def_name, def_res in defense.items():
        if def_name == "Baseline":
            continue
        rows.append({
            "Method":         f"Defense: {def_name}",
            "Clean Acc":      def_res.get("clean_acc", "-"),
            "Adv Acc (FGSM)": def_res.get("fgsm_acc",  "-"),
            "Adv Acc (PGD)":  def_res.get("pgd_acc",   "-"),
            "Adv Acc (CW)":   def_res.get("cw_acc",    "-"),
            "Detection AUC":  "-",
            "FPR":            "-",
            "Notes":          def_name,
        })

    df = pd.DataFrame(rows)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_heatmap(df: pd.DataFrame, save_path: Path):
    numeric_cols = ["Clean Acc", "Adv Acc (FGSM)", "Adv Acc (PGD)",
                    "Adv Acc (CW)", "Detection AUC", "FPR"]
    plot_df = df.set_index("Method")[numeric_cols].replace("-", np.nan)
    plot_df = plot_df.astype(float)

    fig, ax = plt.subplots(figsize=(14, max(5, len(plot_df) * 0.7)))
    sns.heatmap(
        plot_df, annot=True, fmt=".3f", cmap="YlGnBu",
        linewidths=0.5, linecolor="gray",
        cbar_kws={"label": "Score"},
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title("Overall Metrics Heatmap", fontsize=15, pad=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Saved heatmap -> {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report generator
# ─────────────────────────────────────────────────────────────────────────────

REPORT_TEMPLATE = """# Adversarial Attacks in Healthcare AI: Vulnerabilities and Detection Mechanisms
## Project Report

---

## 1. Executive Summary

This project demonstrates end-to-end adversarial robustness evaluation on the
**HAM10000 skin lesion dataset** (10,015 images, 7 classes).  Key findings:

{summary_bullets}

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

{baseline_table}

---

## 4. Adversarial Attacks

| Attack | ε | Steps | ASR | L∞ | SSIM |
|--------|---|-------|-----|----|------|
{attack_rows}

---

## 5. Detection Mechanisms

| Detector | AUC | TPR | FPR |
|----------|-----|-----|-----|
{detect_rows}

---

## 6. Defenses

| Method | Clean Acc | FGSM Acc | PGD Acc | CW Acc |
|--------|-----------|----------|---------|--------|
{defense_rows}

---

## 7. Full Metrics Table

{full_table}

---

## 8. Artifacts

| Artifact | Path |
|----------|------|
| Baseline model | `models/baseline_cnn.pth` |
| AE model | `models/autoencoder.pth` |
| Adv-trained model | `models/adv_trained_cnn.pth` |
| Adversarial images | `adv_images/{{fgsm,pgd,cw}}/adv_tensors.pt` |
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

*Generated automatically by evaluate.py | Seed={seed} | Date: {date}*
"""


def build_report(df: pd.DataFrame,
                 attack_m: dict, detect_m: dict, defense_m: dict,
                 baseline_m: dict) -> str:

    # Summary bullets
    base_clean = defense_m.get("Baseline", {}).get("clean_acc", "?")
    base_pgd   = defense_m.get("Baseline", {}).get("pgd_acc",  "?")
    best_det   = max(detect_m or {}, key=lambda k: detect_m[k].get("AUC", 0), default="LID")
    best_auc   = detect_m.get(best_det, {}).get("AUC", "?")
    adv_clean  = defense_m.get("Adv-Trained", {}).get("clean_acc", "?")
    adv_pgd    = defense_m.get("Adv-Trained", {}).get("pgd_acc",   "?")

    summary = f"""\
- **Baseline clean accuracy**: {base_clean} (target: >0.90)
- **Attack Success Rate (PGD vs baseline)**: {1 - float(base_pgd) if base_pgd not in ('?','-') else '?':.2%}  
- **Best detector**: {best_det} with AUC={best_auc}
- **Adversarial training**: clean acc={adv_clean}, PGD acc={adv_pgd}  
- **Recommendation**: deploy adversarial training + {best_det} ensemble detection
"""

    # Attack rows
    attack_rows = ""
    for aname, am in attack_m.items():
        attack_rows += (f"| {aname.upper()} | {config.FGSM_EPS if 'fgsm' in aname else config.PGD_EPS} "
                        f"| {am.get('steps', '-')} | {am.get('asr', '-')} "
                        f"| {am.get('linf_mean', '-')} | {am.get('ssim', '-')} |\n")

    # Detect rows
    detect_rows = ""
    for dname, dm in detect_m.items():
        detect_rows += f"| {dname} | {dm.get('AUC','-')} | {dm.get('TPR','-')} | {dm.get('FPR','-')} |\n"

    # Defense rows
    defense_rows = ""
    for dname, dm in defense_m.items():
        defense_rows += (f"| {dname} | {dm.get('clean_acc','-')} | {dm.get('fgsm_acc','-')} "
                         f"| {dm.get('pgd_acc','-')} | {dm.get('cw_acc','-')} |\n")

    baseline_table = (f"| Clean Accuracy | {baseline_m.get('clean_accuracy','-')} |\n"
                      f"| Weighted F1    | {baseline_m.get('weighted_f1','-')} |")

    full_table = df.to_markdown(index=False)

    import datetime
    report = REPORT_TEMPLATE.format(
        summary_bullets=summary,
        baseline_table=baseline_table,
        attack_rows=attack_rows or "| - | - | - | - | - | - |",
        detect_rows=detect_rows or "| - | - | - | - |",
        defense_rows=defense_rows or "| - | - | - | - | - |",
        full_table=full_table,
        seed=config.SEED,
        date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    baseline_m = load_json(config.METRIC_DIR / "baseline_metrics.json")
    attack_m   = load_json(config.METRIC_DIR / "attack_metrics.json")
    detect_m   = load_json(config.METRIC_DIR / "detection_metrics.json")
    defense_m  = load_json(config.METRIC_DIR / "defense_metrics.json")

    # Build summary table
    df = build_metrics_table()
    print("\n" + df.to_string(index=False))

    # Save CSV
    df.to_csv(config.METRICS_CSV, index=False)
    print(f"\n[evaluate] Metrics CSV -> {config.METRICS_CSV}")

    # Heatmap
    plot_metrics_heatmap(df, config.PLOT_DIR / "metrics_heatmap.png")

    # Report
    report = build_report(df, attack_m, detect_m, defense_m, baseline_m)
    report_path = Path(__file__).parent / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[evaluate] Report -> {report_path}")


if __name__ == "__main__":
    main()
