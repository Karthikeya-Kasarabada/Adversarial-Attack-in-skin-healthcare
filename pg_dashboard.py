"""
pages/pg_dashboard.py – Dashboard overview page.
"""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load(path: Path) -> dict:
    if path.exists():
        return json.load(open(path))
    return {}


def _metric_card(label: str, value: str, sub: str = "", color: str = "#818cf8"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="background:linear-gradient(135deg,{color},#38bdf8);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent">
             {value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def show():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style="margin:0;font-size:2rem;font-weight:800;
               background:linear-gradient(135deg,#818cf8,#38bdf8,#34d399);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent">
        Adversarial Attacks in Healthcare AI
    </h1>
    <p style="color:#64748b;margin:.3rem 0 1.5rem;font-size:.9rem">
        HAM10000 Skin Lesion Classification · Adversarial Robustness Benchmark
    </p>""", unsafe_allow_html=True)

    # ── Status banner ─────────────────────────────────────────────────────────
    has_baseline  = config.BASELINE_MODEL_PATH.exists()
    has_adv       = (config.ADV_DIR / "pgd" / "adv_tensors.pt").exists()
    has_ae        = config.AE_MODEL_PATH.exists()
    has_adv_train = config.ADV_TRAIN_PATH.exists()

    phases = [
        ("Phase 1: Baseline",     has_baseline),
        ("Phase 2: Attacks",      has_adv),
        ("Phase 3: Detectors",    (config.METRIC_DIR / "detection_metrics.json").exists()),
        ("Phase 4: Defenses",     has_adv_train),
        ("Phase 5: Report",       (config.METRICS_CSV).exists()),
    ]

    cols = st.columns(5)
    for col, (label, done) in zip(cols, phases):
        icon  = "✅" if done else "⏳"
        color = "#22c55e" if done else "#f59e0b"
        col.markdown(f"""
        <div style="background:#1e2235;border:1px solid #2d3561;border-radius:12px;
                    padding:.8rem .6rem;text-align:center">
            <div style="font-size:1.3rem">{icon}</div>
            <div style="font-size:.7rem;color:{color};font-weight:600;margin-top:.3rem">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Metrics Row ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

    bm  = _load(config.METRIC_DIR / "baseline_metrics.json")
    am  = _load(config.METRIC_DIR / "attack_metrics.json")
    dm  = _load(config.METRIC_DIR / "detection_metrics.json")
    dfm = _load(config.METRIC_DIR / "defense_metrics.json")

    clean_acc   = bm.get("clean_accuracy", "—")
    f1          = bm.get("weighted_f1", "—")
    pgd_asr     = am.get("pgd", {}).get("asr", "—")
    best_det    = max(dm, key=lambda k: dm[k].get("AUC", 0)) if dm else "—"
    best_auc    = dm.get(best_det, {}).get("AUC", "—") if dm else "—"
    adv_pgd_acc = dfm.get("Adv-Trained", {}).get("pgd_acc", "—")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: _metric_card("Clean Accuracy",
                           f"{float(clean_acc):.1%}" if clean_acc != "—" else "—",
                           "Baseline ResNet-18", "#818cf8")
    with c2: _metric_card("Weighted F1",
                           f"{float(f1):.3f}" if f1 != "—" else "—",
                           "Clean test set", "#38bdf8")
    with c3: _metric_card("PGD ASR",
                           f"{float(pgd_asr):.1%}" if pgd_asr != "—" else "—",
                           "Attack success rate", "#f472b6")
    with c4: _metric_card("Best Det. AUC",
                           f"{float(best_auc):.3f}" if best_auc != "—" else "—",
                           f"{best_det} detector", "#34d399")
    with c5: _metric_card("Adv-Train PGD",
                           f"{float(adv_pgd_acc):.1%}" if adv_pgd_acc != "—" else "—",
                           "After defense", "#fb923c")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column area ───────────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="section-title">Attack Success Rates</div>', unsafe_allow_html=True)
        if am:
            attacks = list(am.keys())
            asrs    = [am[a].get("asr", 0) for a in attacks]
            ssims   = [am[a].get("ssim", 0) for a in attacks]

            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), facecolor="#1e2235")
            for ax in axes:
                ax.set_facecolor("#252840")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#2d3561")

            colors = ["#818cf8", "#38bdf8", "#f472b6"]
            bars = axes[0].bar(attacks, asrs, color=colors, width=0.5)
            axes[0].set_ylim(0, 1)
            axes[0].set_ylabel("ASR", color="#94a3b8", fontsize=9)
            axes[0].set_title("Attack Success Rate (↑ worse)", color="#c8d0e0", fontsize=9)
            axes[0].tick_params(colors="#94a3b8", labelsize=8)
            axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))
            for bar, v in zip(bars, asrs):
                axes[0].text(bar.get_x()+bar.get_width()/2, v+.01,
                             f"{v:.1%}", ha="center", color="white", fontsize=8, fontweight="bold")

            bars2 = axes[1].bar(attacks, ssims, color=colors, width=0.5)
            axes[1].set_ylim(0, 1)
            axes[1].set_ylabel("SSIM", color="#94a3b8", fontsize=9)
            axes[1].set_title("Perceptual Similarity (↑ more invisible)", color="#c8d0e0", fontsize=9)
            axes[1].tick_params(colors="#94a3b8", labelsize=8)
            for bar, v in zip(bars2, ssims):
                axes[1].text(bar.get_x()+bar.get_width()/2, v+.01,
                             f"{v:.3f}", ha="center", color="white", fontsize=8, fontweight="bold")

            plt.tight_layout(pad=1.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown("""<div class="result-box result-warning">
            ⏳ Attack metrics not yet generated.<br>
            Run <code>python main.py --phase 2</code> first.
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">Detection Performance</div>', unsafe_allow_html=True)
        if dm:
            det_names = list(dm.keys())
            aucs      = [dm[d].get("AUC", 0) for d in det_names]
            fprs      = [dm[d].get("FPR", 0) for d in det_names]

            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1e2235")
            ax.set_facecolor("#252840")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")

            x      = np.arange(len(det_names))
            width  = 0.35
            colors_auc = ["#34d399", "#22c55e", "#86efac", "#4ade80"]
            colors_fpr = ["#f472b6", "#ec4899", "#f9a8d4", "#fbcfe8"]

            b1 = ax.bar(x - width/2, aucs, width, label="AUC", color=colors_auc[:len(det_names)])
            b2 = ax.bar(x + width/2, fprs, width, label="FPR", color=colors_fpr[:len(det_names)])
            ax.set_ylim(0, 1.1)
            ax.set_xticks(x); ax.set_xticklabels(det_names, color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.legend(fontsize=8, facecolor="#2d3561", edgecolor="#2d3561", labelcolor="white")
            ax.set_title("AUC vs FPR per Detector", color="#c8d0e0", fontsize=9)

            for bar, v in zip(b1.patches, aucs):
                ax.text(bar.get_x()+bar.get_width()/2, v+.01,
                        f"{v:.2f}", ha="center", color="white", fontsize=7, fontweight="bold")

            plt.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown("""<div class="result-box result-warning">
            ⏳ Detection metrics not yet generated.<br>
            Run <code>python main.py --phase 3</code> first.
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    class_info = {
        "nv":    ("Melanocytic Nevi",          "6,705", "#818cf8"),
        "mel":   ("Melanoma",                  "1,113", "#f472b6"),
        "bkl":   ("Benign Keratosis",          "1,099", "#38bdf8"),
        "bcc":   ("Basal Cell Carcinoma",        "514", "#fb923c"),
        "akiec": ("Actinic Keratoses",           "327", "#34d399"),
        "vasc":  ("Vascular Lesions",            "142", "#a78bfa"),
        "df":    ("Dermatofibroma",              "115", "#fbbf24"),
    }

    # Class distribution mini chart
    with col_a:
        fig, ax = plt.subplots(figsize=(4, 3.2), facecolor="#1e2235")
        ax.set_facecolor("#1e2235")
        counts = [int(v[1].replace(",","")) for v in class_info.values()]
        colors = [v[2] for v in class_info.values()]
        wedges, texts, autotexts = ax.pie(
            counts, labels=list(class_info.keys()),
            colors=colors, autopct="%1.1f%%", startangle=90,
            textprops={"color": "#94a3b8", "fontsize": 7},
            pctdistance=0.8,
        )
        for at in autotexts:
            at.set_fontsize(6); at.set_color("white")
        ax.set_title("Class Distribution", color="#c8d0e0", fontsize=9, pad=6)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col_b:
        df_info = pd.DataFrame([
            {"Class": k, "Full Name": v[0], "Count": v[1]}
            for k, v in class_info.items()
        ])
        st.dataframe(
            df_info,
            use_container_width=True,
            hide_index=True,
            height=230,
        )

    with col_c:
        st.markdown("""
        <div style="background:#1e2235;border:1px solid #2d3561;border-radius:12px;
                    padding:1.2rem;height:100%">
            <div style="color:#94a3b8;font-size:.78rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:.06em;margin-bottom:.8rem">
                Configuration
            </div>
            <table style="width:100%;font-size:.78rem;border-collapse:collapse">
        """, unsafe_allow_html=True)

        rows = [
            ("Images",     "10,015"),
            ("Classes",    "7"),
            ("Input Size", "224 × 224 px"),
            ("Split",      "70 / 15 / 15"),
            ("Normalise",  "ImageNet μ/σ"),
            ("FGSM ε",     "0.03"),
            ("PGD steps",  "40"),
            ("CW iters",   "100"),
            ("LID k",      "20"),
            ("Seed",       "42"),
        ]
        tbl = "".join(
            f'<tr><td style="color:#64748b;padding:.28rem 0">{k}</td>'
            f'<td style="color:#c8d0e0;text-align:right;font-weight:600">{v}</td></tr>'
            for k, v in rows
        )
        st.markdown(tbl + "</table></div>", unsafe_allow_html=True)

    # ── Existing plots gallery ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">📸 Generated Plots</div>', unsafe_allow_html=True)

    plot_files = sorted(config.PLOT_DIR.glob("*.png"))
    if plot_files:
        cols = st.columns(3)
        for i, p in enumerate(plot_files):
            with cols[i % 3]:
                st.image(str(p), caption=p.stem.replace("_", " ").title(),
                         use_container_width=True)
    else:
        st.markdown("""<div class="result-box result-info">
        ℹ️  No plots generated yet. Run the pipeline phases to generate visualisations.
        </div>""", unsafe_allow_html=True)
