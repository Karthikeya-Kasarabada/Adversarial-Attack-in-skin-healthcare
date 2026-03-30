"""
pages/pg_attack.py – Live adversarial attack generation on an uploaded image.
"""

import sys
from pathlib import Path
import io

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils import get_val_transform, get_device, load_checkpoint, denormalize
from model import build_resnet18


@st.cache_resource
def _load_model():
    device = get_device()
    model  = build_resnet18().to(device)
    load_checkpoint(model, config.BASELINE_MODEL_PATH, device)
    model.eval()
    return model, device


def _preprocess(img: Image.Image) -> torch.Tensor:
    return get_val_transform()(img.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def _classify(model, tensor, device):
    out   = model(tensor.to(device))
    probs = torch.softmax(out, dim=1)[0].cpu().numpy()
    return int(probs.argmax()), probs


def _run_attack(model, tensor, label, atk_name, eps, alpha, steps, device):
    import torchattacks
    model.eval()
    tensor = tensor.to(device)
    lbl    = torch.tensor([label], device=device)

    if atk_name == "FGSM":
        atk = torchattacks.FGSM(model, eps=eps)
    elif atk_name == "PGD":
        atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    elif atk_name == "CW":
        atk = torchattacks.CW(model, c=1, kappa=0, steps=steps, lr=alpha)
    else:
        return tensor.cpu()

    adv = atk(tensor, lbl)
    return adv.cpu()


def _tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """Denormalize a (1,C,H,W) tensor and return H×W×3 numpy uint8."""
    rgb = denormalize(t)[0].permute(1, 2, 0).numpy()
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _diff_amplified(clean_rgb, adv_rgb, factor=10):
    diff = (adv_rgb.astype(float) - clean_rgb.astype(float))
    amp  = diff * factor + 128
    return np.clip(amp, 0, 255).astype(np.uint8)


def show():
    st.markdown("""
    <h2 style="font-size:1.6rem;font-weight:800;
               background:linear-gradient(135deg,#f472b6,#818cf8);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        ⚔️ Adversarial Attack Generator
    </h2>
    <p style="color:#64748b;font-size:.85rem;margin:.3rem 0 1.5rem">
        Generate adversarial examples in real-time. Pick an attack and tune parameters.
    </p>""", unsafe_allow_html=True)

    if not config.BASELINE_MODEL_PATH.exists():
        st.markdown("""<div class="result-box result-warning">
        ⚠️ <b>Baseline model not found.</b> Run <code>python main.py --phase 1</code> first.
        </div>""", unsafe_allow_html=True)
        return

    model, device = _load_model()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col, img_col = st.columns([1, 2])

    with ctrl_col:
        st.markdown('<div class="section-title">⚙️ Attack Settings</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"],
                                    help="Any skin lesion image")

        atk_name = st.selectbox("Attack Method", ["FGSM", "PGD", "CW"])

        eps = st.slider("ε (epsilon / perturbation budget)",
                        min_value=0.001, max_value=0.3,
                        value=0.03, step=0.001, format="%.3f")

        if atk_name in ("PGD", "CW"):
            alpha = st.slider("Step size (α / lr)",
                              min_value=0.001, max_value=0.05,
                              value=0.01, step=0.001, format="%.3f")
            steps = st.slider("Steps / Iterations",
                              min_value=5, max_value=100,
                              value=40 if atk_name == "PGD" else 20, step=5)
        else:
            alpha, steps = 0.01, 1

        run_btn = st.button("⚔️ Generate Adversarial Example", use_container_width=True)

    with img_col:
        if uploaded is None:
            st.markdown("""<div style="background:#1e2235;border:2px dashed #2d3561;
                border-radius:12px;padding:3rem;text-align:center;color:#475569">
                <div style="font-size:2.5rem">🖼️</div>
                <div style="margin-top:.5rem;font-size:.85rem">Upload an image to get started</div>
                </div>""", unsafe_allow_html=True)
            return

        orig_img = Image.open(uploaded).convert("RGB")
        tensor   = _preprocess(orig_img)

        # Clean prediction
        pred_clean, probs_clean = _classify(model, tensor, device)
        cls_clean = config.CLASS_NAMES[pred_clean]

        if run_btn or "adv_tensor" in st.session_state:
            if run_btn:
                with st.spinner(f"Running {atk_name} attack…"):
                    adv_t = _run_attack(model, tensor, pred_clean,
                                        atk_name, eps, alpha, steps, device)
                st.session_state["adv_tensor"] = adv_t
                st.session_state["clean_pred"] = pred_clean
            else:
                adv_t = st.session_state["adv_tensor"]

            pred_adv, probs_adv = _classify(model, adv_t, device)
            cls_adv   = config.CLASS_NAMES[pred_adv]
            success   = pred_adv != pred_clean

            # Linf perturbation
            linf = (adv_t - tensor).abs().max().item()

            # ── 4-panel visualisation ─────────────────────────────────────────
            clean_rgb = _tensor_to_rgb(tensor)
            adv_rgb   = _tensor_to_rgb(adv_t)
            diff_rgb  = _diff_amplified(clean_rgb, adv_rgb, factor=15)

            fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), facecolor="#1e2235")
            for ax in axes:
                ax.set_facecolor("#1e2235"); ax.axis("off")

            axes[0].imshow(clean_rgb)
            axes[0].set_title(f"Clean\n{cls_clean} ({probs_clean[pred_clean]:.1%})",
                              color="#4ade80", fontsize=9, fontweight="bold")

            axes[1].imshow(adv_rgb)
            axes[1].set_title(
                f"{atk_name} Adversarial\n{cls_adv} ({probs_adv[pred_adv]:.1%})",
                color="#f87171" if success else "#fbbf24", fontsize=9, fontweight="bold"
            )

            axes[2].imshow(diff_rgb)
            axes[2].set_title(f"Perturbation ×15\nL∞ = {linf:.4f}",
                              color="#94a3b8", fontsize=9)

            plt.tight_layout(pad=1)
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # ── Status banner ─────────────────────────────────────────────────
            if success:
                st.markdown(f"""<div class="result-box result-danger">
                🚨 <strong>Attack Successful!</strong> Model fooled: 
                <code>{cls_clean}</code> → <code>{cls_adv}</code> &nbsp;|&nbsp; 
                L∞ = {linf:.4f} &nbsp;|&nbsp; ε = {eps:.3f}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-box result-safe">
                ✅ <strong>Attack Failed.</strong> Model still predicts <code>{cls_clean}</code> 
                (conf={probs_adv[pred_clean]:.1%}) &nbsp;|&nbsp; L∞ = {linf:.4f}
                </div>""", unsafe_allow_html=True)

            # ── Side-by-side probability bars ─────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability Shift</div>',
                        unsafe_allow_html=True)

            fig2, axes2 = plt.subplots(1, 2, figsize=(11, 3), facecolor="#1e2235",
                                       sharey=True)
            for ax in axes2:
                ax.set_facecolor("#252840")
                for sp in ax.spines.values(): sp.set_edgecolor("#2d3561")

            palette_clean = ["#34d399" if i == pred_clean else "#2d3561"
                             for i in range(config.NUM_CLASSES)]
            palette_adv   = ["#f87171"  if i == pred_adv   else "#2d3561"
                             for i in range(config.NUM_CLASSES)]

            axes2[0].bar(config.CLASS_NAMES, probs_clean, color=palette_clean, width=0.55)
            axes2[0].set_ylim(0, 1)
            axes2[0].set_title("Clean Probabilities", color="#94a3b8", fontsize=9)
            axes2[0].tick_params(colors="#94a3b8", labelsize=7)
            axes2[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y,_: f"{y:.0%}"))

            axes2[1].bar(config.CLASS_NAMES, probs_adv, color=palette_adv, width=0.55)
            axes2[1].set_ylim(0, 1)
            axes2[1].set_title(f"{atk_name} Adversarial Probabilities", color="#94a3b8", fontsize=9)
            axes2[1].tick_params(colors="#94a3b8", labelsize=7)

            plt.tight_layout(pad=1.2)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

            # Download adversarial image
            adv_pil = Image.fromarray(adv_rgb)
            buf = io.BytesIO()
            adv_pil.save(buf, format="PNG")
            st.download_button(
                "💾 Download Adversarial Image",
                data=buf.getvalue(),
                file_name=f"adversarial_{atk_name.lower()}_eps{eps:.3f}.png",
                mime="image/png",
            )
        else:
            # Just show clean image
            fig, ax = plt.subplots(figsize=(4, 4), facecolor="#1e2235")
            ax.set_facecolor("#1e2235"); ax.axis("off")
            clean_rgb = _tensor_to_rgb(tensor)
            ax.imshow(clean_rgb)
            ax.set_title(f"Clean: {cls_clean} ({probs_clean[pred_clean]:.1%})",
                         color="#4ade80", fontsize=10, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()
            st.info("Adjust the settings on the left and click ⚔️ Generate Adversarial Example")
