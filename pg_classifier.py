"""
pages/pg_classifier.py – Upload an image and classify it.
"""

import sys
from pathlib import Path
import io

import streamlit as st
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from utils import get_val_transform, get_device, load_checkpoint, denormalize
from model import build_resnet18


# ── Cached model loader ───────────────────────────────────────────────────────

@st.cache_resource
def _load_model():
    device = get_device()
    model  = build_resnet18().to(device)
    ckpt   = load_checkpoint(model, config.BASELINE_MODEL_PATH, device)
    model.eval()
    return model, device


def _preprocess(img: Image.Image) -> torch.Tensor:
    tf = get_val_transform()
    return tf(img.convert("RGB")).unsqueeze(0)   # (1, C, H, W)


@torch.no_grad()
def _predict(model, tensor: torch.Tensor, device: torch.device):
    tensor  = tensor.to(device)
    logits  = model(tensor)
    probs   = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred    = int(probs.argmax())
    return pred, probs


def _grad_cam(model, tensor: torch.Tensor, device: torch.device):
    """Simple GradCAM on layer4 of ResNet-18."""
    tensor = tensor.to(device).requires_grad_(False)
    grads, activations = [], []

    def fwd_hook(mod, inp, out):
        activations.append(out)

    def bwd_hook(mod, gin, gout):
        grads.append(gout[0])

    layer = model.layer4[-1]
    fh = layer.register_forward_hook(fwd_hook)
    bh = layer.register_full_backward_hook(bwd_hook)

    out    = model(tensor)
    pred   = out.argmax(1)
    model.zero_grad()
    out[0, pred].backward()

    fh.remove(); bh.remove()

    grad  = grads[0].mean(dim=(2, 3), keepdim=True)
    cam   = (grad * activations[0]).sum(dim=1, keepdim=True)
    cam   = torch.relu(cam)
    cam   = cam / (cam.max() + 1e-8)

    # Upsample to 224×224
    cam_np = cam[0, 0].cpu().numpy()
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(
        (config.IMG_SIZE, config.IMG_SIZE), Image.BILINEAR
    )
    return np.array(cam_img) / 255.0


def _overlay_cam(rgb: np.ndarray, cam: np.ndarray) -> np.ndarray:
    palette = cm.get_cmap("jet")(cam)[:, :, :3]
    return np.clip(0.55 * rgb + 0.45 * palette, 0, 1)


# ── Class descriptions ────────────────────────────────────────────────────────

CLASS_INFO = {
    "akiec": ("Actinic Keratoses", "Pre-cancerous lesion caused by UV damage. Can progress to SCC.", "#fb923c"),
    "bcc":   ("Basal Cell Carcinoma", "Most common form of skin cancer. Arises from basal cells.", "#f472b6"),
    "bkl":   ("Benign Keratosis", "Non-cancerous skin lesion. Includes seborrheic keratosis.", "#38bdf8"),
    "df":    ("Dermatofibroma", "Benign skin nodule. Usually harmless and doesn't need treatment.", "#34d399"),
    "mel":   ("Melanoma", "⚠️ Dangerous form of skin cancer. Early detection is critical.", "#ef4444"),
    "nv":    ("Melanocytic Nevi", "Common benign moles. Usually harmless.", "#818cf8"),
    "vasc":  ("Vascular Lesions", "Lesions from blood vessels e.g. angiomas, angiokeratomas.", "#a78bfa"),
}


def show():
    st.markdown("""
    <h2 style="font-size:1.6rem;font-weight:800;
               background:linear-gradient(135deg,#818cf8,#38bdf8);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0">
        🔬 Skin Lesion Classifier
    </h2>
    <p style="color:#64748b;font-size:.85rem;margin:.3rem 0 1.5rem">
        Upload a dermatoscopic image to classify and visualise model attention
    </p>""", unsafe_allow_html=True)

    # ── Model status ──────────────────────────────────────────────────────────
    if not config.BASELINE_MODEL_PATH.exists():
        st.markdown("""<div class="result-box result-warning">
        ⚠️ <b>Baseline model not found.</b><br>
        Run <code>python main.py --phase 1</code> to train it first, then reload the app.
        </div>""", unsafe_allow_html=True)
        return

    model, device = _load_model()

    # ── Upload ────────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.5])

    with left:
        uploaded = st.file_uploader(
            "Upload a skin lesion image",
            type=["jpg", "jpeg", "png"],
            help="Best results with dermoscopic images similar to HAM10000",
        )

        st.markdown("---")
        st.markdown("<div style='color:#64748b;font-size:.8rem;font-weight:600'>USE A SAMPLE CLASS</div>", unsafe_allow_html=True)
        sample_cls = st.selectbox("", config.CLASS_NAMES, label_visibility="collapsed")

        if st.button("Load sample image from dataset"):
            if config.METADATA_CSV.exists():
                import pandas as pd
                meta = pd.read_csv(config.METADATA_CSV)
                from utils import build_dataframe
                df = build_dataframe()
                filtered_df = df[df["label"] == sample_cls]
                if not filtered_df.empty:
                    row = filtered_df.iloc[0]
                    st.session_state["sample_path"] = row["path"]
                else:
                    st.warning(f"No sample image found for class '{sample_cls}' in the downloaded dataset.")
            else:
                st.warning("Dataset not downloaded yet.")

    with right:
        img_to_use: Image.Image | None = None

        if uploaded:
            img_to_use = Image.open(uploaded).convert("RGB")
            st.image(img_to_use, caption="Uploaded Image", use_container_width=True)
        elif "sample_path" in st.session_state:
            img_to_use = Image.open(st.session_state["sample_path"]).convert("RGB")
            st.image(img_to_use, caption="Sample Image", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if img_to_use is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)

        tensor      = _preprocess(img_to_use)
        pred_idx, probs = _predict(model, tensor, device)
        pred_cls    = config.CLASS_NAMES[pred_idx]
        full_name, desc, color = CLASS_INFO[pred_cls]

        # Main prediction banner
        is_cancer = pred_cls in ("mel", "bcc", "akiec")
        box_cls   = "result-danger" if is_cancer else "result-safe"
        icon      = "🚨" if is_cancer else "✅"
        st.markdown(f"""
        <div class="result-box {box_cls}">
            <span style="font-size:1.3rem">{icon}</span>
            <strong style="font-size:1.1rem">&nbsp;&nbsp;{full_name} ({pred_cls.upper()})</strong><br>
            <span style="font-size:.85rem;opacity:.85">{desc}</span><br>
            <span style="font-size:.8rem;margin-top:.3rem;display:block">
            Confidence: <strong>{probs[pred_idx]:.2%}</strong>
            </span>
        </div>""", unsafe_allow_html=True)

        # Probability bar chart + GradCAM
        col_prob, col_cam = st.columns(2)

        with col_prob:
            sorted_idx = np.argsort(probs)[::-1]
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1e2235")
            ax.set_facecolor("#252840")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d3561")

            bar_colors = ["#818cf8" if i == pred_idx else "#2d3561" for i in sorted_idx]
            bars = ax.barh(
                [config.CLASS_NAMES[i] for i in sorted_idx],
                [probs[i] for i in sorted_idx],
                color=bar_colors, height=0.6,
            )
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probability", color="#94a3b8", fontsize=9)
            ax.set_title("Class Probabilities", color="#c8d0e0", fontsize=10)
            ax.tick_params(colors="#94a3b8", labelsize=8)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            for bar, prob in zip(bars, [probs[i] for i in sorted_idx]):
                ax.text(min(prob + .01, .95), bar.get_y() + bar.get_height()/2,
                        f"{prob:.2%}", va="center", color="white", fontsize=7, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col_cam:
            try:
                tensor_grad = _preprocess(img_to_use).clone()
                cam         = _grad_cam(model, tensor_grad, device)
                rgb_np      = np.array(img_to_use.resize(
                    (config.IMG_SIZE, config.IMG_SIZE))) / 255.0
                overlay     = _overlay_cam(rgb_np, cam)

                fig, axes = plt.subplots(1, 2, figsize=(6, 3), facecolor="#1e2235")
                for ax in axes:
                    ax.set_facecolor("#1e2235")
                    ax.axis("off")
                axes[0].imshow(np.clip(rgb_np, 0, 1))
                axes[0].set_title("Original", color="#94a3b8", fontsize=8)
                axes[1].imshow(overlay)
                axes[1].set_title("GradCAM Attention", color="#94a3b8", fontsize=8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()
            except Exception as e:
                st.info(f"GradCAM: {e}")

        # Disclaimer
        st.markdown("""<div class="result-box result-info" style="font-size:.78rem;margin-top:.5rem">
        ⚠️ <b>Research Only</b> – This prediction is generated by an AI research model and 
        must NOT be used for clinical decision-making. Consult a licensed dermatologist.
        </div>""", unsafe_allow_html=True)
