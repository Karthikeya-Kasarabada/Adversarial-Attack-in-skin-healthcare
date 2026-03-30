"""
config.py – Central configuration for all project hyperparameters and paths.
"""

import os
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Directories ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"          # HAM10000 images land here
MODEL_DIR  = BASE_DIR / "models"
ADV_DIR    = BASE_DIR / "adv_images"
PLOT_DIR   = BASE_DIR / "plots"
METRIC_DIR = BASE_DIR / "metrics"

for _d in [RAW_DIR, MODEL_DIR, ADV_DIR, PLOT_DIR, METRIC_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
METADATA_CSV  = RAW_DIR / "HAM10000_metadata.csv"
IMAGE_DIRS    = [
    DATA_DIR / "demo_images",
    RAW_DIR / "HAM10000_images_part_1",
    RAW_DIR / "HAM10000_images_part_2",
]
CLASS_NAMES   = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES   = len(CLASS_NAMES)
IMG_SIZE      = 224
TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE    = 32
NUM_EPOCHS    = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 0   # Must be 0 for Streamlit Cloud to avoid /dev/shm OOM errors

# ImageNet normalisation statistics
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Attack Hyperparameters ────────────────────────────────────────────────────
FGSM_EPS        = 0.03
PGD_EPS         = 0.03
PGD_ALPHA       = 0.01
PGD_STEPS       = 40
CW_LR           = 0.01
CW_MAX_ITER     = 100
ADV_SAMPLES_PER_CLASS = 1000   # target; capped to available test images

# ── Detection Hyperparameters ─────────────────────────────────────────────────
LID_K           = 20           # k-nearest neighbours for LID
MAHAL_LAYER     = "avgpool"    # feature layer for Mahalanobis
AE_LATENT_DIM   = 128
AE_EPOCHS       = 30
AE_LR           = 1e-3
DETECT_TPR_TARGET = 0.95       # tune threshold so clean TPR ≥ 95 %

# ── Adversarial Training ──────────────────────────────────────────────────────
ADV_TRAIN_EPS   = 0.03
ADV_TRAIN_STEPS = 10
ADV_TRAIN_ALPHA = 0.01

# ── Paths ─────────────────────────────────────────────────────────────────────
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_cnn.pth"
AE_MODEL_PATH       = MODEL_DIR / "autoencoder.pth"
ADV_TRAIN_PATH      = MODEL_DIR / "adv_trained_cnn.pth"
METRICS_CSV         = METRIC_DIR / "metrics.csv"
