"""
utils.py – Data loading, preprocessing, augmentation, and misc helpers.
"""

import os, random, shutil
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

import config


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build image-path → label mapping from metadata CSV
# ─────────────────────────────────────────────────────────────────────────────

def _find_image(image_id: str) -> Path | None:
    """Search across both image folders."""
    for d in config.IMAGE_DIRS:
        p = Path(d) / f"{image_id}.jpg"
        if p.exists():
            return p
    return None


def build_dataframe() -> pd.DataFrame:
    """Return a DataFrame with columns [image_id, path, label, label_idx]."""
    meta = pd.read_csv(config.METADATA_CSV)
    label_map = {name: idx for idx, name in enumerate(config.CLASS_NAMES)}

    rows = []
    for _, row in meta.iterrows():
        p = _find_image(row["image_id"])
        if p is not None:
            rows.append({
                "image_id":  row["image_id"],
                "path":      str(p),
                "label":     row["dx"],
                "label_idx": label_map[row["dx"]],
            })

    df = pd.DataFrame(rows)
    print(f"[utils] Found {len(df)} images across {len(df['label'].unique())} classes.")
    return df


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 split. Fallback to non-stratified if counts are too low."""
    # Check if we can stratify
    label_counts = df["label_idx"].value_counts()
    if label_counts.min() < 2:
        print("[utils] WARNING: Some classes have < 2 samples. Disabling stratification.")
        stratify_col = None
    else:
        stratify_col = df["label_idx"]

    train_df, tmp_df = train_test_split(
        df, test_size=1 - config.TRAIN_SPLIT,
        stratify=stratify_col, random_state=config.SEED
    )

    # Re-check for the second split
    label_counts_tmp = tmp_df["label_idx"].value_counts()
    if label_counts_tmp.min() < 2:
        stratify_col_tmp = None
    else:
        stratify_col_tmp = tmp_df["label_idx"]

    val_df, test_df = train_test_split(
        tmp_df, test_size=config.TEST_SPLIT / (config.VAL_SPLIT + config.TEST_SPLIT),
        stratify=stratify_col_tmp, random_state=config.SEED
    )
    print(f"[utils] Split -> train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_idx"])


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Transforms
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DataLoaders with oversampling to handle class imbalance
# ─────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(df: pd.DataFrame) -> WeightedRandomSampler:
    """Safely build a sampler even if some classes are missing."""
    label_counts   = df["label_idx"].value_counts()
    
    # Map label indices to their weights
    # Weight = 1.0 / count
    weights_map = {idx: 1.0 / count for idx, count in label_counts.items()}
    
    # Assign weight to each sample
    sample_weights = df["label_idx"].map(weights_map).values
    
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_ds = HAM10000Dataset(train_df, get_train_transform())
    val_ds   = HAM10000Dataset(val_df,   get_val_transform())
    test_ds  = HAM10000Dataset(test_df,  get_val_transform())

    sampler = make_weighted_sampler(train_df)

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE,
        sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def denormalize(tensor: torch.Tensor,
                mean=config.MEAN, std=config.STD) -> torch.Tensor:
    """Undo ImageNet normalisation for visualisation."""
    mean_t = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std_t  = torch.tensor(std,  device=tensor.device).view(1, 3, 1, 1)
    return (tensor * std_t + mean_t).clamp(0, 1)


def ssim_batch(clean: torch.Tensor, adv: torch.Tensor) -> float:
    """Average SSIM over a batch (both already in [0,1] space)."""
    try:
        from pytorch_msssim import ssim
        return ssim(clean, adv, data_range=1.0, size_average=True).item()
    except ImportError:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Feature extraction hook
# ─────────────────────────────────────────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """Wraps a ResNet-18 and returns the avgpool 512-d feature vector."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.features = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        return out.flatten(1)          # (B, 512)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Misc
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(state: dict, path: Path):
    torch.save(state, path)
    print(f"[utils] Checkpoint saved -> {path}")


def load_checkpoint(model: nn.Module, path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt
