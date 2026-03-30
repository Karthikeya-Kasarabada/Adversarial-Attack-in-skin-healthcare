"""
model.py – ResNet-18 baseline and Autoencoder definitions.
"""

import torch
import torch.nn as nn
from torchvision import models

import config


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Baseline Classifier – ResNet-18
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int = config.NUM_CLASSES,
                   pretrained: bool = True) -> nn.Module:
    """
    ResNet-18 with ImageNet pre-training.
    Replace final FC layer for `num_classes` output.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model   = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Autoencoder – for anomaly detection (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder operating on 224×224 RGB images.
    Latent dim is spatially 7×7 × 256 (≈ 12 k floats).
    """

    def __init__(self):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            # 3 × 224 × 224 → 64 × 112 × 112
            nn.Conv2d(3, 64, 4, 2, 1),  nn.BatchNorm2d(64),  nn.ReLU(True),
            # 64 × 112 × 112 → 128 × 56 × 56
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            # 128 × 56 × 56 → 256 × 28 × 28
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            # 256 × 28 × 28 → 256 × 14 × 14
            nn.Conv2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            # 256 × 14 × 14 → 128 × 7 × 7
            nn.Conv2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
        )

        # ── Decoder ──────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            # 128 × 7 × 7 → 256 × 14 × 14
            nn.ConvTranspose2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            # 256 × 14 × 14 → 256 × 28 × 28
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            # 256 × 28 × 28 → 128 × 56 × 56
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            # 128 × 56 × 56 → 64 × 112 × 112
            nn.ConvTranspose2d(128, 64,  4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(True),
            # 64 × 112 × 112 → 3 × 224 × 224
            nn.ConvTranspose2d(64,  3,   4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        z     = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE between input and reconstruction (no grad needed)."""
        with torch.no_grad():
            recon = self.forward(x)
            mse   = ((x - recon) ** 2).mean(dim=(1, 2, 3))
        return mse      # shape (B,)
