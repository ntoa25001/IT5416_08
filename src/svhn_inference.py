"""Run the CIFAR-10 ResNet18 checkpoint on SVHN test and export logits.

Expected files:
- best_model.pth at repository root
- data/svhn/test_32x32.mat

Outputs:
- output/svhn_test_logits.npy
- output/svhn_test_labels.npy
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
SVHN_PATH = ROOT / "data" / "svhn" / "test_32x32.mat"
MODEL_PATH = ROOT / "best_model.pth"
OUTPUT_DIR = ROOT / "output"
OUTPUT_LOGITS = OUTPUT_DIR / "svhn_test_logits.npy"
OUTPUT_LABELS = OUTPUT_DIR / "svhn_test_labels.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

# Must match Hung's CIFAR-10 transform_test.
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
CIFAR_STD = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)


def build_model() -> nn.Module:
    """Build the same ResNet18 architecture used for CIFAR-10 training."""
    try:
        model = models.resnet18(weights=None, num_classes=10)
    except TypeError:  # compatibility with older torchvision
        model = models.resnet18(pretrained=False, num_classes=10)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Support DataParallel checkpoints if needed.
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


def load_svhn_test(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(path)
    x = data["X"]  # (32, 32, 3, N)
    y = data["y"].squeeze()
    y[y == 10] = 0

    # Convert to PyTorch NCHW and normalize exactly like CIFAR-10 test images.
    x = np.transpose(x, (3, 2, 0, 1)).astype(np.float32) / 255.0
    x = (x - CIFAR_MEAN) / CIFAR_STD
    return x, y.astype(np.int64)


def main() -> None:
    print("Repository root:", ROOT)
    print("SVHN path:", SVHN_PATH)
    print("Model path:", MODEL_PATH)
    print("Device:", DEVICE)

    if not SVHN_PATH.exists():
        raise FileNotFoundError(f"Missing SVHN file: {SVHN_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading SVHN test...")
    x, y = load_svhn_test(SVHN_PATH)
    print("X shape:", x.shape)
    print("y shape:", y.shape)
    print("X min/max after normalize:", float(x.min()), float(x.max()))

    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("\nLoading model...")
    model = build_model().to(DEVICE)
    load_checkpoint(model, MODEL_PATH)
    model.eval()

    print("\nRunning inference...")
    all_logits = []
    with torch.no_grad():
        for images, _ in loader:
            logits = model(images.to(DEVICE))
            all_logits.append(logits.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    print("Logits shape:", logits.shape)

    np.save(OUTPUT_LOGITS, logits)
    np.save(OUTPUT_LABELS, y)
    print("Saved:", OUTPUT_LOGITS)
    print("Saved:", OUTPUT_LABELS)
    print("DONE")


if __name__ == "__main__":
    main()
