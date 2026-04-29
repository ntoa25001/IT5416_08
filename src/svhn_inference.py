import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
import torchvision.models as models

# =========================
# 1. CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SVHN_PATH = os.path.join(BASE_DIR, "data", "svhn", "test_32x32.mat")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "svhn_test_logits.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

print("SVHN path:", SVHN_PATH)
print("Model path:", MODEL_PATH)

# =========================
# 2. LOAD SVHN
# =========================
print("\nLoading SVHN...")

data = loadmat(SVHN_PATH)

X = data['X']   # (32,32,3,N)
y = data['y']

# reshape -> (N,3,32,32)
X = np.transpose(X, (3, 2, 0, 1))
y = y.squeeze()

# fix label (10 -> 0)
y[y == 10] = 0

print("X shape:", X.shape)

# =========================
# 3. NORMALIZE (GIỐNG CIFAR)
# =========================
# NORMALIZE giống transform_test lúc Hưng train CIFAR-10
X = X.astype(np.float32) / 255.0

mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
std  = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)

X = (X - mean) / std

print("\n=== AFTER NORMALIZE ===")
print("Min:", X.min(), "Max:", X.max())

# =========================
# 4. TO TENSOR
# =========================
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 5. LOAD MODEL (RESNET18)
# =========================
print("\nLoading model...")

model = models.resnet18(num_classes=10)

# 👉 FIX CHO CIFAR (RẤT QUAN TRỌNG)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()

# load weight
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()

# =========================
# 6. INFERENCE
# =========================
print("\nRunning inference...")

all_logits = []

with torch.no_grad():
    for images, _ in loader:
        images = images.to(DEVICE)

        outputs = model(images)  # logits
        all_logits.append(outputs.cpu().numpy())

logits = np.concatenate(all_logits, axis=0)

print("Logits shape:", logits.shape)

# =========================
# 7. SAVE OUTPUT
# =========================
np.save(OUTPUT_PATH, logits)

print("\nSaved logits to:", OUTPUT_PATH)
print("DONE")