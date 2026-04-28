import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat

# =========================
# 1. CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

SVHN_PATH = os.path.join(BASE_DIR, "data", "svhn", "test_32x32.mat")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "svhn_test_logits.npy")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("SVHN path:", SVHN_PATH)
print("Model path:", MODEL_PATH)

# =========================
# 2. LOAD SVHN
# =========================
data = loadmat(SVHN_PATH)

X = data['X']   # (32,32,3,N)
y = data['y']   # (N,1)

print("\n=== RAW SVHN ===")
print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 3. RESHAPE + FIX LABEL
# =========================
# CHUYỂN VỀ (N, 3, 32, 32) → CHUẨN PYTORCH
X = np.transpose(X, (3, 2, 0, 1))  # ⚠️ QUAN TRỌNG
y = y.squeeze()

# SVHN: label 10 = digit 0
y[y == 10] = 0

print("\n=== AFTER RESHAPE ===")
print("X shape:", X.shape)  # phải là (N,3,32,32)
print("y shape:", y.shape)

# =========================
# 4. NORMALIZE (GIỐNG CIFAR)
# =========================
X = X.astype(np.float32) / 255.0

mean = np.array([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)
std  = np.array([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1)

X = (X - mean) / std

print("\n=== AFTER NORMALIZE ===")
print("Min:", X.min(), "Max:", X.max())

# =========================
# 5. TO TENSOR
# =========================
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# =========================
# 6. LOAD MODEL
# =========================
print("\nLoading model...")

model = torch.load(MODEL_PATH, map_location=DEVICE)
model.to(DEVICE)
model.eval()

# =========================
# 7. INFERENCE
# =========================
print("\nRunning inference...")

all_logits = []

with torch.no_grad():
    for images, _ in loader:
        images = images.to(DEVICE)

        outputs = model(images)   # logits (KHÔNG softmax)
        all_logits.append(outputs.cpu().numpy())

logits = np.concatenate(all_logits, axis=0)

print("Logits shape:", logits.shape)

# =========================
# 8. SAVE
# =========================
np.save(OUTPUT_FILE, logits)

print("\nSaved to:", OUTPUT_FILE)
print("DONE")