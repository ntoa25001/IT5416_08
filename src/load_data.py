import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from datetime import datetime
from scipy.io import loadmat

# ==============================
# 1. PATH CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_ROOT = os.path.join(BASE_DIR, 'data')
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')

DATASET = "svhn"  # "cifar" hoặc "svhn"

DATA_DIR = os.path.join(DATA_ROOT, DATASET)
OUTPUT_DIR = os.path.join(OUTPUT_ROOT, DATASET)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n=== DATASET: {DATASET.upper()} ===")
print("Data dir:", DATA_DIR)
print("Output dir:", OUTPUT_DIR)

# ==============================
# 2. HELPER (FIX BLUR - NO CV2)
# ==============================
def show_image(img, upscale=True):
    if upscale:
        img = np.kron(img, np.ones((4, 4, 1)))  # 32 → 128
    plt.imshow(img, interpolation='nearest')

# ==============================
# 3. LOAD CIFAR
# ==============================
def load_batch(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict[b'data'], dict[b'labels']

def load_cifar10(data_dir):
    X_train, y_train = [], []

    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        X_train.append(data)
        y_train += labels

    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)

    X_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))
    y_test = np.array(y_test)

    with open(os.path.join(data_dir, 'batches.meta'), 'rb') as f:
        meta = pickle.load(f, encoding='bytes')

    label_names = [x.decode('utf-8') for x in meta[b'label_names']]

    # reshape CIFAR
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test  = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return X_train, y_train, X_test, y_test, label_names

# ==============================
# 4. LOAD SVHN
# ==============================
def load_svhn(data_dir):
    train = loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    test  = loadmat(os.path.join(data_dir, 'test_32x32.mat'))

    X_train = train['X']  # (32,32,3,N)
    y_train = train['y']

    X_test = test['X']
    y_test = test['y']

    # reshape
    X_train = np.transpose(X_train, (3, 0, 1, 2))
    X_test  = np.transpose(X_test, (3, 0, 1, 2))

    # fix label 10 -> 0
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0

    y_train = y_train.flatten()
    y_test  = y_test.flatten()

    label_names = [str(i) for i in range(10)]

    return X_train, y_train, X_test, y_test, label_names

# ==============================
# 5. LOAD DATASET SWITCH
# ==============================
def load_dataset(name, data_dir):
    if name == "cifar":
        return load_cifar10(data_dir)
    elif name == "svhn":
        return load_svhn(data_dir)
    else:
        raise ValueError("Dataset không hỗ trợ")

# ==============================
# 6. LOAD DATA
# ==============================
X_train, y_train, X_test, y_test, label_names = load_dataset(DATASET, DATA_DIR)

print("\n=== RAW DATA ===")
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ==============================
# 7. PREPROCESS (CHUNG)
# ==============================
X_train = X_train / 255.0
X_test  = X_test / 255.0

# ==============================
# 8. SPLIT
# ==============================
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

print("\n=== AFTER SPLIT ===")
print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)

print("\n=== DATASET INFO ===")
print("Image size:", X_train.shape[1:])
print("Classes:", label_names)

# ==============================
# 9. CLASS DISTRIBUTION
# ==============================
unique, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(10,5))
plt.bar([label_names[i] for i in unique], counts)
plt.xticks(rotation=45)
plt.title(f"{DATASET.upper()} - Class Distribution")

filename = datetime.now().strftime("class_dist_%Y%m%d_%H%M%S.png")
plt.savefig(os.path.join(OUTPUT_DIR, filename))
plt.close()

# ==============================
# 10. SAMPLE IMAGES
# ==============================
for k in range(3):
    plt.figure(figsize=(6,6))

    indices = np.random.choice(len(X_train), 10, replace=False)

    for idx, i in enumerate(indices):
        plt.subplot(2,5,idx+1)
        show_image(X_train[i])
        plt.title(label_names[y_train[i]])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"sample_batch_{k}.png"))
    plt.close()

# ==============================
# 11. SAVE 1 IMAGE PER CLASS
# ==============================
for class_id in range(len(label_names)):
    idx = np.where(y_train == class_id)[0][0]

    plt.figure(figsize=(3,3))
    show_image(X_train[idx])
    plt.title(label_names[class_id])
    plt.axis('off')

    plt.savefig(os.path.join(OUTPUT_DIR, f"class_{label_names[class_id]}.png"))
    plt.close()

# ==============================
# 12. SUMMARY
# ==============================
print("\n=== DATA SUMMARY ===")
print(f"Train: {len(X_train)}")
print(f"Val: {len(X_val)}")
print(f"Test: {len(X_test)}")

print("\nImages saved in:", OUTPUT_DIR)