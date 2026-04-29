# IT5416_08 — Energy-Based OOD Detection Setup

## 1. Mục tiêu

Pipeline cuối dùng ResNet18 đã train trên CIFAR-10 làm classifier. CIFAR-10 test là in-distribution/known, SVHN test là out-of-distribution/unknown. Mô hình không train trên SVHN; SVHN chỉ được đưa qua model để lấy logits rồi tính Energy Score.

## 2. Cấu trúc thư mục đề xuất

```text
IT5416_08/
├── best_model.pth
├── requirements.txt
├── data/
│   ├── cifar/
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── test_batch
│   │   └── batches.meta
│   └── svhn/
│       ├── train_32x32.mat
│       ├── test_32x32.mat
│       └── extra_32x32.mat     # optional
├── output/
│   ├── cifar10_test_logits.npy
│   ├── cifar10_test_labels.npy
│   ├── svhn_test_logits.npy    # sinh ra sau khi chạy src/svhn_inference.py
│   └── ood/
├── src/
│   ├── load_data.py
│   ├── svhn_inference.py
│   ├── energy_ood_utils.py
│   └── run_energy_ood.py
├── notebooks/
│   └── energy_ood_detection.ipynb
└── report/
    └── report_section_energy_ood.md
```

## 3. Setup môi trường Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn activate, chạy một lần:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Sau đó activate lại.

## 4. Chạy SVHN inference

Đảm bảo có:

```text
best_model.pth
data/svhn/test_32x32.mat
```

Chạy:

```powershell
python src/svhn_inference.py
```

Output kỳ vọng:

```text
output/svhn_test_logits.npy
output/svhn_test_labels.npy
```

## 5. Chuẩn bị CIFAR logits

Copy các file Hưng đã gửi vào root hoặc output:

```text
output/cifar10_test_logits.npy
output/cifar10_test_labels.npy
```

Script `src/run_energy_ood.py` cũng tự tìm nếu hai file này đang ở root repo.

## 6. Chạy Energy/OOD evaluation

```powershell
python src/run_energy_ood.py
```

Output chính:

```text
output/ood/ood_metrics_energy_vs_softmax.csv
output/ood/energy_summary_cifar10_vs_svhn.csv
output/ood/energy_hist_cifar10_vs_svhn.png
output/ood/energy_boxplot_cifar10_vs_svhn.png
output/ood/softmax_hist_cifar10_vs_svhn.png
output/ood/cifar10_energy_scores.csv
output/ood/svhn_energy_scores.csv
```

## 7. Quy tắc Unknown

Dùng negative energy:

```text
S(x) = -E(x) = logsumexp(logits), với T = 1
```

Threshold được chọn bằng percentile 5% của CIFAR-10 test/validation để giữ khoảng 95% ID là Known.

```text
S(x) > threshold  -> Known / CIFAR-like
S(x) <= threshold -> Unknown / OOD
```
