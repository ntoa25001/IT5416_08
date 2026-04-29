"""Compute Energy-based OOD detection metrics for CIFAR-10 vs SVHN.

Inputs expected in one of these locations:
- CIFAR logits: output/cifar10_test_logits.npy OR cifar10_test_logits.npy OR output/cifar/cifar10_test_logits.npy
- CIFAR labels: output/cifar10_test_labels.npy OR cifar10_test_labels.npy OR output/cifar/cifar10_test_labels.npy
- SVHN logits:  output/svhn_test_logits.npy OR output/svhn/svhn_test_logits.npy OR svhn_test_logits.npy

Outputs are saved under output/ood/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_ood_utils import (
    classification_accuracy_from_logits,
    choose_threshold_from_id,
    evaluate_ood_scores,
    negative_energy_np,
    save_energy_score_table,
    softmax_confidence_np,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "ood"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def first_existing(candidates: list[Path], description: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    joined = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Cannot find {description}. Tried:\n  - {joined}")


def summarize_scores(name: str, scores: np.ndarray) -> dict:
    return {
        "dataset": name,
        "n": int(len(scores)),
        "mean_negative_energy": float(np.mean(scores)),
        "std_negative_energy": float(np.std(scores)),
        "min_negative_energy": float(np.min(scores)),
        "p05_negative_energy": float(np.percentile(scores, 5)),
        "median_negative_energy": float(np.median(scores)),
        "p95_negative_energy": float(np.percentile(scores, 95)),
        "max_negative_energy": float(np.max(scores)),
    }


def plot_energy_hist(id_scores: np.ndarray, ood_scores: np.ndarray, threshold: float) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(id_scores, bins=60, density=True, alpha=0.55, label="CIFAR-10 ID")
    plt.hist(ood_scores, bins=60, density=True, alpha=0.55, label="SVHN OOD")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    plt.title("Negative energy distribution: CIFAR-10 vs SVHN")
    plt.xlabel("Negative energy score S(x) = -E(x)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "energy_hist_cifar10_vs_svhn.png", dpi=200)
    plt.close()


def plot_energy_boxplot(id_scores: np.ndarray, ood_scores: np.ndarray, threshold: float) -> None:
    plt.figure(figsize=(7, 6))
    plt.boxplot([id_scores, ood_scores], labels=["CIFAR-10 ID", "SVHN OOD"])
    plt.axhline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    plt.title("Negative energy boxplot: CIFAR-10 vs SVHN")
    plt.ylabel("Negative energy score S(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "energy_boxplot_cifar10_vs_svhn.png", dpi=200)
    plt.close()


def plot_softmax_hist(id_conf: np.ndarray, ood_conf: np.ndarray) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(id_conf, bins=60, density=True, alpha=0.55, label="CIFAR-10 ID")
    plt.hist(ood_conf, bins=60, density=True, alpha=0.55, label="SVHN OOD")
    plt.title("Softmax confidence distribution: CIFAR-10 vs SVHN")
    plt.xlabel("Max softmax confidence")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "softmax_hist_cifar10_vs_svhn.png", dpi=200)
    plt.close()


def main() -> None:
    cifar_logits_path = first_existing(
        [
            ROOT / "output" / "cifar10_test_logits.npy",
            ROOT / "cifar10_test_logits.npy",
            ROOT / "output" / "cifar" / "cifar10_test_logits.npy",
        ],
        "CIFAR-10 logits",
    )
    cifar_labels_path = first_existing(
        [
            ROOT / "output" / "cifar10_test_labels.npy",
            ROOT / "cifar10_test_labels.npy",
            ROOT / "output" / "cifar" / "cifar10_test_labels.npy",
        ],
        "CIFAR-10 labels",
    )
    svhn_logits_path = first_existing(
        [
            ROOT / "output" / "svhn_test_logits.npy",
            ROOT / "output" / "svhn" / "svhn_test_logits.npy",
            ROOT / "svhn_test_logits.npy",
        ],
        "SVHN logits. Run python src/svhn_inference.py first.",
    )

    print("Loading:")
    print("-", cifar_logits_path)
    print("-", cifar_labels_path)
    print("-", svhn_logits_path)

    cifar_logits = np.load(cifar_logits_path)
    cifar_labels = np.load(cifar_labels_path)
    svhn_logits = np.load(svhn_logits_path)

    if cifar_logits.ndim != 2 or cifar_logits.shape[1] != 10:
        raise ValueError(f"Unexpected CIFAR logits shape: {cifar_logits.shape}")
    if svhn_logits.ndim != 2 or svhn_logits.shape[1] != 10:
        raise ValueError(f"Unexpected SVHN logits shape: {svhn_logits.shape}")

    cifar_acc = classification_accuracy_from_logits(cifar_logits, cifar_labels)
    id_energy = negative_energy_np(cifar_logits, temperature=1.0)
    ood_energy = negative_energy_np(svhn_logits, temperature=1.0)
    threshold = choose_threshold_from_id(id_energy, target_tpr=0.95)

    energy_metrics = evaluate_ood_scores(id_energy, ood_energy, threshold=threshold)

    id_conf = softmax_confidence_np(cifar_logits)
    ood_conf = softmax_confidence_np(svhn_logits)
    softmax_threshold = choose_threshold_from_id(id_conf, target_tpr=0.95)
    softmax_metrics = evaluate_ood_scores(id_conf, ood_conf, threshold=softmax_threshold)

    metrics_df = pd.DataFrame([
        {"method": "Energy", "cifar10_accuracy_%": 100 * cifar_acc, **energy_metrics.as_percent_dict()},
        {"method": "Softmax confidence", "cifar10_accuracy_%": 100 * cifar_acc, **softmax_metrics.as_percent_dict()},
    ])
    metrics_df.to_csv(OUT_DIR / "ood_metrics_energy_vs_softmax.csv", index=False)

    summary_df = pd.DataFrame([
        summarize_scores("CIFAR-10 ID", id_energy),
        summarize_scores("SVHN OOD", ood_energy),
    ])
    summary_df.to_csv(OUT_DIR / "energy_summary_cifar10_vs_svhn.csv", index=False)

    # Per-sample tables for demo/debugging.
    save_energy_score_table(cifar_logits, cifar_labels, OUT_DIR / "cifar10_energy_scores.csv")
    save_energy_score_table(svhn_logits, None, OUT_DIR / "svhn_energy_scores.csv")

    plot_energy_hist(id_energy, ood_energy, threshold)
    plot_energy_boxplot(id_energy, ood_energy, threshold)
    plot_softmax_hist(id_conf, ood_conf)

    print("\nCIFAR-10 accuracy from logits: {:.2f}%".format(100 * cifar_acc))
    print("\nOOD metrics:")
    print(metrics_df.to_string(index=False))
    print("\nSaved outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
