"""
Utilities for Energy-Based Out-of-Distribution (OOD) detection.

Convention used here:
- Free energy:        E(x)  = -T * logsumexp(logits / T)
- Negative energy:   S(x)  = -E(x) =  T * logsumexp(logits / T)
- Larger S(x) means the sample is more likely in-distribution (ID).
- Predict OOD/Unknown when S(x) <= threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def stable_logsumexp_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Numerically stable logsumexp implemented with NumPy only."""
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def negative_energy_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return S(x) = -E(x) = T * logsumexp(logits / T). Larger score => more ID-like."""
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return temperature * stable_logsumexp_np(logits / temperature, axis=1)


def free_energy_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Return E(x) = -T * logsumexp(logits / T). Lower energy => more ID-like."""
    return -negative_energy_np(logits, temperature=temperature)


def softmax_confidence_np(logits: np.ndarray) -> np.ndarray:
    """Maximum softmax probability for each sample."""
    logits = np.asarray(logits, dtype=np.float64)
    m = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - m)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return np.max(probs, axis=1)


def choose_threshold_from_id(id_scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """
    Choose threshold tau using only ID scores.

    With the convention larger score => ID, selecting the (1-target_tpr) percentile
    makes approximately target_tpr of ID samples satisfy score > tau.
    """
    if not (0 < target_tpr < 1):
        raise ValueError("target_tpr must be in (0, 1)")
    return float(np.percentile(id_scores, 100.0 * (1.0 - target_tpr)))


def predict_unknown(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return True for samples predicted as Unknown/OOD."""
    return np.asarray(scores) <= threshold


def classification_accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    """Top-1 classification accuracy."""
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    return float(np.mean(np.argmax(logits, axis=1) == labels))


@dataclass
class OODMetrics:
    threshold: float
    id_tpr: float
    ood_fpr: float
    id_known_rate: float
    ood_unknown_rate: float
    overall_detector_acc: float
    auroc: Optional[float]
    aupr_in: Optional[float]
    n_id: int
    n_ood: int

    def as_percent_dict(self) -> Dict[str, float]:
        return {
            "threshold": self.threshold,
            "ID TPR / Known rate (%)": 100.0 * self.id_tpr,
            "OOD FPR / OOD misclassified as Known (%)": 100.0 * self.ood_fpr,
            "OOD Unknown rate (%)": 100.0 * self.ood_unknown_rate,
            "Detector accuracy (%)": 100.0 * self.overall_detector_acc,
            "AUROC (%)": np.nan if self.auroc is None else 100.0 * self.auroc,
            "AUPR-In (%)": np.nan if self.aupr_in is None else 100.0 * self.aupr_in,
            "N ID": self.n_id,
            "N OOD": self.n_ood,
        }


def evaluate_ood_scores(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: Optional[float] = None,
    target_tpr: float = 0.95,
) -> OODMetrics:
    """Evaluate ID-vs-OOD detection. ID is the positive class for AUROC/AUPR."""
    id_scores = np.asarray(id_scores)
    ood_scores = np.asarray(ood_scores)
    if threshold is None:
        threshold = choose_threshold_from_id(id_scores, target_tpr=target_tpr)

    id_known = id_scores > threshold
    ood_known = ood_scores > threshold
    id_tpr = float(np.mean(id_known))
    ood_fpr = float(np.mean(ood_known))
    ood_unknown = 1.0 - ood_fpr
    overall_acc = float((np.sum(id_known) + np.sum(~ood_known)) / (len(id_scores) + len(ood_scores)))

    auroc = None
    aupr_in = None
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = np.concatenate([np.ones_like(id_scores, dtype=int), np.zeros_like(ood_scores, dtype=int)])
        y_score = np.concatenate([id_scores, ood_scores])
        auroc = float(roc_auc_score(y_true, y_score))
        aupr_in = float(average_precision_score(y_true, y_score))
    except Exception:
        pass

    return OODMetrics(
        threshold=float(threshold),
        id_tpr=id_tpr,
        ood_fpr=ood_fpr,
        id_known_rate=id_tpr,
        ood_unknown_rate=ood_unknown,
        overall_detector_acc=overall_acc,
        auroc=auroc,
        aupr_in=aupr_in,
        n_id=int(len(id_scores)),
        n_ood=int(len(ood_scores)),
    )


def save_energy_score_table(logits: np.ndarray, labels: Optional[np.ndarray], output_csv: str | Path) -> None:
    """Save per-sample logits-derived scores for later reporting/demo."""
    import pandas as pd

    logits = np.asarray(logits)
    df = pd.DataFrame({
        "sample_id": np.arange(len(logits)),
        "pred": np.argmax(logits, axis=1),
        "negative_energy": negative_energy_np(logits),
        "free_energy": free_energy_np(logits),
        "softmax_confidence": softmax_confidence_np(logits),
    })
    if labels is not None:
        labels = np.asarray(labels)
        df.insert(1, "label", labels)
        df["correct"] = df["pred"] == df["label"]
    df.to_csv(output_csv, index=False)
