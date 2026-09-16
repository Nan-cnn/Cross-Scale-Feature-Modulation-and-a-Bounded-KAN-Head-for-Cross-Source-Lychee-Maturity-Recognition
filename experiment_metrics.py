"""Dependency-light binary classification metrics and calibration measures."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order]
    tp = np.concatenate(([0.0], np.cumsum(y, dtype=float), [float(positives)]))
    fp = np.concatenate(([0.0], np.cumsum(1 - y, dtype=float), [float(negatives)]))
    return float(np.trapz(tp / positives, fp / negatives))


def _average_precision(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y, dtype=float)
    precision = tp / np.arange(1, labels.size + 1)
    return float((precision * y).sum() / positives)


def compute_binary_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float],
    ece_bins: int = 15,
) -> Dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    prob = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    if not (y.size == pred.size == prob.size) or y.size == 0:
        raise ValueError("labels, predictions and probabilities must be non-empty and equal-length")

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    def safe(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision_pos = safe(tp, tp + fp)
    recall_pos = safe(tp, tp + fn)
    f1_pos = safe(2 * precision_pos * recall_pos, precision_pos + recall_pos)
    precision_neg = safe(tn, tn + fn)
    recall_neg = safe(tn, tn + fp)
    f1_neg = safe(2 * precision_neg * recall_neg, precision_neg + recall_neg)
    denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    confidence = np.maximum(prob, 1.0 - prob)
    correctness = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, int(ece_bins) + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean() - confidence[mask].mean()))

    return {
        "n": int(y.size),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float((recall_pos + recall_neg) / 2.0),
        "macro_precision": float((precision_pos + precision_neg) / 2.0),
        "macro_recall": float((recall_pos + recall_neg) / 2.0),
        "macro_f1": float((f1_pos + f1_neg) / 2.0),
        "mcc": safe(tp * tn - fp * fn, denominator),
        "roc_auc": _binary_auc(y, prob),
        "pr_auc": _average_precision(y, prob),
        "nll": float(-(y * np.log(prob) + (1 - y) * np.log(1.0 - prob)).mean()),
        "brier": float(np.square(prob - y).mean()),
        "ece": float(ece),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


__all__ = ["compute_binary_metrics"]
