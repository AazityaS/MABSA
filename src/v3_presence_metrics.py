"""Metrics and dev-only threshold selection for unique-pair presence targets."""

from __future__ import annotations

import numpy as np

from .v3_presence_dataset import LABEL_ORDER, NUM_LABELS


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def evaluate_presence(probabilities, targets, threshold: float) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int8)
    if probabilities.ndim != 2 or probabilities.shape[1] != NUM_LABELS:
        raise ValueError(f"Probabilities must have shape [N,{NUM_LABELS}]")
    if targets.shape != probabilities.shape:
        raise ValueError("Targets and probabilities must have the same [N,18] shape")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and within [0,1]")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Targets must be binary")
    predicted = probabilities >= float(threshold)
    per_label = {}
    f1s = []
    unsupported = []
    for index, name in enumerate(LABEL_ORDER):
        gold = targets[:, index].astype(bool)
        pred = predicted[:, index]
        tp = int(np.logical_and(gold, pred).sum())
        fp = int(np.logical_and(~gold, pred).sum())
        fn = int(np.logical_and(gold, ~pred).sum())
        metrics = _prf(tp, fp, fn)
        metrics["support"] = int(gold.sum())
        per_label[name] = metrics
        f1s.append(metrics["f1"])
        if metrics["support"] == 0:
            unsupported.append(name)
    tp = int(np.logical_and(targets.astype(bool), predicted).sum())
    fp = int(np.logical_and(~targets.astype(bool), predicted).sum())
    fn = int(np.logical_and(targets.astype(bool), ~predicted).sum())
    micro = _prf(tp, fp, fn)
    gold_counts = targets.sum(axis=1)
    pred_counts = predicted.sum(axis=1)
    exact = np.all(targets.astype(bool) == predicted, axis=1)
    return {
        "threshold": float(threshold), "segments": int(len(targets)),
        "micro": micro, "per_label": per_label,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "macro_unsupported_labels": unsupported,
        "macro_warning": (f"Includes fixed-label zero-F1 convention for {len(unsupported)} labels with zero gold support"
                          if unsupported else None),
        "exact_set_accuracy": float(exact.mean()) if len(exact) else 0.0,
        "gold_label_count_mean": float(gold_counts.mean()) if len(gold_counts) else 0.0,
        "predicted_label_count_mean": float(pred_counts.mean()) if len(pred_counts) else 0.0,
        "predicted_label_count_median": float(np.median(pred_counts)) if len(pred_counts) else 0.0,
        "predicted_label_count_distribution": {str(int(k)): int(v) for k, v in
                                                zip(*np.unique(pred_counts, return_counts=True))},
        "cardinality_mae": float(np.abs(pred_counts - gold_counts).mean()) if len(gold_counts) else 0.0,
        "cardinality_signed_bias": float((pred_counts - gold_counts).mean()) if len(gold_counts) else 0.0,
    }


def select_presence_threshold(probabilities, targets, grid=None) -> dict:
    """Select one global threshold on dev; ties go to the highest threshold."""
    if grid is None:
        grid = [i / 20 for i in range(21)]
    candidates = [float(t) for t in grid]
    if not candidates or any(t < 0 or t > 1 for t in candidates):
        raise ValueError("Threshold grid must be non-empty and lie in [0,1]")
    results = [evaluate_presence(probabilities, targets, threshold=t) for t in candidates]
    best = max(results, key=lambda row: (row["micro"]["f1"], row["threshold"]))
    return {"threshold": best["threshold"], "dev_micro_f1": best["micro"]["f1"],
            "grid": candidates}
