"""Unique category-sentiment presence metrics for V4 combined scores."""

from __future__ import annotations

import numpy as np

from .v3_presence_dataset import LABEL_ORDER, NUM_LABELS
from .v3_presence_metrics import _prf, evaluate_presence


def _ranking(probabilities: np.ndarray, targets: np.ndarray) -> dict:
    flat_y, flat_p = targets.ravel(), probabilities.ravel()
    positive = flat_y.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return {"roc_auc": None, "average_precision": (1.0 if n_pos else 0.0),
                "note": "ROC-AUC undefined with only one target class"}
    # Mann-Whitney U with average ranks for ties.
    order_asc = np.argsort(flat_p, kind="stable")
    sorted_p = flat_p[order_asc]
    ranks = np.empty(len(flat_p), dtype=np.float64)
    boundaries = np.r_[0, np.flatnonzero(sorted_p[:-1] != sorted_p[1:]) + 1, len(flat_p)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        ranks[order_asc[left:right]] = (left + 1 + right) / 2.0
    roc = float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    # Non-interpolated average precision, accumulating once per tied-score group.
    order_desc = np.argsort(-flat_p, kind="stable")
    desc_p, desc_y = flat_p[order_desc], positive[order_desc]
    ends = np.r_[np.flatnonzero(desc_p[:-1] != desc_p[1:]), len(desc_p) - 1]
    cum_tp = np.cumsum(desc_y, dtype=np.int64)[ends]
    cum_total = ends + 1
    group_tp = np.diff(np.r_[0, cum_tp])
    ap = float(np.sum((group_tp / n_pos) * (cum_tp / cum_total)))
    return {"roc_auc": roc, "average_precision": ap}


def _binary_metrics(probabilities, targets, mask, threshold):
    p, y, m = np.asarray(probabilities), np.asarray(targets).astype(bool), np.asarray(mask).astype(bool)
    pred = p >= threshold
    return _prf(int((pred & y & m).sum()), int((pred & ~y & m).sum()), int((~pred & y & m).sum()))


def evaluate_hybrid(combined_probabilities, targets, any_probabilities, any_targets,
                    conditional_probabilities, conditional_targets, conditional_mask,
                    threshold: float) -> dict:
    primary = evaluate_presence(combined_probabilities, targets, threshold)
    any_metrics = evaluate_presence(any_probabilities, any_targets, threshold)
    for report, n_rows in ((primary, len(targets)), (any_metrics, len(any_targets))):
        total_cells = int(n_rows * NUM_LABELS)
        report["micro"]["tn"] = total_cells - report["micro"]["tp"] - report["micro"]["fp"] - report["micro"]["fn"]
        for row in report["per_label"].values():
            row["tn"] = int(n_rows - row["tp"] - row["fp"] - row["fn"])
    gold_zero = np.asarray(targets).astype(bool).sum(axis=1) == 0
    pred_zero = np.asarray(combined_probabilities) >= threshold
    pred_zero = pred_zero.sum(axis=1) == 0
    primary["all_zero_diagnostics"] = {"gold_all_zero_segments": int(gold_zero.sum()),
        "predicted_all_zero_segments": int(pred_zero.sum()),
        "gold_all_zero_and_predicted_all_zero": int((gold_zero & pred_zero).sum())}
    cond_macro = []
    for i, label in enumerate(LABEL_ORDER):
        mask = np.asarray(conditional_mask)[:, i].astype(bool)
        vals = _binary_metrics(np.asarray(conditional_probabilities)[:, i],
                               np.asarray(conditional_targets)[:, i], mask, threshold)
        vals["support"] = int((np.asarray(conditional_targets)[:, i].astype(bool) & mask).sum())
        vals["valid_rows"] = int(mask.sum())
        vals["tn"] = int(mask.sum() - vals["tp"] - vals["fp"] - vals["fn"])
        cond_macro.append(vals["f1"])
    conditional = _binary_metrics(conditional_probabilities, conditional_targets,
                                  conditional_mask, threshold)
    valid_count = int(np.asarray(conditional_mask).sum())
    conditional["tn"] = valid_count - conditional["tp"] - conditional["fp"] - conditional["fn"]
    return {"primary_implicit_pair_presence": primary,
            "any_pair": any_metrics,
            "conditional_implicit_given_pair": {
                **conditional,
                "valid_candidate_rows": int(np.asarray(conditional_mask).sum()),
                "positive_support": int((np.asarray(conditional_targets).astype(bool) & np.asarray(conditional_mask).astype(bool)).sum()),
                "per_label_macro_f1": float(np.mean(cond_macro)) if cond_macro else 0.0,
                "per_label": {name: {**_binary_metrics(np.asarray(conditional_probabilities)[:, i],
                    np.asarray(conditional_targets)[:, i], np.asarray(conditional_mask)[:, i], threshold),
                    "support": int((np.asarray(conditional_targets)[:, i].astype(bool) & np.asarray(conditional_mask)[:, i]).sum()),
                    "valid_rows": int(np.asarray(conditional_mask)[:, i].sum()),
                    "tn": int(np.asarray(conditional_mask)[:, i].sum() -
                        _binary_metrics(np.asarray(conditional_probabilities)[:, i], np.asarray(conditional_targets)[:, i],
                                        np.asarray(conditional_mask)[:, i], threshold)["tp"] -
                        _binary_metrics(np.asarray(conditional_probabilities)[:, i], np.asarray(conditional_targets)[:, i],
                                        np.asarray(conditional_mask)[:, i], threshold)["fp"] -
                        _binary_metrics(np.asarray(conditional_probabilities)[:, i], np.asarray(conditional_targets)[:, i],
                                        np.asarray(conditional_mask)[:, i], threshold)["fn"])}
                    for i, name in enumerate(LABEL_ORDER)}},
            "ranking": _ranking(np.asarray(combined_probabilities), np.asarray(targets)),
            "threshold_semantics": ">=", "threshold": float(threshold)}


def select_combined_threshold(probabilities, targets) -> dict:
    """Exact dev sweep over unique scores; >= comparison and highest-threshold tie break."""
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(targets, dtype=np.int8)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != NUM_LABELS:
        raise ValueError("Threshold selection requires aligned [N,18] arrays")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.isin(y, [0, 1]).all():
        raise ValueError("Threshold selection inputs must be finite probabilities and binary targets")
    scores, labels = p.ravel(), y.ravel().astype(bool)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    cumulative_tp = np.cumsum(sorted_labels, dtype=np.int64)
    cumulative_fp = np.cumsum(~sorted_labels, dtype=np.int64)
    # For >= threshold semantics, evaluate only at the final cell of each equal-score group.
    group_ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    tp = cumulative_tp[group_ends]
    fp = cumulative_fp[group_ends]
    positives = int(labels.sum())
    fn = positives - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    recall = np.divide(tp, positives, out=np.zeros_like(tp, dtype=float), where=positives > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    # Also include an empty-prediction operating point at threshold 1 when possible.
    candidates = [(float(sorted_scores[group_ends[i]]), float(f1[i]), int(tp[i]), int(fp[i]), int(fn[i]))
                  for i in range(len(group_ends))]
    if sorted_scores[0] < 1.0:
        candidates.append((1.0, 0.0, 0, 0, positives))
    threshold, best_f1, best_tp, best_fp, best_fn = max(candidates, key=lambda row: (row[1], row[0]))
    best_precision = best_tp / (best_tp + best_fp) if best_tp + best_fp else 0.0
    best_recall = best_tp / positives if positives else 0.0
    return {"threshold": threshold, "precision": best_precision, "recall": best_recall,
            "micro_f1": best_f1, "tp": best_tp, "fp": best_fp, "fn": best_fn,
            "threshold_comparison": ">=", "tie_break": "highest threshold",
            "candidate_count": int(len(candidates))}
