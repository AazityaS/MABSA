"""Permutation-invariant Hungarian loss for HAMOS structured sets."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def hungarian_assignment(cost):
    """Return minimum-cost row/column pairs; preserves duplicate columns as items."""
    if isinstance(cost, torch.Tensor):
        cost = cost.detach().cpu().numpy()
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    if cost.size == 0:
        return [], []
    rows, cols = linear_sum_assignment(cost)
    return rows.tolist(), cols.tolist()


def _pair_cost(outputs, targets, batch_index, gold_indices, weights):
    pred_n = outputs["objectness"].shape[1]
    if not gold_indices:
        return outputs["objectness"].new_zeros((pred_n, 0))
    gold = torch.as_tensor(gold_indices, device=outputs["objectness"].device)
    cat = targets["category"][batch_index, gold]
    sent = targets["sentiment"][batch_index, gold]
    ai = targets["aspect_implicit"][batch_index, gold].float()
    oi = targets["opinion_implicit"][batch_index, gold].float()
    cat_cost = -F.log_softmax(outputs["category"][batch_index], dim=-1)[:, cat]
    sent_cost = -F.log_softmax(outputs["sentiment"][batch_index], dim=-1)[:, sent]
    ai_cost = F.binary_cross_entropy_with_logits(
        outputs["aspect_implicit"][batch_index, :, None].expand(-1, len(gold_indices)),
        ai[None, :].expand(pred_n, -1), reduction="none")
    oi_cost = F.binary_cross_entropy_with_logits(
        outputs["opinion_implicit"][batch_index, :, None].expand(-1, len(gold_indices)),
        oi[None, :].expand(pred_n, -1), reduction="none")
    return (weights["category"] * cat_cost + weights["sentiment"] * sent_cost +
            weights["aspect_implicit"] * ai_cost + weights["opinion_implicit"] * oi_cost)


class StructuredSetLoss(torch.nn.Module):
    def __init__(self, category=1.0, sentiment=1.0, aspect_implicit=0.5,
                 opinion_implicit=0.5, objectness_positive=1.0, objectness_empty=0.1,
                 cardinality=0.0):
        super().__init__()
        self.weights = {"category": category, "sentiment": sentiment,
                        "aspect_implicit": aspect_implicit, "opinion_implicit": opinion_implicit}
        self.objectness_positive = objectness_positive
        self.objectness_empty = objectness_empty
        self.cardinality = cardinality

    def forward(self, outputs, targets):
        batch, slots = outputs["objectness"].shape
        matched_losses, positive_losses, empty_losses = [], [], []
        assignments = []
        for b in range(batch):
            gold_indices = torch.where(targets["mask"][b])[0].tolist()
            if len(gold_indices) > slots:
                raise ValueError(f"Gold set ({len(gold_indices)}) exceeds output slots ({slots})")
            cost = _pair_cost(outputs, targets, b, gold_indices, self.weights)
            pred_idx, local_gold_idx = hungarian_assignment(cost)
            gold_idx = [gold_indices[j] for j in local_gold_idx]
            assignments.append((pred_idx, gold_idx))
            if pred_idx:
                pi = torch.tensor(pred_idx, device=outputs["objectness"].device)
                gi = torch.tensor(gold_idx, device=outputs["objectness"].device)
                matched = (F.cross_entropy(outputs["category"][b, pi], targets["category"][b, gi], reduction="none") * self.weights["category"] +
                           F.cross_entropy(outputs["sentiment"][b, pi], targets["sentiment"][b, gi], reduction="none") * self.weights["sentiment"] +
                           F.binary_cross_entropy_with_logits(outputs["aspect_implicit"][b, pi],
                               targets["aspect_implicit"][b, gi].float(), reduction="none") * self.weights["aspect_implicit"] +
                           F.binary_cross_entropy_with_logits(outputs["opinion_implicit"][b, pi],
                               targets["opinion_implicit"][b, gi].float(), reduction="none") * self.weights["opinion_implicit"])
                matched_losses.append(matched)
                positive_losses.append(F.binary_cross_entropy_with_logits(
                    outputs["objectness"][b, pi], torch.ones_like(outputs["objectness"][b, pi]), reduction="none"))
            else:
                matched_losses.append(outputs["objectness"].new_zeros(()))
            unmatched = sorted(set(range(slots)) - set(pred_idx))
            if unmatched:
                ui = torch.tensor(unmatched, device=outputs["objectness"].device)
                empty_losses.append(F.binary_cross_entropy_with_logits(
                    outputs["objectness"][b, ui], torch.zeros_like(outputs["objectness"][b, ui]), reduction="none"))
        zero = outputs["objectness"].new_zeros(())
        l_matched = torch.cat(matched_losses).mean() if matched_losses else zero
        l_present = torch.cat(positive_losses).mean() if positive_losses else zero
        l_empty = torch.cat(empty_losses).mean() if empty_losses else zero
        # This auxiliary term supervises only the expected set size. It is not
        # part of Hungarian matching and uses gold cardinality only as a target.
        expected_count = outputs["objectness"].float().sigmoid().sum(dim=1)
        gold_count = targets["mask"].sum(dim=1).to(expected_count.dtype)
        l_cardinality = F.smooth_l1_loss(expected_count, gold_count, reduction="mean")
        weighted_cardinality = self.cardinality * l_cardinality
        total = (l_matched + self.objectness_positive * l_present +
                 self.objectness_empty * l_empty + weighted_cardinality)
        return {"loss": total, "matched": l_matched.detach(), "objectness_positive": l_present.detach(),
                "objectness_empty": l_empty.detach(),
                "cardinality": l_cardinality.detach(),
                "cardinality_weighted": weighted_cardinality.detach(),
                "assignments": assignments}
