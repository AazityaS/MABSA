"""Unweighted any-pair BCE plus masked conditional implicitness BCE."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hybrid_loss(any_logits: torch.Tensor, conditional_logits: torch.Tensor,
                any_targets: torch.Tensor, conditional_targets: torch.Tensor,
                conditional_mask: torch.Tensor, conditional_lambda: float = 1.0) -> dict[str, torch.Tensor]:
    if any_logits.shape != any_targets.shape or conditional_logits.shape != conditional_targets.shape:
        raise ValueError("Logits and target tensors must have matching shapes")
    if conditional_mask.shape != conditional_targets.shape:
        raise ValueError("Conditional mask must match target shape")
    if conditional_lambda < 0:
        raise ValueError("conditional_lambda must be non-negative")
    any_loss = F.binary_cross_entropy_with_logits(any_logits.float(), any_targets.float(), reduction="mean")
    valid = conditional_mask.bool()
    if valid.any():
        conditional_loss = F.binary_cross_entropy_with_logits(
            conditional_logits.float()[valid], conditional_targets.float()[valid], reduction="mean")
    else:
        # Differentiable zero, safe for a batch containing no present pairs.
        conditional_loss = conditional_logits.float().sum() * 0.0
    total = any_loss + float(conditional_lambda) * conditional_loss
    return {"loss": total, "any_loss": any_loss, "conditional_loss": conditional_loss,
            "conditional_valid_count": valid.sum()}
