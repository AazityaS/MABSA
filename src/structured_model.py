"""Small fixed-slot prediction head over frozen encoder features."""

from __future__ import annotations

import torch
from torch import nn


class StructuredSetModel(nn.Module):
    """Frozen feature inputs -> 23 learned slots and structured label logits."""

    def __init__(self, text_dim=768, image_dim=512, hidden_size=256, slots=23,
                 languages=3, multimodal=False, dropout=0.1):
        super().__init__()
        if slots != 23:
            raise ValueError("HAMOS structured baseline requires exactly 23 slots")
        self.slots, self.multimodal = slots, multimodal
        self.text_projection = nn.Linear(text_dim, hidden_size)
        self.language_embedding = nn.Embedding(languages, hidden_size)
        if multimodal:
            self.image_projection = nn.Linear(image_dim, hidden_size)
            self.multimodal_fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        # Keep the original text-only representation path unchanged.
        self.text_fusion = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                         nn.Linear(hidden_size, hidden_size), nn.GELU())
        self.slot_embeddings = nn.Parameter(torch.randn(slots, hidden_size) * 0.02)
        self.slot_norm = nn.LayerNorm(hidden_size)
        self.objectness_head = nn.Linear(hidden_size, 1)
        self.category_head = nn.Linear(hidden_size, 6)
        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.aspect_implicit_head = nn.Linear(hidden_size, 1)
        self.opinion_implicit_head = nn.Linear(hidden_size, 1)

    def forward(self, text_features, language_ids, image_features=None, image_mask=None):
        if text_features.ndim != 2:
            raise ValueError("text_features must have shape [batch, hidden]")
        text_context = self.text_projection(text_features) + self.language_embedding(language_ids)
        if self.multimodal:
            if image_features is None or image_mask is None:
                raise ValueError("multimodal model requires image_features and image_mask")
            if image_features.ndim != 3 or image_mask.shape != image_features.shape[:2]:
                raise ValueError("image features/mask must have shapes [batch, images, dim]/[batch, images]")
            weights = image_mask.to(image_features.dtype).unsqueeze(-1)
            pooled = (image_features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            projected_image = self.image_projection(pooled)
            context = self.multimodal_fusion(torch.cat((text_context, projected_image), dim=-1))
        else:
            context = self.text_fusion(text_context)
        slots = self.slot_norm(context[:, None, :] + self.slot_embeddings[None, :, :])
        return {"objectness": self.objectness_head(slots).squeeze(-1),
                "category": self.category_head(slots),
                "sentiment": self.sentiment_head(slots),
                "aspect_implicit": self.aspect_implicit_head(slots).squeeze(-1),
                "opinion_implicit": self.opinion_implicit_head(slots).squeeze(-1)}


def language_ids(languages):
    """Stable metadata mapping. This is not an annotation target."""
    mapping = {"en": 0, "vi": 1, "other": 2}
    return torch.tensor([mapping.get(x, mapping["other"]) for x in languages], dtype=torch.long)
