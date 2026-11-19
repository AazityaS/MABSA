"""Shared XLM-R encoder with explicitly candidate-conditioned two-head scorer."""

from __future__ import annotations

import torch
from torch import nn

from .v3_presence_dataset import CATEGORIES, NUM_LABELS, SENTIMENTS, TEXT_CHECKPOINT, TEXT_REVISION


class HybridCandidateModel(nn.Module):
    def __init__(self, encoder: nn.Module, *, hidden_size: int = 256,
                 category_embedding_dim: int = 64, sentiment_embedding_dim: int = 32,
                 dropout: float = 0.1, encoder_hidden_size: int | None = None):
        super().__init__()
        self.encoder = encoder
        enc_dim = int(encoder_hidden_size or encoder.config.hidden_size)
        self.category_embeddings = nn.Embedding(len(CATEGORIES), category_embedding_dim)
        self.sentiment_embeddings = nn.Embedding(len(SENTIMENTS), sentiment_embedding_dim)
        self.category_projection = nn.Linear(category_embedding_dim, enc_dim)
        self.sentiment_projection = nn.Linear(sentiment_embedding_dim, enc_dim)
        self.scorer = nn.Sequential(
            nn.Linear(enc_dim * 3 + category_embedding_dim + sentiment_embedding_dim, hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        )
        self.register_buffer("candidate_category_ids", torch.arange(len(CATEGORIES)).repeat_interleave(len(SENTIMENTS)), persistent=True)
        self.register_buffer("candidate_sentiment_ids", torch.arange(len(SENTIMENTS)).repeat(len(CATEGORIES)), persistent=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        result = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = result.last_hidden_state[:, 0, :]
        cat_ids = self.candidate_category_ids
        sent_ids = self.candidate_sentiment_ids
        cat = self.category_embeddings(cat_ids).unsqueeze(0).expand(h.size(0), -1, -1)
        sent = self.sentiment_embeddings(sent_ids).unsqueeze(0).expand(h.size(0), -1, -1)
        h_candidates = h.unsqueeze(1).expand(-1, NUM_LABELS, -1)
        cat_proj = self.category_projection(cat)
        sent_proj = self.sentiment_projection(sent)
        conditioned = torch.cat((h_candidates, cat, sent, h_candidates * cat_proj,
                                 h_candidates * sent_proj), dim=-1)
        logits = self.scorer(conditioned)
        if logits.shape != (h.size(0), NUM_LABELS, 2):
            raise AssertionError(f"Unexpected candidate logits shape: {tuple(logits.shape)}")
        any_logits, conditional_logits = logits.unbind(dim=-1)
        p_any = torch.sigmoid(any_logits)
        p_conditional = torch.sigmoid(conditional_logits)
        return {"any_logits": any_logits, "conditional_logits": conditional_logits,
                "combined_probabilities": p_any * p_conditional,
                "any_probabilities": p_any, "conditional_probabilities": p_conditional}


def load_pinned_encoder(local_files_only: bool = True):
    from transformers import AutoModel
    return AutoModel.from_pretrained(TEXT_CHECKPOINT, revision=TEXT_REVISION,
                                     local_files_only=local_files_only)
