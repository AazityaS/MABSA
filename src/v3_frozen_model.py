"""Linear 18-logit head over cached frozen XLM-R sentence embeddings."""

from torch import nn

from .v3_frozen_cache import EXPECTED_DIM
from .v3_presence_dataset import NUM_LABELS


class FrozenXLMRPresenceHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(EXPECTED_DIM, NUM_LABELS)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != EXPECTED_DIM:
            raise ValueError(f"Input must have shape [batch,{EXPECTED_DIM}]")
        return self.classifier(features)
