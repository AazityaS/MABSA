"""Fine-tuned XLM-R classifier for 18 implicit pair-presence labels."""

from __future__ import annotations

from torch import nn

from .v3_presence_dataset import NUM_LABELS, TEXT_CHECKPOINT, TEXT_REVISION


class XLMRImplicitPresenceModel(nn.Module):
    def __init__(self, checkpoint: str = TEXT_CHECKPOINT, revision: str = TEXT_REVISION,
                 dropout: float = 0.1, local_files_only: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.checkpoint = checkpoint
        self.revision = revision
        self.encoder = AutoModel.from_pretrained(
            checkpoint, revision=revision, local_files_only=local_files_only
        )
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.classifier(self.dropout(pooled))

    def enable_gradient_checkpointing(self) -> str:
        """Enable the supported activation checkpointing API for this Transformers version."""
        if hasattr(self.encoder, "activation_checkpointing_enable"):
            self.encoder.activation_checkpointing_enable()
            return "activation_checkpointing_enable"
        self.encoder.gradient_checkpointing_enable()
        return "gradient_checkpointing_enable"
