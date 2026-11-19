"""Train/dev-only candidate-conditioned targets for HAMOS V4."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .v3_presence_dataset import (
    CATEGORIES, LABEL_ORDER, NUM_LABELS, SENTIMENTS, _hotel_id,
    assert_hotel_disjoint, validate_label_order,
)


def build_status_targets(quads: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return any-pair, implicit-given-pair and conditional-valid multi-hot vectors."""
    any_pair = torch.zeros(NUM_LABELS, dtype=torch.float32)
    implicit_pair = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for quad in quads:
        category = quad.get("aspect_category")
        sentiment = str(quad.get("sentiment", "")).lower()
        if category not in CATEGORIES:
            raise ValueError(f"Unknown broad category: {category!r}")
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Unknown sentiment: {sentiment!r}")
        index = CATEGORIES.index(category) * len(SENTIMENTS) + SENTIMENTS.index(sentiment)
        any_pair[index] = 1.0
        if quad.get("aspect_implicit") is True:
            implicit_pair[index] = 1.0
    valid = any_pair.bool()
    # Conditional labels are meaningful only where the candidate pair occurs.
    implicit_pair[~valid] = 0.0
    return any_pair, implicit_pair, valid


class HybridCandidateDataset(Dataset):
    """One segment per item; identifiers and hotel are metadata only."""

    def __init__(self, root: str | Path, split: str):
        if split not in {"train", "dev"}:
            raise ValueError("V4 dataset accepts only official train and dev splits")
        self.root = Path(root).resolve()
        validate_label_order(self.root)
        path = self.root / "splits" / "single_image" / f"{split}.jsonl"
        self.records = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                any_pair, implicit_pair, valid = build_status_targets(row.get("quads", []))
                self.records.append({
                    "text": row["text"], "any_targets": any_pair,
                    "conditional_targets": implicit_pair, "conditional_mask": valid,
                    "metadata": {"segment_id": row["segment_id"],
                                 "source_review_id": row["source_review_id"],
                                 "hotel_id": _hotel_id(row["source_review_id"]),
                                 "split": split},
                })
        ids = [r["metadata"]["segment_id"] for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate segment IDs in {split}")
        self.hotel_ids = {r["metadata"]["hotel_id"] for r in self.records}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"text": row["text"], "any_targets": row["any_targets"].clone(),
                "conditional_targets": row["conditional_targets"].clone(),
                "conditional_mask": row["conditional_mask"].clone(),
                "metadata": dict(row["metadata"])}


def validate_split_pair(train: HybridCandidateDataset, dev: HybridCandidateDataset) -> None:
    assert_hotel_disjoint(train, dev)
    train_ids = {r["metadata"]["segment_id"] for r in train.records}
    dev_ids = {r["metadata"]["segment_id"] for r in dev.records}
    if train_ids & dev_ids:
        raise AssertionError("Train/dev segment IDs overlap")


def summarize_targets(dataset: HybridCandidateDataset) -> dict:
    any_positive = sum(int(r["any_targets"].sum()) for r in dataset.records)
    cond_valid = sum(int(r["conditional_mask"].sum()) for r in dataset.records)
    cond_positive = sum(int(r["conditional_targets"].sum()) for r in dataset.records)
    return {"segments": len(dataset), "candidate_rows": len(dataset) * NUM_LABELS,
            "any_positive_rows": any_positive,
            "any_negative_rows": len(dataset) * NUM_LABELS - any_positive,
            "conditional_valid_rows": cond_valid,
            "conditional_positive_rows": cond_positive,
            "conditional_negative_rows": cond_valid - cond_positive,
            "label_order": list(LABEL_ORDER)}


def audit_repeated_text_targets(*datasets: HybridCandidateDataset) -> dict:
    """Audit duplicate text records without collapsing or altering their targets.

    Normalization is Unicode-preserving casefold plus whitespace collapse; punctuation,
    accents, and token content are retained.
    """
    groups: dict[str, list[tuple[str, tuple[int, ...], tuple[int, ...]]]] = {}
    for dataset in datasets:
        for row in dataset.records:
            key = re.sub(r"\s+", " ", row["text"].casefold()).strip()
            signature = (tuple(row["any_targets"].to(torch.uint8).tolist()),
                         tuple(row["conditional_targets"].to(torch.uint8).tolist()))
            groups.setdefault(key, []).append((row["metadata"]["segment_id"], *signature))
    repeated = [rows for rows in groups.values() if len(rows) > 1]
    conflicts = [rows for rows in repeated if len({(r[1], r[2]) for r in rows}) > 1]
    return {"normalized_text_groups": len(groups), "repeated_text_groups": len(repeated),
            "records_in_repeated_groups": sum(len(rows) for rows in repeated),
            "conflicting_repeated_groups": len(conflicts),
            "conflicting_records": sum(len(rows) for rows in conflicts)}


def collate_hybrid(batch: list[dict], tokenizer, max_length: int = 256) -> dict:
    encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "any_targets": torch.stack([r["any_targets"] for r in batch]),
            "conditional_targets": torch.stack([r["conditional_targets"] for r in batch]),
            "conditional_mask": torch.stack([r["conditional_mask"] for r in batch]),
            "metadata": [r["metadata"] for r in batch]}
