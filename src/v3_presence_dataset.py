"""Train/dev-only segment dataset for implicit category-sentiment presence."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .label_mapping import build_label_mappings

TEXT_CHECKPOINT = "FacebookAI/xlm-roberta-base"
TEXT_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
CATEGORIES = ("AMENITY", "BRANDING", "EXPERIENCE", "FACILITY", "LOYALTY", "SERVICE")
SENTIMENTS = ("negative", "neutral", "positive")
LABEL_ORDER = tuple(f"{category} / {sentiment}" for category in CATEGORIES for sentiment in SENTIMENTS)
NUM_LABELS = 18


def validate_label_order(root: str | Path | None = None) -> None:
    """Fail if the V3 fixed order no longer agrees with the taxonomy mapping."""
    if root is None:
        return
    mappings = build_label_mappings(Path(root) / "taxonomy" / "hotel_absa_taxonomy.xlsx")
    categories = tuple(name for name, _ in sorted(mappings["aspect_categories"].items(), key=lambda x: x[1]))
    sentiments = tuple(name for name, _ in sorted(mappings["sentiments"].items(), key=lambda x: x[1]))
    if categories != CATEGORIES or sentiments != SENTIMENTS:
        raise ValueError(f"V3 label order mismatch: categories={categories}, sentiments={sentiments}")
    if len(LABEL_ORDER) != NUM_LABELS or len(set(LABEL_ORDER)) != NUM_LABELS:
        raise AssertionError("V3 label order must contain 18 unique labels")


def encode_implicit_presence(quads: list[dict]) -> tuple[torch.Tensor, dict]:
    """Return a binary multi-hot vector and counts, collapsing repeated pairs."""
    target = torch.zeros(NUM_LABELS, dtype=torch.float32)
    pairs: list[tuple[str, str]] = []
    for quad in quads:
        if quad.get("aspect_implicit") is not True:
            continue
        category = quad.get("aspect_category")
        sentiment = str(quad.get("sentiment", "")).lower()
        if category not in CATEGORIES:
            raise ValueError(f"Unknown broad category in implicit quad: {category!r}")
        if sentiment not in SENTIMENTS:
            raise ValueError(f"Unknown sentiment in implicit quad: {sentiment!r}")
        pairs.append((category, sentiment))
    for category, sentiment in set(pairs):
        index = CATEGORIES.index(category) * len(SENTIMENTS) + SENTIMENTS.index(sentiment)
        target[index] = 1.0
    validate_target(target)
    return target, {
        "implicit_quad_count": len(pairs),
        "unique_implicit_pair_count": len(set(pairs)),
        "collapsed_repeat_count": len(pairs) - len(set(pairs)),
    }


def validate_target(target: torch.Tensor) -> None:
    if target.shape != (NUM_LABELS,):
        raise ValueError(f"Target must have shape ({NUM_LABELS},), got {tuple(target.shape)}")
    if not torch.isfinite(target).all() or not torch.all((target == 0) | (target == 1)):
        raise ValueError("Target must contain only finite binary values")


def _hotel_id(source_review_id: str) -> str:
    # HAMOS review IDs follow HOTELID_Rxxxxx. Keep this only in metadata.
    match = re.match(r"^(H[^_]+)_R", source_review_id)
    if not match:
        raise ValueError(f"Cannot derive hotel grouping from source review ID {source_review_id!r}")
    return match.group(1)


def load_split_language_metadata(root: str | Path, segment_ids: set[str]) -> dict[str, str | None]:
    """Join language for requested train/dev IDs; decode no non-requested row fields."""
    result: dict[str, str | None] = {}
    path = Path(root) / "metadata" / "segments.jsonl"
    sid_pattern = re.compile(r'"segment_id"\s*:\s*("(?:\\.|[^"\\])*")')
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            match = sid_pattern.search(line)
            if not match:
                raise ValueError("Malformed segment metadata row: missing segment_id")
            sid = json.loads(match.group(1))
            if sid in segment_ids:
                row = json.loads(line)
                result[sid] = row.get("language")
                if len(result) == len(segment_ids):
                    break
    missing = segment_ids - set(result)
    if missing:
        raise ValueError(f"Missing language metadata for {len(missing)} requested split segments")
    return result


class ImplicitPresenceDataset(Dataset):
    """One record per official train/dev segment; IDs and language are metadata."""

    def __init__(self, root: str | Path, split: str, *, language_by_segment: dict | None = None):
        if split not in {"train", "dev"}:
            raise ValueError("V3 dataset accepts only official train and dev splits")
        self.root = Path(root).resolve()
        self.split = split
        validate_label_order(self.root)
        split_path = self.root / "splits" / "single_image" / f"{split}.jsonl"
        with split_path.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        self.records = []
        for row in rows:
            target, counts = encode_implicit_presence(row.get("quads", []))
            sid = row["segment_id"]
            self.records.append({
                "text": row["text"], "target": target,
                "metadata": {
                    "segment_id": sid,
                    "source_review_id": row["source_review_id"],
                    "hotel_id": _hotel_id(row["source_review_id"]),
                    "language": (language_by_segment or {}).get(sid),
                    **counts,
                },
            })
        hotel_ids = {record["metadata"]["hotel_id"] for record in self.records}
        self.hotel_ids = hotel_ids

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"text": row["text"], "target": row["target"].clone(), "metadata": dict(row["metadata"])}


def assert_hotel_disjoint(train: ImplicitPresenceDataset, dev: ImplicitPresenceDataset) -> None:
    overlap = train.hotel_ids & dev.hotel_ids
    if overlap:
        raise AssertionError(f"Train/dev hotel leakage: {len(overlap)} hotels overlap")


def collate_presence(batch: list[dict], tokenizer, max_length: int = 256) -> dict:
    encoded = tokenizer([row["text"] for row in batch], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
    targets = torch.stack([row["target"] for row in batch])
    if targets.shape != (len(batch), NUM_LABELS):
        raise AssertionError(f"Batch target shape mismatch: {tuple(targets.shape)}")
    return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
            "targets": targets, "metadata": [row["metadata"] for row in batch]}
