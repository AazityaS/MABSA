"""Alignment view for cached XLM-R rows and V3 multi-hot targets."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .v3_presence_dataset import NUM_LABELS, validate_target
from .v3_frozen_cache import EXPECTED_DIM


class FrozenPresenceDataset(Dataset):
    def __init__(self, segment_dataset, feature_map: dict[str, torch.Tensor]):
        records = segment_dataset.records
        ids = [row["metadata"]["segment_id"] for row in records]
        if len(ids) != len(set(ids)):
            raise ValueError("Segment dataset contains duplicate IDs")
        if set(ids) != set(feature_map):
            missing = set(ids) - set(feature_map)
            extra = set(feature_map) - set(ids)
            raise ValueError(f"Feature/dataset ID mismatch: missing={len(missing)}, extra={len(extra)}")
        self.records = []
        for row in records:
            sid = row["metadata"]["segment_id"]
            feature = feature_map[sid]
            target = row["target"]
            if feature.shape != (EXPECTED_DIM,) or not bool(torch.isfinite(feature).all()):
                raise ValueError(f"Invalid cached feature for selected segment ({sid})")
            if target.shape != (NUM_LABELS,):
                raise ValueError(f"Invalid target shape for selected segment ({sid})")
            validate_target(target)
            self.records.append({"segment_id": sid, "feature": feature.float().clone(),
                                 "target": target.float().clone(), "metadata": dict(row["metadata"])})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return {"features": row["feature"].clone(), "target": row["target"].clone(),
                "metadata": dict(row["metadata"])}


def collate_frozen(batch):
    features = torch.stack([row["features"] for row in batch])
    targets = torch.stack([row["target"] for row in batch])
    if features.ndim != 2 or features.shape[1] != EXPECTED_DIM:
        raise AssertionError(f"Frozen batch feature shape mismatch: {tuple(features.shape)}")
    if targets.shape != (len(batch), NUM_LABELS):
        raise AssertionError(f"Frozen batch target shape mismatch: {tuple(targets.shape)}")
    return {"features": features, "targets": targets, "metadata": [row["metadata"] for row in batch]}
