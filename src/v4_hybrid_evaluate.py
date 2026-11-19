"""Development-only evaluator for an isolated HAMOS V4 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .v3_presence_dataset import TEXT_CHECKPOINT, TEXT_REVISION
from .v4_hybrid_dataset import HybridCandidateDataset, collate_hybrid
from .v4_hybrid_metrics import evaluate_hybrid, select_combined_threshold
from .v4_hybrid_model import HybridCandidateModel, load_pinned_encoder


def predict(model, loader, device):
    model.eval()
    values = {key: [] for key in ("combined", "any", "conditional", "targets", "any_targets", "conditional_targets", "mask")}
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            for key, src in (("combined", "combined_probabilities"), ("any", "any_probabilities"),
                             ("conditional", "conditional_probabilities")):
                values[key].append(out[src].float().cpu().numpy())
            for key in ("targets", "any_targets", "conditional_targets", "mask"):
                src = {"targets": "conditional_targets", "mask": "conditional_mask"}.get(key, key)
                values[key].append(batch[src].cpu().numpy())
    return {k: np.concatenate(v) for k, v in values.items()}


def evaluate_checkpoint(root, checkpoint_path, output_path, device="cuda"):
    checkpoint_path = Path(checkpoint_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["text_revision"] != TEXT_REVISION or cfg["text_checkpoint"] != TEXT_CHECKPOINT:
        raise ValueError("Checkpoint encoder provenance mismatch")
    ds = HybridCandidateDataset(root, "dev")
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT, revision=TEXT_REVISION,
                                               local_files_only=True)
    loader = DataLoader(ds, batch_size=int(cfg["micro_batch_size"]), shuffle=False,
                        collate_fn=lambda b: collate_hybrid(b, tokenizer, cfg["max_length"]))
    model = HybridCandidateModel(load_pinned_encoder(True), hidden_size=cfg["hidden_size"],
        category_embedding_dim=cfg["category_embedding_dim"],
        sentiment_embedding_dim=cfg["sentiment_embedding_dim"], dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    arrays = predict(model, loader, torch.device(device))
    dev_threshold_sweep = select_combined_threshold(arrays["combined"], arrays["targets"])
    metrics_at_checkpoint_threshold = evaluate_hybrid(arrays["combined"], arrays["targets"], arrays["any"],
        arrays["any_targets"], arrays["conditional"], arrays["conditional_targets"],
        arrays["mask"], float(ckpt["threshold"]))
    selected_threshold = float(dev_threshold_sweep["threshold"])
    metrics_at_selected_threshold = evaluate_hybrid(arrays["combined"], arrays["targets"], arrays["any"],
        arrays["any_targets"], arrays["conditional"], arrays["conditional_targets"],
        arrays["mask"], selected_threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"split": "dev", "test_loaded": False,
        "threshold": selected_threshold, "threshold_before_dev_recheck": ckpt["threshold"],
        "development_threshold_sweep": dev_threshold_sweep,
        "threshold_recomputed_matches_checkpoint": float(dev_threshold_sweep["threshold"]) == float(ckpt["threshold"]),
        "metrics_at_checkpoint_threshold": metrics_at_checkpoint_threshold,
        "metrics": metrics_at_selected_threshold}, indent=2, allow_nan=False), encoding="utf-8")
    return metrics_at_selected_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    evaluate_checkpoint(args.root, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
