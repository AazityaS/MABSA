"""Evaluate a V3 checkpoint on train or dev only; test is intentionally unsupported."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .v3_presence_dataset import (ImplicitPresenceDataset, LABEL_ORDER, collate_presence,
                                  load_split_language_metadata)
from .v3_presence_metrics import evaluate_presence
from .v3_presence_model import XLMRImplicitPresenceModel


def evaluate_checkpoint(root, config_path, checkpoint_path, split="dev", device_name="cuda"):
    if split not in {"train", "dev"}:
        raise ValueError("V3 evaluator permits only train and dev; test evaluation is blocked")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    root, cfg = Path(root).resolve(), json.loads(Path(config_path).read_text(encoding="utf-8"))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("label_order") != list(LABEL_ORDER):
        raise ValueError("Checkpoint label ordering does not match current V3 mapping")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"],
                                              local_files_only=cfg.get("local_files_only", True))
    dataset = ImplicitPresenceDataset(root, split)
    languages = load_split_language_metadata(root, {r["metadata"]["segment_id"] for r in dataset.records})
    for row in dataset.records:
        row["metadata"]["language"] = languages[row["metadata"]["segment_id"]]
    loader = DataLoader(dataset, batch_size=int(cfg["micro_batch_size"]), shuffle=False,
                        num_workers=0, collate_fn=lambda rows: collate_presence(rows, tokenizer, int(cfg["max_length"])))
    device = torch.device(device_name)
    model = XLMRImplicitPresenceModel(cfg["text_checkpoint"], cfg["text_revision"],
                                     dropout=float(cfg["dropout"]),
                                     local_files_only=cfg.get("local_files_only", True)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    probabilities, targets = [], []
    amp = device.type == "cuda" and cfg.get("mixed_precision") == "fp16"
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: batch[k].to(device) for k in ("input_ids", "attention_mask")}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**inputs)
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(batch["targets"].numpy())
    probs, gold = np.concatenate(probabilities), np.concatenate(targets)
    return {"split": split, "segments": len(dataset), "threshold": float(ckpt["threshold"]),
            "metrics": evaluate_presence(probs, gold, float(ckpt["threshold"])),
            "test_split_loaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_presence_v3.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="dev")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite V3 evaluation result: {output}")
    result = evaluate_checkpoint(args.root, args.config, args.checkpoint, args.split, args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
