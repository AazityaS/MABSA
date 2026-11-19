"""Train the isolated linear V3-Frozen model on cache-backed train/dev rows only."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .v3_frozen_cache import EXPECTED_DIM, load_train_dev_text_features
from .v3_frozen_dataset import FrozenPresenceDataset, collate_frozen
from .v3_frozen_model import FrozenXLMRPresenceHead
from .v3_presence_dataset import (LABEL_ORDER, ImplicitPresenceDataset, assert_hotel_disjoint)
from .v3_presence_metrics import evaluate_presence, select_presence_threshold


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _evaluate(model, loader, device):
    model.eval()
    losses, probabilities, targets = [], [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for batch in loader:
            features = batch["features"].to(device)
            gold = batch["targets"].to(device)
            logits = model(features)
            loss = criterion(logits, gold)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(gold.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probabilities), np.concatenate(targets)


def train_frozen(root, config_path, output_dir):
    root, config_path, output_dir = Path(root).resolve(), Path(config_path).resolve(), Path(output_dir).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V3-Frozen; CPU fallback is disabled")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V3-Frozen run: {output_dir}")
    if int(cfg["text_dimension"]) != EXPECTED_DIM:
        raise ValueError("Frozen config feature dimension does not match the validated cache")
    _seed(int(cfg["seed"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    cache_dir = Path(cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    train_features, dev_features, cache_read_stats = load_train_dev_text_features(
        root, cache_dir, return_stats=True)
    train_source = ImplicitPresenceDataset(root, "train")
    dev_source = ImplicitPresenceDataset(root, "dev")
    assert_hotel_disjoint(train_source, dev_source)
    train_ds = FrozenPresenceDataset(train_source, train_features)
    dev_ds = FrozenPresenceDataset(dev_source, dev_features)

    provenance = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if provenance.get("cache_key") != cfg.get("cache_key"):
        raise ValueError(f"Frozen config/cache key mismatch: {cfg.get('cache_key')} vs {provenance.get('cache_key')}")
    selected_provenance = {key: provenance.get(key) for key in (
        "cache_key", "text_checkpoint", "text_revision_requested", "text_revision_resolved",
        "text_dim", "max_length", "feature_inputs_only")}
    selected_provenance.update({"train_embedding_rows": len(train_features),
                                "dev_embedding_rows": len(dev_features),
                                "selective_read_stats": cache_read_stats,
                                "train_dev_hotel_disjoint": True,
                                "pooling": "XLM-R last_hidden_state[:,0] (<s> token)",
                                "test_embedding_bytes_read": False})
    (output_dir / "cache_provenance.json").write_text(json.dumps(selected_provenance, indent=2), encoding="utf-8")
    support = {
        "label_order": list(LABEL_ORDER),
        "train_positive_segments": {label: int(sum(row["target"][i].item() for row in train_ds.records))
                                    for i, label in enumerate(LABEL_ORDER)},
        "dev_positive_segments": {label: int(sum(row["target"][i].item() for row in dev_ds.records))
                                  for i, label in enumerate(LABEL_ORDER)},
        "train_segments": len(train_ds), "dev_segments": len(dev_ds),
        "train_all_zero_targets": int(sum(not bool(row["target"].sum()) for row in train_ds.records)),
        "dev_all_zero_targets": int(sum(not bool(row["target"].sum()) for row in dev_ds.records)),
    }
    (output_dir / "label_supports.json").write_text(json.dumps(support, indent=2), encoding="utf-8")

    batch_size = int(cfg["batch_size"])
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_frozen, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_frozen, pin_memory=True)
    device = torch.device("cuda")
    model = FrozenXLMRPresenceHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    criterion = torch.nn.BCEWithLogitsLoss()
    best_f1, best_epoch, stale = -1.0, None, 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            train_losses = []
            for batch in train_loader:
                features = batch["features"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = criterion(logits, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite V3-Frozen loss at epoch {epoch}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(float(loss.item()))
            dev_loss, dev_prob, dev_gold = _evaluate(model, dev_loader, device)
            selection = select_presence_threshold(dev_prob, dev_gold, cfg["threshold_grid"])
            metrics = evaluate_presence(dev_prob, dev_gold, selection["threshold"])
            total = len(dev_gold) * len(LABEL_ORDER)
            metrics["micro"]["tn"] = int(total - metrics["micro"]["tp"] - metrics["micro"]["fp"] - metrics["micro"]["fn"])
            for i, label in enumerate(LABEL_ORDER):
                row = metrics["per_label"][label]
                row["tn"] = int(len(dev_gold) - row["tp"] - row["fp"] - row["fn"])
            epoch_row = {"epoch": epoch, "train_bce": float(np.mean(train_losses)),
                         "dev_bce": dev_loss, "threshold_selection": selection,
                         "dev_metrics": metrics}
            log.write(json.dumps(epoch_row, allow_nan=False) + "\n")
            log.flush()
            if metrics["micro"]["f1"] > best_f1:
                best_f1, best_epoch, stale = metrics["micro"]["f1"], epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "threshold": selection["threshold"],
                            "epoch": epoch, "dev_metrics": metrics, "label_order": list(LABEL_ORDER),
                            "config": cfg, "cache_provenance": selected_provenance},
                           output_dir / "best_checkpoint.pt")
            else:
                stale += 1
            print(f"epoch={epoch} train_bce={epoch_row['train_bce']:.6f} dev_bce={dev_loss:.6f} "
                  f"dev_micro_f1={metrics['micro']['f1']:.6f} threshold={selection['threshold']:.2f}")
            if stale >= int(cfg["early_stopping_patience"]):
                break

    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, prob, gold = _evaluate(model, dev_loader, device)
    final_metrics = evaluate_presence(prob, gold, float(checkpoint["threshold"]))
    total = len(gold) * len(LABEL_ORDER)
    final_metrics["micro"]["tn"] = int(total - final_metrics["micro"]["tp"] - final_metrics["micro"]["fp"] - final_metrics["micro"]["fn"])
    for label in LABEL_ORDER:
        row = final_metrics["per_label"][label]
        row["tn"] = int(len(gold) - row["tp"] - row["fp"] - row["fn"])
    (output_dir / "best_dev_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    metadata = {"variant": "V3-Frozen", "seed": int(cfg["seed"]), "device": torch.cuda.get_device_name(0),
                "train_segments": len(train_ds), "dev_segments": len(dev_ds), "best_epoch": best_epoch,
                "selected_threshold": float(checkpoint["threshold"]), "dev_micro_f1": best_f1,
                "elapsed_seconds": time.perf_counter() - start,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "test_split_loaded": False, "test_embedding_values_read": False}
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return final_metrics, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_presence_v3_frozen.json")
    parser.add_argument("--output-dir", default="runs/structured/challenger_v3/frozen")
    args = parser.parse_args()
    train_frozen(args.root, args.config, args.output_dir)


if __name__ == "__main__":
    main()
