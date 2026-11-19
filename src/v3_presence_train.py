"""Train the isolated V3 text-only implicit pair-presence model (train/dev only)."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .v3_presence_dataset import (LABEL_ORDER, ImplicitPresenceDataset, assert_hotel_disjoint,
                                  collate_presence, load_split_language_metadata)
from .v3_presence_metrics import evaluate_presence, select_presence_threshold
from .v3_presence_model import XLMRImplicitPresenceModel


def compute_train_pos_weight(train_targets: torch.Tensor, cap: float = 20.0) -> tuple[torch.Tensor, dict]:
    """Derive capped BCE positive weights from training labels only.

    Zero-support labels receive weight 1.0 and are explicitly listed as unsupported;
    because there are no train positives, the model can only receive negative BCE
    supervision for those labels.
    """
    if train_targets.ndim != 2 or train_targets.shape[1] != len(LABEL_ORDER):
        raise ValueError("Training targets must have shape [N,18]")
    if not torch.isfinite(train_targets).all() or not torch.all((train_targets == 0) | (train_targets == 1)):
        raise ValueError("Training targets must be finite and binary")
    if cap < 1:
        raise ValueError("Positive-weight cap must be at least 1")
    positive = train_targets.float().sum(dim=0)
    negative = train_targets.shape[0] - positive
    weights = torch.ones_like(positive)
    supported = positive > 0
    weights[supported] = torch.clamp(negative[supported] / positive[supported], max=float(cap))
    return weights, {
        "source": "train_targets_only", "cap": float(cap),
        "positive_support": {label: int(positive[i].item()) for i, label in enumerate(LABEL_ORDER)},
        "negative_support": {label: int(negative[i].item()) for i, label in enumerate(LABEL_ORDER)},
        "pos_weight": {label: float(weights[i].item()) for i, label in enumerate(LABEL_ORDER)},
        "zero_train_support": [LABEL_ORDER[i] for i in range(len(LABEL_ORDER)) if not supported[i]],
        "zero_support_handling": "weight=1; negative-only BCE; positive class cannot be learned without train positives",
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _eval_epoch(model, loader, device, amp_enabled):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    losses, probabilities, targets = [], [], []
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: batch[k].to(device, non_blocking=True) for k in ("input_ids", "attention_mask")}
            gold = batch["targets"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(**inputs)
                loss = criterion(logits.float(), gold)
            losses.append(float(loss.item()))
            probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
            targets.append(gold.cpu().numpy())
    return float(np.mean(losses)), np.concatenate(probabilities), np.concatenate(targets)


def train(root: str | Path, config_path: str | Path, output_dir: str | Path, *, device_name="cuda") -> Path:
    root, config_path, output_dir = Path(root).resolve(), Path(config_path).resolve(), Path(output_dir).resolve()
    cfg = _load_config(config_path)
    if device_name != "cuda":
        raise ValueError("V3 training is configured for CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V3 training will not fall back to CPU")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite V3 run directory: {output_dir}")
    if cfg.get("class_weighted", False) and "weighted" not in output_dir.name.lower():
        raise ValueError("Weighted V3 runs must use a separate output directory whose name includes 'weighted'")
    _set_seed(int(cfg["seed"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    train_ds = ImplicitPresenceDataset(root, "train")
    dev_ds = ImplicitPresenceDataset(root, "dev")
    assert_hotel_disjoint(train_ds, dev_ds)
    languages = {}
    languages.update(load_split_language_metadata(root, {r["metadata"]["segment_id"] for r in train_ds.records}))
    languages.update(load_split_language_metadata(root, {r["metadata"]["segment_id"] for r in dev_ds.records}))
    for dataset in (train_ds, dev_ds):
        for record in dataset.records:
            record["metadata"]["language"] = languages[record["metadata"]["segment_id"]]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"],
                                              local_files_only=cfg.get("local_files_only", True))
    collate = lambda rows: collate_presence(rows, tokenizer, int(cfg["max_length"]))
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=int(cfg["micro_batch_size"]), shuffle=True,
                              generator=generator, num_workers=int(cfg["num_workers"]), collate_fn=collate,
                              pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=int(cfg["micro_batch_size"]), shuffle=False,
                            num_workers=int(cfg["num_workers"]), collate_fn=collate, pin_memory=True)

    train_targets = torch.stack([row["target"] for row in train_ds.records])
    supports = {label: int(train_targets[:, i].sum().item()) for i, label in enumerate(LABEL_ORDER)}
    if cfg.get("class_weighted", False):
        pos_weight, weight_details = compute_train_pos_weight(train_targets, float(cfg["class_weight_cap"]))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to("cuda"))
    else:
        pos_weight = torch.ones(len(LABEL_ORDER), dtype=torch.float32)
        weight_details = {"source": "disabled", "pos_weight": None, "zero_train_support": []}
        criterion = torch.nn.BCEWithLogitsLoss()
    (output_dir / "class_weighting.json").write_text(json.dumps(weight_details, indent=2), encoding="utf-8")
    (output_dir / "label_supports.json").write_text(json.dumps({
        "train_positive_segments": supports,
        "dev_positive_segments": {label: int(sum(r["target"][i].item() for r in dev_ds.records))
                                  for i, label in enumerate(LABEL_ORDER)},
        "train_segments_with_any_positive": int((train_targets.sum(dim=1) > 0).sum().item()),
        "dev_segments_with_any_positive": int(sum(bool(r["target"].sum()) for r in dev_ds.records)),
        "train_implicit_quads": int(sum(r["metadata"]["implicit_quad_count"] for r in train_ds.records)),
        "train_collapsed_quads": int(sum(r["metadata"]["collapsed_repeat_count"] for r in train_ds.records)),
        "dev_implicit_quads": int(sum(r["metadata"]["implicit_quad_count"] for r in dev_ds.records)),
        "dev_collapsed_quads": int(sum(r["metadata"]["collapsed_repeat_count"] for r in dev_ds.records)),
    }, indent=2), encoding="utf-8")

    model = XLMRImplicitPresenceModel(cfg["text_checkpoint"], cfg["text_revision"],
                                     dropout=float(cfg["dropout"]),
                                     local_files_only=cfg.get("local_files_only", True)).to("cuda")
    if cfg.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    accumulation = int(cfg["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = updates_per_epoch * int(cfg["max_epochs"])
    warmup_steps = int(total_steps * float(cfg["warmup_ratio"]))

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = cfg.get("mixed_precision") == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_f1, best_epoch, stale_epochs = -1.0, None, 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, int(cfg["max_epochs"]) + 1):
            model.train()
            losses = []
            optimizer.zero_grad(set_to_none=True)
            n_batches = len(train_loader)
            for batch_index, batch in enumerate(train_loader):
                inputs = {k: batch[k].to("cuda", non_blocking=True) for k in ("input_ids", "attention_mask")}
                gold = batch["targets"].to("cuda", non_blocking=True)
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, n_batches - group_start)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                    logits = model(**inputs)
                    loss = criterion(logits.float(), gold)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
                losses.append(float(loss.item()))
                scaler.scale(loss / group_size).backward()
                if (batch_index + 1) % accumulation == 0 or batch_index + 1 == n_batches:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
            dev_loss, dev_prob, dev_gold = _eval_epoch(model, dev_loader, torch.device("cuda"), amp_enabled)
            selected = select_presence_threshold(dev_prob, dev_gold, cfg["threshold_grid"])
            metrics = evaluate_presence(dev_prob, dev_gold, selected["threshold"])
            row = {"epoch": epoch, "train_bce": float(np.mean(losses)), "dev_bce": dev_loss,
                   "dev_threshold_selection": selected, "dev_metrics": metrics,
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            if metrics["micro"]["f1"] > best_f1:
                best_f1, best_epoch, stale_epochs = metrics["micro"]["f1"], epoch, 0
                torch.save({"model_state_dict": model.state_dict(), "threshold": selected["threshold"],
                            "epoch": epoch, "dev_metrics": metrics, "label_order": list(LABEL_ORDER),
                            "text_checkpoint": cfg["text_checkpoint"], "text_revision": cfg["text_revision"],
                            "config": cfg}, output_dir / "best_checkpoint.pt")
            else:
                stale_epochs += 1
            print(f"epoch={epoch} train_bce={row['train_bce']:.6f} dev_bce={dev_loss:.6f} "
                  f"dev_micro_f1={metrics['micro']['f1']:.6f} threshold={selected['threshold']:.2f}")
            if stale_epochs >= int(cfg["early_stopping_patience"]):
                break
    metadata = {"seed": int(cfg["seed"]), "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                "transformers_version": __import__("transformers").__version__,
                "best_epoch": best_epoch, "best_dev_micro_f1": best_f1,
                "elapsed_seconds": time.perf_counter() - start,
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                "train_segments": len(train_ds), "dev_segments": len(dev_ds),
                "test_split_loaded": False, "class_weighted": bool(cfg["class_weighted"])}
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_presence_v3.json")
    parser.add_argument("--output-dir", default="runs/structured/challenger_v3/text")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.root, args.config, args.output_dir, device_name=args.device)


if __name__ == "__main__":
    main()
