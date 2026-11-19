"""Train one primary structured set-prediction condition from frozen features."""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .structured_dataset import StructuredDataset
from .structured_features import load_config, resolve_cache_directory
from .structured_loss import StructuredSetLoss
from .structured_metrics import decode_outputs, evaluate_structured, select_objectness_threshold
from .structured_model import StructuredSetModel, language_ids


class FeatureView(Dataset):
    def __init__(self, root, split, mode, text_features, image_features):
        self.base = StructuredDataset(root, split, mode="text", max_slots=23,
                                      load_images=False, verify_images=False)
        self.mode, self.text_features, self.image_features = mode, text_features, image_features
        missing = [r["segment_id"] for r in self.base.instances if r["segment_id"] not in text_features]
        if missing: raise ValueError(f"Feature cache lacks {len(missing)} segment IDs, e.g. {missing[:3]}")
        if mode == "multimodal":
            required = {x["image_id"] for r in self.base.instances for x in r["candidate_images"]}
            absent = required - image_features.keys()
            if absent: raise ValueError(f"Feature cache lacks candidate image IDs: {sorted(absent)[:3]}")

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        sid = item["metadata"]["segment_id"]
        result = {"text_feature": self.text_features[sid], "language": item["inputs"]["language"],
                  "targets": [{"category": x["aspect_category"], "sentiment": x["sentiment"],
                               "aspect_implicit": x["aspect_implicit"],
                               "opinion_implicit": x["opinion_implicit"]} for x in item["targets"]],
                  "metadata": item["metadata"]}
        if self.mode == "multimodal":
            result["image_features"] = [self.image_features[i] for i in item["metadata"]["candidate_image_ids"]]
        return result


def collate_features(samples, mode):
    batch = len(samples)
    max_targets = max(len(s["targets"]) for s in samples)
    targets = {"category": torch.zeros((batch, max_targets), dtype=torch.long),
               "sentiment": torch.zeros((batch, max_targets), dtype=torch.long),
               "aspect_implicit": torch.zeros((batch, max_targets), dtype=torch.float32),
               "opinion_implicit": torch.zeros((batch, max_targets), dtype=torch.float32),
               "mask": torch.zeros((batch, max_targets), dtype=torch.bool)}
    for b, sample in enumerate(samples):
        for j, target in enumerate(sample["targets"]):
            for key in ("category", "sentiment", "aspect_implicit", "opinion_implicit"):
                targets[key][b, j] = target[key]
            targets["mask"][b, j] = True
    inputs = {"text_features": torch.stack([s["text_feature"] for s in samples]).float(),
              "language_ids": language_ids([s["language"] for s in samples])}
    if mode == "multimodal":
        max_images = max(len(s["image_features"]) for s in samples)
        image_dim = samples[0]["image_features"][0].numel()
        image_tensor = torch.zeros((batch, max_images, image_dim), dtype=torch.float32)
        image_mask = torch.zeros((batch, max_images), dtype=torch.bool)
        for b, sample in enumerate(samples):
            for j, feature in enumerate(sample["image_features"]):
                image_tensor[b, j] = feature
                image_mask[b, j] = True
        inputs["image_features"], inputs["image_mask"] = image_tensor, image_mask
    return {"inputs": inputs, "targets": targets, "metadata": [s["metadata"] for s in samples]}


def _move(batch, device):
    return ({k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["inputs"].items()},
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch["targets"].items()})


def _slot_probability_samples(outputs, assignments):
    """Collect sigmoid objectness probabilities by Hungarian positive/negative slot."""
    probabilities = outputs["objectness"].detach().float().sigmoid().cpu()
    positive, negative = [], []
    for batch_index, (positive_indices, _) in enumerate(assignments):
        positive_set = set(positive_indices)
        if positive_indices:
            positive.extend(probabilities[batch_index, positive_indices].tolist())
        negative_indices = [i for i in range(probabilities.shape[1]) if i not in positive_set]
        if negative_indices:
            negative.extend(probabilities[batch_index, negative_indices].tolist())
    return positive, negative


def _probability_quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {"count": int(len(values)), "mean": float(values.mean()), "std": float(values.std()),
            **{f"p{q:02d}": float(value) for q, value in
               zip((1, 5, 25, 50, 75, 95, 99), quantiles)}}


def _cardinality_diagnostics(predictions):
    counts = np.asarray([len(items) for items in predictions], dtype=np.int64)
    unique, frequencies = np.unique(counts, return_counts=True)
    at_least_20 = int(np.sum(counts >= 20))
    return {"mean": float(counts.mean()), "median": float(np.median(counts)),
            "std": float(counts.std()), "min": int(counts.min()), "max": int(counts.max()),
            "histogram": {str(int(k)): int(v) for k, v in zip(unique, frequencies)},
            "at_least_20_count": at_least_20,
            "at_least_20_pct": float(100.0 * at_least_20 / len(counts))}


@torch.inference_mode()
def _run_dev(model, loader, mode, device, criterion, *, detailed_metrics=False):
    model.eval(); raw_outputs = []; gold_sets = []; hotel_ids = []; metadata = []; losses = []
    component_losses = {key: [] for key in
                        ("matched", "objectness_positive", "objectness_empty", "cardinality")}
    positive_scores, negative_scores = [], []
    for batch in loader:
        inputs, targets = _move(batch, device)
        outputs = model(inputs["text_features"], inputs["language_ids"],
                        inputs.get("image_features"), inputs.get("image_mask"))
        loss_out = criterion(outputs, targets)
        losses.append(float(loss_out["loss"].detach().cpu()))
        if detailed_metrics:
            for key in component_losses:
                component_losses[key].append(float(loss_out[key].detach().cpu()))
            pos, neg = _slot_probability_samples(outputs, loss_out["assignments"])
            positive_scores.extend(pos); negative_scores.extend(neg)
        raw_outputs.append({k: v.detach().cpu() for k, v in outputs.items()})
        for b in range(len(batch["metadata"])):
            count = int(targets["mask"][b].sum())
            gold_sets.append([{k: int(targets[k][b, j].item()) for k in
                               ("category", "sentiment", "aspect_implicit", "opinion_implicit")}
                              for j in range(count)])
        hotel_ids.extend(x["hotel_id"] for x in batch["metadata"])
        metadata.extend(batch["metadata"])
    outputs = {key: torch.cat([part[key] for part in raw_outputs], dim=0) for key in raw_outputs[0]}
    if detailed_metrics:
        loss_summary = {"total": float(np.mean(losses)),
                        **{key: float(np.mean(values)) for key, values in component_losses.items()},
                        "cardinality_weighted": float(
                            criterion.cardinality * np.mean(component_losses["cardinality"]))}
        slot_summary = {"positive": _probability_quantiles(positive_scores),
                        "negative": _probability_quantiles(negative_scores)}
    else:
        loss_summary, slot_summary = float(np.mean(losses)), None
    return outputs, gold_sets, hotel_ids, metadata, loss_summary, slot_summary


def train(root, config_path, cache_dir, output_dir, mode="text", device="auto",
          detailed_epoch_metrics=False):
    if mode not in {"text", "multimodal"}: raise ValueError("mode must be text or multimodal")
    run_started = datetime.now(ZoneInfo("Asia/Kolkata"))
    run_timer = time.perf_counter()
    cfg = load_config(config_path)
    cache_dir = resolve_cache_directory(cache_dir, cfg)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu": print("WARNING: CUDA is unavailable to PyTorch; training will run on CPU.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if device.startswith("cuda"): torch.cuda.manual_seed_all(cfg["seed"])
    text_features = torch.load(cache_dir / "text_features.pt", map_location="cpu", weights_only=True)
    image_features = torch.load(cache_dir / "image_features.pt", map_location="cpu", weights_only=True)
    train_ds = FeatureView(root, "train", mode, text_features, image_features)
    dev_ds = FeatureView(root, "dev", mode, text_features, image_features)
    batch_size = int(cfg["train_batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                              collate_fn=lambda rows: collate_features(rows, mode))
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                            collate_fn=lambda rows: collate_features(rows, mode))
    model = StructuredSetModel(text_dim=manifest["text_dim"], image_dim=manifest["image_dim"],
                               hidden_size=cfg["hidden_size"], slots=23, multimodal=mode == "multimodal").to(device)
    loss_cfg = cfg["loss_weights"]
    criterion = StructuredSetLoss(**loss_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    output_dir = Path(output_dir).resolve() / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    best_f1, best_epoch, stale = -1.0, 0, 0
    completed_epochs = 0
    stop_reason = "maximum_epochs_reached"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        epoch_timer = time.perf_counter()
        model.train(); losses = []
        train_components = {key: [] for key in
                            ("matched", "objectness_positive", "objectness_empty", "cardinality")}
        train_positive_scores, train_negative_scores = [], []
        gradient_norm_samples = {key: [] for key in
                                 ("objectness", "category", "sentiment", "aspect_implicit", "opinion_implicit")}
        for batch in train_loader:
            inputs, targets = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                outputs = model(inputs["text_features"], inputs["language_ids"],
                                inputs.get("image_features"), inputs.get("image_mask"))
                loss_out = criterion(outputs, targets)
            loss_out["loss"].backward()
            if detailed_epoch_metrics:
                heads = {"objectness": model.objectness_head, "category": model.category_head,
                         "sentiment": model.sentiment_head, "aspect_implicit": model.aspect_implicit_head,
                         "opinion_implicit": model.opinion_implicit_head}
                for key, head in heads.items():
                    squared = [parameter.grad.detach().float().square().sum()
                               for parameter in head.parameters() if parameter.grad is not None]
                    if squared:
                        gradient_norm_samples[key].append(torch.stack(squared).sum().sqrt())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss_out["loss"].detach().cpu()))
            if detailed_epoch_metrics:
                for key in train_components:
                    train_components[key].append(float(loss_out[key].detach().cpu()))
                pos, neg = _slot_probability_samples(outputs, loss_out["assignments"])
                train_positive_scores.extend(pos); train_negative_scores.extend(neg)
        dev_outputs, dev_gold, _, _, dev_losses, dev_slots = _run_dev(
            model, dev_loader, mode, device, criterion,
            detailed_metrics=detailed_epoch_metrics)
        selected = select_objectness_threshold(dev_outputs, dev_gold, cfg["objectness_threshold_grid"])
        dev_f1 = selected["development_primary_f1"]
        if detailed_epoch_metrics:
            predictions = decode_outputs(dev_outputs, selected["threshold"], cfg["flag_threshold"])
            mappings = train_ds.base.label_mappings
            category_names = [name for name, _ in sorted(mappings["aspect_categories"].items(),
                                                          key=lambda pair: pair[1])]
            sentiment_names = [name for name, _ in sorted(mappings["sentiments"].items(),
                                                           key=lambda pair: pair[1])]
            dev_metrics = evaluate_structured(predictions, dev_gold,
                                              category_names=category_names,
                                              sentiment_names=sentiment_names)
            primary_counts = selected["development_primary_counts"]
            primary_precision = (primary_counts["tp"] / (primary_counts["tp"] + primary_counts["fp"])
                                 if primary_counts["tp"] + primary_counts["fp"] else 0.0)
            primary_recall = (primary_counts["tp"] / (primary_counts["tp"] + primary_counts["fn"])
                              if primary_counts["tp"] + primary_counts["fn"] else 0.0)
            dev_diagnostics = {
                "primary_implicit_category_sentiment": {
                    "precision": primary_precision, "recall": primary_recall,
                    "f1": dev_f1, **primary_counts},
                "complete_set_f1": dev_metrics["complete_set"]["f1"],
                "cardinality": dev_metrics["cardinality"],
                "predicted_cardinality": _cardinality_diagnostics(predictions),
                "aspect_implicit": dev_metrics["aspect_implicit_positive"],
                "opinion_implicit": dev_metrics["opinion_implicit_positive"],
            }
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            peak_gpu_memory_mib = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        else:
            peak_gpu_memory_mib = None
        epoch_duration = time.perf_counter() - epoch_timer
        record = {"epoch": epoch, "train_loss": float(np.mean(losses)),
                  "dev_loss": dev_losses["total"] if detailed_epoch_metrics else dev_losses,
                  "dev_primary_implicit_f1": dev_f1,
                  "dev_threshold": selected["threshold"],
                  "epoch_duration_seconds": round(epoch_duration, 3),
                  "peak_gpu_memory_mib": peak_gpu_memory_mib}
        if detailed_epoch_metrics:
            record["train_loss_components"] = {
                "matched": float(np.mean(train_components["matched"])),
                "objectness_positive": float(np.mean(train_components["objectness_positive"])),
                "objectness_empty": float(np.mean(train_components["objectness_empty"])),
                "objectness_empty_weighted": float(
                    cfg["loss_weights"]["objectness_empty"] * np.mean(train_components["objectness_empty"])),
                "cardinality": float(np.mean(train_components["cardinality"])),
                "cardinality_weighted": float(
                    cfg["loss_weights"].get("cardinality", 0.0) *
                    np.mean(train_components["cardinality"]))}
            record["dev_loss_components"] = {
                **dev_losses,
                "objectness_empty_weighted": (
                    cfg["loss_weights"]["objectness_empty"] * dev_losses["objectness_empty"]),
                "cardinality_weighted": (
                    cfg["loss_weights"].get("cardinality", 0.0) * dev_losses["cardinality"])}
            record["objectness_probability_quantiles"] = {
                "train": {"positive": _probability_quantiles(train_positive_scores),
                          "negative": _probability_quantiles(train_negative_scores)},
                "dev": dev_slots}
            record["dev_diagnostics"] = dev_diagnostics
            record["gradient_norms_pre_clip_mean"] = {
                key: float(torch.stack(values).mean().detach().cpu()) if values else 0.0
                for key, values in gradient_norm_samples.items()}
        completed_epochs = epoch
        with log_path.open("a", encoding="utf-8") as handle: handle.write(json.dumps(record) + "\n")
        print(json.dumps(record))
        if dev_f1 > best_f1:
            best_f1, best_epoch, stale = dev_f1, epoch, 0
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "mode": mode,
                          "model": {"text_dim": manifest["text_dim"], "image_dim": manifest["image_dim"],
                                    "hidden_size": cfg["hidden_size"], "slots": 23},
                          "config": cfg, "cache_key": manifest["cache_key"],
                          "objectness_threshold": selected["threshold"], "flag_threshold": cfg["flag_threshold"],
                          "best_epoch": epoch, "dev_primary_counts": selected["development_primary_counts"]}
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")
        else:
            stale += 1
        if stale >= int(cfg["early_stopping_patience"]):
            stop_reason = "early_stopping_patience"
            break
    run_ended = datetime.now(ZoneInfo("Asia/Kolkata"))
    run_duration = time.perf_counter() - run_timer
    summary = {"mode": mode, "started_at": run_started.isoformat(), "ended_at": run_ended.isoformat(),
               "duration_seconds": round(run_duration, 3), "completed_epochs": completed_epochs,
               "best_epoch": best_epoch, "best_dev_primary_implicit_f1": best_f1,
               "early_stopping_patience": int(cfg["early_stopping_patience"]),
               "stop_reason": stop_reason, "checkpoint": str(output_dir / "best_checkpoint.pt"),
               "warnings": [], "errors": []}
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {output_dir / 'best_checkpoint.pt'} (epoch={best_epoch}, dev F1={best_f1:.6f})")
    print(json.dumps(summary))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--config", default="configs/structured_baseline.json")
    p.add_argument("--cache-dir", default="cache/structured_features")
    p.add_argument("--output-dir", default="runs/structured")
    p.add_argument("--mode", required=True, choices=("text", "multimodal"))
    p.add_argument("--device", default="auto")
    p.add_argument("--detailed-epoch-metrics", action="store_true",
                   help="log train/dev loss components, objectness quantiles and dev set diagnostics")
    a = p.parse_args()
    train(a.root, a.config, a.cache_dir, a.output_dir, a.mode, a.device,
          detailed_epoch_metrics=a.detailed_epoch_metrics)


if __name__ == "__main__": main()
