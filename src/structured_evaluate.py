"""Evaluate selected checkpoints; test threshold is loaded from dev-selected checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .structured_dataset import StructuredDataset
from .structured_metrics import (decode_outputs, evaluate_by_language, evaluate_structured,
                                 paired_hotel_bootstrap)
from .structured_model import StructuredSetModel
from .structured_train import FeatureView, _move, collate_features


@torch.inference_mode()
def evaluate_checkpoint(root, checkpoint_path, cache_dir, *, device="auto", batch_size=32):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mode, cfg = checkpoint["mode"], checkpoint["config"]
    cache_path = Path(cache_dir).resolve() / checkpoint["cache_key"]
    manifest = json.loads((cache_path / "manifest.json").read_text(encoding="utf-8"))
    if device == "auto": device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    text_features = torch.load(cache_path / "text_features.pt", map_location="cpu", weights_only=True)
    image_features = torch.load(cache_path / "image_features.pt", map_location="cpu", weights_only=True)
    dataset = FeatureView(root, "test", mode, text_features, image_features)
    label_mappings = dataset.base.label_mappings
    category_names = [name for name, _ in sorted(label_mappings["aspect_categories"].items(),
                                                  key=lambda pair: pair[1])]
    sentiment_names = [name for name, _ in sorted(label_mappings["sentiments"].items(),
                                                   key=lambda pair: pair[1])]
    language_by_segment = {row["segment_id"]: row["language"] for row in dataset.base.instances}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=lambda rows: collate_features(rows, mode))
    model = StructuredSetModel(**checkpoint["model"], multimodal=mode == "multimodal").to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    outputs_all, gold_sets, hotels, segment_ids, languages = [], [], [], [], []
    for batch in loader:
        inputs, targets = _move(batch, device)
        outputs = model(inputs["text_features"], inputs["language_ids"],
                        inputs.get("image_features"), inputs.get("image_mask"))
        outputs_all.append({k: v.cpu() for k, v in outputs.items()})
        for b in range(len(batch["metadata"])):
            count = int(targets["mask"][b].sum())
            gold_sets.append([{k: int(targets[k][b, j].item()) for k in
                               ("category", "sentiment", "aspect_implicit", "opinion_implicit")}
                              for j in range(count)])
        hotels.extend(x["hotel_id"] for x in batch["metadata"])
        segment_ids.extend(x["segment_id"] for x in batch["metadata"])
        languages.extend(language_by_segment[sid] for sid in segment_ids[-len(batch["metadata"]):])
    outputs_all = {k: torch.cat([part[k] for part in outputs_all], dim=0) for k in outputs_all[0]}
    predictions = decode_outputs(outputs_all, checkpoint["objectness_threshold"], checkpoint["flag_threshold"])
    metrics = evaluate_structured(predictions, gold_sets, category_names=category_names,
                                  sentiment_names=sentiment_names)
    per_language = evaluate_by_language(predictions, gold_sets, languages,
                                        category_names=category_names,
                                        sentiment_names=sentiment_names)
    return {"checkpoint": str(Path(checkpoint_path).resolve()), "mode": mode,
            "objectness_threshold": checkpoint["objectness_threshold"],
            "feature_cache_key": manifest["cache_key"], "metrics": metrics,
            "per_language": per_language,
            "test_support": {"eligible_segments": len(segment_ids),
                             "hotels": len(set(hotels)),
                             "gold_implicit_quads": sum(g["aspect_implicit"] for gold in gold_sets for g in gold)},
            "predictions": predictions, "gold": gold_sets, "hotels": hotels,
            "segment_ids": segment_ids, "languages": languages}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--cache-dir", default="cache/structured_features")
    p.add_argument("--text-checkpoint")
    p.add_argument("--multimodal-checkpoint")
    p.add_argument("--output", default="runs/structured/test_evaluation.json")
    p.add_argument("--device", default="auto")
    p.add_argument("--bootstrap-replicates", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=1729)
    a = p.parse_args()
    results = {}
    if a.text_checkpoint:
        results["text"] = evaluate_checkpoint(a.root, a.text_checkpoint, a.cache_dir, device=a.device)
    if a.multimodal_checkpoint:
        results["multimodal"] = evaluate_checkpoint(a.root, a.multimodal_checkpoint, a.cache_dir, device=a.device)
    if not results: p.error("provide --text-checkpoint and/or --multimodal-checkpoint")
    output = {mode: {k: v for k, v in result.items() if k not in
                     {"predictions", "gold", "hotels", "segment_ids", "languages"}}
              for mode, result in results.items()}
    if "text" in results and "multimodal" in results:
        t, m = results["text"], results["multimodal"]
        if (t["segment_ids"] != m["segment_ids"] or t["gold"] != m["gold"] or
                t["languages"] != m["languages"] or t["hotels"] != m["hotels"]):
            raise AssertionError("Primary test examples/gold sets differ across modalities")
        output["paired_hotel_bootstrap"] = paired_hotel_bootstrap(
            t["predictions"], m["predictions"], t["gold"], t["hotels"],
            replicates=a.bootstrap_replicates, seed=a.bootstrap_seed)
    dest = Path(a.output).resolve(); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Wrote evaluation: {dest}")


if __name__ == "__main__": main()
