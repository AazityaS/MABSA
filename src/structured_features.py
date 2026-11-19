"""Frozen XLM-R/CLIP embedding cache extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data_loader import read_jsonl
from .structured_dataset import StructuredDataset


def load_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cache_key(config):
    fields = {k: config[k] for k in ("text_checkpoint", "text_revision", "image_checkpoint",
                                      "image_revision", "max_length", "image_size")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def resolve_cache_directory(cache_dir, config):
    """Accept either the feature-cache root or its exact keyed directory."""
    requested = Path(cache_dir).resolve()
    keyed = requested if (requested / "manifest.json").is_file() else requested / cache_key(config)
    manifest_path = keyed / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Feature cache manifest not found under {keyed}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_key = cache_key(config)
    if manifest.get("cache_key") != expected_key:
        raise ValueError(f"Feature cache key mismatch: expected {expected_key}, found {manifest.get('cache_key')}")
    return keyed


def extract_cache(root, config_path, output_root, *, device="auto", batch_size=None):
    root, output_root = Path(root).resolve(), Path(output_root).resolve()
    cfg = load_config(config_path)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA is unavailable to this PyTorch build; extracting features on CPU.")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false; no CPU fallback performed.")
    batch_size = batch_size or int(cfg["encoder_batch_size"])
    out = output_root / cache_key(cfg)
    out.mkdir(parents=True, exist_ok=True)
    text_path, image_path, manifest_path = out / "text_features.pt", out / "image_features.pt", out / "manifest.json"
    if text_path.exists() and image_path.exists() and manifest_path.exists():
        print(f"Feature cache already exists: {out}")
        return out

    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

    datasets = {s: StructuredDataset(root, s, mode="text", max_slots=23,
                    image_size=cfg["image_size"], load_images=False, verify_images=False,
                    require_candidate_images=False)
                for s in ("train", "dev", "test")}
    # Build an ordered ID/text set from the primary eligible segments only.
    segment_texts = {}
    image_paths = {}
    for ds in datasets.values():
        for row in ds.instances:
            segment_texts[row["segment_id"]] = row["text"]
            for rec in row["candidate_images"]:
                image_paths[rec["image_id"]] = rec["path"]
    print(f"Caching {len(segment_texts)} segment features and {len(image_paths)} unique candidate images on {device}.")

    tokenizer = AutoTokenizer.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"])
    text_model = AutoModel.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"])
    text_model.eval().to(device)
    for parameter in text_model.parameters(): parameter.requires_grad_(False)
    text_features = {}
    ids = list(segment_texts)
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            batch_ids = ids[start:start + batch_size]
            tokens = tokenizer([segment_texts[x] for x in batch_ids], padding=True, truncation=True,
                               max_length=cfg["max_length"], return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                features = text_model(**tokens).last_hidden_state[:, 0].float().cpu()
            text_features.update({sid: features[i] for i, sid in enumerate(batch_ids)})
    text_dim = int(text_model.config.hidden_size)
    text_commit = getattr(text_model.config, "_commit_hash", None) or cfg["text_revision"]
    del text_model, tokenizer
    if device.startswith("cuda"): torch.cuda.empty_cache()

    processor = CLIPImageProcessor.from_pretrained(cfg["image_checkpoint"], revision=cfg["image_revision"])
    vision_model = CLIPVisionModelWithProjection.from_pretrained(
        cfg["image_checkpoint"], revision=cfg["image_revision"])
    vision_model.eval().to(device)
    for parameter in vision_model.parameters(): parameter.requires_grad_(False)
    image_features = {}
    image_ids = list(image_paths)
    from .image_utils import load_rgb_image
    with torch.inference_mode():
        for start in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[start:start + batch_size]
            pixels = processor(images=[load_rgb_image(image_paths[x]) for x in batch_ids],
                               return_tensors="pt")["pixel_values"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                features = vision_model(pixel_values=pixels).image_embeds.float().cpu()
            image_features.update({iid: features[i] for i, iid in enumerate(batch_ids)})
    image_dim = int(vision_model.config.projection_dim)
    image_commit = getattr(vision_model.config, "_commit_hash", None) or cfg["image_revision"]
    del vision_model, processor
    if device.startswith("cuda"): torch.cuda.empty_cache()

    torch.save(text_features, text_path)
    torch.save(image_features, image_path)
    manifest = {"cache_key": cache_key(cfg), "text_checkpoint": cfg["text_checkpoint"],
                "text_revision_requested": cfg["text_revision"], "text_revision_resolved": text_commit,
                "image_checkpoint": cfg["image_checkpoint"], "image_revision_requested": cfg["image_revision"],
                "image_revision_resolved": image_commit, "max_length": cfg["max_length"],
                "image_size": cfg["image_size"], "text_dim": text_dim, "image_dim": image_dim,
                "device": device, "encoder_batch_size": batch_size,
                "segment_ids": sorted(text_features), "image_ids": sorted(image_features),
                "feature_inputs_only": True}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote feature cache: {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/structured_baseline.json")
    parser.add_argument("--output", default="cache/structured_features")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    extract_cache(args.root, args.config, args.output, device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
