"""Small real-checkpoint CUDA smoke test and extraction-rate benchmark.

This utility processes eight training segment texts and eight metadata-derived
candidate images. It never reads test labels and never writes a full feature cache.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_

from .structured_dataset import StructuredDataset
from .structured_features import load_config
from .structured_loss import StructuredSetLoss
from .structured_model import StructuredSetModel, language_ids
from .image_utils import load_rgb_image


def _finite(tensor):
    return bool(torch.isfinite(tensor).all().item())


def _peak_mib():
    return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)


def _cached_snapshot_exists(repo_id, revision):
    from huggingface_hub.constants import HF_HUB_CACHE

    repo_folder = "models--" + repo_id.replace("/", "--")
    return (Path(HF_HUB_CACHE) / repo_folder / "snapshots" / revision).is_dir()


def run(root, config_path, *, sample_size=8, encoder_batch_size=2):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; real GPU smoke test will not fall back to CPU")
    root = Path(root).resolve()
    cfg = load_config(config_path)
    device = torch.device("cuda")
    dataset = StructuredDataset(root, "train", mode="text", load_images=False,
                                verify_images=False, require_candidate_images=True)

    text_rows = [row for row in dataset.instances if row["text"].strip()][:sample_size]
    if len(text_rows) != sample_size:
        raise RuntimeError(f"Only found {len(text_rows)} text rows; expected {sample_size}")
    # Build an image sample solely from review-level candidate metadata.
    candidate_by_id = {}
    for row in dataset.instances:
        for record in row["candidate_images"]:
            candidate_by_id.setdefault(record["image_id"], record)
            if len(candidate_by_id) >= sample_size:
                break
        if len(candidate_by_id) >= sample_size:
            break
    image_records = list(candidate_by_id.values())[:sample_size]
    if len(image_records) != sample_size:
        raise RuntimeError(f"Only found {len(image_records)} candidate images; expected {sample_size}")

    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

    # Text encoder: load the exact revision and time only the representative extraction loop.
    torch.cuda.reset_peak_memory_stats()
    start_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"])
    text_encoder = AutoModel.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"])
    text_load_seconds = time.perf_counter() - start_load
    text_encoder.eval().to(device)
    for param in text_encoder.parameters():
        param.requires_grad_(False)
    text_ids = [row["segment_id"] for row in text_rows]
    text_map = {}
    text_seconds_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, sample_size, encoder_batch_size):
            batch_rows = text_rows[offset:offset + encoder_batch_size]
            tokens = tokenizer([row["text"] for row in batch_rows], padding=True, truncation=True,
                               max_length=cfg["max_length"], return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                feature = text_encoder(**tokens).last_hidden_state[:, 0].float().cpu()
            if not _finite(feature):
                raise RuntimeError("Non-finite XLM-R features")
            text_map.update({row["segment_id"]: feature[i].clone()
                             for i, row in enumerate(batch_rows)})
    text_extract_seconds = time.perf_counter() - text_seconds_start
    text_dim = int(text_encoder.config.hidden_size)
    text_revision_loaded = getattr(text_encoder.config, "_commit_hash", None)
    text_revision_snapshot_exists = _cached_snapshot_exists(cfg["text_checkpoint"], cfg["text_revision"])
    frozen_text = all(not parameter.requires_grad for parameter in text_encoder.parameters())
    text_eval = not text_encoder.training
    del text_encoder, tokenizer
    torch.cuda.empty_cache()
    text_peak_mib = _peak_mib()

    # Vision encoder: release XLM-R before loading CLIP to fit the laptop GPU.
    torch.cuda.reset_peak_memory_stats()
    start_load = time.perf_counter()
    processor = CLIPImageProcessor.from_pretrained(cfg["image_checkpoint"], revision=cfg["image_revision"])
    vision_encoder = CLIPVisionModelWithProjection.from_pretrained(
        cfg["image_checkpoint"], revision=cfg["image_revision"])
    image_load_seconds = time.perf_counter() - start_load
    vision_encoder.eval().to(device)
    for param in vision_encoder.parameters():
        param.requires_grad_(False)
    image_ids = [record["image_id"] for record in image_records]
    image_map = {}
    image_extract_seconds_start = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, sample_size, encoder_batch_size):
            batch_records = image_records[offset:offset + encoder_batch_size]
            pil_images = [load_rgb_image(record["path"]) for record in batch_records]
            pixels = processor(images=pil_images, return_tensors="pt")["pixel_values"].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                feature = vision_encoder(pixel_values=pixels).image_embeds.float().cpu()
            if not _finite(feature):
                raise RuntimeError("Non-finite CLIP features")
            image_map.update({record["image_id"]: feature[i].clone()
                              for i, record in enumerate(batch_records)})
    image_extract_seconds = time.perf_counter() - image_extract_seconds_start
    image_dim = int(vision_encoder.config.projection_dim)
    image_revision_loaded = getattr(vision_encoder.config, "_commit_hash", None)
    image_revision_snapshot_exists = _cached_snapshot_exists(cfg["image_checkpoint"], cfg["image_revision"])
    frozen_vision = all(not parameter.requires_grad for parameter in vision_encoder.parameters())
    vision_eval = not vision_encoder.training
    del vision_encoder, processor
    torch.cuda.empty_cache()
    image_peak_mib = _peak_mib()

    # Feature keys are checked against the ordered source IDs, not labels or gold associations.
    if list(text_map) != text_ids or list(image_map) != image_ids:
        raise AssertionError("Feature output order/key associations do not match source IDs")
    if any(tuple(x.shape) != (text_dim,) for x in text_map.values()):
        raise AssertionError("Unexpected text feature dimensions")
    if any(tuple(x.shape) != (image_dim,) for x in image_map.values()):
        raise AssertionError("Unexpected image feature dimensions")
    text_snapshots = {key: value.clone() for key, value in text_map.items()}
    image_snapshots = {key: value.clone() for key, value in image_map.items()}

    # Use two real train segments and one metadata candidate image per segment.
    # The chosen images are joined from each segment's metadata candidate pool.
    training_rows = []
    for row in text_rows:
        available = [record for record in row["candidate_images"] if record["image_id"] in image_map]
        if available:
            training_rows.append((row, available[0]))
        if len(training_rows) == 2:
            break
    if len(training_rows) < 2:
        # Expand the same small candidate sample until two eligible segment links exist.
        for row in dataset.instances:
            for record in row["candidate_images"]:
                if record["image_id"] in image_map and row["segment_id"] in text_map:
                    if all(row["segment_id"] != existing[0]["segment_id"] for existing in training_rows):
                        training_rows.append((row, record))
                        break
            if len(training_rows) == 2:
                break
    if len(training_rows) != 2:
        raise RuntimeError("Could not link two actual train segments to sampled metadata images")

    counts = [len(row["quads"]) for row, _ in training_rows]
    max_targets = max(counts)
    targets = {"category": torch.zeros((2, max_targets), dtype=torch.long, device=device),
               "sentiment": torch.zeros((2, max_targets), dtype=torch.long, device=device),
               "aspect_implicit": torch.zeros((2, max_targets), dtype=torch.float32, device=device),
               "opinion_implicit": torch.zeros((2, max_targets), dtype=torch.float32, device=device),
               "mask": torch.zeros((2, max_targets), dtype=torch.bool, device=device)}
    languages, text_features, image_features = [], [], []
    for batch_index, (row, record) in enumerate(training_rows):
        sid = row["segment_id"]
        text_features.append(text_map[sid])
        languages.append(row["language"])
        image_features.append(image_map[record["image_id"]].unsqueeze(0))
        for target_index, quad in enumerate(row["quads"]):
            targets["category"][batch_index, target_index] = dataset.label_mappings["aspect_categories"][quad["aspect_category"]]
            targets["sentiment"][batch_index, target_index] = dataset.label_mappings["sentiments"][quad["sentiment"]]
            targets["aspect_implicit"][batch_index, target_index] = int(bool(quad["aspect_implicit"]))
            targets["opinion_implicit"][batch_index, target_index] = int(bool(quad["opinion_implicit"]))
            targets["mask"][batch_index, target_index] = True
    text_batch = torch.stack(text_features).to(device)
    language_batch = language_ids(languages).to(device)
    image_batch = torch.cat(image_features, dim=0).unsqueeze(1).to(device)
    image_mask = torch.ones(image_batch.shape[:2], dtype=torch.bool, device=device)

    torch.cuda.reset_peak_memory_stats()
    model = StructuredSetModel(text_dim=text_dim, image_dim=image_dim,
                               hidden_size=int(cfg["hidden_size"]), slots=23,
                               multimodal=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = StructuredSetLoss(**cfg["loss_weights"])
    before = model.slot_embeddings.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = model(text_batch, language_batch, image_batch, image_mask)
        loss = criterion(outputs, targets)["loss"]
    output_shapes = {key: list(value.shape) for key, value in outputs.items()}
    expected_shapes = {"objectness": [2, 23], "category": [2, 23, 6],
                       "sentiment": [2, 23, 3], "aspect_implicit": [2, 23],
                       "opinion_implicit": [2, 23]}
    if output_shapes != expected_shapes:
        raise AssertionError(f"Unexpected structured output schema: {output_shapes}")
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite FP16 autocast loss")
    loss_finite = bool(torch.isfinite(loss.detach()).item())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    finite_gradients = bool(gradients) and all(_finite(gradient) for gradient in gradients)
    scaler_used = False
    if not finite_gradients:
        # Retry from the same weights with scaling only if the unscaled FP16 path is unstable.
        model.zero_grad(set_to_none=True)
        model.slot_embeddings.data.copy_(before)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
        scaler = torch.amp.GradScaler("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(text_batch, language_batch, image_batch, image_mask)
            loss = criterion(outputs, targets)["loss"]
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        finite_gradients = bool(gradients) and all(_finite(gradient) for gradient in gradients)
        if not finite_gradients:
            raise RuntimeError("Non-finite gradients persisted after GradScaler retry")
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scaler_used = True
    else:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    optimizer_seconds = time.perf_counter() - started
    changed = not torch.equal(before, model.slot_embeddings.detach())
    if not changed:
        raise AssertionError("Optimizer update did not change model parameters")
    if not all(torch.equal(text_snapshots[k], text_map[k]) for k in text_map):
        raise AssertionError("Text encoder feature tensors changed during head training")
    if not all(torch.equal(image_snapshots[k], image_map[k]) for k in image_map):
        raise AssertionError("Vision encoder feature tensors changed during head training")
    training_peak_mib = _peak_mib()
    del model, optimizer, criterion, outputs, loss
    torch.cuda.empty_cache()

    text_per_second = sample_size / text_extract_seconds
    image_per_second = sample_size / image_extract_seconds
    return {
        "environment": {"python": __import__("sys").version.split()[0],
                        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available(),
                        "gpu": torch.cuda.get_device_name(0),
                        "gpu_memory_total_bytes": torch.cuda.get_device_properties(0).total_memory},
        "checkpoints": {
            "text": {"id": cfg["text_checkpoint"], "requested_revision": cfg["text_revision"],
                     "loaded_revision": text_revision_loaded or cfg["text_revision"],
                     "revision_snapshot_exists": text_revision_snapshot_exists,
                     "load_seconds": round(text_load_seconds, 3),
                     "dim": text_dim, "frozen": frozen_text, "eval_mode": text_eval},
            "vision": {"id": cfg["image_checkpoint"], "requested_revision": cfg["image_revision"],
                       "loaded_revision": image_revision_loaded or cfg["image_revision"],
                       "revision_snapshot_exists": image_revision_snapshot_exists,
                       "load_seconds": round(image_load_seconds, 3),
                       "dim": image_dim, "frozen": frozen_vision, "eval_mode": vision_eval}},
        "feature_sample": {"text_count": len(text_map), "text_segment_ids": list(text_map),
                           "text_shape": [len(text_map), text_dim], "text_all_finite": all(_finite(x) for x in text_map.values()),
                           "image_count": len(image_map), "image_ids": list(image_map),
                           "image_shape": [len(image_map), image_dim], "image_all_finite": all(_finite(x) for x in image_map.values()),
                           "image_source": "review candidate metadata only; no quad image IDs read"},
        "gpu_training_smoke": {"segment_ids": [row["segment_id"] for row, _ in training_rows],
                               "image_ids": [record["image_id"] for _, record in training_rows],
                               "output_shapes": output_shapes,
                               "loss_finite": loss_finite,
                               "gradients_finite": finite_gradients, "optimizer_updated": changed,
                               "grad_scaler_used": scaler_used, "elapsed_seconds": round(optimizer_seconds, 3),
                               "peak_memory_mib": training_peak_mib},
        "benchmark": {"sample_size": sample_size, "encoder_batch_size": encoder_batch_size,
                      "text_extract_seconds": round(text_extract_seconds, 3),
                      "text_features_per_second": round(text_per_second, 3),
                      "text_full_12601_estimate_seconds": round(12601 / text_per_second, 1),
                      "image_extract_seconds": round(image_extract_seconds, 3),
                      "image_features_per_second": round(image_per_second, 3),
                      "image_full_9216_estimate_seconds": round(9216 / image_per_second, 1),
                      "text_peak_memory_mib": text_peak_mib,
                      "image_peak_memory_mib": image_peak_mib,
                      "training_peak_memory_mib": training_peak_mib},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/structured_baseline.json")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(args.root, args.config, sample_size=args.sample_size,
                 encoder_batch_size=args.batch_size)
    serialized = json.dumps(result, indent=2)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
