"""Real pinned XLM-R GPU forward/backward/optimizer smoke; train text only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .v3_presence_dataset import TEXT_CHECKPOINT, TEXT_REVISION
from .v4_hybrid_dataset import HybridCandidateDataset, collate_hybrid
from .v4_hybrid_loss import hybrid_loss
from .v4_hybrid_model import HybridCandidateModel, load_pinned_encoder


def run_smoke(root: str | Path, config_path: str | Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; V4 GPU smoke will not fall back to CPU")
    root = Path(root).resolve()
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if cfg["text_checkpoint"] != TEXT_CHECKPOINT or cfg["text_revision"] != TEXT_REVISION:
        raise ValueError("Smoke config must use the exact pinned XLM-R revision")
    torch.manual_seed(int(cfg["seed"]))
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT, revision=TEXT_REVISION,
                                               local_files_only=True)
    ds = HybridCandidateDataset(root, "train")
    encoded_lens = [len(tokenizer.encode(r["text"], truncation=True, max_length=cfg["max_length"]))
                    for r in ds.records]
    p90 = float(np.percentile(encoded_lens, 90))
    chosen = sorted(range(len(encoded_lens)), key=lambda i: abs(encoded_lens[i] - p90))[:2]
    mini = torch.utils.data.Subset(ds, chosen)
    loader = DataLoader(mini, batch_size=2, shuffle=False,
        collate_fn=lambda b: collate_hybrid(b, tokenizer, int(cfg["max_length"])))
    batch = next(iter(loader))
    encoder = load_pinned_encoder(local_files_only=True)
    model = HybridCandidateModel(encoder, hidden_size=int(cfg["hidden_size"]),
        category_embedding_dim=int(cfg["category_embedding_dim"]),
        sentiment_embedding_dim=int(cfg["sentiment_embedding_dim"]), dropout=float(cfg["dropout"]))
    if cfg.get("gradient_checkpointing"):
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
    model = model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ids, attn = batch["input_ids"].cuda(), batch["attention_mask"].cuda()
    targets = {k: batch[k].cuda() for k in ("any_targets", "conditional_targets", "conditional_mask")}
    start = time.perf_counter()
    initial_scale = float(scaler.get_scale())
    attempts, overflow_backoffs = [], 0
    accepted = False
    outputs = losses = None
    encoder_grad = category_grad = sentiment_grad = False
    for attempt in range(1, 6):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(ids, attn)
            losses = hybrid_loss(outputs["any_logits"], outputs["conditional_logits"],
                targets["any_targets"], targets["conditional_targets"], targets["conditional_mask"],
                float(cfg["conditional_loss_lambda"]))
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(opt)
        grad_finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
        attempts.append({"attempt": attempt, "scale": float(scaler.get_scale()), "finite_gradients": bool(grad_finite)})
        if grad_finite:
            encoder_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.encoder.parameters())
            category_grad = model.category_embeddings.weight.grad is not None and model.category_embeddings.weight.grad.abs().sum().item() > 0
            sentiment_grad = model.sentiment_embeddings.weight.grad is not None and model.sentiment_embeddings.weight.grad.abs().sum().item() > 0
            tracked = model.category_embeddings.weight.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            accepted = not torch.equal(tracked, model.category_embeddings.weight.detach())
            break
        scaler.step(opt)  # GradScaler detects overflow and skips this update.
        previous_scale = float(scaler.get_scale())
        scaler.update()
        if float(scaler.get_scale()) < previous_scale:
            overflow_backoffs += 1
    if not accepted:
        raise RuntimeError(f"No finite-gradient optimizer update after scaler backoff: {attempts}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {"checkpoint": TEXT_CHECKPOINT, "revision": TEXT_REVISION,
        "device": torch.cuda.get_device_name(0), "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "batch_segment_ids": [ds.records[i]["metadata"]["segment_id"] for i in chosen],
        "sequence_lengths": [encoded_lens[i] for i in chosen], "train_length_p90": p90,
        "input_shape": list(ids.shape), "candidate_logit_shape": list(outputs["any_logits"].shape),
        "loss": float(losses["loss"].detach().float().item()),
        "any_loss": float(losses["any_loss"].detach().float().item()),
        "conditional_loss": float(losses["conditional_loss"].detach().float().item()),
        "finite_loss": bool(torch.isfinite(losses["loss"]).item()), "finite_gradients": True,
        "scaler_initial_scale": initial_scale, "scaler_final_scale": float(scaler.get_scale()),
        "gradient_attempts": attempts, "overflow_backoffs": overflow_backoffs,
        "encoder_gradient_flow": bool(encoder_grad), "category_embedding_gradient_flow": bool(category_grad),
        "sentiment_embedding_gradient_flow": bool(sentiment_grad),
        "optimizer_step_performed": accepted, "elapsed_step_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "estimated_train_epoch_seconds_at_measured_step": float(np.ceil(8816 / 2) * elapsed),
        "note": "Single two-segment forward/backward/optimizer step; train epoch estimate extrapolates 4,408 steps and excludes dev evaluation and data-loading overhead."}
    if not (result["finite_loss"] and result["finite_gradients"] and result["encoder_gradient_flow"]
            and result["category_embedding_gradient_flow"] and result["sentiment_embedding_gradient_flow"]):
        raise RuntimeError(f"GPU smoke checks failed: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_hybrid_v4.json")
    parser.add_argument("--output", default="runs/structured/challenger_v4/hybrid_smoke.json")
    args = parser.parse_args()
    result = run_smoke(args.root, args.config)
    target = Path(args.output)
    if target.exists(): raise FileExistsError(f"Refusing to overwrite smoke result: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
