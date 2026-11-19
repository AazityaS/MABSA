"""Real pretrained CUDA smoke test on a small train-only HAMOS batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .v3_presence_dataset import (ImplicitPresenceDataset, collate_presence,
                                  load_split_language_metadata)
from .v3_presence_model import XLMRImplicitPresenceModel


def run_smoke(root: str | Path, config_path: str | Path, batch_size: int = 2, steps: int = 3) -> dict:
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; V3 smoke test will not fall back to CPU")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if steps < 2:
        raise ValueError("Use at least two update steps to estimate warmed throughput")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["text_checkpoint"], revision=cfg["text_revision"],
                                              local_files_only=cfg.get("local_files_only", True))
    dataset = ImplicitPresenceDataset(root, "train")
    # Choose actual training examples around the middle of the length distribution.
    ordered = sorted(range(len(dataset)), key=lambda i: len(dataset.records[i]["text"].split()))
    start = max(0, len(ordered) // 2 - batch_size // 2)
    indices = ordered[start:start + batch_size]
    if len(indices) != batch_size:
        raise AssertionError("Could not form a complete smoke batch")
    samples = [dataset[i] for i in indices]
    batch = collate_presence(samples, tokenizer, int(cfg["max_length"]))
    device = torch.device("cuda")
    memory_before = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = XLMRImplicitPresenceModel(cfg["text_checkpoint"], cfg["text_revision"],
                                     dropout=float(cfg["dropout"]),
                                     local_files_only=cfg.get("local_files_only", True)).to(device)
    checkpointing_api = (model.enable_gradient_checkpointing()
                         if cfg.get("gradient_checkpointing", False) else None)
    model_load_seconds = time.perf_counter() - load_start
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    x = {key: batch[key].to(device) for key in ("input_ids", "attention_mask")}
    y = batch["targets"].to(device)
    before_head = model.classifier.weight.detach().clone()
    before_encoder = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    step_times, losses, logits = [], [], None
    finite_grads, finite_loss = True, True
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            logits = model(**x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        finite_grads = finite_grads and all(bool(torch.isfinite(g).all()) for g in grads)
        finite_loss = finite_loss and bool(torch.isfinite(loss))
        if not bool(torch.isfinite(loss)) or not finite_grads:
            raise FloatingPointError("Non-finite V3 GPU smoke loss or gradient")
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        losses.append(float(loss.item()))
    head_update = float((model.classifier.weight.detach() - before_head).abs().max().item())
    encoder_update = float((model.encoder.embeddings.word_embeddings.weight.detach() - before_encoder).abs().max().item())
    if head_update == 0 or encoder_update == 0:
        raise AssertionError("Expected optimizer updates in both V3 head and fine-tuned XLM-R")
    result = {
        "device": torch.cuda.get_device_name(0),
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "checkpoint": cfg["text_checkpoint"], "revision": cfg["text_revision"],
        "train_only_segment_ids": [row["metadata"]["segment_id"] for row in samples],
        "input_shape": list(batch["input_ids"].shape), "target_shape": list(batch["targets"].shape),
        "logits_shape": list(logits.shape), "finite_logits": bool(torch.isfinite(logits).all()),
        "losses": losses, "finite_loss": finite_loss,
        "finite_gradient_tensors": finite_grads, "gradient_tensor_count": len(grads),
        "optimizer_step_completed": True, "head_max_abs_update": head_update,
        "encoder_max_abs_update": encoder_update, "amp_fp16_grad_scaler": True,
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", False)),
        "checkpointing_api": checkpointing_api,
        "model_load_seconds": model_load_seconds, "batch_step_seconds": step_times,
        "warmed_step_seconds_mean": float(sum(step_times[1:]) / len(step_times[1:])),
        "batch_size": batch_size, "smoke_optimizer_steps": steps,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        "cuda_free_bytes_before_model_load": int(memory_before[0]),
        "cuda_total_bytes": int(memory_before[1]),
        "test_split_loaded": False,
    }
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_presence_v3.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", default="runs/structured/challenger_v3/gpu_smoke.json")
    args = parser.parse_args()
    result = run_smoke(args.root, args.config, args.batch_size, args.steps)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
