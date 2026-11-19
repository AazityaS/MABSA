"""V4 trainer with guarded one-epoch default and explicit checkpoint continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .v3_presence_dataset import LABEL_ORDER, TEXT_CHECKPOINT, TEXT_REVISION
from .v4_hybrid_dataset import (HybridCandidateDataset, collate_hybrid,
                                summarize_targets, validate_split_pair)
from .v4_hybrid_loss import hybrid_loss
from .v4_hybrid_metrics import evaluate_hybrid, select_combined_threshold
from .v4_hybrid_model import HybridCandidateModel, load_pinned_encoder


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluate(model, loader, device, conditional_lambda):
    model.eval()
    sums = {key: [] for key in ("combined", "any", "conditional", "targets",
        "any_targets", "conditional_targets", "mask")}
    loss_rows = []
    with torch.no_grad():
        for batch in loader:
            ids, attention = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            any_gold = batch["any_targets"].to(device)
            cond_gold = batch["conditional_targets"].to(device)
            cond_mask = batch["conditional_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(ids, attention)
            losses = hybrid_loss(out["any_logits"], out["conditional_logits"], any_gold,
                cond_gold, cond_mask, conditional_lambda=conditional_lambda)
            loss_rows.append(float(losses["loss"].item()))
            for key, source in (("combined", "combined_probabilities"), ("any", "any_probabilities"),
                                ("conditional", "conditional_probabilities")):
                sums[key].append(out[source].float().cpu().numpy())
            sums["targets"].append(cond_gold.cpu().numpy())
            sums["any_targets"].append(any_gold.cpu().numpy())
            sums["conditional_targets"].append(cond_gold.cpu().numpy())
            sums["mask"].append(cond_mask.cpu().numpy())
    return float(np.mean(loss_rows)), {key: np.concatenate(value) for key, value in sums.items()}


def _rebase_scheduler(scheduler, optimizer, completed_successful_updates: int) -> float:
    """Place a fresh configured scheduler at the global update reached by epoch 1."""
    if completed_successful_updates < 0:
        raise ValueError("completed_successful_updates must be non-negative")
    if completed_successful_updates == 0:
        return float(optimizer.param_groups[0]["lr"])
    scheduler.last_epoch = int(completed_successful_updates)
    scheduler._step_count = int(completed_successful_updates) + 1
    learning_rates = []
    for group, base_lr, multiplier in zip(optimizer.param_groups, scheduler.base_lrs,
                                           scheduler.lr_lambdas):
        lr = float(base_lr) * float(multiplier(int(completed_successful_updates)))
        group["lr"] = lr
        learning_rates.append(lr)
    scheduler._last_lr = learning_rates
    return learning_rates[0]


def _target_summary_matches(root, train_ds, dev_ds):
    expected = {"train": (8816, 11663, 147025, 451, 11212),
                "dev": (1895, 2501, 31609, 98, 2403)}
    for name, dataset in (("train", train_ds), ("dev", dev_ds)):
        summary = summarize_targets(dataset)
        actual = (summary["segments"], summary["any_positive_rows"],
                  summary["any_negative_rows"], summary["conditional_positive_rows"],
                  summary["conditional_negative_rows"])
        if actual != expected[name]:
            raise ValueError(f"{name} target counts differ from audited counts: {actual} != {expected[name]}")


def train(root, config_path, output_dir, epoch_limit: int = 1,
          resume_checkpoint: str | Path | None = None):
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V4; CPU fallback is disabled")
    if cfg["text_checkpoint"] != TEXT_CHECKPOINT or cfg["text_revision"] != TEXT_REVISION:
        raise ValueError("V4 config does not use the pinned XLM-R revision")
    if cfg.get("mixed_precision") != "fp16":
        raise ValueError("This V4 implementation currently expects fp16 mixed precision")
    if epoch_limit < 1 or epoch_limit > int(cfg["max_epochs"]):
        raise ValueError(f"epoch_limit must be within [1, {cfg['max_epochs']}]")

    resume = Path(resume_checkpoint).resolve() if resume_checkpoint else None
    previous_meta = None
    previous_history = []
    checkpoint = None
    start_epoch = 1
    if resume is None:
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite V4 run: {output_dir}")
        if epoch_limit < 1:
            raise ValueError("Invalid epoch limit")
    else:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Resume output directory does not exist: {output_dir}")
        if resume.parent != output_dir:
            raise ValueError("Resume checkpoint must be in the isolated V4 output directory")
        run_config_path = output_dir / "config.json"
        if not run_config_path.exists() or json.loads(run_config_path.read_text(encoding="utf-8")) != cfg:
            raise ValueError("Existing V4 run config does not match the requested config")
        if not (output_dir / "best_checkpoint.pt").exists():
            raise FileNotFoundError("Existing best V4 checkpoint is missing")
        if not resume.exists():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if checkpoint.get("config") != cfg or checkpoint.get("label_order") != list(LABEL_ORDER):
            raise ValueError("Resume checkpoint config or label ordering does not match V4")
        if checkpoint.get("text_revision") != TEXT_REVISION:
            raise ValueError("Resume checkpoint XLM-R revision mismatch")
        if int(checkpoint.get("epoch", 0)) < 1:
            raise ValueError("Resume checkpoint has no completed epoch marker")
        start_epoch = int(checkpoint["epoch"]) + 1
        if start_epoch > epoch_limit:
            raise ValueError(f"Checkpoint already at epoch {start_epoch - 1}; limit is {epoch_limit}")
        previous_meta_path = output_dir / "training_metadata.json"
        if not previous_meta_path.exists():
            raise FileNotFoundError("Resume requires prior training_metadata.json")
        previous_meta = json.loads(previous_meta_path.read_text(encoding="utf-8"))
        if int(previous_meta.get("completed_epochs", -1)) != int(checkpoint["epoch"]):
            raise ValueError("Checkpoint epoch and recorded completed epoch differ; refusing ambiguous resume")
        log_path = output_dir / "metrics.jsonl"
        previous_history = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        logged_epochs = [int(row["epoch"]) for row in previous_history]
        if logged_epochs != list(range(1, int(checkpoint["epoch"]) + 1)):
            raise ValueError(f"Existing epoch history is not contiguous through checkpoint: {logged_epochs}")
        if "optimizer_step_attempts" not in previous_history[-1]:
            previous_history[-1].update({
                "optimizer_step_attempts": int(previous_meta["optimizer_step_attempts"]),
                "successful_optimizer_steps": int(previous_meta["optimizer_step_attempts"] -
                                                     previous_meta["skipped_optimizer_steps"]),
                "skipped_optimizer_steps": int(previous_meta["skipped_optimizer_steps"]),
                "nonfinite_loss_count": 0,
                "nonfinite_gradient_overflow_count": int(previous_meta["skipped_optimizer_steps"]),
                "grad_scaler_scale_start": float(previous_meta["grad_scaler_scale_events"][0]["scale_before"])
                    if previous_meta.get("grad_scaler_scale_events") else
                    float(previous_meta.get("grad_scaler_final_scale", 65536.0)),
                "grad_scaler_scale_end": float(previous_meta.get("grad_scaler_final_scale", 65536.0)),
                "grad_scaler_scale_events": previous_meta.get("grad_scaler_scale_events", [])})
        for epoch in range(start_epoch, epoch_limit + 1):
            if (output_dir / f"epoch_{epoch:03d}_checkpoint.pt").exists():
                raise FileExistsError(f"Refusing to overwrite existing epoch checkpoint {epoch}")
            if (output_dir / f"epoch_{epoch:03d}_dev_metrics.json").exists():
                raise FileExistsError(f"Refusing to overwrite existing epoch metrics {epoch}")
        if int(checkpoint["epoch"]) == 1 and (output_dir / "epoch_001_checkpoint.pt").exists():
            raise FileExistsError("Refusing to overwrite the epoch-1 checkpoint preservation file")

    if epoch_limit < start_epoch:
        raise ValueError("Epoch limit must be at least the next epoch to train")
    seed_everything(int(cfg["seed"]))
    train_ds = HybridCandidateDataset(root, "train")
    dev_ds = HybridCandidateDataset(root, "dev")
    validate_split_pair(train_ds, dev_ds)
    _target_summary_matches(root, train_ds, dev_ds)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_CHECKPOINT, revision=TEXT_REVISION,
        local_files_only=bool(cfg.get("local_files_only", True)))
    collate = lambda batch: collate_hybrid(batch, tokenizer, int(cfg["max_length"]))
    # A new, deterministic continuation shuffle is required because epoch-1 RNG state
    # was not checkpointed. The global experiment seed remains 1729.
    train_generator = torch.Generator().manual_seed(int(cfg["seed"]) + start_epoch - 1)
    train_loader = DataLoader(train_ds, batch_size=int(cfg["micro_batch_size"]), shuffle=True,
        generator=train_generator, num_workers=int(cfg["num_workers"]), pin_memory=True,
        collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=int(cfg["micro_batch_size"]), shuffle=False,
        num_workers=int(cfg["num_workers"]), pin_memory=True, collate_fn=collate)

    encoder = load_pinned_encoder(bool(cfg.get("local_files_only", True)))
    model = HybridCandidateModel(encoder, hidden_size=cfg["hidden_size"],
        category_embedding_dim=cfg["category_embedding_dim"],
        sentiment_embedding_dim=cfg["sentiment_embedding_dim"], dropout=cfg["dropout"])
    if checkpoint is not None:
        # Restore and strictly validate epoch-1 weights before writing any run artifact.
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if checkpoint.get("rng_state"):
            rng = checkpoint["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
    if cfg.get("gradient_checkpointing"):
        model.encoder.gradient_checkpointing_enable()
        if hasattr(model.encoder.config, "use_cache"):
            model.encoder.config.use_cache = False
    model.to("cuda")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]))
    accumulation = int(cfg["gradient_accumulation_steps"])
    steps_per_epoch = int(np.ceil(len(train_loader) / accumulation))
    total_updates = steps_per_epoch * int(cfg["max_epochs"])
    scheduler = get_linear_schedule_with_warmup(optimizer,
        num_warmup_steps=int(total_updates * float(cfg["warmup_ratio"])),
        num_training_steps=total_updates)
    scaler_init = 65536.0
    schedule_position = 0
    resume_lr = float(optimizer.param_groups[0]["lr"])
    optimizer_continuity = "fresh AdamW state"
    scheduler_continuity = "configured 20-epoch schedule"
    if checkpoint is not None:
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_continuity = "restored optimizer state"
        else:
            optimizer_continuity = "restarted AdamW; epoch-1 moments were not saved"
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scheduler_continuity = "restored scheduler state"
        else:
            schedule_position = int(previous_meta["optimizer_step_attempts"] -
                                    previous_meta["skipped_optimizer_steps"])
            if schedule_position > total_updates:
                raise ValueError("Recorded successful updates exceed configured scheduler horizon")
            resumed_lr = _rebase_scheduler(scheduler, optimizer, schedule_position)
            resume_lr = resumed_lr
            scheduler_continuity = "rebased at epoch-1 successful update count"
        scaler_init = float(previous_meta.get("grad_scaler_final_scale", scaler_init))
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=scaler_init)
    if checkpoint is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    if resume is None:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        (output_dir / "target_support.json").write_text(json.dumps({"train": summarize_targets(train_ds),
            "dev": summarize_targets(dev_ds), "label_order": list(LABEL_ORDER)}, indent=2), encoding="utf-8")
        previous_meta = {}
        previous_history = []
        best_f1, best_epoch = -1.0, 0
        previous_attempts = previous_skipped = 0
        previous_scale_events = []
        previous_total_train_seconds = 0.0
    else:
        old_best = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        best_f1 = float(old_best.get("dev_metrics", {}).get("primary_implicit_pair_presence", {}).get("micro", {}).get("f1", -1.0))
        best_epoch = int(old_best.get("epoch", 0))
        previous_attempts = int(previous_meta.get("optimizer_step_attempts", 0))
        previous_skipped = int(previous_meta.get("skipped_optimizer_steps", 0))
        previous_scale_events = list(previous_meta.get("grad_scaler_scale_events", []))
        previous_total_train_seconds = sum(float(r.get("train_duration_seconds", 0)) for r in previous_history)
        if not (output_dir / "target_support.json").exists():
            raise FileNotFoundError("Existing V4 target-support record is missing")
        epoch1_path = output_dir / "epoch_001_checkpoint.pt"
        epoch1_hash = None
        if int(checkpoint["epoch"]) == 1:
            # Strict model restoration has succeeded. Preserve the verified original
            # epoch-1 file before any possible best-checkpoint update.
            source_hash = _sha256_file(resume)
            shutil.copy2(resume, epoch1_path)
            epoch1_hash = _sha256_file(epoch1_path)
            if epoch1_hash != source_hash:
                raise IOError("Preserved epoch-1 checkpoint copy failed SHA-256 verification")
        continuation = {"resume_checkpoint": str(resume), "resume_epoch": int(checkpoint["epoch"]),
            "next_epoch": start_epoch, "optimizer": optimizer_continuity,
            "scheduler": scheduler_continuity, "scheduler_global_successful_updates": schedule_position,
            "scheduler_total_updates": total_updates, "scheduler_lr_at_resume": float(optimizer.param_groups[0]["lr"]),
            "scaler_state": ("restored" if "scaler_state_dict" in checkpoint else
                             f"reinitialized at saved final scale {scaler_init:g}"),
            "rng_state": ("restored" if "rng_state" in checkpoint else
                          "reseeded with 1729; continuation shuffle uses deterministic epoch seed"),
            "epoch1_checkpoint_weights_strictly_restored_before_writes": True,
            "preserved_epoch1_checkpoint": (str(epoch1_path) if epoch1_hash else None),
            "preserved_epoch1_sha256": epoch1_hash,
            "epoch1_checkpoint_preservation": "verified copy (hard links unavailable on this Windows setup)",
            "max_total_epoch": epoch_limit}
        continuation_path = output_dir / "continuation_metadata.json"
        if continuation_path.exists():
            raise FileExistsError("Refusing to overwrite continuation_metadata.json")
        continuation_path.write_text(json.dumps(continuation, indent=2), encoding="utf-8")
        (output_dir / "metrics.jsonl").write_text("".join(
            json.dumps(row, allow_nan=False) + "\n" for row in previous_history), encoding="utf-8")

    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    started = time.perf_counter()
    continuation_started_at = _utc_now()
    torch.cuda.reset_peak_memory_stats()
    free_before, total_before = torch.cuda.mem_get_info()
    allocated_before = torch.cuda.memory_allocated()
    reserved_before = torch.cuda.memory_reserved()
    epoch_rows = []
    scaler_events = []
    continuation_attempts = continuation_skipped = 0
    prior_f1 = (previous_history[-1]["dev_metrics"]["primary_implicit_pair_presence"]["micro"]["f1"]
                if previous_history else None)
    collapse_streak = 0
    latest_metadata = previous_meta.copy()
    log_mode = "a" if resume is not None else "w"
    with (output_dir / "metrics.jsonl").open(log_mode, encoding="utf-8") as log:
        for epoch in range(start_epoch, epoch_limit + 1):
            epoch_started_at = _utc_now()
            train_started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_rows = []
            epoch_scale_events = []
            epoch_attempts = epoch_skipped = epoch_successful = 0
            consecutive_skips = 0
            pending = 0
            scale_start = float(scaler.get_scale())
            for step, batch in enumerate(train_loader, 1):
                ids = batch["input_ids"].to("cuda", non_blocking=True)
                attention = batch["attention_mask"].to("cuda", non_blocking=True)
                any_y = batch["any_targets"].to("cuda", non_blocking=True)
                cond_y = batch["conditional_targets"].to("cuda", non_blocking=True)
                cond_mask = batch["conditional_mask"].to("cuda", non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(ids, attention)
                    losses = hybrid_loss(out["any_logits"], out["conditional_logits"], any_y,
                        cond_y, cond_mask, float(cfg["conditional_loss_lambda"]))
                if not torch.isfinite(losses["loss"]):
                    raise FloatingPointError(f"Non-finite loss at epoch {epoch}, microbatch {step}")
                scaler.scale(losses["loss"] / accumulation).backward()
                train_rows.append({key: float(losses[key].detach().item())
                                   for key in ("loss", "any_loss", "conditional_loss")})
                if step % 100 == 0 or step == len(train_loader):
                    print(f"epoch={epoch} step={step}/{len(train_loader)} "
                          f"loss={train_rows[-1]['loss']:.6f} scaler={scaler.get_scale():.0f} "
                          f"allocated_gb={torch.cuda.memory_allocated() / 1e9:.3f}", flush=True)
                pending += 1
                final_step = step == len(train_loader)
                if pending == accumulation or final_step:
                    scaler.unscale_(optimizer)
                    if pending < accumulation:
                        correction = accumulation / pending
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                parameter.grad.mul_(correction)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scale_before = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    scale_after = float(scaler.get_scale())
                    epoch_attempts += 1
                    continuation_attempts += 1
                    skipped = scale_after < scale_before
                    if skipped:
                        epoch_skipped += 1
                        continuation_skipped += 1
                        consecutive_skips += 1
                        event = {"epoch": epoch, "optimizer_step_in_epoch": epoch_attempts,
                            "scale_before": scale_before, "scale_after": scale_after,
                            "event": "overflow_step_skipped"}
                        epoch_scale_events.append(event)
                        scaler_events.append(event)
                    else:
                        epoch_successful += 1
                        consecutive_skips = 0
                        scheduler.step()
                        if scale_after != scale_before:
                            event = {"epoch": epoch, "optimizer_step_in_epoch": epoch_attempts,
                                "scale_before": scale_before, "scale_after": scale_after,
                                "event": "scale_changed_after_successful_step"}
                            epoch_scale_events.append(event)
                            scaler_events.append(event)
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                    if consecutive_skips >= 5:
                        raise FloatingPointError(f"Five consecutive GradScaler-skipped updates at epoch {epoch}")
                    if epoch_attempts >= 20 and epoch_skipped / epoch_attempts > 0.05:
                        raise FloatingPointError(f"GradScaler skip rate exceeded 5% at epoch {epoch}")

            torch.cuda.synchronize()
            train_duration = time.perf_counter() - train_started
            training_ended_at = _utc_now()
            dev_started = time.perf_counter()
            dev_loss, arrays = _evaluate(model, dev_loader, torch.device("cuda"),
                                         float(cfg["conditional_loss_lambda"]))
            torch.cuda.synchronize()
            dev_duration = time.perf_counter() - dev_started
            threshold = select_combined_threshold(arrays["combined"], arrays["targets"])
            dev_metrics = evaluate_hybrid(arrays["combined"], arrays["targets"], arrays["any"],
                arrays["any_targets"], arrays["conditional"], arrays["conditional_targets"],
                arrays["mask"], threshold["threshold"])
            f1 = float(dev_metrics["primary_implicit_pair_presence"]["micro"]["f1"])
            epoch_row = {"epoch": epoch, "epoch_started_at_utc": epoch_started_at,
                "training_ended_at_utc": training_ended_at,
                "train_duration_seconds": train_duration,
                "average_training_microbatch_seconds": train_duration / len(train_loader),
                "development_evaluation_duration_seconds": dev_duration,
                "train_loss": float(np.mean([row["loss"] for row in train_rows])),
                "train_any_loss": float(np.mean([row["any_loss"] for row in train_rows])),
                "train_conditional_loss": float(np.mean([row["conditional_loss"] for row in train_rows])),
                "dev_loss": dev_loss, "threshold_selection": threshold, "dev_metrics": dev_metrics,
                "grad_scaler_scale_start": scale_start,
                "grad_scaler_scale_end": float(scaler.get_scale()),
                "optimizer_step_attempts": epoch_attempts,
                "successful_optimizer_steps": epoch_successful,
                "skipped_optimizer_steps": epoch_skipped,
                "nonfinite_loss_count": 0,
                "nonfinite_gradient_overflow_count": epoch_skipped,
                "grad_scaler_scale_events": epoch_scale_events}
            log.write(json.dumps(epoch_row, allow_nan=False) + "\n")
            log.flush()
            (output_dir / f"epoch_{epoch:03d}_dev_metrics.json").write_text(json.dumps({
                "split": "dev", "epoch": epoch, "threshold_selection": threshold,
                "metrics": dev_metrics}, indent=2, allow_nan=False), encoding="utf-8")
            checkpoint_payload = {"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()},
                "threshold": threshold["threshold"], "threshold_selection": threshold,
                "epoch": epoch, "dev_metrics": dev_metrics, "label_order": list(LABEL_ORDER),
                "config": cfg, "text_revision": TEXT_REVISION,
                "global_successful_optimizer_steps": int(schedule_position + sum(
                    row["successful_optimizer_steps"] for row in epoch_rows) + epoch_successful)}
            epoch_checkpoint_path = output_dir / f"epoch_{epoch:03d}_checkpoint.pt"
            torch.save(checkpoint_payload, epoch_checkpoint_path)
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                best_tmp_path = best_checkpoint_path.with_name("best_checkpoint.pt.tmp")
                torch.save(checkpoint_payload, best_tmp_path)
                best_tmp_path.replace(best_checkpoint_path)

            collapse_streak = collapse_streak + 1 if (
                prior_f1 is not None and best_f1 > 0 and f1 < 0.5 * best_f1 and f1 < prior_f1) else 0
            prior_f1 = f1
            print(f"epoch={epoch} train_loss={epoch_row['train_loss']:.6f} dev_loss={dev_loss:.6f} "
                  f"dev_primary_f1={f1:.6f} threshold={threshold['threshold']:.8g} "
                  f"successful_updates={epoch_successful} skipped_updates={epoch_skipped}", flush=True)

            torch.cuda.synchronize()
            free_after, total_after = torch.cuda.mem_get_info()
            latest_metadata = {**previous_meta,
                "variant": "V4 hybrid candidate-conditioned",
                "best_epoch": best_epoch, "completed_epochs": epoch,
                "best_dev_micro_f1": best_f1,
                "training_started_at_utc": previous_meta.get("training_started_at_utc", epoch_started_at),
                "training_ended_at_utc": training_ended_at,
                "epoch_timing": epoch_row,
                "epoch_history": previous_history + epoch_rows + [epoch_row],
                "optimizer_step_attempts": previous_attempts + continuation_attempts,
                "skipped_optimizer_steps": previous_skipped + continuation_skipped,
                "successful_optimizer_steps_total": (previous_attempts - previous_skipped) +
                    sum(row["successful_optimizer_steps"] for row in epoch_rows) + epoch_successful,
                "grad_scaler_final_scale": float(scaler.get_scale()),
                "grad_scaler_scale_events": previous_scale_events + scaler_events,
                "continuation_optimizer_state": optimizer_continuity,
                "continuation_scheduler": scheduler_continuity,
                "continuation_scheduler_start_update": schedule_position,
                "continuation_scheduler_lr_start": resume_lr,
                "continuation_started_at_utc": continuation_started_at,
                "continuation_elapsed_seconds": time.perf_counter() - started,
                "process_elapsed_seconds": float(previous_meta.get("process_elapsed_seconds", 0.0)) +
                    time.perf_counter() - started,
                "total_train_duration_seconds": sum(float(row.get("train_duration_seconds", 0.0))
                    for row in (previous_history + epoch_rows + [epoch_row])),
                "cuda_memory_before_continuation": {"free_bytes": free_before, "total_bytes": total_before,
                    "allocated_bytes": allocated_before, "reserved_bytes": reserved_before},
                "cuda_memory_after_evaluation": {"free_bytes": free_after, "total_bytes": total_after,
                    "allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()},
                "peak_cuda_allocated_bytes": max(int(previous_meta.get("peak_cuda_allocated_bytes", 0)),
                    int(torch.cuda.max_memory_allocated())),
                "peak_cuda_reserved_bytes": max(int(previous_meta.get("peak_cuda_reserved_bytes", 0)),
                    int(torch.cuda.max_memory_reserved())), "test_split_loaded": False,
                "last_epoch_checkpoint": str(epoch_checkpoint_path)}
            latest_metadata["continuation_epoch_rows"] = epoch_rows + [epoch_row]
            (output_dir / "training_metadata.json").write_text(json.dumps(latest_metadata, indent=2),
                                                                 encoding="utf-8")
            epoch_rows.append(epoch_row)

            # Review only a sustained, large deterioration. One epoch falling is not enough.
            if epoch_skipped / max(epoch_attempts, 1) > 0.05:
                print("Stopping after epoch evaluation: GradScaler skipped more than 5% of updates.", flush=True)
                break
            if collapse_streak >= 2:
                print("Stopping after two consecutive strong dev collapses; requesting review.", flush=True)
                break

    return latest_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--config", default="configs/implicit_hybrid_v4.json")
    parser.add_argument("--output-dir", default="runs/structured/challenger_v4/hybrid")
    parser.add_argument("--epoch-limit", type=int, default=1,
                        help="Hard upper bound on total completed epochs")
    parser.add_argument("--resume-checkpoint", default=None,
                        help="Resume model weights from an existing isolated V4 checkpoint")
    args = parser.parse_args()
    train(args.root, args.config, args.output_dir, epoch_limit=args.epoch_limit,
          resume_checkpoint=args.resume_checkpoint)


if __name__ == "__main__":
    main()
