"""Training, checkpointing and precision helpers used by the v1.7 CLI."""
from __future__ import annotations
import math
import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
import torch
import torch.nn.functional as F


def autocast_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"auto", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16 or fp16")
    if precision == "auto":
        if device.type != "cuda":
            return nullcontext()
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 requires CUDA")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def save_checkpoint(path, model, optimizer=None, *, step, metadata=None, scaler=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1, "step": int(step), "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None, "metadata": metadata or {},
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
            temporary = handle.name
        torch.save(state, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer=None, *, map_location="cpu", scaler=None, restore_rng=True):
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or state.get("format_version") != 1 or "model" not in state:
        raise ValueError("invalid Artemide checkpoint")
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        if state.get("python_rng") is not None:
            random.setstate(state["python_rng"])
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return {"step": int(state.get("step", 0)), "metadata": state.get("metadata", {})}


def build_scheduler(optimizer, total_steps, warmup_steps, min_lr_ratio=0.1):
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be >=0 and < total_steps")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0,1]")
    def scale(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train(model, train_data, *, output, steps, batch_size, sequence_length,
          validation_data=None, validation_interval=100, validation_batches=10,
          grad_accum_steps=1, learning_rate=3e-4, weight_decay=0.1,
          warmup_steps=100, precision="auto", checkpoint_interval=500,
          resume=None, seed=42):
    if min(steps, batch_size, sequence_length, grad_accum_steps) < 1:
        raise ValueError("steps, batch, sequence and accumulation must be positive")
    if sequence_length > model.config.max_seq_len:
        raise ValueError("sequence length exceeds model context")
    random.seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = build_scheduler(optimizer, steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and precision in {"auto", "fp16"}))
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = 0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if resume:
        state = load_checkpoint(resume, model, optimizer, map_location=device, scaler=scaler)
        start = state["step"]
        if state["metadata"].get("scheduler"):
            scheduler.load_state_dict(state["metadata"]["scheduler"])
        if state["metadata"].get("sampler_rng") is not None:
            generator.set_state(state["metadata"]["sampler_rng"])
    metrics_path = output / "metrics.jsonl"
    model.train()
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = train_data.sample(batch_size, sequence_length, device, generator)
            with autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite loss at step {step + 1}")
            (loss / grad_accum_steps).backward()
            total_loss += float(loss.detach()) / grad_accum_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        record = {"step": step + 1, "train_loss": total_loss, "grad_norm": float(grad_norm),
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        if validation_data and validation_interval and ((step + 1) % validation_interval == 0 or step + 1 == steps):
            model.eval()
            losses = []
            with torch.no_grad():
                for _ in range(validation_batches):
                    x, y = validation_data.sample(batch_size, sequence_length, device, generator)
                    with autocast_context(device, precision):
                        logits = model(x)
                        losses.append(float(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))))
            record["val_loss"] = sum(losses) / len(losses)
            record["val_perplexity"] = math.exp(min(record["val_loss"], 80))
            model.train()
        with metrics_path.open("a", encoding="utf-8") as handle:
            import json
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        if (checkpoint_interval and (step + 1) % checkpoint_interval == 0) or step + 1 == steps:
            save_checkpoint(output / "latest.pt", model, optimizer, step=step + 1, scaler=scaler,
                metadata={"scheduler": scheduler.state_dict(), "sampler_rng": generator.get_state(),
                          "version": "1.7"})
        print(record, flush=True)
    return str(output / "latest.pt")
