"""Training utilities for Artemide's decoder-only transformer."""
from __future__ import annotations

import os
import random
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16", "auto"}:
        raise ValueError("precision must be one of: fp32, bf16, fp16, auto")
    if precision == "auto":
        if device.type != "cuda":
            return nullcontext()
        precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
        if device.type != "cuda":
            raise ValueError("fp16 autocast is supported here only on CUDA")
    return torch.autocast(device_type=device.type, dtype=dtype)


def train_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    *,
    precision: str = "auto",
    grad_clip: float | None = 1.0,
    scaler: Any = None,
) -> dict[str, float]:
    """Run one next-token prediction update and return scalar diagnostics."""
    if input_ids.ndim != 2 or targets.shape != input_ids.shape:
        raise ValueError("input_ids and targets must have the same [batch, sequence] shape")
    if input_ids.dtype not in (torch.int32, torch.int64) or targets.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_ids and targets must be integer token IDs")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive or None")
    device = input_ids.device
    model.train()
    optimizer.zero_grad(set_to_none=True)
    use_scaler = scaler is not None and scaler.is_enabled()
    with _autocast_context(device, precision):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite training loss; optimizer update skipped")
    if use_scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    if grad_clip is None:
        grad_norm = torch.linalg.vector_norm(torch.stack([
            p.grad.detach().float().norm(2) for p in model.parameters() if p.grad is not None
        ]))
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if use_scaler:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return {"loss": float(loss.detach()), "grad_norm": float(grad_norm)}


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    step: int,
    metadata: dict[str, Any] | None = None,
    scaler: Any = None,
) -> None:
    """Atomically save model, optimizer, scaler and RNG state."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": 1,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata or {},
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as f:
            temp_name = f.name
        torch.save(state, temp_name)
        os.replace(temp_name, destination)
    finally:
        if temp_name and os.path.exists(temp_name):
            os.unlink(temp_name)


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    scaler: Any = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a trusted local checkpoint. Do not load files from unknown sources."""
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        state = torch.load(path, map_location=map_location)
    if state.get("format_version") != 1 or "model" not in state:
        raise ValueError("unsupported or invalid Artemide checkpoint")
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
