"""End-to-end training runner for Artemide Core v1.5.

Training data format: flat little-endian uint16 token IDs (.bin), already tokenized.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from artemide_training import _autocast_context, load_checkpoint, save_checkpoint


class TokenStream:
    """Memory-mapped-by-Torch flat uint16 token stream with random batch sampling."""

    def __init__(self, path: str | Path, vocab_size: int):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"token file not found: {self.path}")
        byte_count = self.path.stat().st_size
        if byte_count % 2:
            raise ValueError("uint16 token file must contain an even number of bytes")
        count = byte_count // 2
        if count < 2:
            raise ValueError("token file must contain at least two token IDs")
        self.tokens = torch.from_file(
            str(self.path), shared=False, size=count, dtype=torch.uint16
        ).to(torch.int64)
        if int(self.tokens.max()) >= vocab_size:
            raise ValueError(
                f"token ID {int(self.tokens.max())} is outside model vocabulary "
                f"(vocab_size={vocab_size})"
            )

    def sample(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        max_start = self.tokens.numel() - sequence_length - 1
        if max_start < 0:
            raise ValueError(
                f"{self.path} needs at least sequence_length + 1 "
                f"({sequence_length + 1}) tokens"
            )
        starts = torch.randint(
            0, max_start + 1, (batch_size,), generator=generator
        ).tolist()
        rows = [self.tokens[s : s + sequence_length + 1] for s in starts]
        batch = torch.stack(rows)
        return batch[:, :-1].to(device), batch[:, 1:].to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to min_lr_ratio."""
    if total_steps < 1 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("require total_steps >= 1 and 0 <= warmup_steps < total_steps")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    data: TokenStream,
    *,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    precision: str = "auto",
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    if batches < 1:
        raise ValueError("validation batches must be at least 1")
    was_training = model.training
    model.eval()
    losses = []
    try:
        for _ in range(batches):
            x, y = data.sample(batch_size, sequence_length, device, generator)
            with _autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite validation loss")
            losses.append(float(loss))
    finally:
        model.train(was_training)
    mean_loss = sum(losses) / len(losses)
    return {
        "val_loss": mean_loss,
        "val_perplexity": math.exp(min(mean_loss, 80.0)),
    }


def _make_scaler(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"auto", "fp16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_training(
    model: torch.nn.Module,
    train_data: TokenStream,
    *,
    output_dir: str | Path,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    min_lr_ratio: float = 0.1,
    grad_accum_steps: int = 1,
    grad_clip: float = 1.0,
    precision: str = "auto",
    validation_data: TokenStream | None = None,
    validation_interval: int = 100,
    validation_batches: int = 10,
    checkpoint_interval: int = 500,
    resume: str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Train for a target number of optimizer updates (not micro-batches)."""
    if steps < 1 or batch_size < 1 or sequence_length < 1 or grad_accum_steps < 1:
        raise ValueError("steps, batch_size, sequence_length and grad_accum_steps must be positive")
    if learning_rate <= 0 or weight_decay < 0 or grad_clip <= 0:
        raise ValueError("learning_rate and grad_clip must be positive; weight_decay non-negative")
    if validation_interval < 0 or checkpoint_interval < 0:
        raise ValueError("validation/checkpoint intervals cannot be negative")
    if sequence_length > model.config.max_seq_len:
        raise ValueError("sequence_length exceeds model max_seq_len")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = build_scheduler(
        optimizer, total_steps=steps, warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )
    scaler = _make_scaler(device, precision)
    sampler_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    start_step = 0
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    checkpoint_path = output / "latest.pt"

    if resume:
        restored = load_checkpoint(
            resume, model, optimizer, map_location=device, scaler=scaler
        )
        start_step = restored["step"]
        scheduler_state = restored["metadata"].get("scheduler")
        if scheduler_state:
            scheduler.load_state_dict(scheduler_state)
        rng_state = restored["metadata"].get("sampler_rng")
        if rng_state is not None:
            sampler_generator.set_state(rng_state)
        if start_step > steps:
            raise ValueError("checkpoint step exceeds requested total steps")

    model.train()
    for step in range(start_step, steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _micro in range(grad_accum_steps):
            x, y = train_data.sample(
                batch_size, sequence_length, device, sampler_generator
            )
            with _autocast_context(device, precision):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1)
                )
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(f"non-finite training loss at update {step + 1}")
            (loss / grad_accum_steps).backward()
            accumulated_loss += float(loss.detach()) / grad_accum_steps

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError(f"non-finite gradient norm at update {step + 1}")
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        completed_step = step + 1
        elapsed = max(time.perf_counter() - started, 1e-9)
        record: dict[str, Any] = {
            "step": completed_step,
            "train_loss": accumulated_loss,
            "grad_norm": float(grad_norm),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "tokens_per_second": (batch_size * sequence_length * grad_accum_steps) / elapsed,
            "seconds": elapsed,
        }

        if validation_data is not None and validation_interval and (
            completed_step % validation_interval == 0 or completed_step == steps
        ):
            record.update(validate(
                model, validation_data, batches=validation_batches,
                batch_size=batch_size, sequence_length=sequence_length,
                device=device, precision=precision,
                generator=sampler_generator,
            ))
            model.train()

        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, sort_keys=True) + "\n")

        should_save = (
            (checkpoint_interval and completed_step % checkpoint_interval == 0)
            or completed_step == steps
        )
        if should_save:
            save_checkpoint(
                checkpoint_path, model, optimizer, step=completed_step, scaler=scaler,
                metadata={
                    "version": "1.5",
                    "scheduler": scheduler.state_dict(),
                    "sampler_rng": sampler_generator.get_state(),
                    "train_data": str(train_data.path),
                    "validation_data": str(validation_data.path) if validation_data else None,
                    "sequence_length": sequence_length,
                    "batch_size": batch_size,
                    "grad_accum_steps": grad_accum_steps,
                },
            )
        print(json.dumps(record, sort_keys=True), flush=True)

    return {"step": steps, "checkpoint": str(checkpoint_path), "metrics": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Artemide from tokenized uint16 files")
    parser.add_argument("--config", required=True, help="JSON ModelConfig file")
    parser.add_argument("--train", required=True, help="training .bin file (uint16 token IDs)")
    parser.add_argument("--validation", help="optional validation .bin file")
    parser.add_argument("--output", default="runs/artemide")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from artemide.model.config import ModelConfig
    from artemide.model.transformer import ArtemideLM

    with Path(args.config).open(encoding="utf-8") as handle:
        config = ModelConfig(**json.load(handle))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtemideLM(config).to(device)
    train_data = TokenStream(args.train, config.vocab_size)
    val_data = TokenStream(args.validation, config.vocab_size) if args.validation else None
    result = run_training(
        model, train_data, output_dir=args.output, steps=args.steps,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        grad_accum_steps=args.grad_accum_steps, grad_clip=args.grad_clip,
        precision=args.precision, validation_data=val_data,
        validation_interval=args.validation_interval,
        validation_batches=args.validation_batches,
        checkpoint_interval=args.checkpoint_interval, resume=args.resume, seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
