"""Command line interface: prepare data, train, or serve Artemide."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from . import ArtemideLM, ModelConfig
from .data import TokenStream, prepare_corpus
from .training import train


def main():
    parser = argparse.ArgumentParser(prog="artemide")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--input", required=True)
    prep.add_argument("--output", default="data/processed")
    prep.add_argument("--validation-fraction", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--license", default="unspecified")
    fit = commands.add_parser("train")
    fit.add_argument("--config", required=True)
    fit.add_argument("--train", required=True)
    fit.add_argument("--validation")
    fit.add_argument("--output", default="runs/artemide")
    fit.add_argument("--steps", type=int, default=1000)
    fit.add_argument("--batch-size", type=int, default=8)
    fit.add_argument("--sequence-length", type=int, default=256)
    fit.add_argument("--grad-accum-steps", type=int, default=1)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--warmup-steps", type=int, default=100)
    fit.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    fit.add_argument("--resume")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_corpus(args.input, args.output,
            validation_fraction=args.validation_fraction, seed=args.seed,
            license_name=args.license), indent=2))
        return
    config = ModelConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArtemideLM(config).to(device)
    train_data = TokenStream(args.train, config.vocab_size)
    val_data = TokenStream(args.validation, config.vocab_size) if args.validation else None
    print("checkpoint:", train(model, train_data, output=args.output, steps=args.steps,
        batch_size=args.batch_size, sequence_length=args.sequence_length,
        validation_data=val_data, grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        precision=args.precision, resume=args.resume))


if __name__ == "__main__":
    main()
