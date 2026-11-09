"""Deterministic held-out evaluation for Artemide checkpoints."""
import argparse, json, math
from pathlib import Path
import torch
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide.tokenizer import ByteTokenizer
from artemide.training.checkpoint import load_checkpoint

@torch.inference_mode()
def evaluate_text(model, text, device="cpu", seq_len=None):
    tokenizer = ByteTokenizer()
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
    if ids.numel() < 2:
        raise ValueError("evaluation text must contain at least two tokens")
    limit = seq_len or model.cfg.max_seq_len
    if limit < 1 or limit > model.cfg.max_seq_len:
        raise ValueError(f"seq_len must be between 1 and model max_seq_len ({model.cfg.max_seq_len})")
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, ids.numel() - 1, limit):
        length = min(limit, ids.numel() - 1 - start)
        x = ids[start:start + length].unsqueeze(0)
        y = ids[start + 1:start + length + 1].unsqueeze(0)
        _, loss = model(x, y)
        count = y.numel()
        total_loss += float(loss) * count
        total_tokens += count
    if was_training:
        model.train()
    mean_loss = total_loss / total_tokens
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 80)), "tokens": total_tokens}

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Artemide checkpoint on held-out text")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = ModelConfig.from_json(args.config)
    if config.vocab_size != ByteTokenizer.vocab_size:
        parser.error("Config vocab_size must match ByteTokenizer (259)")
    model = ArtemideLM(config).to(args.device)
    state = load_checkpoint(args.checkpoint, model, map_location=args.device)
    report = evaluate_text(model, Path(args.text).read_text(encoding="utf-8"), args.device, args.seq_len)
    report.update({"checkpoint": args.checkpoint, "step": int(state.get("step", 0)), "device": args.device})
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
