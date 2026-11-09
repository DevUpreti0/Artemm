"""Export a trained byte-token Artemide checkpoint to ONNX for browser inference.

Run from the repository root after: pip install -e ./core-v2.5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide.training import load_checkpoint


class ExportableLogits(torch.nn.Module):
    """Expose only full-sequence logits; autoregressive sampling stays in JS."""
    def __init__(self, model: ArtemideLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify", action="store_true",
                        help="Compare ONNX Runtime output with PyTorch (requires onnxruntime).")
    args = parser.parse_args()

    config_data = json.loads(args.config.read_text(encoding="utf-8"))
    config = ModelConfig(**config_data)
    if config.vocab_size != 259:
        raise SystemExit(
            "This first browser exporter supports the 259-token ByteTokenizer only. "
            "BPE export needs its matching tokenizer implementation in the browser."
        )

    model = ArtemideLM(config).cpu().eval()
    info = load_checkpoint(args.checkpoint, model, map_location="cpu", restore_rng=False)
    wrapper = ExportableLogits(model).eval()
    sample_len = min(8, config.max_seq_len)
    sample = torch.tensor([[1] + [ord("A") + 3] * (sample_len - 1)], dtype=torch.long)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample,),
        str(args.output),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    metadata = {
        "format": "artemide-onnx-v1",
        "tokenizer": "byte-v1",
        "vocab_size": config.vocab_size,
        "config": config_data,
        "training_step": info["step"],
        "checkpoint_metadata": info.get("metadata", {}),
        "input": {"name": "input_ids", "dtype": "int64", "shape": ["batch", "sequence"]},
        "output": {"name": "logits", "dtype": "float32", "shape": ["batch", "sequence", 259]},
        "generation": "Client performs autoregressive sampling; send the full context each step.",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("--verify requires: pip install onnxruntime") from exc
        session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        ids = sample.numpy()
        expected = wrapper(sample).detach().numpy()
        actual = session.run(["logits"], {"input_ids": ids})[0]
        if actual.shape != expected.shape:
            raise SystemExit(f"ONNX shape mismatch: {actual.shape} != {expected.shape}")
        max_error = float(abs(actual - expected).max())
        if max_error > 2e-3:
            raise SystemExit(f"ONNX numerical check failed; max absolute error={max_error:.6g}")
        print(f"ONNX verification passed; max absolute error={max_error:.6g}")

    print(f"Exported: {args.output}")
    print(f"Metadata: {args.output.with_suffix('.json')}")
    print(f"Training step: {info['step']}")


if __name__ == "__main__":
    main()
