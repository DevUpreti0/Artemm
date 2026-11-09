"""Reproducible cached-vs-uncached generation benchmark for Artemide."""
import argparse
import json
import platform
import statistics
import time

import torch

from artemide.model import ArtemideLM, ModelConfig


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(model, prompt, new_tokens, use_cache, repeats, device):
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = time.perf_counter()
        result = model.generate(
            prompt,
            max_new_tokens=new_tokens,
            temperature=0,
            use_cache=use_cache,
        )
        synchronize(device)
        samples.append(time.perf_counter() - start)
    elapsed = statistics.median(samples)
    generated = max(0, result.shape[1] - prompt.shape[1])
    return {
        "median_seconds": round(elapsed, 6),
        "generated_tokens": generated,
        "tokens_per_second": round(generated / elapsed, 3) if elapsed else None,
        "samples_seconds": [round(x, 6) for x in samples],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    if args.prompt_tokens < 1 or args.new_tokens < 1 or args.repeats < 1 or args.warmup < 0:
        parser.error("token counts and repeats must be positive; warmup cannot be negative")
    if args.prompt_tokens + args.new_tokens > args.max_seq_len:
        parser.error("prompt-tokens + new-tokens cannot exceed max-seq-len")

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if device.type == "cpu" and dtype == torch.float16:
        parser.error("float16 CPU inference is not supported by this benchmark")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = ModelConfig(
        vocab_size=259,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        max_seq_len=args.max_seq_len,
    )
    model = ArtemideLM(config).to(device=device, dtype=dtype).eval()
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_tokens), device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model.generate(prompt, max_new_tokens=min(4, args.new_tokens), temperature=0, use_cache=True)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        cached = measure(model, prompt, args.new_tokens, True, args.repeats, device)
        uncached = measure(model, prompt, args.new_tokens, False, args.repeats, device)

    report = {
        "benchmark": "artemide_cached_vs_uncached_generation",
        "note": "Randomly initialized model; results are hardware/workload specific, not a comparison with other AI services.",
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU",
            "dtype": args.dtype,
        },
        "configuration": vars(args),
        "cached": cached,
        "uncached": uncached,
        "cached_speedup": round(uncached["median_seconds"] / cached["median_seconds"], 3)
        if cached["median_seconds"] else None,
        "peak_cuda_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2)
        if device.type == "cuda" else None,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
