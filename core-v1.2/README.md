# Artemide Core v1.2.0 — inference benchmarking

This version adds a reproducible benchmark for cached versus uncached autoregressive generation.

## Run

From this directory, with PyTorch installed:

```bash
PYTHONPATH=../core-v1.1 python benchmarks/generation.py --device cpu
```

For a CUDA GPU:

```bash
PYTHONPATH=../core-v1.1 python benchmarks/generation.py --device cuda --dtype bfloat16
```

Useful controls include `--prompt-tokens`, `--new-tokens`, `--repeats`, `--warmup`, `--d-model`, `--layers`, and `--heads`. The script emits JSON with median latency, generated tokens/second, per-run timings, device details, and peak CUDA memory when available.

The benchmark imports the standalone transformer from `../core-v1.1/artemide`; keep that folder beside this version. It uses a randomly initialized Artemide model and synthetic token IDs. It measures inference implementation only; it does not measure language quality, training progress, or compare against hosted AI products. Compare cached and uncached runs on the same device, dtype, model shape, and token counts. Results are hardware- and workload-specific.
