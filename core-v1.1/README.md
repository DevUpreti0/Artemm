# Artemide Core v1.1.0 — integrated KV-cached transformer

Standalone decoder-only transformer overlay with RoPE, RMSNorm, SwiGLU, PyTorch scaled-dot-product attention, and incremental KV-cached generation.

Run tests with: PYTHONPATH=. pytest -q

The implementation is randomly initialized, not pretrained or instruction-tuned. Cached decoding is tested for logits equivalence against full-sequence inference. Actual speed depends on hardware and workload; no ChatGPT/Gemini speed claim is made without controlled benchmarks.
