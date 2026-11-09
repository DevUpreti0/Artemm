# Browser model export (first implementation)

Artemide can be run without an always-on inference server by exporting a **trained** checkpoint to ONNX and executing it in the visitor's browser. This is the first export milestone; it does not make the current playground live by itself.

## Export

From the repository root, with the package installed:

```bash
pip install -e ./core-v2.5
pip install onnx
python core-v2.5/scripts/export_onnx.py \
  --config core-v2.5/configs/small.json \
  --checkpoint /path/to/artemide-checkpoint.pt \
  --output ./artemide-model.onnx \
  --verify
```

Install `onnxruntime` too if using `--verify`. The exporter writes an ONNX model and a same-name JSON metadata file. It checks ONNX Runtime output against PyTorch when verification is requested.

## Scope and limitations

- First version supports the 259-token UTF-8 byte tokenizer only. BPE export is intentionally rejected rather than silently using the wrong token IDs.
- The graph returns logits for the full input sequence. Browser code must append one sampled token and call the graph again; this is simpler but slower than a KV-cache graph.
- Model context cannot exceed the configured `max_seq_len`.
- WebGPU availability varies by browser/device; a production client should try WebGPU and fall back to WASM.
- No checkpoint is currently trained or included. The Kaggle notebook must be run first. The small Shakespeare experiment is not an instruction-following assistant.
- ONNX export has not yet been executed against a real trained checkpoint in this repository. Run the command with `--verify` before publishing any model artifact.
- This does not create a centralized public API for third-party server applications. Browser inference is client-side; a shared server API needs compute.

No paid service or external AI provider is used.
