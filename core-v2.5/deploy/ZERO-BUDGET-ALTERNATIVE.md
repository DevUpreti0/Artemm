# Zero-budget alternative: burst training + browser inference

Render/Koyeb-style free containers are too small and unreliable for a PyTorch model API. This plan avoids keeping a cloud machine alive.

## 1. Train on free notebook compute

Use the Kaggle notebook at `notebooks/artemide-kaggle-free-training.ipynb`. Kaggle currently documents free notebook sessions and limited GPU availability; sessions are temporary, not always-on hosting. Run the notebook interactively, save a Kaggle version, and retain the checkpoint artifact. Never use a notebook session as a permanent public web server.

The included notebook uses a small public-domain Shakespeare text corpus and Artemide's randomly initialized 64-dimension, 2-layer Transformer. It is a reproducible training smoke test and will learn a narrow text style, not general assistant capability.

## 2. Serve inference on each visitor's device

The intended long-term ₹0 serving design is:
- Keep the playground as a static site on the existing Vercel free deployment.
- Export Artemide checkpoints to ONNX.
- Use ONNX Runtime Web with WebGPU where supported, and WASM CPU fallback where available.
- Download model weights once per browser and cache them locally.
- Keep prompts and generated text on the visitor's device; no permanent inference server is needed.

Official ONNX Runtime Web docs describe browser inference and WASM/WebGPU execution providers. WebGPU availability varies by browser/device, so a fallback and clear compatibility status are required.

## 3. Keep weights outside Git history

Publish only a small, tested model artifact to a public Hugging Face model repository if storage and usage comply with the Hub's free public-storage policy. Never commit large binary checkpoints into this source repository. Store the model card, data provenance, training config, evaluation metrics, and SHA-256 digest alongside the artifact.

## 4. Why not another free always-on API?

- Kaggle and Colab are notebook compute, not dependable web-hosting services. Colab explicitly disallows using free managed runtimes primarily to power a web UI.
- Hugging Face Spaces are free for static Spaces; compute-backed Docker/Gradio Spaces require paid hardware under the current Spaces documentation.
- Koyeb's free instance is 512 MB RAM and 0.1 vCPU, with no persistent volume and scale-to-zero. That is not a robust place to run PyTorch inference.
- Cloudflare Workers Free has a 10 ms CPU-time limit per request, which is not a realistic environment for autoregressive Transformer inference.

## Status

This document is an architecture plan, not a claim that ONNX export, browser inference, or a trained assistant is already working. The Kaggle notebook must be run by the account owner to produce the first trained checkpoint. No paid resource is required or authorized.
