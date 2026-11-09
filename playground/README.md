# Artemide Playground

Static chat playground with two inference paths:

1. **Local browser inference:** choose the `.onnx` model and matching `.json` metadata produced by the Kaggle notebook's export cell. The page loads ONNX Runtime Web, attempts WebGPU, then falls back to WASM. The model files remain in the browser tab; no model server or external AI provider is called.
2. **Optional server inference:** if no local model is loaded, the page uses the existing Vercel proxy and `ARTEMIDE_API_URL` server-side environment variable.

## Local inference

1. Run all cells in `notebooks/artemide-kaggle-free-training.ipynb`.
2. Save the Kaggle notebook version and download `artemide.onnx` plus `artemide.json` from the output folder.
3. Open the playground, select both files in **Run a model on this device**, then choose **Load model on this device**.
4. Send a prompt. The model runs locally in the current browser tab.

The browser path currently supports the 259-token ByteTokenizer and full-context ONNX graph. It performs one full forward pass per generated token, so generation is slower than KV-cached server inference. The first small Shakespeare model is a narrow text-generation experiment, not an instruction-following assistant. WebGPU depends on browser/device support; WASM is the fallback. ONNX Runtime Web is loaded from jsDelivr, so an internet connection is needed to load the runtime script.

## Server inference

Set optional Vercel server-side environment variables:

- `ARTEMIDE_API_URL`: HTTPS base URL of a running Artemide inference service.
- `ARTEMIDE_API_KEY`: optional bearer token.

The service must expose `GET /health` and `POST /v1/stream`. Credentials are never embedded in browser code. Without a local model or configured server, the playground reports that inference is offline.

No paid service or external AI model provider is used by the local path.
