# Artemide Core v1.7 — unified installable package

This release consolidates the model, KV cache, tokenizer, corpus preparation, training/checkpoint utilities, CLI, and HTTP inference API into one installable Python package. The model is initialized from random weights; no third-party foundation model or model API is used.

## Install

Python 3.10+ is required.

```bash
cd core-v1.7
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -e .
```

PyTorch installation can differ by operating system and CUDA version. If you need a specific CUDA build, install the matching PyTorch wheel before installing this package.

## 1. Prepare your own licensed corpus

```bash
artemide prepare --input ./my-text --output ./data/processed --license "CC-BY-4.0"
```

Input may be a directory of UTF-8 `.txt` and `.jsonl` files. JSONL records are strings or objects containing a string `text` field. The tool splits documents before tokenization and writes `train.bin`, optional `validation.bin`, and a SHA-256 manifest.

The license field is an operator-declared label, not an automatic legal verification. Only use data you have permission to use.

## 2. Train from random initialization

```bash
artemide train --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 1000 \
  --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

Checkpoints and JSONL metrics are written under `runs/artemide`. Use `--resume runs/artemide/latest.pt` to resume a compatible run. The target `--steps` is the total optimizer-update count.

## 3. Run the inference API

Set environment variables:
- `ARTEMIDE_CONFIG=./configs/small.json`
- `ARTEMIDE_CHECKPOINT=./runs/artemide/latest.pt`
- Optional `ARTEMIDE_API_KEY` for bearer-token protection

Then run:

```bash
uvicorn artemide.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /v1/generate` with JSON `{"prompt":"Hello","max_new_tokens":64,"temperature":0.7,"top_k":40}`

If `ARTEMIDE_API_KEY` is set, send `Authorization: Bearer <key>`. Keep the key private and terminate TLS at a trusted reverse proxy before exposing the service publicly.

## 4. Run tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover import, tokenizer round-trip, model forward shape, cached-vs-uncached greedy generation, data splitting, binary sampling, and health route. They are committed but have not been executed in this environment.

## Limitations and honest status

- No dataset is bundled, downloaded, or licensed on your behalf.
- No model training has been launched by this release.
- No paid GPU/cloud resources were created.
- This baseline byte tokenizer uses 259 tokens and is inefficient for long text.
- A small randomly initialized model is a research prototype, not a ChatGPT/Gemini competitor. Quality requires substantial lawful data, compute, and rigorous evaluation.
