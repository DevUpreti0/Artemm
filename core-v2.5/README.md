# Artemide Core v2.5 — trainable tokenizer and self-hosted AI API

v2.5 adds an independently trainable byte-level BPE tokenizer. The default byte tokenizer remains supported for old checkpoints. No pretrained tokenizer or external AI API is required.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"
```

Install PyTorch from the official selector for your CPU/CUDA environment.

## Train a tokenizer on permitted text

```bash
artemide bpe-train --input ./licensed-corpus --output ./runs/tokenizer.json --vocab-size 8000
```

The command prints the actual vocabulary size (the corpus may not produce all requested merges). Set `vocab_size` in your model config to that exact value. Then prepare token files with the same tokenizer:

```bash
artemide prepare --input ./licensed-corpus --output ./data/processed \
  --tokenizer-file ./runs/tokenizer.json
```

Train the model with a matching config and token files. Do not mix byte-token and BPE-token data/checkpoints.

## Initialize and train

```bash
artemide init --config configs/small.json --output runs/artemide/initial.pt
artemide train --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 10000 \
  --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

The init command creates random weights only; it is a startup smoke test, not a useful assistant. Language ability requires properly licensed data, substantial training compute and evaluation.

## Run the API

Set `ARTEMIDE_CONFIG`, `ARTEMIDE_CHECKPOINT`, `ARTEMIDE_API_KEY`, and `ARTEMIDE_ADMIN_API_KEY`. For BPE checkpoints, also set `ARTEMIDE_TOKENIZER=/path/to/tokenizer.json`. The loader verifies that tokenizer and model vocabulary sizes match.

The API includes native text generation, SSE streaming, OpenAI-compatible chat completions, individual developer keys, revocation and usage tracking. Python and JavaScript SDKs are included.

## Deployment

For a ₹0/month setup using hardware you already own, follow [the zero-budget self-hosting guide](deploy/README.md). It uses local CPU inference and optionally a temporary free HTTPS tunnel. No paid cloud resources are started automatically.

## Current limits

Artemide is an independently initialized text Transformer with its own tokenizer/training/inference pipeline. It is not yet instruction-tuned and does not yet generate images, audio or video, browse the web, or execute general agent tasks. Those require additional trained model families and secure tools. This project does not claim commercial-model parity.
