# Artemide Core v1.7 — unified installable package

v1.7 consolidates the decoder-only model, KV cache, byte tokenizer, corpus preparation, training/checkpoint utilities, generation helpers, CLI and optional HTTP API into one installable Python package. It is still a research baseline initialized from random weights—not a pretrained or instruction-tuned assistant.

## Install

Python 3.10+:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\\Scripts\\activate
pip install -e ".[dev,api]"
```

PyTorch installation may need a platform-specific command from the official PyTorch installer to match your CPU/CUDA setup.

## Prepare text you have rights to use

Input can be a directory of `.txt` and `.jsonl` files or one supported file. JSONL rows may be strings or objects with a string `text` field.

```bash
artemide prepare --input ./licensed-corpus --output ./data/processed \
  --validation-fraction 0.05 --seed 42 --license "CC-BY-4.0"
```

Creates `train.bin`, optional `validation.bin`, and `manifest.json` with the byte-token format, document counts and SHA-256 hashes. The license label is supplied by the operator and is not independently verified.

## Configure and train

Create `configs/small.json`:

```json
{
  "vocab_size": 259,
  "d_model": 256,
  "n_layers": 4,
  "n_heads": 4,
  "max_seq_len": 512,
  "dropout": 0.0,
  "ff_mult": 4
}
```

Then run:

```bash
artemide train --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 10000 \
  --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

Training supports gradient accumulation, clipping, warmup/cosine LR scheduling, optional validation, JSONL metrics and atomic checkpoints. Resume with `--resume runs/artemide/latest.pt`; `--steps` is the total target optimizer-update count. Training runs locally on CPU or an already available CUDA device. No cloud GPU is provisioned.

## Generate

```bash
artemide generate --config configs/small.json --checkpoint runs/artemide/latest.pt \
  --prompt "Hello, Artemide:" --max-new-tokens 100 --temperature 0
```

A random-init checkpoint produces untrained output until trained on an appropriate corpus.

## HTTP API

Set environment variables before starting the service:

```bash
export ARTEMIDE_CONFIG=configs/small.json
export ARTEMIDE_CHECKPOINT=runs/artemide/latest.pt
export ARTEMIDE_API_KEY="replace-with-a-long-random-secret"
artemide serve --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /v1/generate` with `{"prompt":"Hello","max_new_tokens":64,"temperature":0}`
- `POST /v1/chat` with a `messages` array of system/user/assistant messages

The API is checkpoint-backed and never silently falls back to another model provider. Set a strong API key before exposing it publicly. Chat role formatting is not instruction tuning; this model may not follow user instructions reliably.

## Architecture and limits

- Decoder-only Transformer, RMSNorm, RoPE, SwiGLU, scaled dot-product causal attention
- Incremental KV cache and greedy/top-k sampling
- UTF-8 byte tokenizer: PAD=0, BOS=1, EOS=2, byte IDs=3..258
- Random initialization; no third-party pretrained weights
- No corpus bundled and no training job launched
- The byte tokenizer is a baseline; token sequences are longer than subword tokenizers
- No claim of ChatGPT/Gemini-level capability or speed

## Tests

```bash
pytest
```

The test suite covers model forward/generation, cache equivalence, Unicode tokenization, corpus preparation, checkpoint restore, scheduler and validation mode. Run it in the target environment before production use.
