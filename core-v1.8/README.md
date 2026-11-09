# Artemide Core v1.8 — unified installable package

v1.8 adds UTF-8-safe token streaming over Server-Sent Events (SSE) to the independently initialized Artemide model. It retains the v1.7 model, tokenizer, corpus preparation, training/checkpoint utilities, CLI and optional HTTP API.

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
```

PyTorch installation may need a platform-specific command from the official PyTorch installer to match your CPU/CUDA setup.

## Prepare text you have rights to use

```bash
artemide prepare --input ./licensed-corpus --output ./data/processed --validation-fraction 0.05 --seed 42 --license "CC-BY-4.0"
```

Input supports .txt and .jsonl documents. Creates uint16 token files and a manifest with hashes. The license label is supplied by the operator and is not independently verified.

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

Train locally on CPU or an already available CUDA device:

```bash
artemide train --config configs/small.json --train data/processed/train.bin --validation data/processed/validation.bin --steps 10000 --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

Supports gradient accumulation, clipping, warmup/cosine LR scheduling, validation, JSONL metrics, atomic checkpoints and resume. No cloud GPU is provisioned.

## Generate and stream

```bash
artemide generate --config configs/small.json --checkpoint runs/artemide/latest.pt --prompt "Hello, Artemide:" --max-new-tokens 100 --temperature 0
```

HTTP API environment:

```bash
export ARTEMIDE_CONFIG=configs/small.json
export ARTEMIDE_CHECKPOINT=runs/artemide/latest.pt
export ARTEMIDE_API_KEY="replace-with-a-long-random-secret"
artemide serve --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /v1/generate` — complete response JSON
- `POST /v1/stream` — Server-Sent Events, each event contains a UTF-8-safe `delta`; final event contains `done: true`
- `POST /v1/chat` — basic role-prefixed prompt formatting

Example stream request:

```bash
curl -N http://localhost:8000/v1/stream \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ARTEMIDE_API_KEY" \
  -d '{"prompt":"Hello","max_new_tokens":64,"temperature":0}'
```

The API has no external model-provider fallback. Set a strong API key before exposing it publicly. Chat role formatting is not instruction tuning.

## Architecture and limits

- Decoder-only Transformer, RMSNorm, RoPE, SwiGLU, scaled-dot-product causal attention
- Incremental KV cache and greedy/top-k sampling
- UTF-8 byte tokenizer: PAD=0, BOS=1, EOS=2, byte IDs=3..258
- Streaming yields incremental decoded text and safely buffers partial UTF-8 sequences
- Random initialization; no third-party pretrained weights
- No corpus bundled and no training job launched
- Byte tokenizer produces longer sequences than subword tokenizers
- No claim of ChatGPT/Gemini-level capability or speed

## Tests

```bash
pytest
```

The suite includes cache equivalence, Unicode tokenizer, corpus prep, checkpoint restore, scheduler, validation mode and streaming/non-streaming greedy parity. Run the suite in the target environment before production use.
