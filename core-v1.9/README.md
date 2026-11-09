# Artemide Core v1.9 — unified package + compatible chat API

v1.9 adds an OpenAI-compatible subset of the Chat Completions HTTP interface while retaining Artemide's own random-initialized model, tokenizer, training pipeline and native streaming API. It does not call or proxy to an external model provider.

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
```

Install PyTorch using the official command appropriate for your CPU/CUDA environment.

## Prepare and train

Use text data you have permission to use. Input supports .txt and .jsonl files.

```bash
artemide prepare --input ./licensed-corpus --output ./data/processed
artemide train --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 10000 \
  --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

Training runs locally on CPU or an already available CUDA device. No paid cloud resources are provisioned. A random-init checkpoint is not a useful assistant until trained and evaluated.

## Start API

```bash
export ARTEMIDE_CONFIG=configs/small.json
export ARTEMIDE_CHECKPOINT=runs/artemide/latest.pt
export ARTEMIDE_API_KEY='replace-with-a-long-random-secret'
artemide serve --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /v1/generate`
- `POST /v1/stream` — native UTF-8-safe Server-Sent Events
- `POST /v1/chat` — Artemide's basic role-prefixed chat format
- `GET /v1/models` — model-list shape compatible with common OpenAI clients
- `POST /v1/chat/completions` — OpenAI-compatible subset, including SSE when `stream: true`

Example compatible request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $ARTEMIDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"artemide-core-1.9","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

The compatibility layer supports messages, model, max_tokens, temperature, top_k and stream. It does not implement every OpenAI API feature (tools, images, logprobs, usage accounting or fine-grained sampling controls). Streaming uses standard chat-completion chunks followed by `[DONE]`.

## Security

Set a strong `ARTEMIDE_API_KEY` before exposing the API publicly. If no key is configured, authentication is disabled for local development. Put the service behind TLS and a trusted reverse proxy for public use.

## Architecture and limitations

- Decoder-only Transformer, RMSNorm, RoPE, SwiGLU, causal attention and KV cache
- UTF-8 byte tokenizer (259 IDs)
- Training, validation, checkpointing and local inference
- No third-party pretrained weights and no external inference fallback
- The API compatibility layer does not make the model instruction-tuned or equivalent to commercial assistants
- No corpus is bundled and no training job is started by this release

## Tests

`pytest` covers core model behavior, tokenizer, corpus preparation, checkpoint restore, streaming parity, API authentication and OpenAI-compatible JSON/SSE responses.
