# Artemide Core v2.2 — independent model runtime and developer SDKs

v2.2 adds first-party Python and JavaScript clients for Artemide's own API. SDK calls go only to the Artemide server configured by the developer; no outside model provider is used.

## Install and train

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
artemide prepare --input ./licensed-corpus --output ./data/processed
artemide train --config configs/small.json --train data/processed/train.bin --validation data/processed/validation.bin --steps 10000
```

Install PyTorch using the official command for your CPU/CUDA environment. Use only data you have permission to use. A random-init checkpoint is not a useful assistant until trained and evaluated.

## Start the API

```bash
export ARTEMIDE_CONFIG=configs/small.json
export ARTEMIDE_CHECKPOINT=runs/artemide/latest.pt
export ARTEMIDE_API_KEY='replace-with-a-long-random-secret'
artemide serve --host 0.0.0.0 --port 8000
```

Routes: `GET /health`, `GET /v1/models`, `POST /v1/generate`, `POST /v1/stream`, `POST /v1/chat`, and `POST /v1/chat/completions`. The last route is an OpenAI-compatible subset with optional SSE streaming. Model ID remains `artemide-core-2.0` for this API generation.

## Python SDK

Install locally with `pip install -e ./sdks/python`:

```python
from artemide_client import ArtemideClient
client = ArtemideClient("http://localhost:8000", api_key="YOUR_SERVER_KEY")
answer = client.chat([{"role": "user", "content": "Hello"}])
print(answer["choices"][0]["message"]["content"])
for text in client.stream_chat([{"role": "user", "content": "Tell me a story"}]):
    print(text, end="", flush=True)
```

## JavaScript SDK (Node 18+)

```js
import { ArtemideClient } from "./sdks/javascript/index.mjs";
const client = new ArtemideClient("http://localhost:8000", "YOUR_SERVER_KEY");
const answer = await client.chat([{ role: "user", content: "Hello" }]);
console.log(answer.choices[0].message.content);
for await (const text of client.streamChat([{ role: "user", content: "Tell me a story" }])) {
  process.stdout.write(text);
}
```

See [docs/API.md](docs/API.md) for request examples and compatibility limits.

## Security and current limits

Set a strong server-side `ARTEMIDE_API_KEY` before public exposure. The current service uses one shared bearer secret; per-customer key issuance, quotas, billing, usage dashboards, distributed rate limiting and abuse controls are not implemented yet. Never put the server key in browser code or mobile bundles. Use a backend proxy, TLS, a trusted reverse proxy and monitoring.

The model is an independently initialized decoder-only Transformer with a byte tokenizer, training pipeline, checkpointing and inference. It does not use third-party pretrained weights or external inference fallbacks. No training corpus is bundled and no paid GPU is started. This research model is not yet instruction-tuned or comparable to mature commercial assistants.

## Tests

The v2.2 GitHub Actions workflow runs the existing core/API suite, Python SDK tests and a JavaScript syntax check on CPU.
