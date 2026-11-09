# Artemide Core v2.4 — self-hostable independent AI API

v2.4 adds a random-init checkpoint command, sample configuration and container deployment files. The API runs Artemide's own model and never falls back to an outside AI provider.

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
```

Install PyTorch using the official selector for your CPU/CUDA environment.

## Initialize and train

```bash
artemide init --config configs/small.json --output runs/artemide/initial.pt
artemide prepare --input ./licensed-corpus --output ./data/processed
artemide train --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 10000 \
  --batch-size 8 --sequence-length 256 --grad-accum-steps 4
```

The init command creates random weights only. It is for verifying that the API boots, not for useful answers. Language ability requires licensed training data, substantial compute and evaluation.

## Run the API locally

```bash
export ARTEMIDE_CONFIG=configs/small.json
export ARTEMIDE_CHECKPOINT=runs/artemide/latest.pt
export ARTEMIDE_API_KEY='use-a-long-random-secret'
export ARTEMIDE_ADMIN_API_KEY='use-a-different-admin-secret'
export ARTEMIDE_REQUIRE_API_KEY=1
artemide serve --host 0.0.0.0 --port 8000
```

Routes include native generate/chat/stream, OpenAI-compatible chat completions, developer key management and usage reporting. See [docs/API.md](docs/API.md).

## Docker Compose

```bash
cp .env.example .env
mkdir -p models
cp configs/small.json models/config.json
# Put a checkpoint at models/latest.pt (random init is only a smoke test).
docker compose up --build -d
```

The default port binding is localhost-only. Use a TLS reverse proxy before remote access. See [deploy/README.md](deploy/README.md).

For CUDA, build with a PyTorch wheel index matching the host's supported CUDA runtime, then use NVIDIA Container Toolkit. Check the official PyTorch selector; do not blindly reuse an example CUDA version.

## SDKs

Python SDK: `pip install -e ./sdks/python`.
JavaScript SDK: Node.js 18+, import `ArtemideClient` from `sdks/javascript/index.mjs`.

## Security and limits

- Use different long random values for server and admin keys.
- Developer keys are stored as hashes in SQLite; raw keys are shown once.
- Per-key RPM and monthly token limits work for a single API instance. A request already in flight may finish before usage is recorded.
- Multi-instance production needs shared database storage, distributed rate limiting, billing, abuse controls, backups, monitoring and load tests.
- Never expose server/admin keys in browser or mobile bundles.
- No paid cloud/GPU resources are provisioned.

## Model reality

Artemide currently has an independently initialized decoder-only text model, byte tokenizer, training and inference stack. It does not yet have image/audio/video generation models, web browsing or a general agent runtime. Those require separate trained model families and permissioned tools. This release does not claim parity with commercial AI systems.
