# Self-hosting Artemide

## CPU quick start

1. Install Docker Engine and Docker Compose.
2. Copy `.env.example` to `.env` and set a strong random `ARTEMIDE_ADMIN_API_KEY`.
3. Create a model folder and config: `mkdir -p models && cp configs/small.json models/config.json`.
4. Create a smoke-test checkpoint: `pip install -e . && artemide init --config configs/small.json --output models/latest.pt`. This is random and untrained; it only verifies startup.
5. Start: `docker compose up --build -d`.
6. Check `http://127.0.0.1:8000/health`. The default port binding is loopback-only.

## Issue a developer key

```bash
curl -X POST http://127.0.0.1:8000/v1/admin/api-keys \
  -H "X-Admin-Key: $ARTEMIDE_ADMIN_API_KEY" -H "Content-Type: application/json" \
  -d '{"label":"my-project","requests_per_minute":60,"monthly_token_limit":100000}'
```

The raw key is shown only once. Keep Docker volume `artemide-data` backed up; it contains hashed keys and usage records.

## GPU hosts

The Dockerfile accepts `TORCH_INDEX_URL`. Build with a PyTorch wheel index matching the target CUDA runtime, install NVIDIA Container Toolkit, and validate driver/runtime compatibility with a local smoke test. Do not launch paid instances without explicit budget approval.

## Production gaps

This is a single-instance starting point. Before public multi-tenant production, add TLS reverse proxy, distributed rate limiting, shared database, request/concurrency controls, billing, alerting, structured logs, backup restore drills, abuse monitoring and load tests. Never expose admin/server keys to client-side apps.
