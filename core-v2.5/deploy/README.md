# Artemide zero-budget self-hosting

This is the recommended backend plan when the hosting budget is **₹0/month**.

## What costs zero, and what does not

- Keep the existing static playground on its current Vercel deployment; do not add paid Vercel services.
- Run the inference API on a computer you already own. No rented VM, GPU, database, or paid API is used.
- The computer must remain powered on while you use the API. CPU inference will be slow.
- A free Cloudflare Quick Tunnel can temporarily expose the local API over HTTPS so the hosted playground can reach it. Its URL changes when restarted; it is not a permanent production endpoint.
- Electricity, internet data, and your own hardware are not included in the ₹0 hosting budget.
- A random-init checkpoint is only a startup smoke test. It does not make Artemide a capable assistant. Training a useful model needs suitable licensed data and compute.

## Local CPU setup (Docker)

From this directory's parent, `core-v2.5`:

```bash
cp .env.example .env
```

Generate a long random admin secret and put it in `.env` as `ARTEMIDE_ADMIN_API_KEY`. Keep it private. Then:

```bash
mkdir -p models
cp configs/small.json models/config.json
python -m pip install -e ".[api]"
artemide init --config configs/small.json --output models/latest.pt
docker compose up --build -d
curl http://127.0.0.1:8000/health
```

The default model is intentionally tiny and randomly initialized. The health response should say `model_loaded: true`; any generated output is untrained and should not be presented as useful AI.

## Optional temporary HTTPS connection to the hosted playground

Only do this after the local API is running and you understand that the temporary URL makes the API reachable from the internet.

1. Install the Cloudflare `cloudflared` client on the same computer.
2. Start a temporary tunnel: `cloudflared tunnel --url http://127.0.0.1:8000`.
3. Copy the generated HTTPS `trycloudflare.com` URL. Do not publish it.
4. In the Vercel project settings for `artemide-playground`, set `ARTEMIDE_API_URL` to that URL and set `ARTEMIDE_API_KEY` to a strong server API key configured for the local API. Redeploy the playground.
5. Keep the computer, API container, and tunnel running. If the tunnel URL changes, update the Vercel variable and redeploy.

Do not expose the admin key. Do not leave the API open without authentication. For permanent production use, use a stable domain, TLS, hardened ingress, backups, monitoring and abuse controls; those may exceed a zero budget.

## Developer API keys

Create individual project keys locally:

```bash
curl -X POST http://127.0.0.1:8000/v1/admin/api-keys \
  -H "X-Admin-Key: $ARTEMIDE_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"label":"my-project","requests_per_minute":30,"monthly_token_limit":10000}'
```

The raw developer key is displayed once. Store it securely. Back up the Docker volume containing hashed keys and usage records.

## Zero-budget operating rules

- No paid cloud GPU or compute.
- No paid inference provider or external model fallback.
- No paid tunnel or managed database.
- Stop the container when not in use.
- Treat this as a personal development/test environment, not a reliable public API SLA.
