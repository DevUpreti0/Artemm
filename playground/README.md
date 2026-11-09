# Artemide Playground

The static chat interface is deployed separately from the model runtime. It does not itself contain an AI model and never substitutes another provider.

## Zero-budget connection

With a ₹0/month hosting budget, run Artemide Core on a computer you already own. Follow [the zero-budget self-hosting guide](../core-v2.5/deploy/README.md). For temporary access from this hosted playground, use the optional Cloudflare Quick Tunnel instructions there.

Configure these server-side Vercel environment variables only after your local API is running:

- `ARTEMIDE_API_URL`: HTTPS base URL of the temporary tunnel.
- `ARTEMIDE_API_KEY`: a developer API key accepted by the local Artemide API. Never use the admin key.

The API must expose `GET /health` and `POST /v1/stream` (SSE). The proxy forwards the stream; it does not generate answers itself.

A Quick Tunnel URL changes when restarted. Keep the computer, Docker service and tunnel running. Update the Vercel variable and redeploy when the URL changes.

## Important model limitation

The bundled small random-initialized checkpoint is only a startup smoke test. It is not trained to follow instructions and its output is not useful assistant-quality language. A useful model requires permitted training data, training time, evaluation and hardware. This guide starts no paid resources.

## Security

Keep API/admin keys server-side. Never commit secrets to GitHub or place them in browser JavaScript. Do not expose the admin endpoint to the public internet.
