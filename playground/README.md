# Artemide Playground

A static chat playground and server-side proxy for the independently built Artemide inference API.

## Deploy

This directory is intended to be the Vercel project root.

## Connect a running model

Set these environment variables in the Vercel project:

- `ARTEMIDE_API_URL`: base URL of the running Artemide inference service (for example, an HTTPS service URL).
- `ARTEMIDE_API_KEY`: optional bearer token for the inference service.

The service must expose:
- `GET /health`
- `POST /v1/generate` accepting JSON `{ "prompt": "...", "max_new_tokens": 256, "temperature": 0.8 }`

The playground intentionally does not use another provider or fabricate responses. Without a connected runtime, it displays an offline status and a clear error when a prompt is submitted.

## Security

API credentials are stored only as Vercel server-side environment variables and are never embedded in browser code. Use HTTPS for the inference service.
