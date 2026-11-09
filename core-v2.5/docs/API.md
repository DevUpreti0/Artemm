# Artemide Developer API

Use the base URL of your own deployed Artemide server. Protected endpoints accept `Authorization: Bearer YOUR_ARTEMIDE_API_KEY`.

## Chat completion

```bash
curl "$ARTEMIDE_URL/v1/chat/completions" \
  -H "Authorization: Bearer $ARTEMIDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"artemide-core-2.0","messages":[{"role":"user","content":"Hello"}],"max_tokens":100,"temperature":0}'
```

Set `stream: true` for SSE chunks in `choices[0].delta.content`, terminated by `data: [DONE]`.

## Python

Install `sdks/python` with `pip install -e ./sdks/python`, then import `ArtemideClient` from `artemide_client`. Methods: `models()`, `generate(prompt)`, `chat(messages)`, and `stream_chat(messages)`.

## JavaScript

Node 18+ imports `ArtemideClient` from `sdks/javascript/index.mjs`. Methods: `models()`, `generate(prompt)`, `chat(messages)`, and async iterator `streamChat(messages)`.

## Compatibility boundary

Supported: system/user/assistant messages, model ID, max_tokens, temperature, top_k, JSON chat responses, SSE streaming and byte-token usage. Not supported yet: tools/function calling, image/audio/video inputs, embeddings, logprobs, fine-tuning endpoints, per-customer keys, quotas, billing, organizations or usage dashboards. This is not full compatibility with another provider.

## Public deployment checklist

Use TLS; keep keys server-side; add distributed rate limiting, per-customer key management, request/concurrency limits, monitoring, backups and abuse controls before public multi-tenant launch.


## Developer key management (v2.3)

Set `ARTEMIDE_ADMIN_API_KEY` on the server. Create a key:

```bash
curl -X POST "$ARTEMIDE_URL/v1/admin/api-keys" \\
  -H "X-Admin-Key: $ARTEMIDE_ADMIN_API_KEY" -H "Content-Type: application/json" \\
  -d '{"label":"my-app","requests_per_minute":60,"monthly_token_limit":100000}'
```

Copy the returned `api_key` immediately; it is not shown again. Use it as `Authorization: Bearer ...`. Admin routes can list metadata and revoke a key. A developer key can query its usage from `GET /v1/usage`. The SQLite database is local to one service instance; multi-instance deployments need shared storage and distributed rate limiting.
