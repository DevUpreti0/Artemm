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
