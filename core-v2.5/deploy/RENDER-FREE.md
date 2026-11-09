# Zero-budget Render prototype

This optional profile runs the Artemide API on Render's Free Web Service. It does not require your own computer to stay on, and it does not call another AI provider.

## Important limits

- Render Free Web Services have 512 MB RAM and 0.5 CPU.
- They spin down after 15 minutes without inbound traffic; the next request can take about a minute to wake them.
- The workspace receives 750 free instance-hours per calendar month. This allowance is shared across its free services.
- The service filesystem is ephemeral. The demo API-key database is stored under /tmp and is lost on restarts or spin-down.
- The container creates a tiny random-initialization checkpoint at build time. It is only a startup/inference smoke test, not a trained assistant. Expect meaningless output.
- The service may be too memory-constrained for PyTorch and can fail to start. If that happens, do not upgrade the plan; keep the budget at zero.

This is a prototype, not a production service or a reliable public developer API. Do not distribute its server API key. Developer keys and usage history are not durable on this free profile.

## Deploy from an Android phone

1. Open https://dashboard.render.com/ and sign in or create a free account.
2. Choose **New** → **Blueprint**.
3. Connect the public GitHub repository `DevUpreti0/Artemm`, select branch `main`, and let Render detect the root `render.yaml`.
4. Review the proposed resource carefully. It must say **Free** (`plan: free`) and show no paid services. If Render asks you to upgrade, add a payment method, or choose a paid plan, stop and do not continue.
5. Create the Blueprint. Render generates `ARTEMIDE_API_KEY` and `ARTEMIDE_ADMIN_API_KEY` as environment secrets; do not paste either into public chats or GitHub.
6. Wait for the build and deployment. Open the service URL ending in `.onrender.com`, then visit `/health`.

A healthy startup response should contain `"model_loaded": true`. This confirms that the API and random smoke-test checkpoint loaded; it does not prove useful language ability.

## API smoke test

Use the service's API URL with a bearer token kept private:

```bash
curl -X POST "https://YOUR-SERVICE.onrender.com/v1/generate" \
  -H "Authorization: Bearer YOUR_ARTEMIDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hello Artemide","max_new_tokens":24,"temperature":0}'
```

The response is expected to be random/untrained text until an actual checkpoint is trained and deployed.

## Before connecting the website or issuing developer keys

Do not connect the public playground to this random checkpoint. First train and evaluate a real checkpoint. Before exposing developer API keys to other users, replace the ephemeral SQLite key store with durable storage and add rate limiting, abuse protection, monitoring, backup/restore and load tests. Keep the API and admin secrets server-side only.

If the service reports out-of-memory or fails its health check, stop here. Do not choose a paid Render instance under the ₹0 budget.
