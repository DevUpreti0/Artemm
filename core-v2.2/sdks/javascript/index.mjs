/** Dependency-free JavaScript client for Artemide (Node.js 18+). */
export class ArtemideAPIError extends Error {
  constructor(message, { statusCode = null, body = null } = {}) {
    super(message); this.name = "ArtemideAPIError"; this.statusCode = statusCode; this.body = body;
  }
}
export class ArtemideClient {
  constructor(baseURL, apiKey, { timeoutMs = 60000, fetchImpl = globalThis.fetch } = {}) {
    if (typeof fetchImpl !== "function") throw new TypeError("Fetch implementation required");
    this.baseURL = baseURL.replace(/\/$/, ""); this.apiKey = apiKey;
    this.timeoutMs = timeoutMs; this.fetchImpl = fetchImpl;
  }
  async request(path, payload, { signal } = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    if (signal) signal.addEventListener("abort", () => controller.abort(), { once: true });
    try {
      const headers = { Accept: "application/json" };
      if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
      if (payload !== undefined) headers["Content-Type"] = "application/json";
      const res = await this.fetchImpl(this.baseURL + path, { method: payload === undefined ? "GET" : "POST",
        headers, body: payload === undefined ? undefined : JSON.stringify(payload), signal: controller.signal });
      const raw = await res.text(); let body = {};
      try { body = raw ? JSON.parse(raw) : {}; } catch { body = { raw }; }
      if (!res.ok) throw new ArtemideAPIError(String(body?.detail ?? body?.error?.message ?? body?.raw ?? res.statusText),
        { statusCode: res.status, body });
      return body;
    } catch (err) {
      if (err instanceof ArtemideAPIError) throw err;
      throw new ArtemideAPIError(err?.name === "AbortError" ? "Artemide request timed out" : err.message);
    } finally { clearTimeout(timer); }
  }
  models(options) { return this.request("/v1/models", undefined, options); }
  generate(prompt, options = {}) {
    const { signal, ...p } = options;
    return this.request("/v1/generate", { prompt, max_new_tokens:p.maxNewTokens ?? 128,
      temperature:p.temperature ?? 0, top_k:p.topK ?? 40 }, { signal });
  }
  chat(messages, options = {}) {
    const { signal, ...p } = options;
    return this.request("/v1/chat/completions", { model:p.model ?? "artemide-core-2.0", messages,
      max_tokens:p.maxTokens ?? 128, temperature:p.temperature ?? 0, top_k:p.topK ?? 40, stream:false }, { signal });
  }
  async *streamChat(messages, options = {}) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), options.timeoutMs ?? this.timeoutMs);
    if (options.signal) options.signal.addEventListener("abort", () => controller.abort(), { once:true });
    try {
      const headers = { Accept:"text/event-stream", "Content-Type":"application/json" };
      if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
      const res = await this.fetchImpl(this.baseURL + "/v1/chat/completions", { method:"POST", headers,
        signal:controller.signal, body:JSON.stringify({ model:options.model ?? "artemide-core-2.0", messages,
          max_tokens:options.maxTokens ?? 128, temperature:options.temperature ?? 0,
          top_k:options.topK ?? 40, stream:true }) });
      if (!res.ok) {
        const raw = await res.text(); let body = {};
        try { body = JSON.parse(raw); } catch { body = { raw }; }
        throw new ArtemideAPIError(String(body?.detail ?? body?.error?.message ?? body?.raw ?? res.statusText),
          { statusCode:res.status, body });
      }
      if (!res.body) throw new ArtemideAPIError("Streaming response body unavailable");
      const reader = res.body.getReader(), decoder = new TextDecoder(); let buffer = "";
      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value ?? new Uint8Array(), { stream:!done });
          const events = buffer.split("\n\n"); buffer = events.pop() ?? "";
          for (const eventText of events) for (const line of eventText.split("\n")) {
            if (!line.startsWith("data:")) continue;
            const data = line.slice(5).trim(); if (data === "[DONE]") return; if (!data) continue;
            let event; try { event = JSON.parse(data); } catch { throw new ArtemideAPIError("Invalid SSE event"); }
            if (event.error) throw new ArtemideAPIError(event.error.message ?? String(event.error));
            const delta = event.choices?.[0]?.delta?.content; if (delta) yield delta;
          }
          if (done) break;
        }
      } finally { await reader.cancel().catch(() => {}); }
    } catch (err) {
      if (err instanceof ArtemideAPIError) throw err;
      throw new ArtemideAPIError(err?.name === "AbortError" ? "Artemide stream timed out" : err.message);
    } finally { clearTimeout(timer); }
  }
}
