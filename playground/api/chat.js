export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });
  const base = process.env.ARTEMIDE_API_URL;
  if (!base) return res.status(503).json({ error: "Artemide inference is not connected yet. The playground is live, but no trained model runtime is configured." });
  const prompt = typeof req.body?.prompt === "string" ? req.body.prompt.trim() : "";
  if (!prompt) return res.status(400).json({ error: "Enter a prompt first." });
  if (prompt.length > 6000) return res.status(413).json({ error: "Prompt is too long (maximum 6,000 characters)." });
  try {
    const headers = { "Content-Type": "application/json" };
    if (process.env.ARTEMIDE_API_KEY) headers.Authorization = "Bearer " + process.env.ARTEMIDE_API_KEY;
    const upstream = await fetch(new URL("/v1/stream", base), {
      method: "POST", headers,
      body: JSON.stringify({ prompt, max_new_tokens: 256, temperature: 0.8 }),
      signal: AbortSignal.timeout(120000)
    });
    if (!upstream.ok) {
      const data = await upstream.json().catch(() => ({}));
      return res.status(upstream.status).json({ error: data.detail || data.error || "Artemide inference request failed." });
    }
    if (!upstream.body) return res.status(502).json({ error: "Artemide returned an empty stream." });
    res.status(200);
    res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
    res.setHeader("Cache-Control", "no-cache, no-transform");
    res.setHeader("Connection", "keep-alive");
    res.setHeader("X-Accel-Buffering", "no");
    for await (const chunk of upstream.body) res.write(chunk);
    return res.end();
  } catch (error) {
    if (!res.headersSent) return res.status(502).json({ error: "The configured Artemide inference service could not be reached." });
    return res.end();
  }
}
