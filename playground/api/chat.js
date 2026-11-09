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
    const upstream = await fetch(new URL("/v1/generate", base), {
      method: "POST", headers,
      body: JSON.stringify({ prompt, max_new_tokens: 256, temperature: 0.8 }),
      signal: AbortSignal.timeout(60000)
    });
    const data = await upstream.json().catch(() => ({}));
    if (!upstream.ok) return res.status(upstream.status).json({ error: data.detail || data.error || "Artemide inference request failed." });
    return res.status(200).json({ response: data.response || data.text || data.generated_text || "", model: data.model || "Artemide" });
  } catch (error) {
    return res.status(502).json({ error: "The configured Artemide inference service could not be reached." });
  }
}