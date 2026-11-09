export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const base = process.env.ARTEMIDE_API_URL;
  if (!base) return res.status(200).json({ connected: false, model: null, version: "v0.6-dev", reason: "ARTEMIDE_API_URL is not configured" });
  try {
    const response = await fetch(new URL("/health", base), { signal: AbortSignal.timeout(5000) });
    const data = await response.json().catch(() => ({}));
    return res.status(200).json({ connected: response.ok, model: data.model || data.model_name || null, version: data.version || "v0.6-dev", upstreamStatus: response.status });
  } catch (error) {
    return res.status(200).json({ connected: false, model: null, version: "v0.6-dev", reason: "Inference service did not respond" });
  }
}