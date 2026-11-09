export default async function handler(req, res) {
  res.setHeader("Cache-Control", "no-store");
  const base = process.env.ARTEMIDE_API_URL;
  if (!base) return res.status(200).json({
    connected: false, model: null, version: "playground-2.5.2", streaming: true,
    reason: "ARTEMIDE_API_URL is not configured"
  });
  try {
    const response = await fetch(new URL("/health", base), { signal: AbortSignal.timeout(5000) });
    const data = await response.json().catch(() => ({}));
    const loaded = data.model_loaded === true;
    return res.status(200).json({
      connected: response.ok && loaded,
      model: data.model || data.model_name || (loaded ? "Artemide Core" : "Not loaded"),
      version: data.version || "unknown",
      streaming: data.streaming === true,
      upstreamStatus: response.status,
      reason: loaded ? null : "The API is reachable, but no model checkpoint is loaded"
    });
  } catch (error) {
    return res.status(200).json({
      connected: false, model: null, version: "unreachable", streaming: false,
      reason: "Inference service did not respond"
    });
  }
}
