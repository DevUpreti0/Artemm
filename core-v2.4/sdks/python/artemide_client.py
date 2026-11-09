"""Dependency-free Python client for the Artemide HTTP API."""
from __future__ import annotations
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class ArtemideAPIError(RuntimeError):
    def __init__(self, message, *, status_code=None, body=None):
        super().__init__(message)
        self.status_code, self.body = status_code, body

class ArtemideClient:
    def __init__(self, base_url, api_key=None, *, timeout=60.0):
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout

    def _request(self, path, payload=None, *, stream=False):
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if payload is not None: headers["Content-Type"] = "application/json"
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        req = Request(self.base_url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers, method="POST" if payload is not None else "GET")
        try: response = urlopen(req, timeout=self.timeout)
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try: body = json.loads(raw)
            except json.JSONDecodeError: body = raw
            detail = body.get("detail", body.get("error", raw)) if isinstance(body, dict) else raw
            raise ArtemideAPIError(str(detail), status_code=exc.code, body=body) from exc
        except URLError as exc:
            raise ArtemideAPIError(f"Could not reach Artemide API: {exc.reason}") from exc
        if stream: return response
        try:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
        finally: response.close()

    def models(self): return self._request("/v1/models")

    def generate(self, prompt, *, max_new_tokens=128, temperature=0.0, top_k=40):
        return self._request("/v1/generate", {"prompt":prompt,"max_new_tokens":max_new_tokens,
            "temperature":temperature,"top_k":top_k})

    def chat(self, messages, *, model="artemide-core-2.0", max_tokens=128, temperature=0.0, top_k=40):
        return self._request("/v1/chat/completions", {"model":model,"messages":messages,
            "max_tokens":max_tokens,"temperature":temperature,"top_k":top_k,"stream":False})

    def stream_chat(self, messages, *, model="artemide-core-2.0", max_tokens=128, temperature=0.0, top_k=40):
        response = self._request("/v1/chat/completions", {"model":model,"messages":messages,
            "max_tokens":max_tokens,"temperature":temperature,"top_k":top_k,"stream":True}, stream=True)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"): continue
                data = line[5:].strip()
                if data == "[DONE]": return
                if not data: continue
                try: event = json.loads(data)
                except json.JSONDecodeError as exc: raise ArtemideAPIError("Invalid SSE event") from exc
                if "error" in event:
                    err = event["error"]
                    raise ArtemideAPIError(str(err.get("message", err)))
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {}).get("content")
                    if delta: yield delta
        finally: response.close()
