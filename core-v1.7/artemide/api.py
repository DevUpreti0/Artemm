"""Minimal FastAPI inference service for a trusted local Artemide checkpoint."""
from __future__ import annotations
import json
import os
from pathlib import Path
from threading import Lock
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from . import __version__, ArtemideLM, ByteTokenizer, ModelConfig
from .training import load_checkpoint

app = FastAPI(title="Artemide API", version=__version__)
_model = None
_tokenizer = ByteTokenizer()
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_lock = Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_k: int = Field(default=40, ge=0, le=259)


def _load_model():
    global _model
    if _model is not None:
        return _model
    checkpoint = os.environ.get("ARTEMIDE_CHECKPOINT")
    config_path = os.environ.get("ARTEMIDE_CONFIG")
    if not checkpoint or not config_path:
        raise RuntimeError("Set ARTEMIDE_CHECKPOINT and ARTEMIDE_CONFIG")
    config = ModelConfig(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    if config.vocab_size != _tokenizer.vocab_size:
        raise RuntimeError(
            f"API byte tokenizer vocabulary ({_tokenizer.vocab_size}) "
            f"does not match model vocabulary ({config.vocab_size})"
        )
    model = ArtemideLM(config).to(_device)
    load_checkpoint(checkpoint, model, map_location=_device, restore_rng=False)
    model.eval()
    _model = model
    return model


@app.get("/health")
def health():
    return {
        "status": "ready" if _model is not None else "not_loaded",
        "model": "Artemide",
        "version": __version__,
        "device": str(_device),
    }


@app.post("/v1/generate")
def generate(request: GenerateRequest, authorization: str | None = Header(default=None)):
    expected = os.environ.get("ARTEMIDE_API_KEY")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Invalid API key")
    try:
        model = _load_model()
        prompt_ids = _tokenizer.encode(request.prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=_device)
        with _lock:
            output = model.generate(
                input_ids, request.max_new_tokens, request.temperature, request.top_k
            )
        generated = output[0, len(prompt_ids):].tolist()
        return {
            "text": _tokenizer.decode(generated),
            "model": "Artemide",
            "version": __version__,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
