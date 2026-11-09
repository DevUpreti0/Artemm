"""Optional FastAPI service with native and OpenAI-compatible endpoints."""
import json
import os
import time
import uuid
from typing import Optional

from .inference import load_model, generate_text, stream_text


def create_app(model=None, tokenizer=None, device=None, *, api_key: Optional[str] = None):
    try:
        from fastapi import FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install API extras: pip install 'artemide-core[api]'") from exc

    class GenerateRequest(BaseModel):
        prompt: str = Field(min_length=1, max_length=20000)
        max_new_tokens: int = Field(default=128, ge=1, le=2048)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_k: int = Field(default=40, ge=0, le=259)

    class ChatMessage(BaseModel):
        role: str
        content: str = Field(min_length=1, max_length=20000)

    class ChatRequest(BaseModel):
        messages: list[ChatMessage] = Field(min_length=1, max_length=32)
        max_new_tokens: int = Field(default=128, ge=1, le=2048)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_k: int = Field(default=40, ge=0, le=259)

    class OpenAIChatRequest(BaseModel):
        model: str = Field(default="artemide-core-1.9", min_length=1, max_length=128)
        messages: list[ChatMessage] = Field(min_length=1, max_length=32)
        max_tokens: int = Field(default=128, ge=1, le=2048)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_k: int = Field(default=40, ge=0, le=259)
        stream: bool = False

    app = FastAPI(title="Artemide API", version="1.9.0")
    state = {"model": model, "tokenizer": tokenizer, "device": device}
    if state["model"] is None:
        config = os.getenv("ARTEMIDE_CONFIG")
        checkpoint = os.getenv("ARTEMIDE_CHECKPOINT")
        if config and checkpoint:
            state["model"], state["tokenizer"], state["device"] = load_model(config, checkpoint)
    secret = api_key if api_key is not None else os.getenv("ARTEMIDE_API_KEY", "")

    def authorize(header):
        if secret and header != f"Bearer {secret}":
            raise HTTPException(status_code=401, detail="Invalid API key")
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Artemide checkpoint is not configured")

    def validate_messages(messages):
        if any(message.role not in {"system", "user", "assistant"} for message in messages):
            raise HTTPException(status_code=400, detail="Invalid message role")
        return "".join(f"{message.role}: {message.content}\n" for message in messages) + "assistant:"

    @app.get("/health")
    def health():
        return {
            "status": "ok" if state["model"] is not None else "offline",
            "model_loaded": state["model"] is not None,
            "version": "1.9.0",
            "streaming": True,
            "openai_compatible": True,
        }

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        return {
            "object": "list",
            "data": [{
                "id": "artemide-core-1.9",
                "object": "model",
                "created": 0,
                "owned_by": "artemide",
            }],
        }

    @app.post("/v1/generate")
    def generate(request: GenerateRequest, authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        try:
            output = generate_text(
                state["model"], state["tokenizer"], request.prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"text": output, "model": "artemide-core-1.9", "finish_reason": "length"}

    @app.post("/v1/stream")
    def stream(request: GenerateRequest, authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        try:
            fragments = stream_text(
                state["model"], state["tokenizer"], request.prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def events():
            try:
                for fragment in fragments:
                    yield "data: " + json.dumps({"delta": fragment}, ensure_ascii=False) + "\n\n"
                yield "data: " + json.dumps({"done": True, "model": "artemide-core-1.9"}) + "\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"error": "Generation failed: " + type(exc).__name__}) + "\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat")
    def chat(request: ChatRequest, authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        prompt = validate_messages(request.messages)
        try:
            output = generate_text(
                state["model"], state["tokenizer"], prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "message": {"role": "assistant", "content": output},
            "model": "artemide-core-1.9",
            "note": "Chat formatting is not instruction tuning.",
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: OpenAIChatRequest, authorization: Optional[str] = Header(default=None)):
        authorize(authorization)
        prompt = validate_messages(request.messages)
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        model_name = request.model or "artemide-core-1.9"

        if request.stream:
            try:
                fragments = stream_text(
                    state["model"], state["tokenizer"], prompt, device=state["device"],
                    max_new_tokens=request.max_tokens, temperature=request.temperature, top_k=request.top_k,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            def chunks():
                try:
                    for fragment in fragments:
                        payload = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": fragment}, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    final_payload = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    }
                    yield "data: " + json.dumps(final_payload) + "\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as exc:
                    payload = {"error": {"message": "Generation failed: " + type(exc).__name__, "type": "server_error"}}
                    yield "data: " + json.dumps(payload) + "\n\n"

            return StreamingResponse(
                chunks(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )

        try:
            output = generate_text(
                state["model"], state["tokenizer"], prompt, device=state["device"],
                max_new_tokens=request.max_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app
