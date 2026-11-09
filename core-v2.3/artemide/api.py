"""Optional FastAPI service with native and OpenAI-compatible endpoints."""
import hmac
import json
import os
import time
import uuid
from typing import Optional

from .inference import load_model, generate_text, generate_token_ids, stream_text
from .auth import APIKeyStore, RateLimitExceeded, QuotaExceeded


def create_app(model=None, tokenizer=None, device=None, *, api_key: Optional[str] = None,
               admin_api_key: Optional[str] = None, key_db_path: Optional[str] = None):
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
        model: str = Field(default="artemide-core-2.0", min_length=1, max_length=128)
        messages: list[ChatMessage] = Field(min_length=1, max_length=32)
        max_tokens: int = Field(default=128, ge=1, le=2048)
        temperature: float = Field(default=0.0, ge=0.0, le=2.0)
        top_k: int = Field(default=40, ge=0, le=259)
        stream: bool = False

    class CreateAPIKeyRequest(BaseModel):
        label: str = Field(min_length=1, max_length=80)
        requests_per_minute: int = Field(default=60, ge=1, le=10000)
        monthly_token_limit: int = Field(default=100000, ge=1, le=1000000000)

    app = FastAPI(title="Artemide API", version="2.3.0")
    state = {"model": model, "tokenizer": tokenizer, "device": device}
    if state["model"] is None:
        config = os.getenv("ARTEMIDE_CONFIG")
        checkpoint = os.getenv("ARTEMIDE_CHECKPOINT")
        if config and checkpoint:
            state["model"], state["tokenizer"], state["device"] = load_model(config, checkpoint)
    secret = api_key if api_key is not None else os.getenv("ARTEMIDE_API_KEY", "")
    admin_secret = admin_api_key if admin_api_key is not None else os.getenv("ARTEMIDE_ADMIN_API_KEY", "")
    key_store = APIKeyStore(key_db_path or os.getenv("ARTEMIDE_KEY_DB", "artemide_api_keys.sqlite3"))

    def authorize(header):
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="Artemide checkpoint is not configured")
        bearer = header[7:] if header and header.startswith("Bearer ") else ""
        identity = None
        if secret and hmac.compare_digest(bearer, secret):
            identity = {"key_id": None, "usage_id": None, "label": "server-master"}
        elif bearer:
            try:
                identity = key_store.authenticate(bearer)
            except RateLimitExceeded as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
            except QuotaExceeded as exc:
                raise HTTPException(status_code=429, detail=str(exc)) from exc
        if identity is None:
            if secret or key_store.has_active_keys() or os.getenv("ARTEMIDE_REQUIRE_API_KEY", "").lower() in {"1","true","yes"}:
                raise HTTPException(status_code=401, detail="Missing or invalid API key")
            identity = {"key_id": None, "usage_id": None, "label": "development-open"}
        return identity

    def record_usage(identity, prompt, completion):
        if identity and identity.get("key_id") is not None:
            p_count = len(state["tokenizer"].encode(prompt))
            c_count = len(state["tokenizer"].encode(completion))
            key_store.record_usage(identity, p_count, c_count)

    def require_admin(header):
        if not admin_secret:
            raise HTTPException(status_code=503, detail="Set ARTEMIDE_ADMIN_API_KEY to enable key administration")
        if not header or not hmac.compare_digest(header, admin_secret):
            raise HTTPException(status_code=401, detail="Invalid admin key")

    def validate_messages(messages):
        if any(message.role not in {"system", "user", "assistant"} for message in messages):
            raise HTTPException(status_code=400, detail="Invalid message role")
        return "".join(f"{message.role}: {message.content}\n" for message in messages) + "assistant:"

    @app.get("/health")
    def health():
        return {
            "status": "ok" if state["model"] is not None else "offline",
            "model_loaded": state["model"] is not None,
            "version": "2.3.0",
            "streaming": True,
            "openai_compatible": True,
        }

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        return {
            "object": "list",
            "data": [{
                "id": "artemide-core-2.0",
                "object": "model",
                "created": 0,
                "owned_by": "artemide",
            }],
        }

    @app.post("/v1/generate")
    def generate(request: GenerateRequest, authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        try:
            output = generate_text(
                state["model"], state["tokenizer"], request.prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_usage(identity, request.prompt, output)
        return {"text": output, "model": "artemide-core-2.0", "finish_reason": "length"}

    @app.post("/v1/stream")
    def stream(request: GenerateRequest, authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        try:
            fragments = stream_text(
                state["model"], state["tokenizer"], request.prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def events():
            try:
                collected = []
                for fragment in fragments:
                    collected.append(fragment)
                    yield "data: " + json.dumps({"delta": fragment}, ensure_ascii=False) + "\n\n"
                record_usage(identity, request.prompt, "".join(collected))
                yield "data: " + json.dumps({"done": True, "model": "artemide-core-2.0"}) + "\n\n"
            except Exception as exc:
                yield "data: " + json.dumps({"error": "Generation failed: " + type(exc).__name__}) + "\n\n"

        return StreamingResponse(
            events(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/chat")
    def chat(request: ChatRequest, authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        prompt = validate_messages(request.messages)
        try:
            output = generate_text(
                state["model"], state["tokenizer"], prompt, device=state["device"],
                max_new_tokens=request.max_new_tokens, temperature=request.temperature, top_k=request.top_k,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_usage(identity, prompt, output)
        return {
            "message": {"role": "assistant", "content": output},
            "model": "artemide-core-2.0",
            "note": "Chat formatting is not instruction tuning.",
        }

    @app.post("/v1/chat/completions")
    def chat_completions(request: OpenAIChatRequest, authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        prompt = validate_messages(request.messages)
        completion_id = "chatcmpl-" + uuid.uuid4().hex
        created = int(time.time())
        model_name = "artemide-core-2.0"
        if request.model != model_name:
            raise HTTPException(status_code=400, detail=f"Unsupported model. Use {model_name}.")

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
                    collected = []
                    for fragment in fragments:
                        collected.append(fragment)
                        payload = {
                            "id": completion_id, "object": "chat.completion.chunk",
                            "created": created, "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": fragment}, "finish_reason": None}],
                        }
                        yield "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
                    record_usage(identity, prompt, "".join(collected))
                    final_payload = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model_name,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
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
            output_ids = generate_token_ids(
                state["model"], state["tokenizer"], prompt, device=state["device"],
                max_new_tokens=request.max_tokens, temperature=request.temperature, top_k=request.top_k,
            )
            output = state["tokenizer"].decode(output_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        prompt_tokens = len(state["tokenizer"].encode(prompt))
        completion_tokens = len(output_ids)
        record_usage(identity, prompt, output)
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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

 
    @app.post("/v1/admin/api-keys")
    def create_api_key(request: CreateAPIKeyRequest, x_admin_key: Optional[str] = Header(default=None)):
        require_admin(x_admin_key)
        return key_store.create_key(request.label, requests_per_minute=request.requests_per_minute,
                                    monthly_token_limit=request.monthly_token_limit)

    @app.get("/v1/admin/api-keys")
    def list_api_keys(x_admin_key: Optional[str] = Header(default=None)):
        require_admin(x_admin_key)
        return {"data": key_store.list_keys()}

    @app.delete("/v1/admin/api-keys/{key_id}")
    def revoke_api_key(key_id: str, x_admin_key: Optional[str] = Header(default=None)):
        require_admin(x_admin_key)
        if not key_store.revoke(key_id):
            raise HTTPException(status_code=404, detail="API key not found or already revoked")
        return {"id": key_id, "revoked": True}

    @app.get("/v1/usage")
    def usage(authorization: Optional[str] = Header(default=None)):
        identity = authorize(authorization)
        if identity.get("key_id") is None:
            return {"tracked": False, "message": "Usage is not tracked for the server master or development identity"}
        return {"tracked": True, **key_store.usage(identity["key_id"])}

    return app
