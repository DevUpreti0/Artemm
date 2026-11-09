"""Optional FastAPI service; no external model provider or fallback."""
import os
from typing import Optional
from .inference import load_model,generate_text

def create_app(model=None,tokenizer=None,device=None,*,api_key:Optional[str]=None):
    try:
        from fastapi import FastAPI,Header,HTTPException
        from pydantic import BaseModel,Field
    except ImportError as exc:
        raise RuntimeError("Install API extras: pip install 'artemide-core[api]'") from exc
    class GenerateRequest(BaseModel):
        prompt:str=Field(min_length=1,max_length=20000)
        max_new_tokens:int=Field(default=128,ge=1,le=2048)
        temperature:float=Field(default=0.0,ge=0.0,le=2.0)
        top_k:int=Field(default=40,ge=0,le=259)
    class ChatMessage(BaseModel):
        role:str
        content:str=Field(min_length=1,max_length=20000)
    class ChatRequest(BaseModel):
        messages:list[ChatMessage]=Field(min_length=1,max_length=32)
        max_new_tokens:int=Field(default=128,ge=1,le=2048)
        temperature:float=Field(default=0.0,ge=0.0,le=2.0)
        top_k:int=Field(default=40,ge=0,le=259)
    app=FastAPI(title="Artemide API",version="1.7.0")
    state={"model":model,"tokenizer":tokenizer,"device":device}
    if state["model"] is None:
        config=os.getenv("ARTEMIDE_CONFIG"); checkpoint=os.getenv("ARTEMIDE_CHECKPOINT")
        if config and checkpoint:
            state["model"],state["tokenizer"],state["device"]=load_model(config,checkpoint)
    secret=api_key if api_key is not None else os.getenv("ARTEMIDE_API_KEY","")
    def authorize(header):
        if secret and header!=f"Bearer {secret}":
            raise HTTPException(status_code=401,detail="Invalid API key")
        if state["model"] is None:
            raise HTTPException(status_code=503,detail="Artemide checkpoint is not configured")
    @app.get("/health")
    def health():
        return {"status":"ok" if state["model"] is not None else "offline",
                "model_loaded":state["model"] is not None,"version":"1.7.0"}
    @app.post("/v1/generate")
    def generate(request:GenerateRequest,authorization:Optional[str]=Header(default=None)):
        authorize(authorization)
        try:
            output=generate_text(state["model"],state["tokenizer"],request.prompt,device=state["device"],
                max_new_tokens=request.max_new_tokens,temperature=request.temperature,top_k=request.top_k)
        except ValueError as exc:
            raise HTTPException(status_code=400,detail=str(exc)) from exc
        return {"text":output,"model":"artemide-core-1.7","finish_reason":"length"}
    @app.post("/v1/chat")
    def chat(request:ChatRequest,authorization:Optional[str]=Header(default=None)):
        authorize(authorization)
        if any(message.role not in {"system","user","assistant"} for message in request.messages):
            raise HTTPException(status_code=400,detail="Invalid message role")
        prompt="".join(f"{message.role}: {message.content}\n" for message in request.messages)+"assistant:"
        output=generate_text(state["model"],state["tokenizer"],prompt,device=state["device"],
            max_new_tokens=request.max_new_tokens,temperature=request.temperature,top_k=request.top_k)
        return {"message":{"role":"assistant","content":output},"model":"artemide-core-1.7",
                "note":"Chat formatting is not instruction tuning."}
    return app
