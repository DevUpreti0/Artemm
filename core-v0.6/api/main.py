import os, json
from functools import lru_cache
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from artemide.inference import InferenceRuntime

app=FastAPI(title="Artemide Core API",version="0.6.0")
API_KEY=os.getenv("ARTEMIDE_API_KEY","")
class GenerateRequest(BaseModel):
    prompt: str=Field(min_length=1,max_length=6000)
    max_new_tokens: int=Field(default=128,ge=1,le=512)
    temperature: float=Field(default=0.8,ge=0.05,le=2.0)
    top_k: int=Field(default=40,ge=0,le=259)

@lru_cache(maxsize=1)
def runtime():
    checkpoint=os.getenv("ARTEMIDE_CHECKPOINT")
    config=os.getenv("ARTEMIDE_CONFIG","configs/small.json")
    if not checkpoint or not os.path.isfile(checkpoint):
        raise RuntimeError("ARTEMIDE_CHECKPOINT does not point to a trained checkpoint")
    return InferenceRuntime(checkpoint,config)

def auth(key):
    if not API_KEY or key != f"Bearer {API_KEY}":
        raise HTTPException(401,"Valid API key required")

@app.get("/health")
def health():
    try:
        model=runtime()
        return {"status":"ok","model":"Artemide Core","version":"0.6.0","generation_ready":True,"step":model.step}
    except Exception as exc:
        return {"status":"degraded","model":"Artemide Core","version":"0.6.0","generation_ready":False,"reason":str(exc)}

@app.get("/v1/metrics")
def metrics(authorization: str|None=Header(default=None)):
    auth(authorization)
    try:
        with open("runs/metrics.jsonl",encoding="utf-8") as f: rows=[json.loads(x) for x in f if x.strip()]
    except FileNotFoundError: rows=[]
    return {"points":rows[-500:]}

@app.post("/v1/generate")
def generate(body: GenerateRequest, authorization: str|None=Header(default=None)):
    auth(authorization)
    try: model=runtime()
    except Exception as exc: raise HTTPException(503,str(exc))
    try:
        text=model.generate(body.prompt,body.max_new_tokens,body.temperature,body.top_k)
        return {"response":text,"model":"Artemide Core","step":model.step}
    except Exception as exc: raise HTTPException(500,f"Generation failed: {exc}")

@app.get("/dashboard",response_class=HTMLResponse)
def dashboard():
    try: return open("dashboard/index.html",encoding="utf-8").read()
    except FileNotFoundError: return "<h1>Artemide training dashboard</h1>"
