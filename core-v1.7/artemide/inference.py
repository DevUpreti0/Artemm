"""Load trusted checkpoints and generate text."""
import json
from pathlib import Path
import torch
from .model.config import ModelConfig
from .model.transformer import ArtemideLM
from .tokenizer import ByteTokenizer
from .training import load_checkpoint

def load_model(config_path, checkpoint_path, *, device=None):
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with Path(config_path).open(encoding="utf-8") as handle:
        config=ModelConfig(**json.load(handle))
    model=ArtemideLM(config).to(device)
    load_checkpoint(checkpoint_path,model,map_location=device,restore_rng=False)
    model.eval()
    return model,ByteTokenizer(),device

@torch.inference_mode()
def generate_text(model,tokenizer,prompt,*,device,max_new_tokens=128,temperature=0.0,top_k=40):
    if not isinstance(prompt,str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    ids=torch.tensor([tokenizer.encode(prompt)],dtype=torch.long,device=device)
    result=model.generate(ids,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k,use_cache=True)
    return tokenizer.decode(result[0,ids.shape[1]:].tolist())
