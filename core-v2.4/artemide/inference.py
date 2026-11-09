"""Load trusted checkpoints and generate text."""
import codecs
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .inference_cache import KVCache
from .model.config import ModelConfig
from .model.transformer import ArtemideLM
from .tokenizer import ByteTokenizer
from .training import load_checkpoint


def load_model(config_path, checkpoint_path, *, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with Path(config_path).open(encoding="utf-8") as handle:
        config = ModelConfig(**json.load(handle))
    model = ArtemideLM(config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device, restore_rng=False)
    model.eval()
    return model, ByteTokenizer(), device


@torch.inference_mode()
def generate_token_ids(model, tokenizer, prompt, *, device, max_new_tokens=128, temperature=0.0, top_k=40):
    """Return generated token IDs so API accounting uses the exact tokenizer output."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    result = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, use_cache=True)
    return result[0, ids.shape[1]:].tolist()


def generate_text(model, tokenizer, prompt, *, device, max_new_tokens=128, temperature=0.0, top_k=40):
    token_ids = generate_token_ids(
        model, tokenizer, prompt, device=device, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k,
    )
    return tokenizer.decode(token_ids)


def stream_text(model, tokenizer, prompt, *, device, max_new_tokens=128, temperature=0.0, top_k=40):
    """Yield UTF-8-safe text fragments as newly sampled tokens arrive."""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    if max_new_tokens < 0 or top_k < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("invalid generation settings")

    def generate():
        with torch.inference_mode():
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            if input_ids.shape[1] > model.config.max_seq_len:
                input_ids = input_ids[:, -model.config.max_seq_len:]
            if max_new_tokens == 0 or input_ids.shape[1] >= model.config.max_seq_len:
                return
            cache = KVCache(max_length=model.config.max_seq_len)
            logits = model(input_ids, cache=cache, use_cache=True)[:, -1, :]
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            for index in range(max_new_tokens):
                if temperature == 0:
                    next_id = logits.argmax(-1, keepdim=True)
                else:
                    scores = logits / temperature
                    if top_k:
                        k = min(top_k, scores.shape[-1])
                        cutoff = torch.topk(scores, k, dim=-1).values[..., -1, None]
                        scores = scores.masked_fill(scores < cutoff, float("-inf"))
                    next_id = torch.multinomial(F.softmax(scores, dim=-1), 1)
                token_id = int(next_id.item())
                if token_id == tokenizer.eos_token_id:
                    tail = decoder.decode(b"", final=True)
                    if tail:
                        yield tail
                    return
                if 3 <= token_id < tokenizer.vocab_size:
                    fragment = decoder.decode(bytes([token_id - 3]), final=False)
                    if fragment:
                        yield fragment
                if index + 1 >= max_new_tokens or input_ids.shape[1] + index + 1 >= model.config.max_seq_len:
                    break
                logits = model(next_id, cache=cache, use_cache=True)[:, -1, :]
            tail = decoder.decode(b"", final=True)
            if tail:
                yield tail

    return generate()
