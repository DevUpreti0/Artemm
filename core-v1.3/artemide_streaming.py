"""Streaming and low-overhead generation helpers for Artemide v1.1+."""
import math
from typing import Iterator

import torch
import torch.nn.functional as F

from artemide.inference_cache import KVCache
from artemide.model.transformer import ArtemideLM


class StreamingArtemideLM(ArtemideLM):
    """ArtemideLM with token streaming and one-shot output assembly.

    The model architecture and weights remain compatible with ArtemideLM.
    stream_generate yields one [batch, 1] token tensor per decoding step.
    """

    @staticmethod
    def _sample(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)
        scores = logits / temperature
        if top_k:
            k = min(top_k, scores.shape[-1])
            cutoff = torch.topk(scores, k, dim=-1).values[..., -1, None]
            scores = scores.masked_fill(scores < cutoff, float("-inf"))
        return torch.multinomial(F.softmax(scores, dim=-1), 1)

    def _validate_generation(self, input_ids, max_new_tokens, temperature, top_k):
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer")
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be non-empty [batch, sequence]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("input_ids must contain integer token IDs")

    @torch.inference_mode()
    def stream_generate(
        self, input_ids: torch.Tensor, max_new_tokens: int = 128,
        temperature: float = 0.8, top_k: int = 40, use_cache: bool = True,
    ) -> Iterator[torch.Tensor]:
        """Yield each next-token tensor; consumers can send it immediately."""
        self._validate_generation(input_ids, max_new_tokens, temperature, top_k)
        prompt = input_ids[:, -self.config.max_seq_len:]
        if max_new_tokens == 0 or prompt.shape[1] >= self.config.max_seq_len:
            return
        cache = KVCache(max_length=self.config.max_seq_len) if use_cache else None
        logits = self.forward(prompt, cache=cache, use_cache=use_cache)[:, -1, :]
        for step in range(min(max_new_tokens, self.config.max_seq_len - prompt.shape[1])):
            token = self._sample(logits, temperature, top_k)
            yield token
            if step + 1 >= min(max_new_tokens, self.config.max_seq_len - prompt.shape[1]):
                break
            if use_cache:
                logits = self.forward(token, cache=cache, use_cache=True)[:, -1, :]
            else:
                # Rebuild context only when caching is disabled.
                prefix = torch.cat((prompt, *generated), dim=1) if generated else prompt
                logits = self.forward(torch.cat((prefix, token), dim=1)[:, -self.config.max_seq_len:])[:, -1, :]
            if not use_cache:
                generated.append(token)

    @torch.inference_mode()
    def generate_fast(
        self, input_ids: torch.Tensor, max_new_tokens: int = 128,
        temperature: float = 0.8, top_k: int = 40, use_cache: bool = True,
    ) -> torch.Tensor:
        """Return prompt + completion, concatenating once rather than per token."""
        self._validate_generation(input_ids, max_new_tokens, temperature, top_k)
        prompt = input_ids[:, -self.config.max_seq_len:]
        pieces = [prompt]
        for token in self.stream_generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, use_cache=use_cache,
        ):
            pieces.append(token)
        return torch.cat(pieces, dim=1)
