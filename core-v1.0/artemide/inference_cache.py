"""Experimental per-layer KV cache primitives for incremental transformer decoding.

This module is deliberately model-agnostic. A model must explicitly pass and
consume cached key/value tensors for this cache to reduce end-to-end latency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    """Append-only, bounded key/value cache indexed by transformer layer.

    Tensors use the conventional [batch, heads, sequence, head_dim] layout.
    Sequence is dimension -2. Appends are validated before state is mutated.
    """

    def __init__(self, max_length: Optional[int] = None):
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive or None")
        self.max_length = max_length
        self._layers: Dict[int, LayerKV] = {}

    def __len__(self) -> int:
        return len(self._layers)

    def clear(self) -> None:
        self._layers.clear()

    def get(self, layer: int) -> Optional[LayerKV]:
        return self._layers.get(layer)

    def sequence_length(self, layer: Optional[int] = None) -> int:
        if layer is not None:
            item = self._layers.get(layer)
            return 0 if item is None else int(item.key.shape[-2])
        if not self._layers:
            return 0
        lengths = {int(item.key.shape[-2]) for item in self._layers.values()}
        if len(lengths) != 1:
            raise RuntimeError("KV cache layers have inconsistent sequence lengths")
        return lengths.pop()

    def update(
        self, layer: int, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(layer, int) or layer < 0:
            raise ValueError("layer must be a non-negative integer")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("key and value must have [batch, heads, seq, dim] shape")
        if key.shape != value.shape:
            raise ValueError("key and value shapes must match")
        if key.shape[-2] < 1:
            raise ValueError("cannot append an empty sequence")
        previous = self._layers.get(layer)
        if previous is None:
            combined_key, combined_value = key, value
        else:
            if previous.key.shape[:2] != key.shape[:2] or previous.key.shape[-1] != key.shape[-1]:
                raise ValueError("new key/value tensors are incompatible with cached tensors")
            combined_key = torch.cat((previous.key, key), dim=-2)
            combined_value = torch.cat((previous.value, value), dim=-2)
        if self.max_length is not None:
            combined_key = combined_key[..., -self.max_length:, :]
            combined_value = combined_value[..., -self.max_length:, :]
        self._layers[layer] = LayerKV(combined_key, combined_value)
        return combined_key, combined_value
