"""Bounded per-layer key/value cache for autoregressive decoding."""
from dataclasses import dataclass
from typing import Dict, Optional
import torch


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


class KVCache:
    def __init__(self, max_length: Optional[int] = None):
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive or None")
        self.max_length = max_length
        self._layers: Dict[int, LayerKV] = {}

    def __len__(self):
        return len(self._layers)

    def clear(self):
        self._layers.clear()

    def get(self, layer: int):
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

    def update(self, layer: int, key: torch.Tensor, value: torch.Tensor):
        if not isinstance(layer, int) or layer < 0:
            raise ValueError("layer must be non-negative")
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise ValueError("key/value must have matching [batch, heads, seq, dim] shapes")
        if key.shape[-2] < 1:
            raise ValueError("cannot append empty sequence")
        old = self._layers.get(layer)
        if old is None:
            k, v = key, value
        else:
            if (old.key.shape[:2] != key.shape[:2] or old.key.shape[-1] != key.shape[-1]
                    or old.key.device != key.device or old.key.dtype != key.dtype):
                raise ValueError("new tensors are incompatible with cached tensors")
            k, v = torch.cat((old.key, key), -2), torch.cat((old.value, value), -2)
        if self.max_length is not None:
            k, v = k[..., -self.max_length:, :], v[..., -self.max_length:, :]
        self._layers[layer] = LayerKV(k, v)
        return k, v
