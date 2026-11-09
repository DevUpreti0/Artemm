# Artemide v1.0 inference-cache foundation

This folder is an additive source overlay for the Artemide Core repository.
It introduces a tested, model-agnostic per-layer key/value cache primitive.

## KV cache contract
- Tensor layout: [batch, heads, sequence, head_dim].
- KVCache.update(layer, key, value) appends new positions and returns the accumulated tensors.
- Optional max_length retains only the most recent positions.
- Invalid appends are rejected before cache state is changed.
- clear() resets state between requests.

## Important integration status
This is a cache primitive, not yet an end-to-end generation speedup. The current tracked repository does not contain the complete transformer implementation (artemide/model/transformer.py) needed to wire cache reads/writes into attention and generation. Once that model source is committed, the next integration step is to add past_key_values to each attention layer, apply rotary positions using the absolute token offset, and compare cached vs uncached logits for numerical equivalence.

No external model provider or pretrained weights are used. Do not advertise a measured latency improvement until an end-to-end benchmark confirms one.
