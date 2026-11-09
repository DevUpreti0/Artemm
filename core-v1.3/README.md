# Artemide Core v1.3 — streaming generation

v1.3 adds incremental token delivery and a lower-overhead generation method while preserving the v1.1 model architecture and checkpoint compatibility.

## Layout and installation

Keep `core-v1.1` and `core-v1.3` side by side. Install PyTorch, then run from the repository root:

```bash
PYTHONPATH=core-v1.1:core-v1.3 python -c "from artemide_streaming import StreamingArtemideLM"
```

## Stream tokens

```python
import torch
from artemide.model.config import ModelConfig
from artemide_streaming import StreamingArtemideLM

model = StreamingArtemideLM(ModelConfig()).eval()
prompt = torch.tensor([[1, 42, 17]], dtype=torch.long)
for next_token in model.stream_generate(prompt, max_new_tokens=40, temperature=0):
    # Decode this [batch, 1] token and send it to the client immediately.
    print(next_token, flush=True)
```

`stream_generate` yields one token tensor per step, so a server can forward tokens without waiting for the entire completion. `generate_fast` returns the prompt and completion together and performs a single final concatenation. It supports greedy decoding (`temperature=0`), temperature sampling, top-k filtering, KV cache on/off, batches, and the configured context limit.

## Tests

Run from the repository root:

```bash
PYTHONPATH=core-v1.1:core-v1.3 pytest core-v1.1/tests core-v1.3/tests
```

The model is still randomly initialized. Streaming improves delivery behavior, not learned intelligence. Benchmark on the target hardware before claiming a speedup.
