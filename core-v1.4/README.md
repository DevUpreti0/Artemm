# Artemide Core v1.4 — training foundations

v1.4 adds reusable next-token training and atomic checkpoint/resume utilities for the v1.1 transformer.

## Layout

Keep `core-v1.1` and `core-v1.4` side by side. Install PyTorch and set both folders on `PYTHONPATH`.

## One training update

```python
import torch
from artemide.model.config import ModelConfig
from artemide.model.transformer import ArtemideLM
from artemide_training import train_batch, save_checkpoint, load_checkpoint

model = ArtemideLM(ModelConfig()).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# x and y are integer token batches of equal [batch, sequence] shape.
metrics = train_batch(model, optimizer, x.cuda(), y.cuda(), precision="auto")
save_checkpoint("runs/latest.pt", model, optimizer, step=1, metadata={"dataset": "licensed-corpus-v1"})
```

`precision="auto"` selects CUDA bfloat16 when supported, otherwise CUDA float16; CPU uses full precision. For fp16, supply a configured `torch.amp.GradScaler("cuda")` as `scaler`. The utility unscales before clipping. For bfloat16 and fp32 no scaler is needed.

## Resume

```python
state = load_checkpoint("runs/latest.pt", model, optimizer, map_location="cuda")
next_step = state["step"] + 1
```

Checkpoints include model and optimizer state, optional scaler state, metadata, and Python/PyTorch/CUDA RNG states. Writes use a temporary file followed by atomic replacement. Load only checkpoints you trust: PyTorch checkpoint files can contain pickled objects.

## Tests

```bash
PYTHONPATH=core-v1.1:core-v1.4 pytest core-v1.4/tests
```

This is a training foundation, not a trained model or a corpus. Use text you have rights to train on, keep a held-out validation split, and monitor loss/perplexity before scaling training. No paid compute is provisioned by this code.
