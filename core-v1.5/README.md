# Artemide Core v1.5 — end-to-end training runner

v1.5 adds a CLI training loop on top of the v1.1 transformer and v1.4 checkpoint utilities.

## What's included

- Flat uint16 token-file reader with random contiguous next-token batches
- Gradient accumulation across micro-batches; one optimizer update and one gradient clip per update
- Linear warmup followed by cosine learning-rate decay
- Optional held-out validation loss and perplexity, with model train/eval mode restored
- JSONL metrics (loss, gradient norm, learning rate, tokens/sec, elapsed time)
- Periodic and final atomic checkpoints; optimizer, scaler, scheduler and sampler RNG resume state
- CPU and CUDA execution; fp32, bf16, fp16 and auto precision selection

## Data format

The runner expects already-tokenized **little-endian uint16** token IDs in flat `.bin` files. Training and validation must use the same tokenizer and vocabulary as the model. IDs must be smaller than `vocab_size`. The file must contain at least `sequence_length + 1` tokens.

No corpus or tokenizer is bundled. Only use text and tokenization assets you have the right to use. The v1.1 model's default vocabulary is 259; use a tokenizer that matches that vocabulary or change the model config and tokenizer together.

## Setup

Keep `core-v1.1`, `core-v1.4` and `core-v1.5` side by side. Install PyTorch and set the directories on `PYTHONPATH`.

Example config file (`configs/small.json`):

```json
{
  "vocab_size": 259,
  "d_model": 256,
  "n_layers": 4,
  "n_heads": 4,
  "max_seq_len": 512,
  "dropout": 0.0,
  "ff_mult": 4
}
```

Run:

```bash
PYTHONPATH=core-v1.1:core-v1.4:core-v1.5 python core-v1.5/artemide_training_runner.py \
  --config configs/small.json --train data/train.bin --validation data/val.bin \
  --steps 10000 --batch-size 8 --sequence-length 256 \
  --grad-accum-steps 4 --warmup-steps 500 --precision auto
```

To resume, pass `--resume runs/artemide/latest.pt`. The `--steps` value is the total target optimizer-update count, not additional updates. Checkpoints are written to `<output>/latest.pt`; metrics append to `<output>/metrics.jsonl`.

## Important limitations

This is a training pipeline, not evidence that Artemide has learned language. No dataset is included, no training was launched, and no cloud/GPU resources were provisioned. A random-initialized model trained on a small corpus will not be comparable to ChatGPT or Gemini. Start with a small, legally usable dataset and evaluate held-out loss before scaling.

## Tests

```bash
PYTHONPATH=core-v1.1:core-v1.4:core-v1.5 pytest core-v1.5/tests
```
