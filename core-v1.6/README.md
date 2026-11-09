# Artemide Core v1.6 — tokenizer and corpus preparation

v1.6 adds a transparent baseline tokenizer and a deterministic, document-level corpus preparation tool. Its 259-token vocabulary matches the default v1.1 `ModelConfig`.

## Token IDs

| ID | Meaning |
|---:|---|
| 0 | PAD |
| 1 | BOS |
| 2 | EOS |
| 3–258 | UTF-8 byte value + 3 |

This is a byte tokenizer, not BPE or a production tokenizer. It is reversible for valid UTF-8 text and handles Unicode, but sequences are longer than subword tokenization.

## Prepare a corpus

Supported inputs:
- A directory containing `.txt` and/or `.jsonl` files (searched recursively)
- A single `.txt` or `.jsonl` file
- JSONL records may be JSON strings or objects with a string `text` field

Example:

```bash
PYTHONPATH=core-v1.6 python core-v1.6/artemide_prepare_corpus.py \
  --input ./my-licensed-text --output ./data/processed \
  --validation-fraction 0.05 --seed 42 --license "CC-BY-4.0"
```

Outputs:
- `train.bin`
- `validation.bin` (when validation fraction is greater than zero)
- `manifest.json` with tokenizer format, declared license label, document counts, token counts, and SHA-256 hashes

The split is performed by document before tokenization, avoiding train/validation overlap at document level. At least two documents are required for a non-empty validation split. Set validation fraction to zero to prepare a single-document training corpus.

## Train with v1.5

Keep `core-v1.1`, `core-v1.4`, `core-v1.5`, and `core-v1.6` available. The model config must retain `vocab_size: 259`.

```bash
PYTHONPATH=core-v1.1:core-v1.4:core-v1.5:core-v1.6 \
  python core-v1.5/artemide_training_runner.py \
  --config configs/small.json --train data/processed/train.bin \
  --validation data/processed/validation.bin --steps 10000
```

## Data rights and limitations

The `--license` value is an operator-supplied label, not an automated license verification. Use only text you have permission to use for model training, and retain provenance. No training corpus is included or downloaded. This release does not start training or provision cloud compute.
