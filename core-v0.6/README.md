# Artemide Core v0.6.0

Independent random-initialization language-model research stack. This release replaces the v0.5 synthetic-token training loop with actual corpus-based next-token training.

## Quick start
1. Install: `pip install -e .`
2. Prepare licensed text: `python -m artemide.training.prepare --input ./my-corpus --output ./data/processed --license CC0`
3. Train: `python -m artemide.training.train --config configs/small.json --data ./data/processed --steps 1000`
4. Serve: set `ARTEMIDE_CHECKPOINT=runs/final.pt` and run `uvicorn api.main:app --host 0.0.0.0 --port 8000`.

Use only text you have rights to use. This byte tokenizer is a baseline, not a production tokenizer. Small random-init runs are pipeline smoke tests, not ChatGPT-level models.

## v0.6 improvements
- Corpus preparation with deterministic split, byte encoding, and SHA-256 manifest.
- Real next-token training batches; no synthetic random-token training.
- Optional validation loss, checkpoint resume, gradient clipping, telemetry.
- Checkpoint-backed generation API with health reporting.
