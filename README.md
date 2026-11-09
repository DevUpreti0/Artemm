# Artemide

Independent, randomly initialized language-model research project.

## Try the current build
- **Playground:** https://artemide-playground.vercel.app
- The Playground UI is deployed and connected to this repository. It reports model availability honestly; real generation stays disabled until a trained Artemide checkpoint is served by the inference API.

## Core releases
- v0.4.0: reconstructed Transformer baseline and byte tokenizer.
- v0.5.0: training/operations scaffold, telemetry, evaluation, and API/dashboard scaffold.
- v0.6.0: real corpus preparation, token-window next-token training, validation loss, resumable checkpoints, and checkpoint-backed inference API.

## v0.6 source
The current v0.6 implementation is in `core-v0.6/`. A complete installable package archive is available as [artemide-core-v0.6.zip](https://github.com/DevUpreti0/Artemm/tree/main/releases) once the archive is published there. The current workspace package passed 9 tests and an end-to-end CPU smoke test (corpus preparation → training → checkpoint load → generation).

Use only text you have the rights to use. A tiny random-init model is an engineering smoke test, not a capable general-purpose assistant. No third-party model API or pretrained weights are used as the foundation.
