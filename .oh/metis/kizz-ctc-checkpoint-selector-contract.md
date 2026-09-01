---
id: kizz-ctc-checkpoint-selector-contract
title: "Kizz checkpoint selection must normalize logits and score deployment suffixes"
---

The Kizz distillation trainer previously passed raw logits to a CTC scorer that
expects log probabilities and selected the maximum over every internal endpoint.
Firmware instead log-softmax normalizes logits and evaluates suffix windows
ending at the newest frame. That mismatch selected misleading checkpoints even
though the exact-audio qualification evaluator was correct.

Checkpoint selection now uses the same accelerated suffix forward-sum scorer as
deployment, with log-softmax normalization and only the latest endpoint. It also
prefers fewer false accepts at the 90% recall floor when no checkpoint qualifies.
Legacy checkpoints must be hash-verified and rescored before comparison.

Cached float16 frontend features differ from exact deployment features by mean
absolute error 0.0003148, so feature caching did not explain the discrepancy.
The selector contract, not merely tensor shape, is part of model provenance.
