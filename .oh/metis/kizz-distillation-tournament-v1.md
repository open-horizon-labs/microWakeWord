---
id: kizz-distillation-tournament-v1
title: "Decision-boundary transfer helped Kizz Control, but no float student qualified"
---

Temperature posterior KD, transcript-conditioned CTC occupation KD, teacher
representation KD, their combination, and a bounded temporal-residual student
were evaluated against the same validation and locked evidence. The exact-delay
occupation model removed nearly all measured teacher/student lag, but did not
preserve recall. A +30 ms delay made it worse.

The strongest candidate was the 96,212-parameter temporal-residual student with
teacher sequence ranking and tail-separation losses. It reached 78.05% recall at
zero validation false accepts, 14/26 frozen aligned positives, 15/24 target-
channel positives, and 0/62 household false wakes. The gates require 90%
validation recall, 23/26 aligned, and 20/24 target-channel positives.

This evidence favors explicit decision-boundary transfer over frame-posterior or
representation imitation alone, but it does not justify quantization or flash.
Locked evidence remains evaluation-only and must not become a fine-tuning set.

The causal-window extension transferred teacher decisions at exact student
endpoints and tested a 94,836-parameter, 1.93-second dilated causal-memory
student. The old architecture did not improve. The long-memory student reached
82.93% zero-FP validation recall and 16/26 aligned positives, but only 12/24
target-channel positives, with 0/62 false wakes. This isolates two learnings:
the prior architecture did lose useful temporal context, and longer context
alone does not transfer target-channel invariance.

Clip-label ranking losses must not be applied to randomly sampled causal
prefixes. A positive clip's early prefix is not necessarily positive. The
failed continuation changed FP@90 from 1 to 72 within 100 steps; this invalid
combination is now rejected by the trainer.
