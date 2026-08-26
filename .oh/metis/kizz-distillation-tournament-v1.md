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
