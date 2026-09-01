---
id: kizz-aligned-distillation-transfer-diagnostic
title: "Aligned INT8 student is the current control, but transfer loses short noisy positives"
---

The corrected full-context C teacher reached 88.10% recall with zero observed
false accepts on the 1,456-second negative test exposure and accepted 0/15
held-out household false wakes. The aligned INT8 student reached 72.62% recall
at its own zero-observed-FP point and accepted 0/15 held-out false wakes.

The teacher-caught/student-missed set contains 49 of the 252 positive test
clips: 42 noisy overlays, 6 direct Deepgram clips, and 1 ElevenLabs clip. The
misses have a median duration of 0.673 seconds versus 0.871 seconds for teacher
catches. This is evidence of a concentrated transfer/temporal robustness gap,
not yet evidence that the student architecture is too small.

The 72.62% INT8 artifact is the current student control. The 252 test positives
and 15 held-out false wakes remain frozen. Future work should mine analogous
teacher-high/student-low examples from train/validation, preserve the teacher's
sequence ranking, and test temporal-offset augmentation before changing model
capacity.
