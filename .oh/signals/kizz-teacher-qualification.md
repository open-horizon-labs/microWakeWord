---
id: kizz-teacher-qualification
type: metric
threshold: ">=90% recall and <=0.10 false accepts/hour, with 0/15 held-out household false wakes"
---

Measure every teacher on the same manifest-bound test set and the quarantined
household false-wake set. Select the operating point on the test negatives,
never on the held-out household captures. A teacher is eligible for
distillation only when the recall/FAPH gate and the held-out zero-accept gate
both pass. Fixed-window reports must be labeled preliminary; final
qualification requires bounded streaming sliding-window exposure.
