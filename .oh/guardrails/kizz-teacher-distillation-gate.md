---
id: kizz-teacher-distillation-gate
severity: hard
statement: "Do not distill or promote a Kizz teacher unless it passes the identical recall/FAPH test and accepts zero held-out household false wakes."
outcome: kizz-voice-reliability
---

D demonstrates why this is hard: it reached the recall floor but still
accepted 13/15 held-out household false wakes and produced 72.68 false
accepts/hour. Training loss, pretrained backbone size, and recall alone are
not qualification evidence. The gate must be evaluated from the exact
source-manifest-bound artifact and must distinguish preliminary fixed-window
results from full sliding-window ambient exposure.
