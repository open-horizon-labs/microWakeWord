---
id: kizz-student-promotion-gate
severity: hard
statement: "Do not quantize, continuously evaluate, package, or flash a Kizz student until its exact float checkpoint passes every frozen clip gate."
outcome: kizz-voice-reliability
---

The float gate is validation recall at least 90% with at most 0.1 observed FAPH,
at least 23/26 frozen aligned positives, at least 20/24 frozen target-channel
positives, and 0/62 locked household false wakes. Threshold selection is
validation-only and uses deployment-equivalent CTC scoring.

Quantization can alter ranking but cannot legitimize a float checkpoint that
already failed recall. The 100-hour continuous corpus is consulted only after
the float clip gate, and firmware packaging/flash only after the INT8 artifact
passes its own bound qualification. Diagnostic progress is not promotion.
