---
id: kizz-clean-slate-v2-teacher-comparison
title: "C materially outperformed D under identical fixed-window gates"
---

The clean-slate corpus was sufficient to separate the teacher approaches. On
the same 252 positive test clips, 560 negative test clips, and 15 held-out
household false wakes, C reached 90.08% recall with zero false accepts/hour and
zero held-out accepts. D reached the same recall floor but produced 72.68 false
accepts/hour and accepted 13/15 held-out false wakes.

The earlier intuition that a larger pretrained waveform teacher would
automatically be stronger was wrong for this corpus and operating point. C is
the only teacher candidate eligible for the next distillation experiment.
The C result remains a fixed-window result until bounded raw-audio
sliding-window qualification is complete.
