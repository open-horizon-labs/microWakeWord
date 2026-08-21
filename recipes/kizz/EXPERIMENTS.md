# HiPhi Kizz model experiments

Synthetic experiments are rejection evidence, not hardware qualification.
All acceptance rates below use the exact held-out split, reset recurrent state
between clips, and score the quantized streaming model with a five-frame moving
window.

## Combined full-phrase model (v6)

Training used 27,000 positives across nine spellings and 34,650 hard negatives,
including prefix/suffix conjunction mining. At cutoff 0.39, intended phrase
acceptance ranged from 75.2% to 95.4%, while `Hiffy Kiss` accepted 25.0%,
`Hee Fye Kiss` 24.2%, and `high fee kids` 21.1%. At the zero-observed ambient
FAPH cutoff 0.78, intended acceptance fell to 44.0%–68.0%.

Unseen `High Phi Kizz`, `High Pie Kizz`, `Hi Pie Kizz`, and `Hippie Kizz`
probes accepted only 31.3%–44.3% at cutoff 0.39. The model was rejected.

## Fee-family acoustic cluster

The cluster trained on `Hi-Fi`, `High-Fi`, `Hee Fee`, `High Fee`, and `Hiffy
Kizz`. The first run reached 94.5%–98.6% intended acceptance at cutoff 0.08,
but accepted bare `Kizz` 34.0%, `kiss` 40.4%, and `Hiffy Kiss` 66.9%.

A second run used 3× hard-negative sampling pressure, 2× penalty, and included
the hard-negative archive in the checkpoint-selection false-accept metric. At
cutoff 0.29, intended acceptance was 83.8%–95.3%, while bare `Kizz` accepted
11.2%, `kiss` 11.9%, `Hiffy Kiss` 38.4%, and `*-kids` conjunctions
31.6%–35.1%. Raising the cutoff to 0.47 reduced intended acceptance to
73.8%–88.7% and still left conjunction false accepts at 11.2%–29.2%.

This cluster was rejected and the remaining clusters were not trained. The
result is consistent with noisy synthetic pronunciation of the invented word
`Kizz`; actual device-microphone wake captures are required before another
model search is justified.
