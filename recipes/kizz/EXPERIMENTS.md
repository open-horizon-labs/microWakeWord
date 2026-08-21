# Kizz synthetic model results

**Decision:** stop synthetic-only search; collect StackChan microphone audio.

All results use held-out synthetic audio, the quantized streaming model, a
five-frame moving window, and reset state between clips.

| Model and training change | Cutoff | Intended phrase acceptance | Incorrect or unseen acceptance | Result |
| --- | ---: | ---: | --- | --- |
| Combined v6: 27,000 positives across nine spellings; 34,650 hard negatives, including prefix/suffix conjunctions | 0.39 | 75.2%–95.4% | `Hiffy Kiss` 25.0%; `Hee Fye Kiss` 24.2%; `high fee kids` 21.1%; unseen `High Phi Kizz`, `High Pie Kizz`, `Hi Pie Kizz`, and `Hippie Kizz` 31.3%–44.3% | Rejected: confusable acceptance remained high and unseen pronunciation recall was weak. |
| Combined v6 at the zero-observed ambient-FAPH cutoff | 0.78 | 44.0%–68.0% | — | Rejected: intended acceptance fell too far. |
| Fee-family v1: `Hi-Fi Kizz`, `High-Fi Kizz`, `Hee Fee Kizz`, `High Fee Kizz`, and `Hiffy Kizz` | 0.08 | 94.5%–98.6% | bare `Kizz` 34.0%; `kiss` 40.4%; `Hiffy Kiss` 66.9% | Rejected: the model did not separate the full phrase from its suffix and near words. |
| Fee-family v2: 3× hard-negative sampling, 2× error penalty, and hard negatives included in checkpoint selection | 0.29 | 83.8%–95.3% | bare `Kizz` 11.2%; `kiss` 11.9%; `Hiffy Kiss` 38.4%; `*-kids` conjunctions 31.6%–35.1% | Rejected: stronger negative pressure reduced some collisions but left conjunction failures. |
| Fee-family v2 at a higher cutoff | 0.47 | 73.8%–88.7% | conjunction false accepts 11.2%–29.2% | Rejected: the recall loss did not buy enough collision reduction. |

Synthetic speech renders `Kizz` too inconsistently to settle the class boundary.
Collect device-microphone captures before another model search.
