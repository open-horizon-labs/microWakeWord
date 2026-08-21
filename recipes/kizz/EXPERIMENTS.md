# Kizz synthetic model results

**Decision:** stop the synthetic-only model search and collect real StackChan
microphone audio. Every model below was rejected.

All acceptance rates use the held-out split and the quantized streaming
model with a five-frame moving window. The evaluator resets streaming state
between clips. These results measure synthetic audio only.

| Model and training change | Cutoff | Intended phrase acceptance | Incorrect or unseen acceptance | Result |
| --- | ---: | ---: | --- | --- |
| Combined v6: 27,000 positives across nine spellings; 34,650 hard negatives, including prefix/suffix conjunctions | 0.39 | 75.2%–95.4% | `Hiffy Kiss` 25.0%; `Hee Fye Kiss` 24.2%; `high fee kids` 21.1%; unseen `High Phi Kizz`, `High Pie Kizz`, `Hi Pie Kizz`, and `Hippie Kizz` 31.3%–44.3% | Rejected: confusable acceptance remained high and unseen pronunciation recall was weak. |
| Combined v6 at the zero-observed ambient-FAPH cutoff | 0.78 | 44.0%–68.0% | — | Rejected: intended acceptance fell too far. |
| Fee-family v1: `Hi-Fi Kizz`, `High-Fi Kizz`, `Hee Fee Kizz`, `High Fee Kizz`, and `Hiffy Kizz` | 0.08 | 94.5%–98.6% | bare `Kizz` 34.0%; `kiss` 40.4%; `Hiffy Kiss` 66.9% | Rejected: the model did not separate the full phrase from its suffix and near words. |
| Fee-family v2: 3× hard-negative sampling, 2× error penalty, and hard negatives included in checkpoint selection | 0.29 | 83.8%–95.3% | bare `Kizz` 11.2%; `kiss` 11.9%; `Hiffy Kiss` 38.4%; `*-kids` conjunctions 31.6%–35.1% | Rejected: stronger negative pressure reduced some collisions but left conjunction failures. |
| Fee-family v2 at a higher cutoff | 0.47 | 73.8%–88.7% | conjunction false accepts 11.2%–29.2% | Rejected: the recall loss did not buy enough collision reduction. |

We did not train the remaining acoustic clusters. The working hypothesis is that
synthetic speech renders the invented word `Kizz` too inconsistently to settle
the class boundary. The next experiment must use actual device-microphone wake
captures before another model search is justified.
