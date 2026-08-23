# Kizz model decisions

**Decision:** stop synthetic-only search; collect StackChan microphone audio.

Historical synthetic results used random clip holdouts, the quantized streaming
model, a five-frame moving window, and reset state between clips. Those clip
splits could reuse speaker identities across train and test; treat them as model
search diagnostics, not speaker-generalization evidence.

| Model and training change | Cutoff | Intended phrase acceptance | Incorrect or unseen acceptance | Result |
| --- | ---: | ---: | --- | --- |
| Combined v6: 27,000 positives across nine spellings; 34,650 hard negatives, including prefix/suffix conjunctions | 0.39 | 75.2%–95.4% | `Hiffy Kiss` 25.0%; `Hee Fye Kiss` 24.2%; `high fee kids` 21.1%; unseen `High Phi Kizz`, `High Pie Kizz`, `Hi Pie Kizz`, and `Hippie Kizz` 31.3%–44.3% | Rejected: confusable acceptance remained high and unseen pronunciation recall was weak. |
| Combined v6 at the zero-observed ambient-FAPH cutoff | 0.78 | 44.0%–68.0% | — | Rejected: intended acceptance fell too far. |
| Fee-family v1: `Hi-Fi Kizz`, `High-Fi Kizz`, `Hee Fee Kizz`, `High Fee Kizz`, and `Hiffy Kizz` | 0.08 | 94.5%–98.6% | bare `Kizz` 34.0%; `kiss` 40.4%; `Hiffy Kiss` 66.9% | Rejected: the model did not separate the full phrase from its suffix and near words. |
| Fee-family v2: 3× hard-negative sampling, 2× error penalty, and hard negatives included in checkpoint selection | 0.29 | 83.8%–95.3% | bare `Kizz` 11.2%; `kiss` 11.9%; `Hiffy Kiss` 38.4%; `*-kids` conjunctions 31.6%–35.1% | Rejected: stronger negative pressure reduced some collisions but left conjunction failures. |
| Fee-family v2 at a higher cutoff | 0.47 | 73.8%–88.7% | conjunction false accepts 11.2%–29.2% | Rejected: the recall loss did not buy enough collision reduction. |

Synthetic speech renders `Kizz` too inconsistently to settle the class boundary.
Collect device-microphone captures before another model search.

## Initial quality-masked device candidate — 2026-08-22

**Decision:** reject this comparison as qualification evidence. It scored the
device training corpus and was useful only for choosing a physical experiment.

Eight reviewed human positives span 560–880 ms (840 ms median). Comparing those
spans with all 67,150 generated clips rejected 148: 143 had an overly long
voiced span and 64 could lose their beginning in the two-second augmented window;
some clips failed both checks. The mask SHA-256 is
`528a76f3b8aa45a14088131659ab3b8d65e75585ff2c3de2c128a24b619ddc5a`.

The recipe now spells accepted kids-like realizations as `Hi-Fi Kids` variants
and keeps the distinct `high five kids` reading as a hard negative. At the
physical operating point (`0.70`, one-frame window), direct inference over the
device training corpus produced:

| Candidate | Positive accepts | Hard-negative accepts | Ambient accepts |
| --- | ---: | ---: | ---: |
| Previously flashed broad model | 11/17 | 1/8 | 0/1 |
| Label-corrected control | 14/17 | 6/8 | 0/1 |
| Quality-masked candidate `91e7052d…` | 17/17 | 0/8 | 0/1 |

Sixteen of seventeen positives were explicit training inputs. The remaining
recording was also Muness but had been assigned a different speaker ID and test
split. That invalid split is now removed. The selected candidate's weakest
training-corpus positive scored `.949`; its
strongest direct-corpus hard negative scored `.639`. That direct result did not
transfer to room-scale playback: only 1/8 reviewed positive recordings crossed
`.70` when replayed through a speaker into Kizz at the 4x microphone setting.
None of eight replayed hard negatives woke it. Each detected turn re-armed.

The synthetic corpus uses hundreds of LibriTTS-R speaker embeddings, but they
are not age-labeled and its former random clip split reused voices across train
and test. The generator now reserves disjoint speaker IDs before synthesis. A
100-clip-per-phrase diagnostic at `.70` also showed weak aggregate positive
recall despite only 4/2,233 hard-negative accepts. Judge a new candidate on the
speaker-independent synthetic test, then on registered physical test speakers.

## Human-span mask revision — 2026-08-22

**Decision:** retrain with a stricter human-anchored positive mask, then compare
the candidate with the initial model on physical Kizz.

The initial mask allowed positive voiced spans from 322 to 1,540 ms and rejected
only 148/67,150 generated clips. The eight reviewed human phrases occupy a much
tighter range: 560–880 ms, with a 840 ms median. Synthetic positives run from
440 to 2,380 ms, with a 900 ms median and a 1,260 ms 95th percentile.

The revised default takes the human 5th and 95th percentiles and adds a 25%
margin. For this corpus, that admits positive spans from 483 to 1,100 ms. It
rejects 6,229/67,150 generated clips: 6,189 have a positive span that is too
long, 36 are too short, and 64 exceed the source length safe for augmentation;
some clips have more than one reason. It retains 32,074/38,300 positives and
28,847/28,850 hard negatives. The mask SHA-256 is
`446105170131ba7da791ef4fe8325623d61cc514e70d1529fd1491937ce954d2`.

Raw level is deliberately not a rejection boundary. The reviewed microphone
phrases are quieter than generated speech, but gain augmentation and the device
channel are expected to vary level. Matching raw RMS would discard useful voice
diversity without showing that the resulting features better match Kizz.
