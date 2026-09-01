# Canonical-v3 teacher and phonetic-identifiability gate

Date: 2026-08-25

## Decision

Do not distill or flash another **Hi-Fi Kizz** model under the current evidence.
The repaired scratch sequence teacher fails deployment anchors, and a pinned
open-weight phoneme recognizer finds target-like speech inside exact pre-wake
trigger contexts. The device remains on the stock ESP-IDF Mycroft model.

This is an operational stop for this phrase in the captured household, not a
claim that the phrase is mathematically unlearnable or that all custom
wake-word training is infeasible. A longer, less confusable phrase can reuse
the repaired data, qualification, distillation, and firmware machinery after
it passes the same gates.

## Repaired C teacher

The final canonical-v3 C screen used:

- a full-context microfrontend teacher with two ordered states per phone
  (16 outputs including background and silence);
- a 24--50 frame completion duration, or 720 ms--1.50 s;
- 303 unique training parents expanded to 1,515 clean/overlay examples;
- a validation-only threshold; and
- 14 aligned test positives, eight independently localized household
  positives, five natural positives whose missing phrase spans fail closed,
  and all 62 quarantined false-wake observations.

The materialized positive mix was not healthy:

| Split | Unique parents | Dominant family |
| --- | ---: | ---: |
| train | 303 | Kokoro 285/303 (94.1%) |
| validation | 21 | Kokoro 19/21 (90.5%) |
| test | 14 | Kokoro 13/14 (92.9%) |

The common duration-aware qualifier selected `-33.6744` from 21 validation
positives and a deterministic 1,024-window reconstruction of the run's
validation archives. At that frozen threshold:

| Gate | Result |
| --- | ---: |
| validation positives | 19/21 (90.5%) |
| reconstructed validation negatives | 0/1,024 observed accepts |
| deduplicated aligned + localized held-out positives | 9/22 (40.9%) |
| unscorable natural-positive anchors | 5 (missing phrase spans) |
| quarantined false-wake observations | 2/62 accepted |

The accepted false wakes were `wake-cd98c172-2` and `wake-bcf497e9-1`. This
teacher is rejected before distillation.

## Pretrained phoneme teacher

The next test deliberately removed the scratch frontend as an explanation. It
used the open-weight
`facebook/wav2vec2-lv-60-espeak-cv-ft` phoneme recognizer at revision
`ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4`; the model weights hash is:

```text
3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e
```

The detector summed CTC paths for:

```text
h -> aɪ -> f -> aɪ -> k -> ɪ -> z
```

It also required that this path outrank explicit `kiss`, `kids`, and
`high-five` competitors. The operating point came only from aligned validation
positives and a deterministic 256-clip public-speech validation slice.

| Gate | Result |
| --- | ---: |
| aligned validation positives | 23/25 (92.0%) |
| public validation negatives | 0/256 over 2,439.14 s |
| aligned test positives | 13/14 (92.9%) |
| household positives | 3/13 (23.1%) |
| quarantined false wakes | 2/62 accepted |

The false-wake evaluator uses only the two seconds ending at the recorded
device wake timestamp. It still accepted two anchors:

| Anchor | Score | Best pre-wake window | Greedy phone decode |
| --- | ---: | --- | --- |
| `wake-cd98c172-2` | `-0.2996` | 1.529--2.495 s; wake at 3.000 s | `h aɪ f aɪ k ɪ z` |
| `wake-faa049dc-1` | `-0.9925` | 1.764--2.727 s; wake at 3.000 s | `h aɪ v aɪ k ɛ z` |

Both are far above the validation threshold `-2.0231`, and both best windows
end before the wake timestamp. The C teacher's accepted pair was instead
`wake-cd98c172-2` and `wake-bcf497e9-1`; only `cd98` replicates across the two
teachers. The independently pinned Whisper audit separately rendered three
anchors in words:

- `wake-cd98c172-2`: “Hi-fi kids.”
- `wake-5201cfa7-1`: “High five kiss.”
- `wake-bcf497e9-1`: “High five, kids.”

Several intended household positives received the same `kids`/`kiss`/`high
five` readings. At least one false wake therefore looks *more* canonical to a
large phoneme model than most real device positives. Blindly adding that clip
as a negative risks teaching channel, speaker, or recording shortcuts rather
than a general acoustic distinction.

## Deterministic guardrails now enforced

1. Feature provenance carries provider, source group, speaker/voice, ancestry,
   and parent identity.
2. Source balance is measured over unique parents before augmentation. No
   provider may exceed 50% of train parents or 60% of validation/test parents;
   train needs four families and validation/test each need three.
3. Every qualification row must resolve to an existing audio file whose actual
   SHA-256 matches the manifest. Validation and held-out groups must be
   disjoint by audio, source lineage, and, where declared, speaker, voice, and
   session identity.
4. The C qualifier constructs the declared topology, applies the declared
   duration bounds, chooses a threshold from validation only, deduplicates by
   audio identity, requires all 62 locked false-wake anchors, and rejects any
   natural anchor that cannot be scored from an explicit phrase span.
5. The pretrained phoneme gate pins model revision, dependency versions,
   tokenizer/config hashes, model-weight hash, manifests, phone sequences,
   window geometry, threshold selection, trigger metadata, and exact pre-wake
   context.
6. Distillation now requires two model-bound reports: the clip/anchor teacher
   prequalification and a separate untouched-continuous qualification. The
   latter must show at least 90% recall, zero accepted locked anchors, and a
   95% FAPH upper bound no greater than 0.1 over at least 100 hours. A report
   with the wrong model hash, scope, evidence declaration, or limits fails
   closed.

## Replacement-phrase pre-screen

A non-qualifying screen rendered 24 macOS `say` clips per phrase (12 English
voices at two rates), selected each phrase's threshold at 90% of those renders,
and replayed all 62 false wakes through the same phoneme teacher. Candidate
scores use the best direct CTC fit over windows up to the firmware's two-second
context; they do not reuse the stopped phrase's hand-written collision list.

| Candidate | False wakes accepted | Nearest decoded-phone edit distance |
| --- | ---: | ---: |
| Kizz Control | 0/62 | 7 |
| Kizz Listen | 0/62 | 5 |
| Hi-Fi Control | 2/62 | 6 |
| Hi-Fi Kizz | 16/62 | 0 |

This is a phrase screen, not model qualification. `Kizz Listen` also has the
obvious connected-speech collision “kids, listen,” so the current preliminary
recommendation is **Kizz Control**. It still needs multi-provider positives,
untouched connected speech, real device replay, a qualified teacher, a
qualified INT8 student, and exact-artifact hardware testing before replacing
Mycroft.

## Reproducible artifacts

- C report: `/private/tmp/kizz-training/canonical-v3/teacher-c-v8-double-blends-1000/qualification-v4-hardened.json`
- phoneme-teacher report: `/private/tmp/kizz-training/canonical-v3/phoneme-teacher-v2-trigger-context-qualification.json`
- replacement-phrase pre-screen: `/private/tmp/kizz-phrase-screen-v1/report-reproducible.json`

Their SHA-256 values are:

```text
C teacher       3834de2a869ca250c2ec5d970b7863528c4fb316cd3770164c2c197e20e515e8
phoneme teacher 4cf6f519a8afc2ba154a9944afd0bde0c538ac907d1d3d5b3e9518750f1b72aa
phrase screen   972d8dd1d7db65e8065b5741f4056b9a8df5bdba7f48785486a2d2f9128c6df6
```

The replacement screen is implemented by
`tools/screen_kizz_wake_phrase_candidates.py`. It requires repeated
`--candidate ID=TEXT` values, the pre-rendered positive-audio directory, the
locked false-wake manifest, and an output path. It pins and hashes the phoneme
model and every audio input, derives thresholds from candidate positives only,
and always records `qualified: false` because this screen cannot authorize
training or deployment.

The paths are local evidence locations. Each committed evaluator records the
input and model hashes needed to detect stale or substituted artifacts.
