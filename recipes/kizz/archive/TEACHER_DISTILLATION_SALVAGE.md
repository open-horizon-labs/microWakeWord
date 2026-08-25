# Kizz teacher phase: fact-check and salvage

Date: 2026-08-24

## Accuracy report

| Claim | Status | Primary evidence | Correction / qualification |
|---|---|---|---|
| Teacher v2 had public-validation AUC 0.8934 | Verified | `/private/tmp/kizz-training/teacher-kizz-v2/validation-score.json`; independently recomputed from its positive/negative score arrays | This is ranking quality, not a production detector qualification. |
| The 62 reviewed false wakes were represented by 1,184 training and 180 held-out windows | Verified | `/private/tmp/kizz-training/false-wake-feature-cache-v1/manifest.json` | Held-out status is by observation, not by independent speaker or environment. |
| Teacher v3 collapsed | Verified with edit | `/private/tmp/kizz-training/teacher-kizz-v3/score-connected.json` | Its connected-slice pairwise AUC was 0.5289047; “collapsed” is shorthand for that measured failure, not a diagnosis. |
| Teacher v4 fixed the background/silence loss mismatch | Verified | `microwakeword/kizz_teacher.py` in commit `7724269` | The loss now uses joint `background ∪ silence` rejection evidence. |
| Teacher v4 passed or nearly passed qualification | Incorrect | `/private/tmp/kizz-training/teacher-kizz-v4/qualification-connected-only.json` | It was rejected: no threshold met ≥90% recall and ≤0.1 FAPH on 11,593.4 seconds of connected negatives. |
| The shared cause is definitely the data/window/objective recipe | Requires expert review | v3/v4 controlled failures plus the earlier direct-student failures | This is the leading local hypothesis, not a proven single cause. |
| Corrected teacher v4 was distilled or flashed | Unsupported / false if stated | No v4 qualification report passed; firmware status remained unchanged | No corrected v4 distillation or firmware flash should be claimed. |
| The qualification guard is enforced before distillation | Verified | `tools/distill_kizz_student.py`; an unqualified report raised `ValueError` before model construction/training | The guard must remain mandatory. |
| The relevant implementation tests pass | Verified | `tests.test_kizz_teacher`, `tests.test_distillation`, `tests.test_teacher_qualification`: 11 passing | This validates plumbing, not detector quality. |

## Salvage report

### Reason and original aim

This phase is being salvaged because the approach reversed several times and
the finish line moved from “train a teacher” to “prove the teacher is strong
before distillation.” The original aim was to use a larger offline teacher to
produce a compact Kizz student with materially better precision while retaining
recall.

### Learnings

1. AUC and rejection of known device anchors are insufficient. The teacher must
   pass the same recall/FAPH gate that the student would face.
2. Frame-state cross-entropy is not the deployed sequence objective. Enabling
   sequence loss alone did not rescue the recipe: v3 failed badly.
3. Forcing all negative frames into `silence` is also wrong. The decoder treats
   background and silence as a rejection set; v4 now models that union.
4. The corrected rejection loss still did not qualify v4. Therefore the next
   investigation must audit positive/window geometry and negative coverage, not
   merely add more teacher capacity or another distillation run.
5. Training-loss checkpoint selection is not detector selection. A future run
   must retain candidate checkpoints and qualify them on held-out streams before
   choosing a teacher.

### Frame shift

**Old frame:** a bigger full-context model will provide a useful teacher if it
learns phone states and rejects the observed false wakes.

**New frame:** the model recipe does not yet establish a reliable mapping from
continuous audio to a calibrated whole-phrase decision. The teacher may be
larger, but it inherits the frontend, windowing, target geometry, negative
sampling, and evaluation mismatch that likely harmed the direct students.

### New guardrails

- Never distill from a teacher without a passing qualification report.
- Qualification must use held-out positive occurrences, untouched negative
  exposure, per-item/stream false accepts per hour, and held-out false-wake
  anchors.
- Hash-bind qualification reports to the exact teacher weights and cache.
- Do not treat AUC, training loss, or one-window scores as detector readiness.
- Do not let quarantined false wakes enter positive/negative training data
  without an explicit, reproducible split and provenance record.
- Do not modify or flash firmware while the teacher or student gate is red.

### Missing context

- Whether the aligned 23-state positive targets reflect real phone boundaries
  rather than a convenient duration convention.
- Whether positive and negative windows have the same padding, phrase position,
  frontend normalization, and continuous-stream geometry as deployment.
- Whether the connected hard-negative slice is representative of the household
  speech that generated the live false wakes.
- Whether the teacher should be trained from weak phrase-level labels, CTC-like
  alignments, or a stronger phonetic teacher instead of the current synthetic
  state targets.

### Ownership / coordination breakdown

The implementation phase allowed “new teacher exists” to stand in for “teacher
is strong enough to teach.” The qualification decision should have been a
mandatory boundary before any student run. The new hash-bound gate fixes the
software handoff, but model-quality ownership still requires an explicit
accept/reject decision against the measured operating-point criteria.

### Reusable fragments

- Hash-bound teacher-logit cache and quarantine-aware false-wake split.
- Batched exact Viterbi scorer used by the teacher qualification gate.
- Joint background/silence rejection loss.

### Pretrained waveform teacher D (2026-08-24)

The M4 Pro path was implemented and tested with `microsoft/wavlm-base-plus`
and a raw-waveform temporal event head. A frozen-backbone run reached no
qualified operating point; its false-accept rate at the 90% recall floor was
319.18/hour on 5,865 seconds of connected negatives. Unfreezing the final two
WavLM layers did not help (356.62/hour). This is a measured rejection of the
first D recipe, not evidence that the Mac or pretrained-teacher strategy is
infeasible. The detailed artifact and decision are in
`PRETRAINED_TEACHER_D_V1.md`.
- Weighted negative-source sampling.
- Qualification-before-distillation enforcement.

### Corrected D follow-up (2026-08-24)

The original D result did contain a concrete implementation bug: frozen WavLM
parameters were placed under `model.train()`, so dropout remained active in
the frozen representation. The run also chose checkpoints by training-batch
loss. That defect was corrected, along with explicit padding masks, weighted
public-speech negatives, a temporal localization/rejection auxiliary loss,
and validation-based checkpoint selection.

The correction improved but did not rescue D:

| run | change | untouched-test FAPH at 90% recall | decision |
|---|---|---:|---|
| original D | frozen WavLM, old recipe | 63.36 | reject |
| D0 | corrected frozen control | 53.77 | reject |
| D1 | D0 + final two WavLM blocks unfrozen at `2e-5` | 31.75 | reject |

The same 252-positive / 560-negative test set and 26,646.6-second sliding
exposure were used. D1's best validation point was 14.44 FAPH, but it did not
generalize to the untouched test. This is evidence for a recipe/data/window
boundary problem after the implementation bug is fixed, not evidence that all
pretrained encoders are unusable.

### Corrected-D guardrails

- A frozen pretrained encoder must be in evaluation mode during head training;
  `requires_grad=False` alone is insufficient.
- Checkpoints must be selected by the deployment qualification metric on a
  validation split, never by a single training-batch loss.
- Any claimed padding-mask fix must be exercised in both training and scoring.
- A materially lower FAPH on validation is not evidence of improvement until
  the untouched sliding-window test agrees.
- Do not spend additional compute on D distillation while the same hard gate is
  red; change data/window/target geometry first.

### Fresh-start recommendation

Start with a data/target audit and a teacher-only experiment. Verify positive
phrase geometry and continuous negative-window construction first. Train a
teacher whose objective directly matches the stream decision, retain several
checkpoints, and qualify every checkpoint on untouched public negatives plus
held-out device anchors. If no checkpoint passes, stop teacher distillation
entirely and revisit the wake phrase, frontend, or label/data design before
training another student.
