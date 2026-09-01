# Kizz teacher distillation v1

Status: **implemented, measured, and rejected for firmware** on 2026-08-24.

The aim was to train a genuinely new offline teacher against the Kizz ordered
phone-state topology, freeze its outputs, and distill those sequence decisions
into the existing tiny streaming student. The teacher is not a firmware model:
it is a seven-block, full-window, non-causal dilated-convolution network over
the existing 40-bin micro-speech frontend. The student remains the only model
that could run on the ESP32-S3.

## Data and quarantine boundary

The 62 human-confirmed false-wake recordings remain under the quarantined
evidence directory and were not added to `device-corpus.json`. A deterministic
cache split created 1,184 training windows from 50 observations and 180 held-out
windows from 12 different observations:

- source manifest SHA-256:
  `5558d3e7ed2bd4446c3777da0cfadd8b43085f66cbc267a01af0b81f3d56b0aa`
- feature-cache manifest: `/private/tmp/kizz-training/false-wake-feature-cache-v1/manifest.json`
- held-out observations never entered teacher training or the distillation cache

The training cache is hashable and records all source paths, positive-array
hashes, teacher-weight hash, and teacher-logit hash. It is not a corpus
promotion mechanism.

## Teacher result

Teacher artifacts:

`/private/tmp/kizz-training/teacher-kizz-v2/`

The teacher was trained for 3,000 balanced steps with public connected speech,
speech/background sources, mined negatives, and the quarantined false-wake
training windows. On the public validation score distribution its pairwise AUC
was 0.8934. On the 62 reviewed device anchors, the maximum teacher score was
`-331.90`; the positive validation maximum was `-225.46`. The teacher therefore
learned a useful device-anchor separation, but public negatives still overlap
the positive distribution and the teacher is not itself a deployable detector.

## Student result

Frozen teacher logits were cached at:

`/private/tmp/kizz-training/teacher-kizz-v2/distill-cache.json`

The int8 streaming conversion succeeded for all candidates. The compact graph
has 50,023 parameters, an 84,056-byte TFLite artifact, a `(1, 3, 40)` int8
input, and a `(1, 1, 23)` uint8 state-logit output. The candidate results below
were measured with exact streaming state, quantized dequantization, and the
existing ordered-state decoder contract on 500 fresh positive occurrences:

The v4 artifact SHA-256 is
`db77e7be001401b3fce3a5dd50e45055b7ff9a7b6a8cc837f75205603a0d17ec`.

| Candidate | Objective | Recall | Decision |
| --- | --- | ---: | --- |
| v1 | teacher v1, framewise scratch | 20.2% | reject |
| v3 | teacher v2, framewise scratch | 14.6% | reject |
| v4 | teacher v2, framewise + throttled sequence loss | 40.8% | reject |
| v5 | teacher v2, initialized ordered-state student, negative hard state = silence | 6.4% | reject |

The required qualification gate is at least 90% fresh-positive recall and at
most 0.1 false accepts/hour. No student reached the recall floor, so negative
hours were not used to select an operating point and no model was copied into
the firmware repository.

The important implementation correction discovered during this pass is that
background and silence are both rejection states in the decoder. Treating every
negative frame as background in the student hard loss can erase a valid silence
representation. The distillation tool now makes the negative hard state
explicit (`--negative-state`, default silence), but that correction did not
rescue the candidate.

## Firmware readiness

**Not ready for flash.** The quantized conversion and desktop streaming scorer
are ready, but the model qualification gate is not met. The firmware repository
was inspected and its pre-existing dirty changes were preserved; no firmware
files were modified, no model was embedded, and no flash was attempted.

Even after a student qualifies, firmware integration requires an explicit
23-state TFLite invocation and the ordered-state decoder in the Kizz component;
the current production path is not allowed to infer this from the existing
scalar wake-model endpoint. The exact StackChan serial device and an exact
artifact hardware resource/boot test are also required before calling the
firmware ready.

## Decision

The teacher implementation, quarantine-aware feature cache, frozen-logit
distillation path, int8 conversion, and exact qualification checks are retained
as reusable infrastructure. The current teacher/student recipe is rejected for
deployment. The next experiment must change a measured mechanism—most likely
student training supervision/architecture or the teacher-to-student temporal
target—not lower the firmware gate or flash this artifact.

## Mandatory pre-distillation teacher gate

The pipeline now refuses to distill unless a separate qualification report is
present, marked `qualified`, and hash-matches the teacher weights in the frozen
logit cache. The qualification gate requires, on the supplied untouched
negative exposure:

- at least 90% positive recall;
- at most 0.1 false accepts/hour using per-item maxima; and
- zero accepts on the held-out false-wake feature cache by default.

This is implemented by
[`qualify_kizz_teacher.py`](../../../tools/qualify_kizz_teacher.py) and enforced by
[`distill_kizz_student.py`](../../../tools/distill_kizz_student.py). Running the
current teacher against the 11,593-second connected-speech validation slice
produced `qualified: false` because no threshold met both recall and the false
accept ceiling. Attempting to distill with that report now fails before model
construction or training.

## Recipe correction experiment

The first correction pass changed three shared failure modes rather than
distilling another weak teacher:

1. sequence loss is enabled by default for new teacher runs, on a throttled
   schedule to keep the full-context objective tractable;
2. negative sampling accepts explicit source weights, allowing quarantined
   device hard negatives to receive deployment-relevant exposure; and
3. negative frame loss scores `background ∪ silence` as one rejection set
   instead of forcing every negative into one arbitrary class.

Teacher v3 (sequence objective plus forced silence targets) collapsed on the
connected validation slice. Teacher v4 fixed the rejection-set loss but still
produced no operating point satisfying 90% recall and 0.1 false accepts/hour.
The hard gate rejected it before distillation. This confirms that the shared
problem is not only the missing sequence loss or a single background/silence
label; the positive/window/negative recipe remains the next investigation.

## Review gate

**Aim:** improve Kizz precision without giving up the fresh-speaker recall
floor, using an offline teacher and a single firmware student.

**Status:** salvage/adjust, not complete for firmware.

- Necessary and aligned: yes; the teacher and distillation path directly test
  the requested architecture.
- Mechanism clear: partly; the full-context teacher learns a better separation,
  but the causal student does not retain enough of that separation under the
  current frame/state objective.
- Risks retired: teacher quality, quarantine leakage, frozen-logit provenance,
  int8 conversion, exact streaming scoring, and the no-flash gate were tested.
- Completion gap: no student met 90% recall and 0.1 false accepts/hour, and no
  exact-device resource or boot test is therefore authorized.

The frame has not collapsed into “the teacher idea is wrong”; it has narrowed
to a student-target mismatch. The next run should compare sequence posterior
targets or teacher hidden representations against the current frame-state
imitation, with the same held-out anchor split.

## Dissent gate

The strongest case for shipping v4 is that it is a genuine teacher-distilled,
single-model, quantized artifact and improves substantially over the first
distillation runs. The contrary evidence is decisive: 40.8% recall is less
than half the qualification floor, and the teacher's public-negative overlap
means it cannot be treated as an oracle. A six-month failure would most likely
come from distilling a full-context state alignment that a causal student
cannot reproduce, then mistaking a low training loss and successful conversion
for a usable detector.

The remaining safe decision is **RECONSIDER the student target, preserve the
teacher/cache infrastructure, and do not flash**. Human verification still
needed: independent review of the 62 weak-label false-wake classifications and
an exact StackChan resource/boot test after a future student qualifies.

The accuracy audit and restart kit are preserved in
[`TEACHER_DISTILLATION_SALVAGE.md`](TEACHER_DISTILLATION_SALVAGE.md).
