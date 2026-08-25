# Kizz aligned teacher → student experiment

Status: **measured offline candidate; rejected for live deployment** on
2026-08-25. This record began before firmware integration. The student was
later integrated and tested on StackChan; frequent live false wakes made its
precision unacceptable, so deployment reverted to the ESP-IDF wake-word model.

This is the reproducible record for the corrected Recipe C teacher, aligned
distillation, and fixed-context/stateful-INT8 evaluation. It supersedes the
deployment conclusions in [`TEACHER_DISTILLATION_V1.md`](TEACHER_DISTILLATION_V1.md)
for this run, while retaining that document as the earlier failed-recipe
record.

## Aim and decision boundary

The aim is to transfer a stronger full-context Kizz detector into one compact
streaming model that can eventually run on the ESP32-S3. The teacher is
offline-only. The student is the only artifact eligible for firmware.

The teacher was accepted for this experiment at an explicitly relaxed gate:

- minimum recall: 88%;
- maximum observed FAPH: 0.1;
- zero accepts on 15 held-out household false wakes.

This is an experiment gate, not a production-quality claim. The negative test
exposure is only 1,456 seconds, so one or two errors produce a very unstable
FAPH estimate.

## Data contract

The source manifest is:

```text
/private/tmp/kizz-training/clean-slate-v2/training-manifest/manifest.json
```

It contains 4,706 examples: 1,848 positives and 2,858 negatives. The test
split contains 252 positives and 560 negatives. Fifteen held-out household
false wakes are kept in a separate manifest and never enter teacher training,
the distillation cache, or threshold selection.

Old Piper data is excluded. The training mix contains AssemblyAI, Deepgram,
ElevenLabs, Kokoro, device-rendered positives, noisy overlays, public speech,
music, and background noise. False-wake observations remain quarantined
evidence and do not modify `device-corpus.json`.

## Corrected feature preparation

The first C run was invalid because short positive clips were padded after
feature extraction. That made the zero-feature tail a class-correlated cue and
did not match raw streaming.

`tools/build_kizz_teacher_features_v1.py` now:

1. resamples to 16 kHz and folds stereo to mono;
2. centers or crops PCM to 41,920 samples;
3. runs the repository microfrontend on that PCM context;
4. requires exactly 260 frontend frames × 40 bins;
5. writes targets for an 87-frame teacher timeline.

Padding happens in PCM, not by appending zeros to an extracted feature matrix.
The corrected outputs are under:

```text
/private/tmp/kizz-training/clean-slate-v2/c-features-aligned-v1/
/private/tmp/kizz-training/clean-slate-v2/c-heldout-aligned-v1/
```

## Teacher recipe

`microwakeword.kizz_teacher.build_teacher` uses:

- input `[260, 40]` microfrontend features;
- LayerNorm and a 128-channel local convolution;
- seven residual dilated context convolutions;
- full-context, non-causal operation;
- 23 ordered-state logits: background, silence, and 21 Kizz phone states;
- 87 output frames at the 30 ms state cadence.

The 87-frame output is material. The old teacher emitted 66 states from only
the first 198 feature frames, while the causal student’s 66 outputs correspond
to a later latency-shifted portion of the 260-frame input. The old index-to-
index distillation target was not temporally aligned.

The bounded teacher run used hidden size 128, seven context blocks, batch size
32, sequence-only loss (`frame_weight=0`, `sequence_weight=1`), and seed 24107.
The best checkpoint was saved after 500 completed steps of the configured
3,000-step run.

```text
/private/tmp/kizz-training/clean-slate-v2/C-teacher-aligned-v1/best.weights.h5
SHA-256: 3fee91bc5b62b434b5c3798dfd2fc46aa9e9e35ccf79ff29b939238fd6acdcf6
```

## Teacher qualification

`tools/qualify_kizz_teacher_npy.py` and
`tools/qualify_kizz_teacher.fast_sequence_scores` score the exact aligned
slice used for distillation: teacher frames `[21:87]`, or 66 frames.

Accepted report:

```text
/private/tmp/kizz-training/clean-slate-v2/C-teacher-aligned-v1/qualification-accepted-88-recall.json
```

| Measure | Result |
| --- | ---: |
| Positive test clips | 252 |
| Teacher recall | **88.10% (222/252)** |
| Negative clips | 560 |
| Negative exposure | 1,456 s |
| Observed false accepts | **0/560** |
| Observed FAPH | **0.00** |
| Held-out false wakes | **0/15** |
| Threshold | 21.8075 |

At the stricter 90% recall floor, the best available point was 94.05% recall
with four false accepts, or 9.89 FAPH on this short exposure. The teacher was
accepted only for the controlled student experiment at 88.10% recall and zero
observed false accepts.

## Distillation recipe

`tools/cache_kizz_teacher_logits.py` freezes teacher logits and hard targets.
It uses `--alignment-offset 21` and `--student-output-frames 66`, computes the
teacher’s 87-frame output, and stores only `[21:87]`:

```text
features:       [32000, 260, 40]
hard targets:   [32000, 66]
teacher logits: [32000, 66, 23]
```

Cache:

```text
/private/tmp/kizz-training/clean-slate-v2/C-distill-aligned-v1/cache*
cache SHA-256: 3e9551caa6d5dd2bb0dd1bf7598ed488a66bb7ae7d4dc2fa2a6e766237cfba95
```

The student is the existing ordered-state causal architecture: a 48-filter
stride-3 first convolution, four 96-filter causal MixConv blocks with kernels
`[3]`, `[5]`, `[7]`, and `[9]`, and 23 state logits per 30 ms step. It has
50,023 parameters and an approximately 670 ms acoustic receptive field.

The primary run used:

- 3,000 steps, batch size 64, learning rate 0.001;
- temperature 2;
- teacher KL weight 1, hard frame weight 0;
- sequence-label weight 1;
- negative state 1 (silence/rejection);
- seed 24109.

The trainer is `tools/distill_kizz_student.py`. Best loss was 0.2210:

```text
/private/tmp/kizz-training/clean-slate-v2/C-distill-aligned-v1/student/best.weights.h5
```

Teacher-only KD, offset 20, and direct teacher sequence-margin matching were
also tested. None exceeded the primary INT8 result below.

## Evaluation process

### Float fixed-context evaluation

The Keras student emits 66 logits from a `[260, 40]` feature window. Scores are
computed with the same ordered-state sequence scorer used by the teacher.
Thresholds are selected from negative test scores; held-out false-wake scores
are checked only after threshold selection.

### Stateful INT8 evaluation

`tools/convert_distilled_student.py` creates the firmware-shaped artifact.
Evaluation feeds the quantized TFLite model 3 frontend frames at a time,
resets state only at source boundaries, and dequantizes all 23 logits. The
artifact emits 86 outputs for a 260-frame source; the first 20 are warm-up, so
the valid comparison tail is `[20:86]`.

The current INT8 artifact is:

```text
/private/tmp/kizz-training/clean-slate-v2/C-distill-aligned-v1/firmware/student_stream_state_internal_quant.tflite
SHA-256: 9ba9280c497abf422b59626b3145406ff53a4e6641f3e375d5fcb1fa7bd3dd3d
Size: 84,056 bytes
```

Its best observed zero-FP point is:

| Measure | Result |
| --- | ---: |
| Recall | **72.62% (183/252)** |
| False accepts | **0/560** |
| Observed FAPH | **0.00** |
| Held-out false wakes | **0/15** |
| Threshold | 16.1674 |

At the first point reaching 90% recall, it achieved 90.48% recall with two
false accepts, or 4.95 FAPH on the 1,456-second exposure. That FAPH is a
small-sample observation, not an hourly-rate estimate.

The teacher-caught/student-missed diagnostic found 49 clips: 42 noisy overlays,
6 direct Deepgram clips, and 1 ElevenLabs clip. Their median duration was
0.673 s versus 0.871 s for teacher catches. The 252 test positives remain
frozen and must not be used for hard-positive retraining.

## Firmware integration and live result

At the time of this experiment, firmware expected a scalar wake probability:

```text
input:  [1, 3, 40] int8
output: [1, 1] uint8
```

The ordered-state artifact produces:

```text
input:  [1, 3, 40] int8
output: [1, 1, 23] uint8
```

The ordered-state decoder was later added and the artifact was flashed for an
explicit hardware test. That test produced frequent false positives in the
household environment. The artifact is therefore preserved as research
evidence, not a deployable replacement for `hiphi_kizz.tflite`.

## Guardrails

- Do not call 72.62% a production qualification.
- Do not train on the 252 test positives or 15 held-out false wakes.
- Do not lower the teacher gate silently; 88% is explicit and experimental.
- Do not infer scalar-probability compatibility from matching input shapes.
- The firmware decoder and exact StackChan test now exist, but the live result
  failed; do not deploy this artifact.
- Preserve this INT8 artifact as the current student control.

## Reproduction map

- `microwakeword/kizz_teacher.py` — teacher architecture and batches;
- `microwakeword/ordered_state_model.py` — causal student;
- `microwakeword/distillation.py` — KD and sequence losses;
- `tools/build_kizz_teacher_features_v1.py` — PCM-context features;
- `tools/train_kizz_teacher.py` — teacher training;
- `tools/qualify_kizz_teacher_npy.py` — teacher gate;
- `tools/cache_kizz_teacher_logits.py` — aligned frozen targets;
- `tools/distill_kizz_student.py` — student training;
- `tools/convert_distilled_student.py` — INT8 conversion;
- `tools/score_ordered_state_streams.py` — stateful stream scoring.
