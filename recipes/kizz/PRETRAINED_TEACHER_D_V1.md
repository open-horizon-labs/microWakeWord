# Kizz pretrained waveform teacher D v1

Date: 2026-08-24

## Aim

Test whether a pretrained speech representation can provide a materially
different offline teacher for Kizz than the failed micro-speech/state recipe.
The intended deployment remains one compact ESP32 student; this teacher is
Mac/offline only.

## Implementation

- Backbone: `microsoft/wavlm-base-plus`.
- Input: raw mono waveform resampled to 16 kHz, fixed 2-second context.
- Head: temporal convolution over WavLM frames with differentiable
  multiple-instance pooling into one stream-level Kizz event score.
- Positive construction: canonical Kizz clips mixed into connected household
  speech backgrounds so zero-padding and clip duration are not positive
  shortcuts.
- Negatives: canonical generated hard-negative families, connected household
  speech, and 50 reviewed device false-wake anchors.
- Held-out device evidence: 12 false-wake anchors, excluded from training.
- Hardware: Mac16,8, 25.8 GB unified memory, MPS execution.

The implementation is in `microwakeword/kizz_pretrained_teacher.py` and the
optional tools are:

- `tools/write_kizz_pretrained_manifest.py`
- `tools/train_kizz_pretrained_teacher.py`
- `tools/qualify_kizz_pretrained_teacher.py`

Install the optional environment from `requirements-kizz-teacher.txt`.

## Qualification gate

The same hard gate used for the existing teacher path applies:

- positive recall at least 90%;
- no more than 0.1 false accepts/hour on connected validation negatives; and
- zero accepts on the 12 held-out device false wakes.

### Frozen-backbone run

Artifact: `/private/tmp/kizz-training/teacher-d/v1-frozen-1000/`

- 1,000 steps, batch size 8, WavLM backbone frozen.
- Connected-negative exposure: 5,865.04 seconds.
- Positive validation items: 2,408.
- Negative validation items: 5,076.
- No threshold satisfied the gate.
- At the 90% recall floor: 319.18 false accepts/hour.
- Held-out false-wake score maximum: 1.33; no operating point existed to
  evaluate accepts because the recall/FAPH gate already failed.

### Last-two-layer adaptation run

Artifact: `/private/tmp/kizz-training/teacher-d/v2-unfreeze2-500/`

- 500 steps, batch size 8, final two WavLM layers unfrozen.
- Same validation exposure and gate.
- No threshold satisfied the gate.
- At the 90% recall floor: 356.62 false accepts/hour.
- Held-out false-wake score maximum: -4.73; again there was no qualified
  operating point.

## Decision

D is feasible on the M4 Pro and is genuinely different from the old state
teacher, but this first WavLM recipe is not a strong teacher. It is rejected
before distillation. No student was trained from these artifacts and no
firmware was changed or flashed.

The result does not prove that all pretrained encoders fail. It does prove
that changing the representation alone does not overcome the current
positive/negative distribution and event-label recipe. The next D experiment
would need a measured change such as real device-channel positive mixtures,
stronger continuous negative sampling, or a teacher objective with explicit
event localization—not merely more training steps.
