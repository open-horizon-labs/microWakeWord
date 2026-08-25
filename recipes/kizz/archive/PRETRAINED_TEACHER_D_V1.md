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

### Corrected clean-slate-v2 runs

The first clean-slate-v2 D run exposed a real recipe defect: freezing WavLM
parameters while calling `model.train()` left dropout active in the supposedly
fixed representation. It also selected `best.pt` by minimum training-batch
loss rather than by validation detector quality. The corrected trainer now:

- forces a frozen backbone into `eval()` mode;
- passes explicit padding masks to WavLM;
- samples balanced positive/negative batches with 3x public-speech negative
  pressure;
- adds a temporal localization/rejection auxiliary loss; and
- selects checkpoints on validation recall/FAPH rather than training loss.

The corrected frozen control (D0) used 1,000 steps, batch size 8, temporal
weight 0.25, and the same clean-slate-v2 manifest. Its best validation point
was 90.20% recall at 35.88 FAPH. On the untouched sliding-window test it
reached 90% recall at 53.77 FAPH.

D1 additionally unfroze the final two WavLM blocks, used learning rate
`2e-5`, and kept the corrected recipe. Its best validation point was 90.20%
recall at 14.44 FAPH. On the untouched test it reached 90% recall at 31.75
FAPH. Neither run produced a qualifying operating point; held-out false-wake
acceptance is therefore not reported as zero or passable.

Artifacts and reports were kept under:

- `/private/tmp/kizz-training/clean-slate-v2/D-teacher-corrected/`
- `/private/tmp/kizz-training/clean-slate-v2/D-teacher-corrected-unfreeze2/`

These results revise the diagnosis: the original implementation was partly
buggy, but correcting it and adapting WavLM still leaves a large ordinary
speech overlap. D remains rejected for distillation.

## Decision

D is feasible on the M4 Pro and is genuinely different from the old state
teacher, but this first WavLM recipe is not a strong teacher. It is rejected
before distillation. No student was trained from these artifacts and no
firmware was changed or flashed.

The result does not prove that all pretrained encoders fail. It does prove
that fixing the first implementation defect, adding explicit temporal
rejection pressure, and lightly adapting WavLM do not overcome the current
positive/window/negative boundary. A future D attempt needs a genuinely
different data or target geometry—not merely more training steps, another
checkpoint, or more unfreezing.

## Review and dissent

Phase: execute → review/dissent. Commit under review: `60ea6ae`.

### Review: Adjust

The implementation is bounded to the offline training fork, keeps the ESP32
student and firmware untouched, has focused tests, and records exact
qualification artifacts. The success criterion is not met: neither D variant
qualified, so no distillation or flash is authorized.

### Dissent: Reconsider the current D recipe

The strongest contrary evidence is that both a frozen WavLM representation and
partial WavLM adaptation produce hundreds of false accepts per hour at the
recall floor. More model capacity and more training steps are therefore not
the next justified action. The surviving hypothesis is narrower: a pretrained
encoder may still help, but only after the waveform/channel distribution and
event supervision are changed and measured independently.

Human verification required: none for the offline measurements; a maintainer
must approve any future teacher qualification before student distillation or
firmware testing.
