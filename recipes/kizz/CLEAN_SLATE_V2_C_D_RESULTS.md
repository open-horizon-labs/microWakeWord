# Clean-slate v2 teacher comparison

Date: 2026-08-24

This is the pre-distillation comparison record. The later aligned C teacher and
student are documented in
[`TEACHER_DISTILLATION_ALIGNED_V1.md`](TEACHER_DISTILLATION_ALIGNED_V1.md).
That student was tested on StackChan and rejected because live false positives
made its precision unacceptable.

## Aim

Find a teacher that can separate the Kizz wake phrase from ordinary household
speech, music, and noise well enough to justify student distillation.

## Shared corpus

Both teachers consumed the same source-manifest-bound clean-slate-v2 training
manifest:

The source-balance gate for this run is recorded in
`recipes/kizz/data-balance-contract-clean-slate-v2.yaml`.

- 4,706 examples: 1,848 positives and 2,858 negatives.
- Train: 1,290 positives / 1,738 negatives.
- Test: 252 positives / 560 negatives.
- Held out: 15 household false wakes, never used for training.
- Positive sources: AssemblyAI, Deepgram, ElevenLabs, Kokoro, device-rendered
  anchors, and deterministic speech/music/noise overlays.
- Negative sources: LibriSpeech speech plus MUSAN music/noise/speech.
- Old Piper data is excluded.

The raw corpus contains 215.268 eligible negative hours. The data is suitable
for a controlled candidate run, not yet a production qualification claim:
natural household positives and long-form domestic/TV audio remain gaps.

## Recipe C — microfrontend ordered-state teacher

Recipe:

- Existing 16 kHz microfrontend, fixed `[260, 40]` feature windows.
- Full-context dilated convolution teacher, hidden size 128, seven context
  blocks, 23 ordered-state outputs.
- 3,000 steps, batch size 32, seed 24103.
- Sequence objective only (`frame_weight=0`, `sequence_weight=1`).

Same-metric fixed-window result:

| metric | result |
|---|---:|
| positive test examples | 252 |
| recall at selected operating point | 90.08% (227/252) |
| negative exposure | 26,646.6 s |
| false accepts/hour | 0.00 |
| held-out household false wakes | 0/15 |

C passes this fixed-window gate. The result is not yet an ambient-hours claim;
full sliding-window scoring is still required before distillation or firmware.

## Recipe D — pretrained waveform teacher

Recipe:

- `microsoft/wavlm-base-plus` frozen backbone with a temporal convolution head.
- 1,000 steps, batch size 8, learning rate `2e-4`, MPS device, seed 24103.
- Positive contexts receive randomized background mixing during training.

Same-metric fixed-window result:

| metric | result |
|---|---:|
| positive test examples | 252 |
| recall floor | 90.08% (227/252) |
| negative exposure | 26,646.6 s |
| false accepts/hour at recall floor | 72.68 |
| held-out household false wakes at recall-floor threshold | 13/15 |
| qualification | **failed** |

The earlier full sliding-window D report also failed: no operating point met
90% recall and `<=0.10` false accepts/hour. D is not eligible for distillation.

## Decision at this phase

At this phase, C was the only surviving teacher candidate. The later aligned
teacher run documented the corrected context geometry and applied the following
gates before distillation:

1. score C over the same raw test audio with bounded streaming sliding windows;
2. require the hard false-activation gate and zero held-out household accepts;
3. only then distill C into a causal firmware-sized student.

The aligned student passed its short offline screen but failed live StackChan
precision. It is preserved as research evidence, not a deployment candidate.

The C and D checkpoints, manifests, and reports remain local under
`/private/tmp/kizz-training/clean-slate-v2/`; this repository records the
recipe and evidence, not the large binary artifacts.
