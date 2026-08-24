# Kizz ordered-state model outline

Status: architecture scaffold implemented under `muness/roon-knob#240`; the
first scratch candidate was trained and rejected under `muness/roon-knob#241`.
No ordered-state artifact qualified or was flashed.

## Why this is a new model

V19 and the v32 canonical reboot both use a whole-window binary target. V19
kept useful live recall but woke repeatedly on household speech. V32 sharply
improved rejection but collapsed held-out recall. That evidence says another
round of binary fine-tuning is not the controlled next step: the detector needs
an inference invariant that ordinary speech cannot satisfy merely by producing
one high whole-window score.

The new invariant is the complete canonical phone sequence:

```text
/h aɪ f aɪ k ɪ z/
```

This is one neural acoustic model followed by a tiny deterministic dynamic
program. It is not a concurrently resident verifier model. The acoustic model
emits local state logits; the dynamic program permits only self-loops and a
one-state advance through the wake phrase.

## Fixed architecture contract

- Preserve the product frontend: 16 kHz mono PCM and 40 `micro_speech`
  features.
- Use a causal MixedNet acoustic encoder initialized from scratch.
- Emit 23 logits every 30 ms: background, silence, and beginning/middle/end
  states for each of the seven canonical phones.
- Require all 21 phrase states in order. A path may remain in its current state
  or advance one state; it cannot skip, reorder, or start at an interior state.
- Compare each phrase-state emission with the combined silence/background
  rejection evidence.
- Use the same logit interpretation, transition probabilities, and optional
  evidence floor in the differentiable Viterbi head and deployed decoder.
- Train the sequence score directly, with optional aligned frame-state
  cross-entropy when reviewed phone spans exist.
- Keep false-wake observations quarantined unless a human explicitly promotes
  them. The untouched captures remain deployment anchors.

The state tensor has this stable layout:

| Index | Meaning |
| ---: | --- |
| 0 | background / other speech |
| 1 | silence |
| 2–4 | `/h/` beginning, middle, end |
| 5–7 | first `/aɪ/` beginning, middle, end |
| 8–10 | `/f/` beginning, middle, end |
| 11–13 | second `/aɪ/` beginning, middle, end |
| 14–16 | `/k/` beginning, middle, end |
| 17–19 | `/ɪ/` beginning, middle, end |
| 20–22 | `/z/` beginning, middle, end |

No text alias becomes a positive implicitly. A frame-aligned positive must
declare exactly the sequence above. A reviewed phrase span without phone spans
may contribute to the sequence objective but not to frame-state supervision.

The normal training entrypoint recognizes the `ordered_state` model. Its
training wrapper turns the state-logit sequence into one differentiable Viterbi
completion logit, so the existing optimizer, source weighting,
checkpointing, and binary endpoint metrics remain usable without changing the
binary MixedNet path:

```yaml
training_loss:
  name: ordered_state_sequence
  state_evidence_floor: null
  self_loop_probability: 0.6
  next_state_probability: 0.4
  frame_weight: 0.25
  frame_supervision:
    directory: work/kizz/aligned-frame-training
    batch_size: 64
    seed: 231
```

```sh
python -m microwakeword.model_train_eval \
  --training_config work/kizz/ordered-state-training.yaml \
  --train 1 \
  --tflite_roc_split none \
  ordered_state
```

The aligned directory contains fixed `features.npy` and `targets.npy` arrays,
plus optional per-example `weights.npy`. Target `-1` masks an unaligned frame;
other values use the stable 0–22 state layout. Each endpoint update is followed
by an auxiliary frame-state update when this source is configured. The endpoint
uses logits-native binary cross-entropy so very negative scratch scores retain
non-zero gradients.

The standard scalar TFLite ROC evaluator is intentionally disabled for this
model. Quantized uint8 output must be dequantized with the tensor's recorded
scale and zero point, then decoded as logits. The ordered-state TFLite adapter
and conversion test enforce that boundary. Training writes
`ordered-state-decoder.json` beside the weights to bind logit mode, transition
probabilities, evidence floor, state count, and frame cadence to evaluation.

## Resource envelope proven by the scaffold

The default causal encoder has 48,119 learned-model parameters and a 670 ms
maximum local receptive field. The converted streaming graph reports 50,023
total parameters after adding its ring buffers. Its signature consumes one
`(1, 3, 40)` feature stride and emits one `(1, 1, 23)` state vector. Full
integer TensorFlow Lite conversion succeeds with int8 input and uint8 output;
the artifact is 80,168 bytes and the converter reports about 0.093 million
operations, or 0.047 million MACs, per invocation. Persistent ring buffers
occupy an estimated 1,904 bytes and a float32-score/int32-coordinate decoder
adds 168 bytes. A 2,000-invocation Apple Silicon run measured 0.00313 ms median
and 0.00333 ms p95 per TFLite invocation. These are desktop conversion and
latency measurements, not ESP32-S3 latency or tensor-arena measurements.
Firmware integration remains gated on a qualifying model and an exact-device
resource report. The checked-in resource reporter reproduces the measurement.

The 21-state path imposes a 630 ms minimum decoded phrase extent at the 30 ms
output cadence. Event coordinates use an inclusive start frame and an
end-exclusive timestamp: a path from frame `N` through `N+20` therefore spans
630 ms. Fast natural speakers are an explicit recall risk to measure, not an
assumption to hide.

## Training and evaluation contract

The first controlled run compares:

1. the frozen v19 control at its deployed operating point;
2. a canonical binary scratch control using the same split evidence; and
3. the ordered-state scratch candidate.

Training uses source/session-disjoint positives, connected household speech,
phonetic collisions such as “high five kids,” and diverse high-scoring windows
mined from the broad negative archive. A deterministic random-negative reserve
stays in every pass so mining does not collapse the negative distribution.

Cutoffs and transition settings are selected on validation only. Evaluation
preserves decoder state inside each continuous source, resets only at source
boundaries or a declared post-trigger re-arm, and reports event coordinates,
positive recall, false rejections, false accepts per negative hour, and a
one-sided 95% Poisson upper bound. At least 100 validation-negative hours and
100 untouched test-negative hours are required. The qualification target is
at least 90% recall on fresh real-speaker positives and no more than 0.1 false
accepts/hour; the stretch target is one false accept per 24 hours.

If no candidate clears that bar, no model is flashed and the failed run remains
falsifying evidence. The next decision must then change a measured mechanism
(state duration, capacity, frontend, data coverage, or wake phrase), rather
than silently reopening the test set.

## First-candidate result and stop decision

The first candidate used the fixed 23-state topology, the 48,119-parameter
causal encoder, the differentiable sequence loss, and `0.25` auxiliary aligned
frame loss. Its negative inventory was frozen before training: 124.43 train
hours, 112.17 validation hours, and 665.66 untouched test hours. The frozen
inventory SHA-256 is
`bb6850d3007f4a1e1931a308a7115a05381768b7bea58d5dbff6ecc2c46bff11`.
The separate quarantined deployment-anchor snapshot froze 62 observations on
2026-08-24 (manifest SHA-256
`5558d3e7ed2bd4446c3777da0cfadd8b43085f66cbc267a01af0b81f3d56b0aa`);
it is distinct from v32's 47-observation 2026-08-23 snapshot and was not used
for model selection.

The candidate never approached the joint gate. Its best false-accept checkpoint
was step 10,000: 50.0% endpoint recall and 28 ambient false accepts over the
112.17-hour validation exposure, or 0.24963 false accepts/hour. The required
operating point was at least 90% fresh-speaker/device recall and at most 0.1
false accepts/hour. Step 15,000 regressed to 0.76672 false accepts/hour, and
step 20,000 regressed to 55.51578 false accepts/hour while recall remained only
54.8%. Training was deliberately terminated during the step-25,000 validation
pass; the untouched test partition was never opened and no firmware was flashed.

These are bounded in-training endpoint measurements, not a completed quantized
qualification report. They are nevertheless sufficient to stop this run: the
best fixed endpoint missed the recall floor by 40 points while exceeding the
false-accept ceiling, and additional training made false accepts unstable without
recovering recall. A quantized decoder-margin sweep was not used to claim a
stronger result. The machine-readable checkpoint ledger is
[`ordered-state-v1-results.json`](ordered-state-v1-results.json).

The failed result narrows the next question. A sequence constraint is cheap
enough for the device, but this particular combination of local state encoder,
fixed 630 ms minimum path, and equal-third B/M/E frame supervision did not
produce a usable separation boundary. Preserve the implementation as a reusable
research scaffold; do not continue this exact recipe, tune on the untouched
test set, or treat the auxiliary frame targets as measured phone boundaries.

Qualification reports must pass the artifact's `--decoder-contract` and
`--require-declared-exposure` to the streaming evaluator. The contract supplies
the 30 ms frame step, validates every score timestamp against that cadence, and
prevents an omitted final stride from biasing false-accepts/hour. A conflicting
decoder argument is rejected. Cooldown belongs either to the decoder or the
evaluator, never both.

## Salvage learnings (2026-08-24)

This attempt is salvaged before another firmware pass. The following findings
change the next experiment:

- The converted candidate was not teacher-distilled. It was a scratch-trained
  ordered-state student. Its failure is evidence against this exact scratch
  recipe, not against teacher distillation or every sequential model.
- Direct quantized streaming evaluation on the 500-item canonical validation
  set found 357/500 recall at completion margin `0` and 383/500 (76.6%) at the
  most permissive tested margin `-2`. The required floor is 90%, so threshold
  tuning cannot rescue this artifact.
- The CHiME6 validation slice alone produced one false activation in 34,813.68
  seconds (0.1034 false accepts/hour), already above the 0.1 ceiling. The
  larger FSD50K slice was not needed to reject the candidate after recall had
  failed; the untouched test set remained closed.
- The evaluator initially rejected regenerated manifests because the frozen
  inventory's record hash was being reused as the scorer's live directory hash.
  These are distinct provenance contracts. Manifests now preserve
  `frozen_path_sha256` and independently compute `expected_path_sha256`.
- The 630 ms minimum path and equal-third beginning/middle/end supervision are
  still plausible recall bottlenecks. They must be changed or ablated in the
  next experiment; simply extending this run is not justified.
- A qualifying model is a prerequisite for firmware integration. With no
  qualifying artifact and no enumerated StackChan serial port, build/flash
  work would only create false confidence and must remain blocked.

### Fresh-start recommendation

Keep the deterministic manifests, source/session-disjoint splits, quarantine
boundary, exact streaming scorer, and no-flash gate. Start a new experiment
with an explicitly measured teacher (teacher logits or posterior state
alignments), a student objective that preserves teacher sequence decisions,
and ablations for path duration, temporal capacity, and frontend. Promote only
after the quantized student clears both validation and untouched test gates,
then integrate it behind an exact-device resource report before flashing.

## Research lineage

This design follows the low-compute shape described in Apple's
[on-device voice-trigger account](https://machinelearning.apple.com/research/hey-siri):
local phone-state posteriors plus cheap temporal integration with self-loop and
next-state transitions. The differentiable sequence objective is informed by
[State Sequence Pooling Training of Acoustic Models for Keyword Spotting](https://www.isca-archive.org/interspeech_2020/opatka20_interspeech.html)
and [Optimize what matters: Training DNN-HMM Keyword Spotting Model Using End
Metric](https://arxiv.org/abs/2011.01151). The retained mix of similar-sounding
hard negatives and broad random negatives is consistent with the failure mode
described by [Re-Weighted Interval Loss for Handling Data Imbalance Problem of
End-to-End Keyword Spotting](https://www.isca-archive.org/interspeech_2020/zhang20u_interspeech.html).
