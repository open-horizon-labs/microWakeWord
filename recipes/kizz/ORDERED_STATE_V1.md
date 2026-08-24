# Kizz ordered-state model outline

Status: architecture scaffold implemented under `muness/roon-knob#240`; no
trained candidate has qualified yet. Training and qualification are tracked by
`muness/roon-knob#241`.

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

The 21-state path imposes a 630 ms minimum decoded phrase duration at the
30 ms output cadence. Fast natural speakers are therefore an explicit recall
risk to measure, not an assumption to hide.

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

Qualification reports must pass the artifact's `--decoder-contract` and
`--require-declared-exposure` to the streaming evaluator. The contract supplies
the 30 ms frame step, validates every score timestamp against that cadence, and
prevents an omitted final stride from biasing false-accepts/hour. A conflicting
decoder argument is rejected. Cooldown belongs either to the decoder or the
evaluator, never both.

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
