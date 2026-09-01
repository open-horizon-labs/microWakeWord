![microWakeWord logo](etc/logo.png)

# microWakeWord for real devices

Build a custom wake word, find out what it confuses with, and test the same
streaming model against the microphones and rooms where it must work.

This Open Horizon Labs fork builds on
[Kevin Ahrendt's microWakeWord](https://github.com/kahrendt/microWakeWord) and
retains its TensorFlow training and streaming TensorFlow Lite export.

## What this fork adds

Upstream microWakeWord provides the TensorFlow trainer, audio augmentation,
checkpoint selection, streaming conversion, and quantized TensorFlow Lite
export. This fork adds the workflow for developing and qualifying a custom wake
word across real devices.

| Upstream provides | This fork adds |
| --- | --- |
| Piper sample generation and notebook-driven training | Versioned recipes, source hashes, per-WAV synthesis provenance, speaker-independent splits, and age-labeled supplemental voice cohorts |
| Weighted training sources and ambient false-accept metrics | Named pronunciation and confusable-speech cohorts used in training, checkpoint selection, and evaluation |
| Streaming TensorFlow Lite export | Per-phrase evaluation with isolated-state and carry-until-detection replay |
| Model training and test datasets | A device-corpus contract that retains every commanded attempt, including detector misses, aligns phrases inside long captures, and keeps indoor/outdoor training backgrounds separate from stress evidence |
| A model artifact | Registered physical speakers, device-profile comparisons, a qualification scope gate, and a physical checklist |

Use upstream when you need the trainer and exporter. Use this fork when you also
need reproducible experiments and evidence that a candidate works on its target
microphones.

## The training and qualification loop

```text
phrase + pronunciations + confusable speech
                  ↓
       reproducible synthetic corpus
                  ↓
     on-device-compatible audio features
                  ↓
       quantized streaming candidate
                  ↓
 pronunciation + collision + ambient tests
                  ↓
 real microphone corpus, including detector misses
                  ↓
 held-out comparison by device profile and cohort
                  ↓
       physical artifact qualification
```

When a gate fails, revise the recipe, corpus balance, or training configuration.
Record rejected candidates; aggregate scores never satisfy a release gate.

## Recommended: the Kizz Control cascade

Start with the [v10 machine recipe](recipes/kizz/CASCADE_V10_RECIPE.md). It is
the reproducible baseline for a new run. Do not begin by retraining every
network. First qualify the complete v10 cascade on the target device. If
physical use exposes repeatable false wakes, follow the
[v15 refinement](recipes/kizz/CASCADE_V15_HARDWARE_REFINEMENT.md) and retrain
only the compact verifier from reviewed device recordings.

| Stage | When it runs | Job |
| --- | --- | --- |
| Ordered INT8 detector | Continuously | Preserve wake recall with a permissive threshold |
| Compact INT8 DS-CNN verifier | On detector candidates | Reject common microphone- and room-specific collisions cheaply |
| Independent ordered verifier | On compact survivors | Make the sparse final decision |

Follow this order:

1. Run the checked-in v10 recipe and freeze its model, threshold, and test
   evidence.
2. Flash the fixed AOT/ESP-NN firmware path and test positive recall, false
   wakes, latency, queue depth, heap, and wake-to-command coexistence.
3. If physical false wakes remain, use the
   [false-wake retraining runbook](recipes/kizz/FALSE_WAKE_RETRAINING.md) to
   quarantine and review them, retrain the compact stage, then repeat the same
   physical tests with a sealed negative guard.

The v10 reference retained 12/12 fresh StackChan-channel positives and produced
23 false wakes over a locked 100.47-hour negative corpus (`0.229/hour`). Its
compact gate forwarded 4.36% of detector candidates to the ordered verifier.
The v15 cascade retained 12/12 post-flash physical wakes, reduced
false accepts from 17 to 5 on the same adversarial 25-minute playback, and
accepted 0/20 candidates on a sealed unseen guard. A corrected production
profile also completed one physical wake-to-STT-to-Roon command with about
12 KiB of internal-heap margin.

These results meet the maintainer's practical operating point. They do not pass
the formal `0.1/hour` upper-confidence gate or establish broad multi-human,
multi-room product qualification.

### How we got there

| Attempt | What the evidence showed | Decision retained in the recipe |
| --- | --- | --- |
| **HiPhi Kizz** phrase | Positives and false wakes overlapped with “Hi-Fi Kids,” “High Five Kiss,” and nearby speech | Change the phrase to **Kizz Control** |
| Teacher and single-student paths | No candidate passed the frozen promotion gates, and the experimental single-student firmware failed live precision | Preserve the experiments, but do not use them as the deployment path |
| Old int16/xwide detector and verifier | The runtime left StackChan effectively CPU-locked | Use compact INT8 models, fixed AOT schedules, static arenas, ESP-NN kernels, and sparse verifier invocation |
| V9 device adaptation | It reached `0.388` false wakes/hour, but the short capture lead created invalid zero-padded verifier context | Rebuild the device corpus with at least 2.3 seconds of real pre-roll |
| V10 clean cascade | It retained 12/12 physical replay recall at `0.229` false wakes/hour while forwarding only 4.36% of candidates | Keep v10 as the long-duration reference and reproducible starting point |
| V15 physical hard negatives | Retraining the compact stage cut the same 25-minute adversarial replay from 17 false accepts to 5 without losing the 12/12 positive replay | Keep the outer stages frozen; adapt the compact verifier from reviewed physical failures |
| Wake/STT coexistence test | The optional enrollment client exhausted socket and internal-heap headroom during normal voice use | Disable enrollment in production and retest the exact wake-to-command path |

Exact v10 models and hashes are checked in under
[`recipes/kizz/reference-cascade-v10`](recipes/kizz/reference-cascade-v10/README.md).
The [Kizz research overview](recipes/kizz/README.md) links the full experiment
history and explains why the rejected teacher, distillation, v9, and
single-student paths remain as evidence rather than recommendations.

## Start here

- [Usage guide](documentation/USAGE.md): train and qualify a model.
- [Techniques and references](documentation/techniques.md): methods, sources,
  and fork-specific policies.
- [Device enrollment](documentation/device_enrollment.md): collect bounded
  microphone attempts, including provisional misses.
- [Data sources](documentation/data_sources.md): corpus licensing and provenance.

## What a run produces

- recipe, generation manifest, and source hashes;
- speaker-independent train, validation, and test features, including declared
  age cohorts when the recipe requires them;
- provenance-bound indoor/outdoor background and stress cohorts;
- quantized streaming TensorFlow Lite candidate and cohort reports;
- versioned device corpus with leak-safe splits;
- evidence for a flash or release decision.

## Status and qualification boundary

This is experimental training and qualification tooling. Synthetic evaluation
can reject a model, not qualify it. Release requires representative device
corpora, frozen held-out evaluation, the tested artifact, and physical
acceptance on each claimed target.

The [device profile catalog](device-profiles.json) records microphone capability
and acoustic-domain metadata. A catalog entry does not imply enrollment firmware
or real recordings. Kizz enrollment is a reference implementation, not a device
limit.

## How microWakeWord detects a wake phrase

The detector first converts mono 16 kHz audio into 40 features every 10 ms with
the TensorFlow Lite Micro
[`micro_speech` frontend](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech):
a 30 ms window with 20 ms overlap, noise reduction, and gain normalization.

Then a streaming neural network consumes each new feature slice and emits a
wake probability. It uses
[MixConv](https://arxiv.org/abs/1907.09595) depthwise convolutions and code
derived from Google Research's
[`kws_streaming`](https://github.com/google-research/google-research/tree/master/kws_streaming)
described in [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/abs/2005.06720).
Several consecutive high probabilities trigger a wake.

Training uses whole spectrograms and exports an incremental streaming model. It
supports
[SpecAugment](https://arxiv.org/abs/1904.08779), weighted sampling and penalties,
ambient false-accept estimation, streaming conversion, and integer
quantization. The [technique ledger](documentation/techniques.md) maps methods
to implementations and sources.

## Models and training data

Upstream-compatible published models are available from
[ESPHome's micro wake word model repository](https://github.com/esphome/micro-wake-word-models).
It can consume upstream [pre-generated negative features](https://huggingface.co/datasets/kahrendt/microwakeword).
See [data sources](documentation/data_sources.md) for corpora and licenses.

## Upstream relationship

This fork retains upstream attribution and compatibility where possible. General
improvements should be upstreamable; recipes, device profiles, and qualification
evidence may remain fork-specific. Bug reports should say whether the problem reproduces on
[`kahrendt/microWakeWord`](https://github.com/kahrendt/microWakeWord) or only on
this fork.

[![A library from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-library.png)](https://www.openhomefoundation.org/)

## Acknowledgements

microWakeWord was created and is maintained upstream by
[Kevin Ahrendt](https://github.com/kahrendt). The original acknowledgements
thank [balloob](https://github.com/balloob),
[dscripka](https://github.com/dscripka),
[jesserockz](https://github.com/jesserockz),
[kbx81](https://github.com/kbx81),
[synesthesiam](https://github.com/synesthesiam),
[ESPHome](https://github.com/esphome),
[Nabu Casa](https://github.com/NabuCasa), and the
[Open Home Foundation](https://www.openhomefoundation.org/) for feedback,
collaboration, and development support.

The repository is licensed under the [Apache License 2.0](LICENSE).
