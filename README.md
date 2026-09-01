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

## Worked recipe: Kizz Control

The current [Kizz Control cascade recipe](recipes/kizz/CASCADE_V9_RECIPE.md)
is a reproducible three-stage path for StackChan: a permissive ordered INT8
detector, a compact device-adapted INT8 DS-CNN verifier, and an independent
ordered verifier. The machine-readable graph and resumable runner parallelize
source synthesis and locked continuous evaluation while enforcing validation-
only selection, frozen fresh test, evidence hashes, and explicit paid-call
authority.

The reference cascade retained 12/12 fresh StackChan-channel positives. Over a
locked 100.47-hour negative corpus it accepted 39 false wakes (`0.388/hour`,
about one every 2.6 hours), and only 5.19% of detector candidates reached the
expensive ordered stage. The maintainer accepted this practical operating
point. It does not pass the stricter formal `0.1/hour` upper-confidence gate,
and host evidence is not physical ESP32-S3 performance qualification.

Exact reference models and hashes are checked in under
[`recipes/kizz/reference-cascade-v9`](recipes/kizz/reference-cascade-v9/README.md).
Earlier HiPhi Kizz and single-student attempts remain as failure evidence in
the [experiment ledger](recipes/kizz/EXPERIMENTS.md); they are not the current
recipe.

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
