![microWakeWord logo](etc/logo.png)

# microWakeWord for real devices

Build a custom wake word, find out what it confuses with, and test the same
streaming model against the microphones and rooms where it must work.

This Open Horizon Labs fork extends
[Kevin Ahrendt's microWakeWord](https://github.com/kahrendt/microWakeWord). It
keeps the upstream TensorFlow training and streaming TensorFlow Lite export,
then adds the workflow they lack: reproducible recipes,
confusable speech, cohort-level evaluation, real-microphone enrollment, and
artifact qualification.

## When this fork is useful

- the wake phrase is a name, brand, or invented word without a ready-made model;
- people may pronounce the phrase several ways;
- nearby words and partial phrases must not wake the device;
- one candidate must be compared across different microphone frontends;
- a detector miss must become training evidence instead of disappearing.

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

A failed gate sends the work back to the recipe, corpus balance, or training
configuration. Record rejected candidates in the experiment ledger. Aggregate
scores do not satisfy the release gate.

## Worked recipe: Kizz

The included [Kizz recipe](recipes/kizz/README.md) is one application of the
framework. It treats natural readings of **HiPhi Kizz** as one wake class and
explicitly tests `Kizz`, `kids`, `kiss`, and `quiz`, plus valid prefixes with
the wrong final word and wrong prefixes followed by `Kizz`.

The current synthetic candidates are rejected. Their per-phrase results exposed
confusable acceptance and weak unseen-pronunciation recall that an aggregate
result would hide. See the [experiment ledger](recipes/kizz/EXPERIMENTS.md).
The repository does not publish a hardware-qualified Kizz model.

## Start here

- Follow the [end-to-end usage guide](documentation/USAGE.md).
- Read the [technique and reference ledger](documentation/techniques.md) to see
  what comes from upstream research and what this fork adds.
- Use the [standalone device enrollment service](documentation/device_enrollment.md)
  to collect bounded real-microphone attempts, including provisional misses.
- Check [data-source and license notes](documentation/data_sources.md) before
  building or distributing a corpus.

## What a run produces

- a recipe and generation manifest with source hashes and phrase counts;
- deterministic train, validation, and test feature sets;
- a quantized streaming TensorFlow Lite candidate;
- separate reports for pronunciations, confusable phrases, ambient audio,
  device profiles, and prior detector outcomes;
- a versioned real-device corpus with audio hashes and leak-safe splits;
- evidence for a flash or release decision.

## Status and qualification boundary

This is experimental training and qualification tooling. Synthetic evaluation
can reject a model; it cannot qualify one for a room or microphone it has never
heard. A model release still requires representative real-device corpora,
frozen held-out evaluation, the tested quantized artifact, and physical
acceptance on every claimed target.

The [device profile catalog](device-profiles.json) records microphone capability
and acoustic-domain metadata so product corpora can be compared through one
contract. A catalog entry does not imply enrollment firmware or real recordings.
The included Kizz enrollment path is a reference implementation, not a limit on
the devices this framework can qualify.

## How microWakeWord detects a wake phrase

The detector has two stages. First, mono 16 kHz audio is converted into 40
features every 10 ms by the TensorFlow Lite Micro
[`micro_speech` frontend](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech).
The frontend uses a 30 ms window with 20 ms of overlap and applies noise
reduction and gain normalization suited to small devices.

Second, a streaming neural network updates from the newest feature slice and
emits a wake probability. The model uses
[MixConv](https://arxiv.org/abs/1907.09595) depthwise convolutions and code
derived from Google Research's
[`kws_streaming`](https://github.com/google-research/google-research/tree/master/kws_streaming)
work described in
[Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/abs/2005.06720).
The runtime requires several consecutive high probabilities before declaring a
wake.

Training uses whole spectrograms, then exports a streaming model for incremental
inference. The upstream pipeline supports
[SpecAugment](https://arxiv.org/abs/1904.08779), weighted sampling and penalties,
ambient false-accept estimation, streaming conversion, and integer
quantization. The [technique ledger](documentation/techniques.md) maps each
method to its implementation and source.

## Models and training data

Upstream-compatible published models are available from
[ESPHome's micro wake word model repository](https://github.com/esphome/micro-wake-word-models).
The framework can consume upstream
[pre-generated negative features](https://huggingface.co/datasets/kahrendt/microwakeword).
The [data-source notes](documentation/data_sources.md) identify the underlying
public corpora and their licenses.

## Upstream relationship

This fork preserves upstream attribution and aims to retain compatibility with
microWakeWord. General improvements should be upstreamable when possible. Open
Horizon Labs recipes, device profiles, and product-qualification evidence may
remain fork-specific. Bug reports should say whether the problem reproduces on
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
