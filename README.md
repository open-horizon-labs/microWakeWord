![microWakeWord logo](etc/logo.png)

# microWakeWord — Open Horizon Labs fork

This is Open Horizon Labs' experimental fork of
[Kevin Ahrendt's microWakeWord](https://github.com/kahrendt/microWakeWord). It
keeps the upstream TensorFlow training and streaming-export architecture, then
adds a reproducible path for developing and qualifying custom wake words against
the microphones that will run them.

The first complete recipe targets **HiPhi Kizz**. The framework itself is not
Kizz-specific: its device-corpus contract represents every microphone-equipped
HiPhi controller by `device_profile`, so one shared model can be evaluated across
devices before evidence warrants separate models.

> **Status:** research and model-development tooling. A successful synthetic
> evaluation is not hardware qualification. No device profile is currently
> marked as having a collected real corpus.

## Start here

- Follow the end-to-end [usage guide](documentation/USAGE.md).
- Read the [HiPhi Kizz recipe](recipes/kizz/README.md) for its phrase and
  collision policy.
- See the [technique and reference ledger](documentation/techniques.md) for
  what is inherited, what this fork adds, and the evidence behind each choice.
- Use [device enrollment](documentation/device_enrollment.md) to collect hits
  **and misses** through a standalone training endpoint.
- Review the [experiment log](recipes/kizz/EXPERIMENTS.md) before changing the
  wake class or its hard negatives.

## Why this fork exists

Upstream microWakeWord provides the neural-network architecture, training loop,
audio frontend, and a starter notebook. This fork extends that foundation with
the pieces needed to repeat and audit a product wake-word program:

| Addition | Why it exists |
| --- | --- |
| Manifest-driven recipes | Recreate generated corpora from explicit phrases, variation settings, seeds, and source hashes. |
| Pronunciation and collision design | Train natural readings of `HiPhi` as one full-phrase class while testing nearby speech that must not wake the device. |
| Confusable-aware model selection | Make hard negatives affect training and checkpoint selection instead of relying on aggregate recall alone. |
| Standalone device enrollment | Collect bounded microphone captures without coupling training to UHC or assuming the training and voice endpoints share a host. |
| Versioned device-corpus contract | Retain provisional detector misses, preserve explicit splits, prevent speaker/session leakage, and group evidence by acoustic profile. |
| Cohort-level evaluation | Report results by phrase, pronunciation, truth, device profile, and prior detector outcome so weak cohorts cannot hide in an average. |

The detailed implementation map and primary references are in
[Techniques and references](documentation/techniques.md).

## What this fork does not claim

- Synthetic speech alone does not establish real-room recall or false accepts
  per hour.
- A cataloged microphone profile does not imply enrollment firmware or a real
  corpus exists for that target.
- The current Kizz thresholds and corpus weights are experiment inputs, not
  universal defaults for every phrase or device.
- Training remains compute-intensive and requires representative negative audio
  plus held-out recordings from the intended hardware and environment.

## How microWakeWord works

The detector has two stages. First, mono 16 kHz audio is converted into 40
features every 10 ms by the TensorFlow Lite Micro
[`micro_speech` frontend](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech).
The frontend uses a 30 ms window, with adjacent windows overlapping by 20 ms,
and applies noise reduction and gain normalization suited to small devices.

Second, a streaming neural network updates from the newest feature slice and
emits a wake probability. The model uses
[MixConv](https://arxiv.org/abs/1907.09595) depthwise convolutions and code
derived from Google Research's
[`kws_streaming`](https://github.com/google-research/google-research/tree/master/kws_streaming)
work described in
[Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/abs/2005.06720).
Several consecutive high probabilities are required before the runtime declares
a wake.

Training uses whole spectrograms, then exports a streaming TensorFlow Lite model
for incremental inference. The upstream training pipeline supports
[SpecAugment](https://arxiv.org/abs/1904.08779), weighted sampling and penalties,
ambient false-accept estimation, streaming conversion, and integer
quantization. See the [technique ledger](documentation/techniques.md) for the
precise relationship between those inherited capabilities and this fork's
extensions.

## Models and data

Upstream-compatible published models are available from
[ESPHome's micro wake word model repository](https://github.com/esphome/micro-wake-word-models).
The training framework can consume the upstream
[pre-generated negative feature datasets](https://huggingface.co/datasets/kahrendt/microwakeword).
Source and license notes for the underlying public corpora are in
[Data sources](documentation/data_sources.md).

## Upstream relationship

This fork intends to preserve attribution and compatibility with upstream
microWakeWord. General improvements should be suitable for upstreaming when
possible; Open Horizon Labs recipes, device profiles, and product qualification
evidence can remain fork-specific. When reporting a problem, state whether it
reproduces on
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
