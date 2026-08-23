# HiPhi Kizz wake-word recipe

This recipe trains one wake class from natural readings of **HiPhi Kizz**. Bare
`Kizz`, `kids`, `kiss`, `quiz`, `Hi-Fi`, and similar speech are negatives. A
one-syllable **Kizz** detector needs a separate model.

Hard negatives pair accepted HiPhi-like prefixes with wrong suffixes and wrong
prefixes with `Kizz`, preventing suffix shortcuts.

Start with synthetic data. A model is not hardware-qualified until real device
corpora evaluate it through the standalone
[device enrollment service](../../documentation/device_enrollment.md):

1. Positive StackChan recordings across positions, voices, speeds, and noise.
2. Long room conversation and music recordings for false accepts per hour.

The same enrollment contract applies to every microphone-equipped profile in
[`device-profiles.json`](../../device-profiles.json). Compare results by
`device_profile` first; split only when held-out evidence requires it. Catalog,
enrollment support, and collected corpus are separate statuses.

## Generate the phrase corpus

Use Python 3.10 or newer and install this repository plus
the [Open Horizon Labs `piper-sample-generator` fork](https://github.com/open-horizon-labs/piper-sample-generator).
Download the LibriTTS-R model linked by that project, then run:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --generator-source ../piper-sample-generator \
  --output work/kizz/generated \
  --batch-size 16
```

The command reserves different LibriTTS speakers for train, validation, and
test, writes per-WAV speaker/synthesis provenance, and resumes by phrase cohort.
`--dry-run` prints the plan. LibriTTS speaker IDs have no reliable age label;
this corpus alone does not claim child coverage. The recipe therefore requires
age-labeled adult and child voices held out by voice ID. Design and render the
supplement before feature building:

```sh
python tools/design_elevenlabs_voice_catalog.py \
  --spec recipes/kizz/elevenlabs-voice-designs.yaml \
  --output work/kizz/elevenlabs-voices.yaml \
  --preview-dir work/kizz/voice-previews

python tools/add_labeled_voice_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --voice-catalog work/kizz/elevenlabs-voices.yaml
```

Before production feature generation, compare the synthetic corpus with reviewed
human phrase spans:

```sh
python tools/build_synthetic_quality_mask.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --reference-corpus work/device-corpus \
  --output work/kizz/generated/quality-mask.json
```

This is a framework quality gate, not a Kizz tuning heuristic. Positive span
limits follow the reviewed human distribution with a 25% margin by default. The
mask removes silence, clipping, implausible positive spans, and clips that can be
truncated by the configured training window; it does not apply positive timing
limits to hard negatives. Inspect its per-phrase summary before training.

Build the `micro_speech` features used on-device, adding actual indoor/outdoor
noise and room responses. Training speech remains louder than the background;
clean holdouts are not augmented:

```sh
python tools/prepare_background_corpus.py \
  --output work/backgrounds \
  --esc50 ../ESC-50 \
  --device-corpus work/device-corpus

python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --quality-mask work/kizz/generated/quality-mask.json \
  --output work/kizz/features \
  --background-indoor work/backgrounds/indoor/train \
  --background-outdoor work/backgrounds/outdoor/train \
  --impulses room-impulses

python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained \
  --output work/kizz/training_parameters.yaml
```

To rebuild one class, pass `--class-name hard_negative` or `positive`; the
builder still validates the recipe, manifest, and corpus. Pass the alternate
feature root to the config writer with `--features-dir`. For acoustic-cluster
work, repeat `--positive-text` with recipe phrase labels.

The config uses hard negatives as both sampled training data and long-form
evaluation data, preventing a flattering FAPH while accepting `Kizz` or `kiss`.
Change sampling and penalty pressure with
`--hard-negative-sampling-weight` and `--hard-negative-penalty-weight`.

After export, measure each spelling separately:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --cutoff 0.96 \
  --output work/kizz/pronunciation_metrics.json
```

`probes.yaml` holds plausible HiPhi pronunciations outside training. Generate
and score them after export to test generalization.

## Quality bar

Minimize ambient false accepts before recall. A release bundle includes the
quantized `.tflite`, ESPHome v2 JSON, recipe, config, metrics, provenance, and
SHA-256 hashes. Synthetic validation cannot release a model.
Kizz qualification requires held-out adult and child human speakers; pass both
`--required-age-group adult` and `--required-age-group child` to the device
evaluator.
