# HiPhi Kizz wake-word recipe

This recipe trains one wake class from several natural readings of **HiPhi
Kizz**. It intentionally treats bare `Kizz`, `kids`, `kiss`, `quiz`, `Hi-Fi`,
and similar speech as negatives. A one-syllable **Kizz** detector belongs in a
separate model so its difficult near-word boundary cannot weaken the full brand
phrase.

The hard-negative corpus also includes conjunction-mining pairs: accepted
HiPhi-like prefixes followed by the wrong final word, and wrong prefixes
followed by the exact word `Kizz`. This prevents a multi-pronunciation model
from learning the common suffix as a shortcut.

The first model is synthetic-data-first, but it is not considered hardware
qualified until it is evaluated against two device-microphone corpora:

1. Positive recordings spoken to the actual StackChan from varied positions,
   voices, speeds, and room noise.
2. Long negative recordings from normal conversation and music in the intended
   room, used to measure false accepts per hour.

## Generate the phrase corpus

Use Python 3.10 or newer and install this repository plus
`piper-sample-generator`. Download the LibriTTS-R generator linked by that
project, then run:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --generator-source ../piper-sample-generator \
  --output work/kizz/generated \
  --batch-size 16
```

The command is resumable per phrase and writes a manifest containing hashes of
the recipe and generator model. `--dry-run` prints the complete generation plan
without downloading or generating anything.

Then augment and convert the audio to the exact micro-speech features used on
device. Pass actual room music/noise and impulse-response directories whenever
available:

```sh
python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --output work/kizz/features \
  --background room-backgrounds \
  --impulses room-impulses

python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained \
  --output work/kizz/training_parameters.yaml
```

When a later recipe revision changes only one class, pass `--class-name
hard_negative` or `--class-name positive`; the complete recipe/manifest and all
corpus directories are still validated before the selected class is rebuilt.

After exporting the quantized streaming model, measure every spelling
separately so a strong aggregate score cannot hide a weak pronunciation:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --cutoff 0.96 \
  --output work/kizz/pronunciation_metrics.json
```

`probes.yaml` defines additional plausible HiPhi pronunciations that are kept
out of training. Generate and score them after export to check acoustic
generalization rather than memorization of the corpus spellings.

## Quality bar

Model selection must minimize ambient false accepts before maximizing recall.
The release bundle must include the quantized streaming `.tflite`, ESPHome v2
model JSON, recipe and training config, metrics, corpus provenance, and SHA-256
hashes. A model is not releasable based only on synthetic validation results.
