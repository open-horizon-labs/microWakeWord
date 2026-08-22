# Usage

This guide covers setup, corpus generation, feature building, training,
evaluation, and device qualification. Kizz is the worked example, not a
framework limit.

## Training workflow at a glance

Synthetic audio defines the wake class and rejects weak models. Device audio
tests the survivors in their intended rooms.

| Stage | What we do | Exit gate |
| --- | --- | --- |
| [Set up](#1-install-the-trainer) | Install and test the trainer; obtain the Piper generator. | The test suite passes and the generator model is available. |
| [Define and generate](#3-inspect-and-generate-the-kizz-corpus) | List accepted pronunciations, nearby speech that must not wake the device, and unseen probes; generate varied synthetic speech. | The recipe, source hashes, phrase counts, and generation manifest validate. |
| [Screen and build features](#4-screen-synthetic-audio-and-build-features) | Compare generated speech with reviewed human phrase spans, exclude objective outliers, mix representative noise and room responses, and extract the on-device `micro_speech` features. | A provenance-bound quality mask and deterministic feature sets are ready. |
| [Train and select](#6-write-the-training-configuration-and-train) | Train the streaming model, minimize ambient false accepts first, then maximize viable recall, and export a quantized candidate. | A candidate and its frozen validation cutoff are ready for challenge testing. |
| [Challenge the candidate](#7-evaluate-every-trained-pronunciation) | Score every trained pronunciation, confusable phrase, unseen probe, and ambient negative cohort separately. | All declared synthetic gates pass. A failure sends us back to the recipe, data balance, or training configuration. |
| [Verify enrollment](#8-exercise-enrollment-without-hardware) | Run the simulated microphone path and prove that directed captures, including provisional detector misses, enter the corpus. | The standalone enrollment and corpus contracts pass end to end without hardware. |
| [Collect, retrain, and compare](#9-add-real-device-evidence) | Record real attempts through available microphone profiles, add those features, retrain one shared model, and inspect held-out cohort results. | Every claimed profile meets the frozen gates; split models only if held-out evidence requires it. |
| [Qualify the artifact](#10-qualification-checklist) | Freeze the model and cutoff, flash it, and run physical recall and false-wake tests. | The tested artifact is eligible for release on the targets that passed. |

Setup through enrollment verification needs no hardware. Record rejected models
and reasons in the [experiment log](../recipes/kizz/EXPERIMENTS.md).

## 1. Install the trainer

Use Python 3.10+ and a virtual environment:

```sh
git clone https://github.com/open-horizon-labs/microWakeWord.git
cd microWakeWord
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install piper-sample-generator
```

Run tests before long generation or training jobs:

```sh
python -m unittest discover -s tests -v
```

GPU is recommended for generation and training; commands are unchanged on CPU.

## 2. Obtain the Piper generator model

Kizz uses the LibriTTS-R multi-speaker generator from
[`piper-sample-generator`](https://github.com/rhasspy/piper-sample-generator):

```sh
mkdir -p models
curl -L \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  -o models/en_US-libritts_r-medium.pt
```

If the installed package lacks the generator module, clone it beside this repo
and add
`--generator-source ../piper-sample-generator` to generation commands.

## 3. Inspect and generate the Kizz corpus

[`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) defines phrases,
counts, pronunciations, confusables, and Piper variation. Inspect it first:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --output work/kizz/generated \
  --dry-run
```

Generate the corpus:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --output work/kizz/generated \
  --batch-size 16
```

Generation skips completed phrase directories and rejects surplus, incomplete,
or mismatched corpora. `generation-manifest.json` records recipe and model hashes.
To reuse unchanged phrase audio from an earlier run, repeat
`--reuse-generated work/earlier/generated`. Reuse requires the same phrase,
sample count, generator model, and synthesis command. The new manifest records
the source.

## 4. Screen synthetic audio and build features

Before production training, mark the intended phrase in at least three reviewed
human-positive captures. This can be audio sent through the enrollment simulator;
the target device need not be attached. Apply reviewed spans without rewriting
the source WAVs:

```sh
python tools/apply_phrase_spans.py \
  --corpus work/device-corpus \
  --spans work/reviewed-phrase-spans.json
```

The spans file maps capture IDs to `start_ms` and `end_ms`. Build a quality mask
that compares every generated WAV with the recorded span distribution:

```sh
python tools/build_synthetic_quality_mask.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --reference-corpus work/device-corpus \
  --output work/kizz/generated/quality-mask.json
```

The report summarizes source duration, voiced span, and level by phrase. It
rejects silence, clipping, implausible positive spans, and source clips that can
lose audio when placed into the training window. Recipe, generation-manifest,
and reference-corpus hashes make the decision auditable. Inspect the grouped
report before using it; a mask that removes a large share of one phrase usually
indicates a recipe or generator problem.

Add representative room recordings and impulse responses when available. Both
arguments are repeatable:

```sh
python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --quality-mask work/kizz/generated/quality-mask.json \
  --output work/kizz/features \
  --background room-backgrounds \
  --impulses room-impulses
```

The builder verifies that the mask belongs to the recipe and generated corpus,
excludes rejected WAVs, creates device-compatible `micro_speech` features, and
preserves deterministic splits.

## 5. Supply general negative features

The recipe config expects prepared negative feature archives under:

```text
work/kizz/negative-datasets/
├── dinner_party/
├── dinner_party_eval/
├── no_speech/
└── speech/
```

Use upstream [pre-generated features](https://huggingface.co/datasets/kahrendt/microwakeword)
or equivalent licensed archives. The starter
[`basic_training_notebook.ipynb`](../notebooks/basic_training_notebook.ipynb)
shows acquisition and preparation. Never reuse evaluation audio for training.

## 6. Write the training configuration and train

```sh
python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained \
  --output work/kizz/training_parameters.yaml

python -m microwakeword.model_train_eval \
  --training_config work/kizz/training_parameters.yaml \
  --train 1 \
  mixednet
```

The config weights confusable speech in training and retains it for evaluation.
Checkpoint selection minimizes ambient false accepts first, then maximizes
viable recall. Training exports the selected model to:

```text
work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

Resume with `--restore_checkpoint 1`; review the learning-rate schedule first.

## 7. Evaluate every trained pronunciation

Freeze the cutoff from held-out validation before testing. `0.96` is illustrative:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --split test \
  --cutoff 0.96 \
  --output work/kizz/pronunciation_metrics.json
```

Review each positive spelling and hard-negative phrase; do not use one aggregate.

[`recipes/kizz/probes.yaml`](../recipes/kizz/probes.yaml) holds readings absent
from training. Generate them separately and evaluate with `--split all`. Never
add scored probes to training; add new held-out probes after recipe changes.

## 8. Exercise enrollment without hardware

The training endpoint is a LAN service, separate from UHC and production voice.
It may run on another host.

Start the service and a simulated microphone device in separate terminals:

```sh
python tools/run_enrollment_service.py \
  --corpus work/device-corpus \
  --port 8091

python tools/simulate_enrollment_device.py \
  --endpoint ws://trainer-host:8091/v1/device \
  --device-id simulated-mic-1 \
  --device-profile my_microphone_profile_v1 \
  --no-detected
```

Use [device corpus enrollment](device_enrollment.md) to queue bounded attempts.
The simulator retains provisional misses. Replace its profile with one from
`device-profiles.json`.

## 9. Add real-device evidence

Validate real audio and build features without changing declared splits:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus

python tools/build_device_corpus_features.py \
  --corpus work/device-corpus \
  --output work/device-features

python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained-with-devices \
  --device-features-dir work/device-features \
  --device-truncation-strategy random \
  --output work/kizz/device_training_parameters.yaml
```

Device recordings are often longer than the model window and the wake phrase
may occur anywhere in them. `random` samples across the recording during
training. Use `truncate_start` or `truncate_end` only when capture timing
guarantees the phrase is aligned to that edge; the chosen strategy is written
into the training configuration. For a longer recording, add
`phrase_span.start_ms` and `phrase_span.end_ms` to its manifest entry. The device
feature builder then uses the phrase with 250 ms of context without changing the
source WAV.

Train, then evaluate the frozen model on held-out device audio. Replace `0.96`
with the validation cutoff:

```sh
python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained-with-devices/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --cutoff 0.96 \
  --output work/kizz/device_test_metrics.json
```

Review results by truth, phrase, pronunciation, profile, speaker, session, and
detector outcome. `--split all` is useful for a corpus-wide diagnostic, but
release claims must use held-out validation and test speakers.
Start with one model across profiles; split only for held-out acoustic failure.

## 10. Qualification checklist

- recipe, generator model, training config, and corpus hashes are recorded;
- no speaker or recording session crosses train, validation, and test splits;
- held-out positive pronunciations and confusable phrases meet explicit gates;
- ambient false accepts are measured over representative duration;
- provisional detector misses are present in the device evaluation;
- the tested quantized artifact is flashed and accepted on each claimed target.

Synthetic-only results belong in the [experiment log](../recipes/kizz/EXPERIMENTS.md),
not a hardware-qualified release claim.

## Troubleshooting

- **Generation stops on an existing phrase directory:** its WAV count differs
  from the recipe. Move that phrase directory aside and regenerate it; do not
  blend old and new recipes.
- **Feature building rejects the corpus:** the recipe hash, phrase directories,
  or WAV counts no longer match `generation-manifest.json`.
- **The quality mask rejects many examples:** inspect its per-phrase reasons and
  source metrics. Fix the generator or phrase recipe instead of weakening the
  mask until the clips pass.
- **Training cannot open a feature set:** populate every negative archive named
  in the generated YAML, or deliberately edit the YAML and record the changed
  experiment.
- **Aggregate recall looks good but a reading fails:** use the per-phrase report
  and unseen probes. Aggregate recall is not the acceptance gate.
- **A real wake attempt is absent:** enrollment must record the bounded attempt
  independently of the provisional wake decision. `detected` is evidence, not
  an inclusion filter.
