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
git clone https://github.com/open-horizon-labs/piper-sample-generator.git ../piper-sample-generator
git -C ../piper-sample-generator checkout 35d2f2d
```

Run tests before long generation or training jobs:

```sh
python -m unittest discover -s tests -v
```

GPU is recommended for generation and training; commands are unchanged on CPU.

## 2. Obtain the Piper generator model

Kizz uses the LibriTTS-R multi-speaker generator through the Open Horizon Labs
[`piper-sample-generator` fork](https://github.com/open-horizon-labs/piper-sample-generator).
The fork reserves disjoint speaker ranges and records per-WAV provenance:

```sh
mkdir -p models
curl -L \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  -o models/en_US-libritts_r-medium.pt
```

Pass `--generator-source ../piper-sample-generator` to generation commands.

## 3. Inspect and generate the Kizz corpus

[`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) defines phrases,
counts, pronunciations, confusables, and Piper variation. Inspect it first:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --output work/kizz/generated \
  --generator-source ../piper-sample-generator \
  --dry-run
```

Generate the corpus:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --output work/kizz/generated \
  --generator-source ../piper-sample-generator \
  --batch-size 16
```

Generation assigns non-overlapping LibriTTS speaker IDs to train, validation,
and test before rendering audio. It skips complete cohorts and rejects surplus,
incomplete, or mismatched corpora. `generation-manifest.json` records recipe,
model, speaker pairs, synthesis settings, and hashes. LibriTTS does not provide
reliable age metadata; its cohorts are `unknown`, not evidence of child voices.
To reuse unchanged phrase audio from an earlier run, repeat
`--reuse-generated work/earlier/generated`. Reuse requires the same phrase,
sample count, generator model, and synthesis command. The new manifest records
the source.

Add age-labeled voices before feature building. The Kizz recipe requires two
adult and two child voices in train, plus distinct adult and child identities in
validation and test. Design them with ElevenLabs, or supply an equivalent
catalog from another licensed source:

```sh
export ELEVENLABS_API_KEY=...
python tools/design_elevenlabs_voice_catalog.py \
  --spec recipes/kizz/elevenlabs-voice-designs.yaml \
  --output work/kizz/elevenlabs-voices.yaml \
  --preview-dir work/kizz/voice-previews

python tools/add_labeled_voice_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --voice-catalog work/kizz/elevenlabs-voices.yaml
```

The first command saves the provider previews and resolves them to persistent
voice IDs. The second renders every positive and confusable phrase, recording
voice identity, declared age group, split, model, seed, and settings. A voice ID
may appear in one split only. The API key stays outside the repository. Start
from [`synthetic_voice_catalog.example.yaml`](synthetic_voice_catalog.example.yaml)
when using existing voices instead of Voice Design.

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
that compares every generated WAV with the human training-span distribution;
validation and test speakers do not influence the mask:

```sh
python tools/build_synthetic_quality_mask.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --reference-corpus work/device-corpus \
  --output work/kizz/generated/quality-mask.json
```

The report compares source duration, voiced span, and level by phrase. For
positive clips, the default span limits use the human 5th and 95th percentiles
with a 25% margin. The mask also rejects silence, clipping, and source clips that
can lose audio when placed into the training window. Hard-negative duration is
not forced into the positive-phrase envelope.

The recipe, generation manifest, reference corpus, and span policy are hashed or
recorded in the report. Inspect its grouped results before using it: a phrase
with an unusually high rejection rate usually needs a generator or recipe fix.
The mask catches measurable defects; it does not prove that pronunciation or
prosody sounds natural.

Add representative indoor and outdoor recordings and room impulse responses.
The builder screens the source clips first, then applies acoustic variation only
while creating training features. Backgrounds are mixed 3–20 dB below speech by
default; validation and test remain clean:

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
```

The preparation step assigns ESC-50 folds 1–4 to training augmentation and fold
5 to `indoor/stress` or `outdoor/stress`; it also includes real training-only
device room tone. `background-corpus.json` records the source revision, license,
category, device profile where applicable, split, and file hash.

The builder verifies that the mask belongs to the recipe and generated corpus,
rejects speaker overlap or missing age cohorts, excludes masked WAVs, creates
device-compatible `micro_speech` features, and preserves the declared speaker
cohorts. The training transform includes gain, EQ, mild distortion, pitch and
band filtering, colored noise, background mixing, and room response. Its source
categories, SNR range, and probabilities are recorded in
`feature-build-manifest.json`.

Feature splits can be rebuilt independently after a verified partial run. Remove
the incomplete split, then select its class and name:

```sh
python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --quality-mask work/kizz/generated/quality-mask.json \
  --output work/kizz/features \
  --class-name positive \
  --feature-split testing
```

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

For controlled comparisons, use a stratified sampling plan instead of allowing
source file counts to set the batch mixture. The plan assigns shares to named
positive and negative groups, then expands them into a training config:

```sh
python tools/write_stratified_training_config.py \
  --base-config work/kizz/training_parameters.yaml \
  --sampling-plan work/kizz/sampling-plan.yaml \
  --output work/kizz/stratified_training_parameters.yaml
```

The generated config records both the planned sample shares and the effective
positive/negative pressure after source penalties and class weights. A plan can
set `balance_guard.maximum_negative_sampling_share` and
`balance_guard.maximum_negative_weighted_pressure_share` to reject an accidental
imbalance before training. Keep each sampling group single-class; split positive
and negative sources into separate groups even when they share a provider.

## 7. Evaluate every trained pronunciation

Freeze the cutoff from held-out validation before testing. `0.96` is illustrative:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --recipe recipes/kizz/corpus.yaml \
  --quality-mask work/kizz/generated/quality-mask.json \
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
  --port 8091 \
  --public-base-url http://trainer-host:8091

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
  --qualification \
  --required-age-group adult \
  --required-age-group child \
  --cutoff 0.96 \
  --output work/kizz/device_test_metrics.json
```

Review results by truth, phrase, pronunciation, profile, speaker, session, and
detector outcome. `--qualification` rejects train or incomplete test corpora.
`--split all` is a diagnostic and is marked as containing training data.
Start with one model across profiles; split only for held-out acoustic failure.

Run a second report with runtime-like streaming state:

```sh
python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained-with-devices/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --cutoff 0.96 \
  --state-mode carry_until_detection \
  --output work/kizz/device_test_runtime_state.json
```

The default, `reset_per_capture`, measures each recording independently.
`carry_until_detection` preserves model state across modeled misses and resets
after an accept, matching the detector's rearm boundary. It exposes sensitivity
to prior audio but cannot recreate ambient gaps that were not recorded. Treat
both reports as diagnostics; qualification still requires the flashed artifact.

## 10. Qualification checklist

- recipe, generator model, training config, and corpus hashes are recorded;
- every human speaker is registered once, assigned one split, and identity-attested;
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
