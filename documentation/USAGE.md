# Usage

This fork trains and evaluates custom wake words for real microphones. The
commands below build a candidate and collect the evidence needed for a release
decision; they do not make a model deployable by themselves. The usual path is:

```text
define phrases → generate or collect audio → build features → train
→ evaluate held-out audio → test the exact artifact on hardware
```

Kizz is the worked example. Its detailed history is in the
[training reference](../recipes/kizz/TRAINING_REFERENCE.md).

## Current Kizz status

V19 had useful recall but poor live precision. The clean-slate teacher → student
attempt also failed in the house: its student produced frequent false positives
on StackChan. Deployment reverted to the ESP-IDF wake-word model.

The recipe is useful research infrastructure, not a deployable model. The
private audio and checkpoints used for the reported run are not in this repo.
Read the [Kizz recipe](../recipes/kizz/README.md) and
[salvage report](../recipes/kizz/SALVAGE_TEACHER_STUDENT_V1.md) before using it.

## 1. Install and test

Use Python 3.10 or newer:

```sh
git clone https://github.com/open-horizon-labs/microWakeWord.git
cd microWakeWord
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
```

Training and synthesis run on CPU, but a GPU is useful for long jobs.

## 2. Define the corpus

Put accepted pronunciations, close phrases, and unseen probes in a versioned
recipe. Keep positives and negatives separate. Record speaker, session, source,
and split for every clip. Never let a held-out clip enter training.

The general Kizz recipe is [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml).
Its tooling supports Piper and designed voices, but the old Kizz Piper corpus is
not part of the current clean-slate baseline. Re-admit it only as a named,
isolated experiment.

The clean-slate Kizz run used 4,706 examples from a hashed manifest:

- 1,848 positives from AssemblyAI, Deepgram, ElevenLabs, Kokoro,
  device-rendered anchors, and speech/music/noise overlays;
- 2,858 negatives from LibriSpeech and MUSAN speech, music, and noise;
- 15 household false wakes held out from every training and threshold decision.

See the [C/D comparison](../recipes/kizz/CLEAN_SLATE_V2_C_D_RESULTS.md) for
the exact evidence and limits.

## 3. Generate a conventional recipe corpus

This is the general Piper-compatible path, not the clean-slate Kizz baseline:

```sh
git clone https://github.com/open-horizon-labs/piper-sample-generator.git ../piper-sample-generator
git -C ../piper-sample-generator checkout 35d2f2d
mkdir -p models
curl -L \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  -o models/en_US-libritts_r-medium.pt

python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --generator-source ../piper-sample-generator \
  --output work/kizz/generated \
  --batch-size 16
```

The generator keeps speaker IDs disjoint across train, validation, and test.
`generation-manifest.json` records the recipe, model, settings, and hashes.

## 4. Build features and train

Review phrase spans before feature extraction. For device recordings:

```sh
python tools/apply_phrase_spans.py \
  --corpus work/device-corpus \
  --spans work/reviewed-phrase-spans.json

python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --output work/kizz/features \
  --background-indoor work/backgrounds/indoor/train \
  --background-outdoor work/backgrounds/outdoor/train \
  --impulses room-impulses

python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained \
  --output work/kizz/training_parameters.yaml

python -m microwakeword.model_train_eval \
  --training_config work/kizz/training_parameters.yaml \
  --train 1 mixednet
```

The feature manifest must identify the exact recipe, source files, split, and
augmentation settings. Source counts must not silently determine batch mix;
use [`write_stratified_training_config.py`](../tools/write_stratified_training_config.py)
for controlled comparisons.

## 5. Evaluate before hardware

Select the cutoff on validation only. Then evaluate each pronunciation,
confusable phrase, unseen probe, negative source, and device cohort separately:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --recipe recipes/kizz/corpus.yaml \
  --split test \
  --cutoff 0.96 \
  --output work/kizz/metrics.json

python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --state-mode carry_until_detection \
  --cutoff 0.96 \
  --output work/kizz/device_metrics.json
```

`carry_until_detection` preserves model state between audio chunks. It is the
runtime-like check; `reset_per_capture` is a useful diagnostic, not a substitute.

For a teacher → student experiment:

1. Bind raw files, features, teacher weights, teacher outputs, student, and
   reports to one manifest hash.
2. Qualify the teacher on held-out positives, untouched negative exposure, and
   held-out device false wakes before distillation.
3. Evaluate the quantized student in the same stateful streaming mode firmware
   will use.
4. Stop if any gate fails. Do not tune against frozen test clips.

The aligned Kizz distillation record is
[`TEACHER_DISTILLATION_ALIGNED_V1.md`](../recipes/kizz/TEACHER_DISTILLATION_ALIGNED_V1.md).
Its student passed a short offline screen but failed live StackChan precision.
That exact artifact was rejected; deployment reverted to the ESP-IDF wake-word
model.

## 6. Collect device evidence

The enrollment service is separate from the production voice gateway:

```sh
python tools/run_enrollment_service.py \
  --corpus work/device-corpus \
  --port 8091 \
  --public-base-url http://trainer-host:8091
```

Every attempt should be retained, including detector misses. False-wake audio
is quarantined evidence; it does not enter `device-corpus.json` or training
until a human explicitly promotes it.

Validate and build device features with:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus
python tools/build_device_corpus_features.py \
  --corpus work/device-corpus \
  --output work/device-features
```

## 7. Hardware qualification

Do not call a model qualified because its desktop metrics look good. Before
deployment, record the exact model hash, firmware commit, cutoff, target, room,
speakers, sessions, wake attempts, misses, false wakes, and ambient exposure.

The artifact must pass all of these:

- held-out positive recall;
- representative long-form household/TV/noise exposure;
- zero or contract-compliant false wakes on held-out device evidence;
- exact quantized streaming behavior;
- physical StackChan or target-device testing.

The current Kizz student failed the last item. Keep the ESP-IDF wake-word model
deployed until a replacement passes the full set.

## Further reading

- [Kizz recipe and tool map](../recipes/kizz/README.md)
- [Experiment ledger](../recipes/kizz/EXPERIMENTS.md)
- [Clean-slate C/D results](../recipes/kizz/CLEAN_SLATE_V2_C_D_RESULTS.md)
- [Teacher/student salvage report](../recipes/kizz/SALVAGE_TEACHER_STUDENT_V1.md)
- [Device enrollment](device_enrollment.md)
- [Data sources](data_sources.md)
- [Techniques and references](techniques.md)
