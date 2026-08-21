# Usage

This guide covers the reproducible path supplied by the Open Horizon Labs fork:
install the trainer, generate a recipe corpus, build features, train and evaluate
a quantized streaming model, then qualify it with held-out device recordings.
The HiPhi Kizz recipe is the worked example.

## 1. Install the trainer

Use Python 3.10 or newer. A local virtual environment keeps TensorFlow and audio
dependencies isolated:

```sh
git clone https://github.com/open-horizon-labs/microWakeWord.git
cd microWakeWord
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install piper-sample-generator
```

Run the tests before starting a long generation or training job:

```sh
python -m unittest discover -s tests -v
```

GPU acceleration is strongly recommended for corpus generation and training,
but the commands are the same on CPU.

## 2. Obtain the Piper generator model

The Kizz recipe uses the LibriTTS-R multi-speaker generator published by
[`piper-sample-generator`](https://github.com/rhasspy/piper-sample-generator):

```sh
mkdir -p models
curl -L \
  https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt \
  -o models/en_US-libritts_r-medium.pt
```

If the installed package does not expose the current generator module, clone
that repository beside this one and add
`--generator-source ../piper-sample-generator` to generation commands.

## 3. Inspect and generate the Kizz corpus

[`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) is the source of truth
for phrases, sample counts, pronunciation variants, confusables, and Piper
variation settings. Inspect the complete plan without creating audio:

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

Generation skips a phrase directory that already contains the exact requested
count and refuses a directory with surplus files. The feature builder later
rejects incomplete or mismatched corpora. The resulting
`generation-manifest.json` records the recipe and generator-model hashes.

## 4. Build on-device-compatible features

Use representative room recordings and room impulse responses when available.
Both arguments are repeatable, and omitting them disables those two augmentation
classes:

```sh
python tools/build_recipe_features.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --output work/kizz/features \
  --background room-backgrounds \
  --impulses room-impulses
```

The builder validates the generation manifest before it creates the exact
40-feature `micro_speech` representation used by the device model. It preserves
deterministic train, validation, and test splits.

## 5. Supply general negative features

The recipe config expects prepared negative feature archives under:

```text
work/kizz/negative-datasets/
├── dinner_party/
├── dinner_party_eval/
├── no_speech/
└── speech/
```

Use the upstream
[pre-generated feature dataset](https://huggingface.co/datasets/kahrendt/microwakeword)
or build equivalent archives from appropriately licensed sources. The starter
[`basic_training_notebook.ipynb`](../notebooks/basic_training_notebook.ipynb)
demonstrates upstream acquisition and feature preparation; source and license
notes are in [Data sources](data_sources.md). Do not reuse evaluation audio as
training data.

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

The generated config gives confusable speech explicit sampling and penalty
weight and uses it again as evaluation-only evidence. Checkpoint
selection first minimizes ambient false accepts per hour to its configured
target, then maximizes viable recall. Training exports the selected quantized
streaming model to:

```text
work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite
```

To resume an interrupted run from its checkpoint, repeat the training command
with `--restore_checkpoint 1`. Review the learning-rate schedule before doing
so; restoring weights does not decide an appropriate new schedule for you.

## 7. Evaluate every trained pronunciation

Choose a cutoff from held-out validation evidence, then keep it fixed for the
test pass. The `0.96` below illustrates the command shape; it is not a recommended
Kizz threshold:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --split test \
  --cutoff 0.96 \
  --output work/kizz/pronunciation_metrics.json
```

The report separates each positive spelling and hard-negative phrase. Do not
replace those cohorts with one aggregate acceptance rate.

[`recipes/kizz/probes.yaml`](../recipes/kizz/probes.yaml) contains plausible
HiPhi readings deliberately absent from training. Generate it into a separate
directory and evaluate it with `--split all` to test acoustic generalization.
Never merge probe audio into the training corpus after observing its score; add
new held-out probes first if the training recipe changes.

## 8. Exercise enrollment without hardware

The training endpoint is standalone. It is not UHC, it is not the production
voice endpoint, and it may run on another host.

Start the service and a simulated microphone device in separate terminals:

```sh
python tools/run_enrollment_service.py \
  --corpus work/device-corpus \
  --port 8091

python tools/simulate_enrollment_device.py \
  --endpoint ws://trainer-host:8091/v1/device \
  --device-id simulated-stackchan-1 \
  --device-profile m5stack_stackchan_k151_cores3_v1 \
  --no-detected
```

Follow [Device corpus enrollment](device_enrollment.md) to queue bounded
positive, hard-negative, and ambient attempts. The simulator is a first-class
fixture: it proves an attempt marked as missed by the provisional detector is
still retained by the corpus contract.

## 9. Add real-device evidence

After collecting real audio, validate it and build features without changing
its declared splits:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus

python tools/build_device_corpus_features.py \
  --corpus work/device-corpus \
  --output work/device-features

python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained-with-devices \
  --device-features-dir work/device-features \
  --output work/kizz/device_training_parameters.yaml
```

Train a new model from that configuration. Evaluate the frozen model against
the held-out device test split. As above, replace the illustrative `0.96` with
the cutoff frozen from validation:

```sh
python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained-with-devices/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --cutoff 0.96 \
  --output work/kizz/device_test_metrics.json
```

The report groups results by truth, phrase, pronunciation, device profile, and
whether the source detector hit or missed. Evaluate one shared model across all
available microphone profiles first. Fork a device-specific model only when
held-out results show a material acoustic-domain failure.

## 10. Qualification checklist

A candidate is ready for physical qualification only when:

- recipe, generator model, training config, and corpus hashes are recorded;
- no speaker or recording session crosses train, validation, and test splits;
- held-out positive pronunciations and confusable phrases meet explicit gates;
- ambient false accepts are measured over representative duration;
- provisional detector misses are present in the device evaluation;
- the exact quantized artifact is flashed and accepted on each claimed target.

Synthetic-only results belong in the
[experiment log](../recipes/kizz/EXPERIMENTS.md), not in a hardware-qualified
release claim.

## Troubleshooting

- **Generation stops on an existing phrase directory:** its WAV count differs
  from the recipe. Move that phrase directory aside and regenerate it; do not
  blend old and new recipes.
- **Feature building rejects the corpus:** the recipe hash, phrase directories,
  or WAV counts no longer match `generation-manifest.json`.
- **Training cannot open a feature set:** populate every negative archive named
  in the generated YAML, or deliberately edit the YAML and record the changed
  experiment.
- **Aggregate recall looks good but a reading fails:** use the per-phrase report
  and unseen probes. Aggregate recall is not the acceptance gate.
- **A real wake attempt is absent:** enrollment must record the bounded attempt
  independently of the provisional wake decision. `detected` is evidence, not
  an inclusion filter.
