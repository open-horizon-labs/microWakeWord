# Kizz wake-word research and Kizz Control recipe

The current implementation is the
[Kizz Control three-stage cascade v10](CASCADE_V10_RECIPE.md), refined on
physical StackChan failures by the
[v15 hardware pass](CASCADE_V15_HARDWARE_REFINEMENT.md). It combines a
high-recall ordered detector, a device-adapted compact verifier, and an
independent ordered verifier. The checked-in
[machine recipe](control-cascade-v10.yaml),
[`run_kizz_control_cascade_recipe.py`](../../tools/run_kizz_control_cascade_recipe.py),
and [reference artifacts](reference-cascade-v10/README.md) are the starting
point for a new run.

The v10 exact firmware retained 12/12 fresh held-out StackChan-channel
positives. It passed all three startup AOT/reference equivalence checks and ran
the continuous detector at about 8 ms p99 without ring overflow. On the locked
100.47-hour negative corpus it produced 23 false wakes (`0.229/hour`), with only
4.36% of detector candidates reaching the expensive ordered verifier.

The older **HiPhi Kizz** phrase remains stopped under phonetic-identifiability
evidence: intended positives and observed false wakes overlap with “Hi-Fi
Kids,” “High Five Kiss,” and nearby speech. V19, v34, clean-slate teachers, and
the failed single-student firmware path remain documented because their
failure boundaries shaped the cascade. See [Training reference](TRAINING_REFERENCE.md),
[Experiments](EXPERIMENTS.md), and the
[canonical-v3 phonetic gate](CANONICAL_V3_PHONETIC_GATE.md).

The corrected teacher → student run, including PCM-context preparation,
87→66 temporal alignment, qualification, distillation, stateful INT8 scoring,
experimental firmware integration, and the live-precision failure, is documented in
[TEACHER_DISTILLATION_ALIGNED_V1.md](TEACHER_DISTILLATION_ALIGNED_V1.md).
The later [Kizz Control distillation tournament](DISTILLATION_TOURNAMENT_V1.md)
implements temperature KD, sequence-conditioned occupation KD, intermediate
representation KD, and a bounded temporal-residual student. D3 was the best
float candidate, but no candidate passed the frozen promotion gate; therefore
none was quantized, packaged, or flashed.
The [clean-slate C/D comparison](CLEAN_SLATE_V2_C_D_RESULTS.md) explains why C
survived and D was rejected; the [salvage report](SALVAGE_TEACHER_STUDENT_V1.md)
is the restart contract.

### Current disposition

V10 is the long-duration Kizz Control reference and v15 is the active
hardware-refined middle-verifier method. The exact v15 firmware accepted 12/12
post-flash wake replays, reduced false accepts from 17 to 5 on the same
adversarial 25-minute schedule, and accepted 0/20 candidates on a fresh unseen
guard. A later production-profile correction disabled the optional enrollment
connection during normal use. The corrected firmware retained 11/12 physical
wakes with the live gateway and completed an event-gated physical
wake→STT→Roon volume command while preserving about 12 KiB of internal-heap
low-water. Extended multi-human, multi-room, and command-heavy soak tests remain
required before claiming complete product qualification. Do not treat v19,
v34, the clean-slate C teacher, or the earlier single-student score as current
evidence.

The later canonical-v3 C teacher reached only 9/22 held-out positives and
accepted 2/62 quarantined false wakes. A pinned pretrained IPA/CTC teacher then
reached 13/14 aligned tests but only 3/13 household positives and also accepted
2/62 false wakes from exact two-second pre-wake contexts; one decoded as the
canonical phone sequence. No Hi-Fi Kizz teacher may proceed to distillation or
firmware under this evidence.

The commands in this document exercise the general recipe tooling. They do not
recreate the reported clean-slate C/D run without its private audio, manifests,
feature caches, and checkpoints.

## What v19 established—and failed to establish

V19 combined evidence from different acoustic domains instead of letting the
largest directory determine training:

| Source group | Configured share | Realized share | Purpose |
| --- | ---: | ---: | --- |
| Piper positives | 15% | 15.625% | Broad speaker and pronunciation coverage |
| Piper hard negatives | 15% | 15.625% | Short-word and phrase collisions |
| Designed positives | 25% | 25.000% | Independent adult and teen synthetic voices |
| Designed negatives | 10% | 9.375% | Controlled confusable phrases in the same voice families |
| Kizz microphone positives | 15% | 14.063% | Adaptation to the real microphone and room channel |
| Targeted negatives | 20% | 20.313% | Kizz hard negatives, household speech, and background-only audio |

The training configuration used:

- a two-second, 16 kHz `micro_speech` input window;
- batches of 64 with deterministic seed `231`;
- binary focal cross-entropy with gamma `2.0`;
- one time mask up to four frames and one frequency mask up to three bins;
- 200 steps at `5e-6`, followed by 200 steps at `1e-6`;
- frozen batch-normalization statistics while adapting v18a weights;
- a MixedNet with stride `3`, 48 first-convolution filters, a five-frame first
  kernel, four 96-filter pointwise blocks, and MixConv kernels
  `[5]`, `[7,11]`, `[9,15]`, and `[23]`.

Checkpoint 100 was selected. Checkpoints 200 and 400 reduced confusable accepts
but lost too much replay recall, so the exported end-of-run model is not the v19
baseline. The deployed quantized artifact has SHA-256:

```text
76250d0cef49f893df4724ea6cce0e87b8a8d0d63cf10fbe23c0e624298871ff
```

Its initial weights, device audio, and generated corpora are not committed to
Git. Reproducing the artifact requires those inputs and their recorded hashes.
The checked-in recipe and tools reproduce the method; they do not manufacture
the missing private evidence.

V19 also predates one label correction: it trained `High Five Kizz` as a
positive. The current `corpus.yaml` treats that extra `/v/` sound as a hard
negative. The checked-in corpus therefore defines the next candidate, not a
byte-for-byte rebuild of v19.

## Tool map

This is the runnable path. The [training reference](TRAINING_REFERENCE.md)
also documents the eight designed ElevenLabs voices, 1,600 ElevenLabs renders,
67,150 Piper renders, Kokoro and macOS voice pilots, human captures, physical
speaker-to-Kizz re-recordings, Deepgram-assisted alignment, background sources,
augmentation settings, and the v1–v34 experiment lineage. The experiment ledger
also records the rejected ordered-state v1 run.

| Task | Tools |
| --- | --- |
| Generate speaker-split Piper speech | [`generate_recipe_samples.py`](../../tools/generate_recipe_samples.py) |
| Design and render age-labeled voices | [`design_elevenlabs_voice_catalog.py`](../../tools/design_elevenlabs_voice_catalog.py), [`add_labeled_voice_samples.py`](../../tools/add_labeled_voice_samples.py) |
| Review phrase spans and screen synthetic sources | [`apply_phrase_spans.py`](../../tools/apply_phrase_spans.py), [`build_synthetic_quality_mask.py`](../../tools/build_synthetic_quality_mask.py) |
| Prepare noise and build augmented features | [`prepare_background_corpus.py`](../../tools/prepare_background_corpus.py), [`build_recipe_features.py`](../../tools/build_recipe_features.py) |
| Collect and build device evidence | [`run_enrollment_service.py`](../../tools/run_enrollment_service.py), [`validate_device_corpus.py`](../../tools/validate_device_corpus.py), [`build_device_corpus_features.py`](../../tools/build_device_corpus_features.py) |
| Control training mixture and comparisons | [`write_recipe_training_config.py`](../../tools/write_recipe_training_config.py), [`write_stratified_training_config.py`](../../tools/write_stratified_training_config.py), [`audit_training_ablation.py`](../../tools/audit_training_ablation.py), [`audit_source_ablation.py`](../../tools/audit_source_ablation.py) |
| Evaluate models and select a cutoff | [`evaluate_recipe_model.py`](../../tools/evaluate_recipe_model.py), [`select_recipe_cutoff.py`](../../tools/select_recipe_cutoff.py), [`evaluate_device_corpus_model.py`](../../tools/evaluate_device_corpus_model.py) |
| Train/evaluate the ordered-state candidate | [`ordered_state_model.py`](../../microwakeword/ordered_state_model.py), [`evaluate_ordered_state.py`](../../tools/evaluate_ordered_state.py), [`report_ordered_state_resources.py`](../../tools/report_ordered_state_resources.py) |
| Qualify teachers and pre-screen replacement phrases | [`qualify_kizz_teacher_v3.py`](../../tools/qualify_kizz_teacher_v3.py), [`qualify_kizz_phoneme_teacher.py`](../../tools/qualify_kizz_phoneme_teacher.py), [`screen_kizz_wake_phrase_candidates.py`](../../tools/screen_kizz_wake_phrase_candidates.py) |

## Training workflow

### 1. Generate independent synthetic sources

Install this repository and the
[Open Horizon Labs `piper-sample-generator` fork](https://github.com/open-horizon-labs/piper-sample-generator),
then generate the Piper corpus:

```sh
python tools/generate_recipe_samples.py \
  --recipe recipes/kizz/corpus.yaml \
  --model models/en_US-libritts_r-medium.pt \
  --generator-source ../piper-sample-generator \
  --output work/kizz/generated \
  --batch-size 16
```

The generator reserves different speaker IDs for train, validation, and test.
Its LibriTTS-R speaker IDs do not provide reliable age labels, so add designed
voices whose immutable voice IDs remain in one split:

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

`corpus.yaml` balances accepted pronunciation variants and defines near-word,
prefix, suffix, and conjunction negatives. `counterexamples.yaml` retains
false wakes found in room use. `probes.yaml` contains unseen pronunciation
spellings for post-training evaluation, not training.

### 2. Screen sources before augmentation

Review phrase boundaries in real training recordings, then build the synthetic
quality mask:

```sh
python tools/apply_phrase_spans.py \
  --corpus work/device-corpus \
  --spans work/reviewed-phrase-spans.json

python tools/build_synthetic_quality_mask.py \
  --recipe recipes/kizz/corpus.yaml \
  --generated work/kizz/generated \
  --reference-corpus work/device-corpus \
  --output work/kizz/generated/quality-mask.json
```

The mask rejects silence, clipping, implausible positive spans, and clips that
would lose speech at the training-window boundary. Positive span limits come
from training-only human recordings with a 25% margin. It does not impose those
limits on hard negatives. Clean source audio remains unchanged.

### 3. Build multi-condition features

Prepare independently split backgrounds, then derive augmented training
features:

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

The builder derives gain, filtering, mild distortion, timing, noise, background,
and room-response variants while preserving clean holdouts. Augmentation
parameters and source provenance are recorded in the feature manifest.

Build real-device features from the manifest-assigned split rather than
re-randomizing captures:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus

python tools/build_device_corpus_features.py \
  --corpus work/device-corpus \
  --split train \
  --output work/kizz/device-features
```

The standalone [device enrollment service](../../documentation/device_enrollment.md)
collects every requested attempt, including detector misses. It is independent
of UHC and the production voice endpoint. The same contract supports other
microphone profiles without hard-coding StackChan.

### 4. Materialize and audit the training plan

First write a base microWakeWord configuration. Then expand a declarative
sampling plan into declared source groups:

```sh
python tools/write_recipe_training_config.py \
  --workspace work/kizz \
  --train-dir work/kizz/trained \
  --output work/kizz/base-training.yaml

python tools/write_stratified_training_config.py \
  --base-config work/kizz/base-training.yaml \
  --sampling-plan work/kizz/sampling-plan.yaml \
  --output work/kizz/training.yaml
```

The stratified writer reports configured shares, realized batch quotas, and
weighted pressure after penalties and class weights. Its balance guard can stop
a run whose effective negative pressure exceeds the declared limit. Use
`tools/audit_training_ablation.py` and `tools/audit_source_ablation.py` when
comparing recipes so an undeclared data or configuration change cannot masquerade
as the tested variable.

Train the quantized streaming model with the architecture pinned in the YAML:

```sh
python -m microwakeword.model_train_eval \
  --training_config work/kizz/training.yaml \
  --train 1 \
  mixednet
```

### 5. Select a cutoff without reading test

Score each phrase and cohort, then choose a cutoff from voice-held-out validation
data only:

```sh
python tools/evaluate_recipe_model.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --recipe recipes/kizz/corpus.yaml \
  --quality-mask work/kizz/generated/quality-mask.json \
  --split validation \
  --cutoff 0.70 \
  --output work/kizz/pronunciation-metrics.json

python tools/select_recipe_cutoff.py \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --generated work/kizz/generated \
  --recipe recipes/kizz/corpus.yaml \
  --quality-mask work/kizz/generated/quality-mask.json \
  --output work/kizz/cutoff-selection.json
```

Run device-corpus evaluation in both isolated-capture and runtime-like streaming
state modes. Neither report substitutes for a physical test:

```sh
python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --cutoff 0.70 \
  --sliding-window 1 \
  --state-mode reset_per_capture \
  --output work/kizz/device-test-isolated.json

python tools/evaluate_device_corpus_model.py \
  --corpus work/device-corpus \
  --model work/kizz/trained/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite \
  --split test \
  --cutoff 0.70 \
  --sliding-window 1 \
  --state-mode carry_until_detection \
  --output work/kizz/device-test-runtime.json
```

## Qualification bar

A candidate must pass held-out adult and child speakers, normal distances,
retained live misses, confusable speech, and long ambient playback on the
quantized artifact that will be flashed. Report results by speaker, age cohort,
pronunciation, session, and device profile. A clean synthetic test, a good
aggregate device score, or one successful room replay cannot qualify a model.

For the rationale and references behind each method, see
[Techniques and references](../../documentation/techniques.md). For the complete
framework path, see [Usage](../../documentation/USAGE.md).
