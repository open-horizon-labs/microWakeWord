# Techniques and references

This ledger distinguishes the upstream methods retained by the fork from the
Open Horizon Labs additions. “Reference” means either a primary publication or
the upstream project that defines the method. Fork-specific product and corpus
policies cite their implementation and experiment evidence instead of implying
that they came from a paper.

## Detection and model architecture

| Technique | Use here | Implementation | Reference |
| --- | --- | --- | --- |
| 16 kHz `micro_speech` frontend | Convert mono PCM into 40 integer-compatible features on the same frontend family used at inference. | [`microwakeword/audio/spectrograms.py`](../microwakeword/audio/spectrograms.py) | [TensorFlow Lite Micro `micro_speech`](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/examples/micro_speech) |
| Streaming keyword spotting | Train with complete feature windows, then convert to a stateful model that consumes new feature slices. | [`microwakeword/layers/stream.py`](../microwakeword/layers/stream.py), [`microwakeword/model_train_eval.py`](../microwakeword/model_train_eval.py) | [Streaming Keyword Spotting on Mobile Devices](https://arxiv.org/abs/2005.06720), [Google Research `kws_streaming`](https://github.com/google-research/google-research/tree/master/kws_streaming) |
| Mixed depthwise convolutions | Use multiple temporal kernel sizes efficiently in the small streaming network. | [`microwakeword/mixednet.py`](../microwakeword/mixednet.py) | [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595) |
| Integer quantization | Export a smaller, faster TensorFlow Lite streaming artifact using representative training features. | [`microwakeword/utils.py`](../microwakeword/utils.py) | [TensorFlow model optimization: post-training integer quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training) |
| Complete-stride calibration | Exclude only an incomplete trailing streaming stride from representative calibration, rather than discarding valid terminal frames. | [`microwakeword/utils.py`](../microwakeword/utils.py), [`tests/test_kizz_recipe.py`](../tests/test_kizz_recipe.py) | Fork correctness fix; verified by regression test. |

## Corpus generation and augmentation

| Technique | Use here | Implementation | Reference |
| --- | --- | --- | --- |
| Multi-speaker synthetic speech | Produce enough controlled positive and confusable samples to begin model development before device enrollment. | [`tools/generate_recipe_samples.py`](../tools/generate_recipe_samples.py) | [`piper-sample-generator`](https://github.com/rhasspy/piper-sample-generator) |
| Speech-rate, synthesis-noise, and speaker interpolation variation | Cycle explicit Piper parameters across each phrase instead of relying on one voice or cadence. | [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) | [`piper-sample-generator` generator options](https://github.com/rhasspy/piper-sample-generator#generator) |
| Time and frequency masking | Regularize feature windows during training. | [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py), [`microwakeword/train.py`](../microwakeword/train.py) | [SpecAugment](https://arxiv.org/abs/1904.08779) |
| Noise, gain, filtering, pitch, and distortion augmentation | Vary level and acoustic/channel characteristics during feature creation. | [`tools/build_recipe_features.py`](../tools/build_recipe_features.py) | [Audio augmentation for speech recognition](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html), [`audiomentations`](https://iver56.github.io/audiomentations/) |
| Background mixing and room impulse responses | Mix representative noise/music and convolve speech with measured room responses when supplied. | [`tools/build_recipe_features.py`](../tools/build_recipe_features.py) | [Audio augmentation for speech recognition](https://www.isca-archive.org/interspeech_2015/ko15_interspeech.html); corpus provenance is listed in [Data sources](data_sources.md). |
| Manifest and content hashes | Make recipe/model provenance explicit and reject a generated tree that no longer matches its recipe. | [`tools/generate_recipe_samples.py`](../tools/generate_recipe_samples.py), [`tools/build_recipe_features.py`](../tools/build_recipe_features.py) | Fork reproducibility contract. |
| Selectable feature roots | Rebuild or compare one phrase cluster without silently changing the complete recipe manifest or the negative corpus. | [`tools/build_recipe_features.py`](../tools/build_recipe_features.py), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Fork experiment-isolation control. |
| Pinned dependency ranges and frontend API compatibility | Keep the tested TensorFlow/audio stack installable while accepting the current `pymicro-features` binding signature. | [`setup.py`](../setup.py), [`microwakeword/audio/audio_utils.py`](../microwakeword/audio/audio_utils.py) | Fork compatibility work; regression coverage is in [`tests/test_kizz_recipe.py`](../tests/test_kizz_recipe.py). |

## Recipe-specific collision design (Kizz)

| Technique | Use here | Implementation | Reference |
| --- | --- | --- | --- |
| One full-phrase wake class | Train natural readings of **HiPhi Kizz** together; keep bare **Kizz** out of the positive class. | [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) | Fork product policy based on rejected bare-word experiments in [`EXPERIMENTS.md`](../recipes/kizz/EXPERIMENTS.md). |
| Explicit pronunciation variants | Represent plausible spoken readings of `HiPhi` as labeled positive phrases. | [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) | Fork corpus design; evaluate each label separately. |
| Near-word hard negatives | Oversample `Kizz`, `kids`, `kiss`, `quiz`, and related speech that must not wake the device. | [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Fork collision policy; failures and revisions are recorded in [`EXPERIMENTS.md`](../recipes/kizz/EXPERIMENTS.md). |
| Conjunction-mining negatives | Pair valid HiPhi-like prefixes with wrong suffixes and wrong prefixes with `Kizz`, forcing evidence from both halves of the phrase. | [`recipes/kizz/corpus.yaml`](../recipes/kizz/corpus.yaml) | Fork anti-shortcut design; no external-method claim. |
| Unseen pronunciation probes | Hold plausible spellings out of training and score them after export to test acoustic generalization. | [`recipes/kizz/probes.yaml`](../recipes/kizz/probes.yaml) | Fork evaluation design. The probes are not training data. |
| Phrase-labeled evaluation | Report each pronunciation and foil independently so an aggregate cannot hide a weak cohort. | [`tools/evaluate_recipe_model.py`](../tools/evaluate_recipe_model.py) | Fork acceptance contract. |
| Independent streaming state | Reset recurrent/streaming state between unrelated clips so one sample cannot affect the next. | [`tools/evaluate_recipe_model.py`](../tools/evaluate_recipe_model.py), [`microwakeword/test.py`](../microwakeword/test.py) | Fork evaluation correctness fix; covered by [`tests/test_kizz_recipe.py`](../tests/test_kizz_recipe.py). |

## Training and checkpoint selection

| Technique | Use here | Implementation | Reference |
| --- | --- | --- | --- |
| Weighted source sampling | Control how often positive, general-negative, confusable, and real-device sources appear in batches. | [`microwakeword/data.py`](../microwakeword/data.py), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Inherited microWakeWord mechanism; fork recipe supplies explicit weights. |
| Per-source error penalties | Make errors on confusable and device samples more costly without pretending those corpora are larger than they are. | [`microwakeword/data.py`](../microwakeword/data.py), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Inherited microWakeWord mechanism; fork recipe supplies explicit penalties. |
| Confusables as train and evaluation evidence | Sample hard negatives during training and include the same labeled archive as a zero-sampling checkpoint-selection source. | [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Fork model-selection design. |
| Ambient false-accept-first selection | Minimize estimated ambient false accepts per hour to the configured gate before maximizing viable recall. | [`microwakeword/train.py`](../microwakeword/train.py), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Inherited microWakeWord selection mechanism; the included recipe's metric and gate are policy. |
| Retained checkpoint candidates | Evaluate every 1,000 steps in the included recipe's 30,000-step schedule and retain each candidate's weights for comparison. | [`microwakeword/train.py`](../microwakeword/train.py), [`tools/write_recipe_training_config.py`](../tools/write_recipe_training_config.py) | Fork experiment-support change. |

## Device corpus and qualification

| Technique | Use here | Implementation | Reference |
| --- | --- | --- | --- |
| Separate enrollment service | Expose a dedicated HTTP/WebSocket training API whose full URL is configured independently of UHC and the production voice endpoint. | [`microwakeword/enrollment.py`](../microwakeword/enrollment.py), [`tools/run_enrollment_service.py`](../tools/run_enrollment_service.py) | Fork deployment contract documented in [Device enrollment](device_enrollment.md). |
| Hardware-independent simulator | Exercise routing, capture, persistence, and miss retention without physical hardware. | [`tools/simulate_enrollment_device.py`](../tools/simulate_enrollment_device.py), [`tests/test_enrollment.py`](../tests/test_enrollment.py) | Fork test architecture. |
| Capture independently of provisional detection | Store every commanded attempt and record `detected` only as an outcome, including wake attempts the current detector missed. | [`microwakeword/enrollment.py`](../microwakeword/enrollment.py), [`tests/test_enrollment.py`](../tests/test_enrollment.py) | Fork corpus contract. |
| Versioned device/audio profiles | Identify microphone frontend, gain, preprocessing, and product acoustic domain instead of hard-coding a product. | [`device-profiles.json`](../device-profiles.json), [`microwakeword/device_profiles.py`](../microwakeword/device_profiles.py) | Fork multi-device qualification design. |
| Immutable, leak-safe corpus manifest | Require unique capture IDs and hashes, explicit truth/splits, registered profiles, and no speaker or session crossing splits. | [`microwakeword/device_corpus.py`](../microwakeword/device_corpus.py), [`tests/test_device_corpus.py`](../tests/test_device_corpus.py) | Fork data-integrity contract. |
| Predetermined device splits | Build features from manifest-assigned train/validation/test captures without re-randomizing them. | [`tools/build_device_corpus_features.py`](../tools/build_device_corpus_features.py) | Fork evaluation-integrity design. |
| Multi-dimensional held-out reporting | Group device results by truth, phrase, pronunciation, profile, and provisional detector outcome. | [`tools/evaluate_device_corpus_model.py`](../tools/evaluate_device_corpus_model.py) | Fork qualification contract. |
| Shared-model-first comparison | Evaluate one model across all microphone-equipped product corpora, then split models only when held-out evidence shows a profile-specific failure. | [`device-profiles.json`](../device-profiles.json), [Device enrollment](device_enrollment.md) | Fork product strategy; not a claim that all profiles are already qualified. |

## Public training data

The framework can use FSD50K, FMA, WHAM!, VOICES, Common Voice, and DiPCo for
negative or ambient evaluation data. Their primary papers, project pages, and
licenses are collected in [Data sources](data_sources.md). Dataset availability
does not grant permission to redistribute derived audio or models; check every
source's current terms before publishing an artifact.
