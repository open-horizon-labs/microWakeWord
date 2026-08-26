# Kizz Control phoneme teacher/student v1

Status: experimental, not a deployment approval.

This recipe uses one offline full-context phoneme teacher to supervise one
compact streaming student. The firmware decision is intended to be a single
20-logit ordered-state model plus deterministic CTC path scoring; it is not a
second neural verifier.

## Data and teacher recipe

The source gate is pronunciation-first. Provider positives are audited with
Allosaurus against the canonical `Kizz Control` phone prefix, with the audit
bound to the exact source-manifest hash and applied to every train, validation,
and test split. Ambiguous or collision pronunciations (`Kids`, `Kiss`, `His`,
missing initial /k/, and similar readings) are excluded from the canonical
positive set. The clean aligned corpus contains 142 accepted canonical
provider positives out of 225 aligned inputs, plus 16 accepted device positives
in train and 12 in validation:

- train: 91 positives and 1,154 negatives;
- validation: 41 positives and 1,078 negatives;
- test: 26 positives and 1,083 negatives.

The resulting corpus is
`/private/tmp/kizz-training/kizz-control-c1/phoneme-distill-corpus-v5-pronunciation-clean`.
Its corpus SHA-256 is `84d9b945a0fe155f7c9a88cfe83f74ef3dac9b3bbc0e58ab7ca87460638259f2`.

The teacher adaptation manifest is the deduplicated pronunciation-clean
manifest at
`/private/tmp/kizz-training/kizz-control-c1/phoneme-teacher-adaptation-v4-pronunciation-clean-dedup/manifest.json`
(SHA-256
`d1c0f090465effe26cd4a6cb5488aca6a351aaa796ffeba1ef47031bfe9818f9`).
Materialized `device_channel_positive` corpus rows are excluded when the
qualified device captures are added, preventing duplicate device positives.

The teacher is the local full-model adaptation
`full24-proj-dense-240-symmetric-margin-full`: all 24 encoder layers are
adapted with a dense feature projection, gradient checkpointing, 240 steps,
and symmetric collision-margin supervision. The positive collision loss
requires canonical speech to beat every declared collision path; the negative
collision loss requires an explicit collision to beat the canonical path.
Generic max-collision supervision alone is not sufficient because it can train
against an easier, unrelated collision.

The compact phone contract is fixed and hash-bound. Canonical phones are
`k, ɪ, z, ə, n, t, ɹ, oʊ, l, d`; the 20 outputs are blank, those phones,
declared collision phones, and `OTHER`. Declared collision paths include
`kidskontrol`, `kiskontrol`, `thiskontrol`, `hiskontrol`, `kizkontroller`,
`kizkontrold`, `kizpatrol`, `kitchenkontrol`, and `kidskantrol`.

## Deterministic guards

The pipeline must fail before model construction or distillation when any of
these contracts drift:

1. The pronunciation audit must match the exact source manifest and must use
   `gate_mode=all`, not only the reserved holdout.
2. Every accepted positive must retain its source/provider/voice identity and
   split; qualification evidence cannot enter training or distillation.
3. Device positives must be capture-qualified and must not be duplicated by
   materialized corpus rows.
4. The teacher must pass its clip/anchor gate before student distillation.
5. Feature, posterior, sequence-score, and streaming-window caches must hash
   their source corpus, teacher weights, decoder contract, and lock files.
6. Positive providers and variants are sampled by a declared schedule, not by
   filesystem abundance. The v11 schedule uses equal AssemblyAI, Deepgram,
   ElevenLabs, and Kokoro shares, with clean/device/overlay variants and equal
   negative groups for public speech, phonetic collisions, device collisions,
   and no-speech/noise.
7. Thresholds are selected from validation only. Test, device anchors, and the
   100-hour continuous corpus are evaluation evidence, never threshold-tuning
   data.

These guards are implemented primarily by:

- `tools/audit_kizz_control_source_pronunciations.py`;
- `tools/build_kizz_phoneme_distillation_corpus.py`;
- `tools/build_kizz_phoneme_teacher_adaptation_manifest.py`;
- `tools/adapt_kizz_phoneme_teacher.py`;
- `tools/distill_kizz_phoneme_student.py`;
- `tools/qualify_kizz_phoneme_student.py`; and
- `tools/package_kizz_phoneme_student_firmware.py`.

## Teacher qualification

The teacher qualification report is
`/private/tmp/kizz-training/kizz-control-c1/phoneme-teacher-adaptation-v4-pronunciation-clean-dedup/full24-proj-dense-240-symmetric-margin-full/qualification.json`.

The v11 distillation guard is bound specifically to the teacher's
adaptation-threshold reports, not merely to that strict summary:

- clip/anchor report:
  `full24-proj-dense-240-symmetric-margin-full/qualification-adaptation-threshold.json`,
  SHA-256
  `6c4d382f79927e1d1237db3816e9ef8389e9bea131d0964f9bbd3771824f7142`;
- continuous report:
  `full24-proj-dense-240-symmetric-margin-full/continuous-100h-adaptation-threshold.json`,
  SHA-256
  `dbe8278ce668d979792a68ccdd2fa93b48e22607fdf7fee1609ddbb18168009f`.

Both paths are relative to the pronunciation-clean teacher-adaptation root
above. Their hashes are part of the distillation provenance contract.

At the clip/anchor operating point it passes:

- validation threshold: `-1.1675665378570557`;
- aligned test: 24/26 accepted (`92.3077%`);
- natural positives: 22/24 accepted;
- false-wake anchors: 0/62 accepted.

The teacher checkpoint is `best/model.safetensors`, SHA-256
`bdd12478178eb3d71c0f699498226d5dd687c037635208c8f1b39de037afdb28`.

The strict continuous report is not green: on 100.0000888 hours it produced
6 false accepts, FAPH `0.06`, but the 95% upper bound was `0.1184238513`, above
the `0.1` gate. A separate adaptation-threshold report found a zero-false-
accept point at threshold `-0.2657920718193054` with upper bound
`0.0299572961`, but that is a changed operating point and does not erase the
failed original teacher report. The teacher is therefore useful for the
experiment and for distillation provenance, but is not a fully deployment-
qualified teacher under the strict continuous gate.

## v11 distillation and evaluation

v11 is recorded by
`/private/tmp/kizz-training/kizz-control-c1/phoneme-student-v11-multichannel-checkpoint-sweep-v1`.
The distillation recipe is `kizz_control_compact_ctc_distillation_v6`, 3,000
steps, batch size 64, with:

- a compact streaming student with 48 first-convolution filters, four
  96-filter pointwise blocks, mix-convolution kernels 3/5/7/9, stride 3, and
  20 output logits;
- forward-sum CTC teacher sequence targets for non-explicit collision rows;
- hard targets, pairwise ranking, tail ranking, CTC loss, and explicit positive
  and negative collision-path margins;
- 24,000 realized samples in each of four negative groups;
- 24,000 realized positive samples for each AssemblyAI, Deepgram, ElevenLabs,
  and Kokoro provider, with equal clean/device/overlay variant shares;
- ambient/device hard negatives, connected mined and random speech, and
  public-speech negatives.

The deterministic forward-sum decoder uses beta `0.0`, one-frame score hops,
window lengths `[19, 23, 27, 32, 39, 47, 54]`, and contract SHA-256
`a277b32a2411775d9c2c8a2a7dea0bf6ce26c52ab6f8bafe958ec059fd967d5e`.
The compact-phone contract hash is
`e1e272ff15a60b178a4fe3c9cf52b9796498aa4352e4cdf59dc78d122aa7d3e3`.

### Multichannel checkpoint selection

The original v11 checkpoint selection was validation-only and used the ordered
key:

```text
(qualified, zero_false_accept_recall, recall, separation)
```

That selector was later found not to be deployment-equivalent: it consumed raw
logits as log probabilities and maximized over internal endpoints. It is
preserved here as v11 provenance, not as the current contract. The corrected
selector and rescored A/B/C/D tournament are documented in
[`DISTILLATION_TOURNAMENT_V1.md`](DISTILLATION_TOURNAMENT_V1.md).

Ties resolve to the earliest evaluation step. The multichannel validation set
contains 1,131 items: 1,119 clean items and 12 device positives. The selected
checkpoint is step 500, with weights SHA-256
`a190a42fe4fddef9100c3176f095bf0c56fd1c5ef952fff7f25835caefcaec35`.

## Step-500 hardware candidate

The step-500 INT8 artifact is an experimental hardware candidate only:

- artifact: `kizz_control_student_streaming_int8.tflite`;
- bytes: `83,696`;
- model SHA-256:
  `1753026675313fb0e77d2a93c002c1942e0069d8ce6459437ba8584666f1c140`;
- input: `int8[1,3,40]`;
- output: `uint8[1,1,20]`;
- packaged firmware model name: `hiphi_kizz_ordered.tflite`;
- experimental packaged raw threshold:
  `-2.1612006732086204`;
- package contract header SHA-256:
  `dd97604b08cc13100dfdf62594422fbd04ebded93efe19ad4b97a704b1b66395`;
- package provenance SHA-256:
  `f2e21686cc4155e27e19ee2a91f2007e422b6b8dbb1384449ae58c4821ef3db0`;
- bound student qualification SHA-256:
  `c791c7e0e187d0bbaad615b271f1d3d78ff4ce81a2a2af2aa8991e5f415bb255`.

The initial 90%-recall qualification sweep scored 53/53 eligible validation
positives but produced 7 false accepts, equivalent to `8.9223753` FAPH on the
short validation exposure. At the selected zero-false-accept threshold
`-2.1612006732086204`, validation recall was 38/53 (`71.6981%`). Re-evaluation
with the repaired streaming evaluator produced:

- held-out aligned test: 11/26 (`42.3077%`);
- StackChan target-channel replays: 11/24 (`45.8333%`);
- locked device false-wake anchors: 0/62 accepted;
- continuous negatives: 6 events in `100.0000888` hours, FAPH `0.06`, 95%
  upper bound `0.1184238513`.

Five files produced the six continuous events: one LibriVox speech file, one
RFM music file, two events in one HD Classical file, one FMA music file, and
one additional HD Classical file. They are next-round hard-negative evidence,
not training input for this frozen evaluation. The report correctly says:

```text
aligned_test_recall_below_minimum
target_channel_recall_below_minimum
continuous_negative_gate_failed
```

The package is retained as an explicitly unqualified hardware experiment. The
packager records the failed gates and continuous-negative statistics rather
than silently binding the firmware to an older report that omitted them. It
must not be called a production model, and its threshold must not be treated as
a qualified deployment operating point.

## Hardware integration status

The firmware pins `micro_wake_word_standalone` commit
`13581e0aacc2e73b3aa384b43463c953517cfe07`, whose ordered-state runtime uses
the same forward-sum CTC algorithm, beta, score windows, and raw threshold as
the package. The wake input is exclusively the StackChan microphone path:
`M5.Mic.record()` feeds `ExternalAudioMicrophone`, which feeds the compact
student. No Mycroft decision and no second neural verifier participates.

The runtime retains its frontend and TFLite allocations across the shared-I2S
pause/re-arm cycle. This avoids the allocator fragmentation that previously
ended in `ESP_ERR_NO_MEM` during Wi-Fi PHY work.

The exact hardware-evaluation firmware flashed on August 26, 2026 was built
from firmware commit `ad642f55e34345762fdda7585d2e7eb6b36d8f6e` and has ELF
SHA-256 `079a8bb761fa1d36be64eb3710985ac6d3856338e3231b0d5cd336196fdb53f1`
and application binary SHA-256
`b67f35953a62f341da14adbacef6944485b78f3f61d6cc24b47d77d2354c7e6c`.
It booted at 240 MHz with the WebRTC VAD frontend and reported the packaged
model hash and threshold above. An accepted held-out target-channel positive
was played from the Mac speaker and captured by the StackChan microphone. The
exact final image crossed the `0.700` display cutoff at `0.717`, entered the
existing UHC voice session, quarantined the observation, received the expected
single-prompt non-command response, and re-armed without reconstructing the
tensor arena. This replay also exercised the server-terminal-before-local-
commit ordering: the firmware stopped the remaining audio flush, suppressed
the redundant device commit, deferred microphone ownership until the blocking
send returned, and then resumed microphone frames and wake scoring without the
prior AFE ring-buffer-full flood.

This is a hardware-path and lifecycle check, not model qualification;
household recall and false-wake behavior still need direct observation.
