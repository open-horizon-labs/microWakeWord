# Retraining the Kizz Control verifier from physical false wakes

This is the repeatable counterexample loop used to turn false wakes observed on
StackChan into the v15 compact verifier. It assumes the
[v10 candidate-conditioned cascade](CASCADE_V10_RECIPE.md) already exists and
re-trains only its compact middle stage. The continuous detector and final
ordered verifier stay frozen so a before/after result has one interpretation.

```text
false trigger on the exact device
              |
              v
reviewed, provenance-bound 5 s microphone recording
              |
              v
candidate windows emitted by the frozen detector
              |
              v
bounded hard-negative sampling + positive recall repairs
              |
              v
INT8 conversion -> flash -> same physical replay -> unseen guard
```

Do not put this workspace in `/tmp`. Use local SSD or directly attached NVMe.
The examples below use:

```sh
export KIZZ_WORKSPACE=/Volumes/Training/kizz-control
export KIZZ_DEVICE_CORPUS="$KIZZ_WORKSPACE/device/hard-negatives-train"
mkdir -p "$KIZZ_WORKSPACE"/{device,manifests,models,reports,runs}
```

## 1. Preserve the failure before changing anything

There are two supported inputs. A spontaneous wake observation is quarantined
by the enrollment service and requires human review. A controlled replay starts
from a locked source manifest whose truth is already known. Neither path makes
an unreviewed wake training-eligible automatically.

For spontaneous observations, run enrollment against a durable corpus:

```sh
.venv/bin/python tools/run_enrollment_service.py \
  --corpus "$KIZZ_DEVICE_CORPUS" \
  --host 0.0.0.0 --port 8091 \
  --public-base-url http://TRAINING_HOST_IP:8091
```

After the test, freeze the time-bounded observations before inspecting or
promoting individual files:

```sh
.venv/bin/python tools/freeze_wake_observation_snapshot.py \
  --corpus "$KIZZ_DEVICE_CORPUS" \
  --output "$KIZZ_WORKSPACE/reports/false-wake-snapshot-001" \
  --since 2026-01-01T12:00:00Z \
  --frozen-at 2026-01-01T12:30:00Z \
  --reviewer REVIEWER \
  --review-note "Confirmed no Kizz Control command was spoken"

.venv/bin/python tools/analyze_wake_observations.py \
  --corpus "$KIZZ_DEVICE_CORPUS" \
  --reliability-url http://127.0.0.1:8088/voice/reliability \
  --window-seconds 120 \
  --output "$KIZZ_WORKSPACE/reports/false-wake-analysis-001.json"
```

Listen to the complete context and promote only a confirmed false wake. The
speaker and session must already belong to the requested split, and a session
may not cross splits:

```sh
.venv/bin/python tools/promote_false_wake.py \
  --corpus "$KIZZ_DEVICE_CORPUS" \
  --observation OBSERVATION_ID \
  --reviewer REVIEWER \
  --split train \
  --speaker-id REGISTERED_TRAIN_SPEAKER \
  --session-id room-2026-01-01-a \
  --reason "Reviewed 5 s context; ordinary speech, no target phrase"

.venv/bin/python tools/validate_device_corpus.py \
  --corpus "$KIZZ_DEVICE_CORPUS"
```

Promotion copies the WAV into `audio/`, appends a `hard_negative` row, retains
the original quarantined observation, and binds the review and audio SHA-256.
Use a separate, sealed corpus for an unseen guard; never promote guard audio
into the training corpus after looking at its scores.

The compact trainer includes promoted `hard-negative-*` rows through normal
proportional negative sampling. Its special bounded reservation recognizes only
the deterministic `hardneg-*` IDs written by the controlled replay tool below.
Do not rename capture IDs to change that behavior: reproduce the reviewed
failure through a locked replay matrix when it needs reserved sampling weight.

## 2. Reproduce failures with a locked physical replay matrix

The successful v11 collection was built from 19 sources: 15 connected
near-phrase clips and four excerpts around an instrumental-classical false-wake
region. Each source manifest row bound `source_id`, truth, source kind, speaker,
description, absolute or manifest-relative path, and audio SHA-256. The matrix
used gains `0.35`, `0.45`, and `0.55`, with two repeats at each gain.

```json
{
  "examples": [
    {
      "source_id": "near-kids-control-part1",
      "speaker_id": "physical-near-speech-01",
      "truth": "hard_negative",
      "source": "synthetic_playback",
      "phrase": "Kids control the volume. The kids controlled the game.",
      "path": "/absolute/path/near-kids-control-part1.wav",
      "audio_sha256": "LOWERCASE_SHA256",
      "conditions": {
        "physical_failure_family": "kids-control"
      }
    }
  ]
}
```

Start enrollment, connect the exact StackChan training firmware, and capture:

```sh
.venv/bin/python tools/capture_kizz_control_hard_negative_replays.py \
  --source-manifest "$KIZZ_WORKSPACE/manifests/hard-negative-sources.json" \
  --corpus "$KIZZ_DEVICE_CORPUS" \
  --selection "$KIZZ_DEVICE_CORPUS/hard-negative-selection-v1.json" \
  --service-url http://TRAINING_HOST_IP:8091 \
  --device-id stackchan-DEVICE_ID \
  --device-profile m5stack_stackchan_k151_cores3_room_scale_v2 \
  --volume 0.35 --volume 0.45 --volume 0.55 \
  --repeats 2 --duration-ms 5000 --lead-seconds 2.5
```

The selection is written and hashed before playback. Re-running the command
resumes missing deterministic capture IDs but rejects a changed selection.
Every source hash is checked before `afplay` runs. Keep at least 2.3 seconds of
real pre-roll; an earlier short-lead attempt padded the decisive pre-trigger
context with zeros and was not valid training evidence.

The historical run produced 114 five-second captures: 90 near-speech and 24
ambient/music. Those recordings yielded 71 frozen-detector candidates. The 43
captures with no detector event were retained in provenance but correctly
produced no verifier row.

## 3. Mine only the windows the frozen detector would forward

Do not train the verifier on arbitrary windows from all 114 recordings. Extend
the immutable v10 candidate corpus with only events from the exact frozen
detector, threshold, and frontend used in firmware:

```sh
BASE_DATASET="$KIZZ_WORKSPACE/candidate-verifier-dataset-v10"
BASE_SHA256=$(shasum -a 256 "$BASE_DATASET/corpus.json" | awk '{print $1}')

.venv/bin/python tools/extend_kizz_candidate_verifier_with_device_corpus.py \
  --base-dataset "$BASE_DATASET" \
  --base-corpus-sha256 "$BASE_SHA256" \
  --device-corpus "$KIZZ_DEVICE_CORPUS/device-corpus.json" \
  --detector-metadata "$KIZZ_WORKSPACE/models/detector/firmware-artifact.json" \
  --detector-model "$KIZZ_WORKSPACE/models/detector/detector.tflite" \
  --detector-threshold-report "$KIZZ_WORKSPACE/models/detector/threshold.json" \
  --output "$KIZZ_WORKSPACE/candidate-verifier-dataset-physical-v1" \
  --top-k-per-file 4 \
  --target-phrase "Kizz Control"
```

This tool opens audio only for `train` rows. Validation/test rows and any row
marked `locked_holdout` or `deployment_anchor` remain quarantined. Its ledger
must show the expected input hashes, zero opened holdouts, zero unexpected
duplicates, and the count of candidate-producing and no-candidate captures.

The audited run bound these immutable inputs:

| Input | SHA-256 |
| --- | --- |
| v10 base `corpus.json` | `f390066c5fcfa26a1fd0e589afc8789ddd754d2db90fc71286b6377f3b89e7ad` |
| device `device-corpus.json` | `a55a65789680dace58b9ef962beb926584df1e69b055f2db4066272da5c1f826` |
| detector INT8 TFLite | `f07d2c010fba020e923c23734e54ba8e86751dfd1b0f23a018eb5ff79b969ae3` |
| detector metadata | `f2aa435611fed316764283533d7ea075cb047414fda0eaad9615d76a5472fe25` |
| detector threshold report | `089def64219ac3180d1b474602d3b290237f27bd49c79649d426fdda273fe336` |

These hashes are an audit record, not defaults for a new run. A new run must
bind its own actual files and reject drift.

## 4. Train with bounded emphasis, then repair recall

The first v11 comparison exposed the main trap. Proportional sampling retained
all 51/51 validation and 33/33 test positive candidates, but still accepted
24/71 physical negatives for seed 1056. Uniform group sampling rejected as
many as 71/71 consumed physical negatives, but recall fell to 40/51 validation
and 23/33 test positives for seed 1157. Memorizing the small physical set was
not a usable win.

The working policy was:

- 75% of each 64-example batch is negative;
- sample negative examples proportionally rather than giving every source
  group equal weight;
- reserve only 2% of those negative slots for `hardneg-` physical captures;
- use `compact_relu6`, strong training-only feature augmentation, and the
  `int8_lsb1` robustness profile;
- require 100% candidate-conditioned validation recall when choosing a
  checkpoint and threshold;
- run independent seeds because this small targeted set has high variance.

```sh
DATASET="$KIZZ_WORKSPACE/candidate-verifier-dataset-physical-v1"
CORPUS_SHA256=$(shasum -a 256 "$DATASET/corpus.json" | awk '{print $1}')

for seed in 1157 1258 1359 1460 1561 1662 1763 1864 1965 2066; do
  .venv/bin/python tools/train_kizz_candidate_verifier.py \
    --dataset "$DATASET" \
    --corpus-sha256 "$CORPUS_SHA256" \
    --output "$KIZZ_WORKSPACE/runs/physical-v1-seed-$seed" \
    --steps 6000 --batch-size 64 --seed "$seed" \
    --conditional-recall-floor 1.0 \
    --model-variant compact_relu6 \
    --augmentation-profile strong \
    --device-robustness-profile int8_lsb1 \
    --negative-sampling-share 0.75 \
    --negative-group-sampling proportional_example \
    --physical-hard-negative-share 0.02 &
done
wait
```

Physical-negative adaptation initially exposed physical positive misses. We did
not lower the recall gate or tune on the test split. V12 added the missed
positive as train-only device evidence; v14 added qualified room-positive
diversity; v15 added six qualified weak-voice room captures spanning source
tempo 0.96–1.06 and playback gain -4 to +6 dB. Each addition used the same
device-corpus extension path and quality gate, leaving the immutable validation,
test, and unseen guards unchanged.

The bounded-emphasis sweep tried physical shares `0.02`, `0.03`, `0.05`,
`0.10`, and `0.20`. A `0.02` share was retained: even `0.05` produced seeds
with only 30/33 or 31/33 test-positive recall. Do not choose the share solely by
how completely it rejects the consumed false wakes.

## 5. Convert, compare, and qualify without leakage

Select float finalists using validation recall plus the already-consumed
physical regression set, then convert each finalist to INT8 with a predeclared
safety margin. Reject any threshold-decision mismatch:

```sh
.venv/bin/python tools/convert_kizz_candidate_verifier.py \
  --training-report "$KIZZ_WORKSPACE/runs/physical-v1-seed-2066/training-report.json" \
  --weights "$KIZZ_WORKSPACE/runs/physical-v1-seed-2066/best.weights.h5" \
  --output "$KIZZ_WORKSPACE/models/seed2066" \
  --quantization-mode int8 \
  --quantization-logit-safety-margin 0.2 \
  --threshold-decision-mismatch-fraction 0
```

Score the quantized artifact at its frozen deployed threshold. The comparison
tool applies one shared `--threshold` to every `--verifier` in an invocation,
so models with different frozen thresholds require separate reports. Never
retune a threshold on the consumed physical failures:

```sh
.venv/bin/python tools/compare_kizz_physical_hard_negative_verifiers.py \
  --candidate-corpus "$DATASET/corpus.json" \
  --device-corpus "$KIZZ_DEVICE_CORPUS/device-corpus.json" \
  --verifier seed2066="$KIZZ_WORKSPACE/models/seed2066/firmware-artifact.json" \
  --threshold -0.5345823287963869 \
  --output "$KIZZ_WORKSPACE/reports/physical-seed2066.json"
```

Only after the model, threshold, and firmware binary are frozen may the unseen
guard be opened. Flash and run, in order:

1. a positive physical replay that exercises the accepted, expensive path;
2. the exact physical false-wake schedule used before retraining;
3. a source- and session-disjoint unseen negative guard;
4. a wake/STT/command coexistence test while recording latency, audio drops,
   queue depth, heap low-water, socket failures, crashes, and reboots.

The selected v15 seed 2066 retained 12/12 post-flash physical wakes, reduced
the same 25-minute adversarial schedule from 17 to 5 accepted false wakes, and
accepted 0/20 candidates from the sealed guard. That is the evidence for this
method; the consumed 71-candidate training set is not an unbiased precision
test and must never be reported as one.

## Stop conditions

Restart the evidence loop instead of promoting when any of these occurs:

- a source, recording, model, threshold, schedule, or report lacks a hash;
- pre-trigger context is zero-padded or shorter than the firmware window;
- an observation has not been human-reviewed or its truth is ambiguous;
- validation/test/guard audio was opened by a training step;
- physical negatives improve while physical positive recall regresses;
- a threshold was tuned after looking at the test split or unseen guard;
- the exact quantized artifact was not flashed and exercised on hardware.
