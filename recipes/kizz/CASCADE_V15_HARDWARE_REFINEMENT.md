# Kizz Control v15 physical-hard-negative refinement

This is the hardware-refinement pass for the
[v10 three-stage Kizz Control cascade](CASCADE_V10_RECIPE.md). V10 remains the
long-duration reference; v15 retrains only the candidate-triggered compact
verifier. The continuous detector and final ordered verifier remain frozen.

```text
permissive ordered detector (continuous)
                  |
                  v
compact ReLU6 INT8 verifier (every candidate, physically adapted)
                  |
                  v
independent ordered verifier (sparse final vote)
```

The separate students are deliberate. The detector is optimized for recall,
the compact verifier rejects microphone-channel false candidates cheaply, and
the ordered verifier supplies an independent final boundary. A single student
was worse at this combination of jobs.

## Exact v15 hardware result

The selected compact verifier is seed `2066`, has 15,793 parameters and about
2.80 million MACs per candidate, and uses threshold
`-0.5345823287963869`. Its SHA-256 is
`9b8a8963f5619b045cf724d6ce0dacedbc71b1dbc7448e5a089bbc99667246f5`.

| Exact StackChan v0.2 evidence | Result |
| --- | ---: |
| Post-flash physical wake replay | 12/12 accepted |
| Same adversarial 25-minute schedule, v10 | 17 false accepts |
| Same adversarial 25-minute schedule, v15 | 5 false accepts (70.6% fewer) |
| Fresh untouched negative guard | 0/20 candidates accepted |
| Candidates reaching ordered verifier | 14/79 (17.7%) |
| Detector hop p99 | about 6 ms |
| Compact verifier physical latency | 94.4–107.0 ms |
| Ordered verifier physical latency | 293.4–330.2 ms |
| Cascade duty during adversarial replay | about 5.63% |
| Audio overflow / partial read or write | 0 / 0 |
| Compact arena | 82,480 / 98,304 PSRAM bytes |
| Internal heap low-water | 32 bytes |

The 25-minute schedule is deliberately dense with near phrases and must not be
converted into a normal-world false-accepts-per-hour claim. The 32-byte heap
low-water is a real qualification risk even though the run had no reboot,
overflow, or audio corruption.

## Durable workspace and immutable guards

Use local SSD or directly attached NVMe. Do not put corpora, features,
checkpoints, or reports in `/tmp`.

```sh
export KIZZ_WORKSPACE=/Volumes/Training/kizz-control
export KIZZ_PORT=/dev/cu.usbmodem2101
mkdir -p "$KIZZ_WORKSPACE"/{device,models,reports,runs}
```

Before training, create and hash two disjoint manifests:

- a train-only physical-hard-negative manifest whose audio may be consumed;
- a sealed unseen guard whose sources and room captures are not opened until
  the model, threshold, firmware binary, and positive schedule are frozen.

Keep every attempted capture, including detector misses. Bind source audio,
selection manifests, device revision, room/session, playback gain, model,
threshold, firmware binary, and final reports by SHA-256. Once a suite has been
used to choose a model, call it a regression suite rather than a fresh test.

## 1. Capture physical hard negatives

Start the device enrollment service described by the v10 recipe, connect the
exact StackChan target, and replay a locked mix of near phrases, ordinary
speech, music, and ambient audio. Vary playback level and preserve at least 2.3
seconds of real pre-roll.

```sh
.venv/bin/python tools/capture_kizz_control_hard_negative_replays.py \
  --source-manifest "$KIZZ_WORKSPACE/manifests/hard-negatives.json" \
  --corpus "$KIZZ_WORKSPACE/device/hard-negatives-train" \
  --selection "$KIZZ_WORKSPACE/device/hard-negatives-train/selection.json" \
  --service-url http://TRAINING_HOST_IP:8091 \
  --device-id stackchan-DEVICE_ID \
  --device-profile m5stack_stackchan_k151_cores3_room_scale_v2 \
  --volume 0.35 --volume 0.45 --volume 0.55 \
  --repeats 2 --lead-seconds 2.5
```

Add only qualified train captures to the train source manifest/features and
detector traces described in the v10 recipe. Never add the sealed guard. Build
the detector-conditioned 260-frame candidate corpus:

```sh
.venv/bin/python tools/build_kizz_candidate_verifier_dataset.py \
  --source-manifest "$KIZZ_WORKSPACE/manifests/candidate-sources-v15.json" \
  --source-features "$KIZZ_WORKSPACE/features/candidate-sources-v15" \
  --detector-traces "$KIZZ_WORKSPACE/traces/detector-v15" \
  --locked-holdout-manifest "$KIZZ_WORKSPACE/manifests/locked-holdouts.json" \
  --output "$KIZZ_WORKSPACE/candidate-verifier-dataset-v15" \
  --pre-context-frames 220 --post-context-frames 39
```

V15 also added six qualified room-captured positive variants from a weak voice,
spanning source tempo 0.96–1.06 and playback gain -4 to +6 dB. This improves
physical diversity without flooding training with synthetic duplicates.

## 2. Train a seed tournament

Hash `corpus.json`, then launch independent seeds in parallel. The hard-negative
share is within the 75% negative portion of each batch, not of the whole batch.
The implementation rejects incompatible sampling modes and caps this share at
50% to prevent a small mined set from dominating training.

```sh
CORPUS_SHA256=$(shasum -a 256 \
  "$KIZZ_WORKSPACE/candidate-verifier-dataset-v15/corpus.json" | awk '{print $1}')

for seed in 1157 1258 1359 1460 1561 1662 1763 1864 1965 2066; do
  .venv/bin/python tools/train_kizz_candidate_verifier.py \
    --dataset "$KIZZ_WORKSPACE/candidate-verifier-dataset-v15" \
    --corpus-sha256 "$CORPUS_SHA256" \
    --output "$KIZZ_WORKSPACE/runs/v15-seed-$seed" \
    --steps 6000 --seed "$seed" \
    --model-variant compact_relu6 \
    --augmentation-profile strong \
    --device-robustness-profile int8_lsb1 \
    --negative-sampling-share 0.75 \
    --negative-group-sampling proportional_example \
    --physical-hard-negative-share 0.02 &
done
wait
```

Promote only seeds that retain full voice-disjoint validation and recorded
physical-regression recall. Compare their independently quantized artifacts on
the consumed physical negatives; do not inspect the sealed guard:

```sh
.venv/bin/python tools/compare_kizz_physical_hard_negative_verifiers.py \
  --candidate-corpus "$KIZZ_WORKSPACE/candidate-verifier-dataset-v15/corpus.json" \
  --device-corpus "$KIZZ_WORKSPACE/device/hard-negatives-train/device-corpus.json" \
  --verifier seed2066="$KIZZ_WORKSPACE/models/seed2066/firmware-artifact.json" \
  --threshold -0.5345823287963869 \
  --output "$KIZZ_WORKSPACE/reports/seed-comparison.json"
```

## 3. Convert and bind the INT8 graph

Convert each finalist with the same limits, including the predeclared 0.2-logit
integer safety margin. Reject conversion if any threshold decision changes.

```sh
.venv/bin/python tools/convert_kizz_candidate_verifier.py \
  --training-report "$KIZZ_WORKSPACE/runs/v15-seed-2066/training-report.json" \
  --weights "$KIZZ_WORKSPACE/runs/v15-seed-2066/best.weights.h5" \
  --output "$KIZZ_WORKSPACE/models/seed2066" \
  --quantization-mode int8 \
  --quantization-logit-safety-margin 0.2 \
  --threshold-decision-mismatch-fraction 0
```

The selected conversion had maximum/mean logit error `0.2783/0.1091`, maximum
probability error `0.0165`, and zero threshold mismatches. Firmware uses a fixed
C execution schedule with ESP-NN kernels and static PSRAM arena reuse. Preserve
the optimized kernels: generic generated C is not automatically faster.

Bind the model SHA-256 and threshold in firmware. Run startup reference checks
for the frozen detector and ordered verifier. For the compact graph, retain the
TFLite reference outputs and separately bind the deterministic exact-hardware
AOT fingerprint; random stress vectors can amplify ESP-NN integer-rounding
differences even when silence, saturation, and physical decisions agree.

## 4. Build, flash, and qualify the exact artifact

Build and flash the release profile, then record the binary and ELF hashes. Do
not qualify a model file separately from the flashed firmware that contains it.

```sh
source "$IDF_PATH/export.sh"
idf.py -B build-stackchan build
idf.py -B build-stackchan -p "$KIZZ_PORT" flash
```

Run three provenance-bound physical schedules against the flashed artifact:

```sh
.venv/bin/python tools/run_kizz_physical_playback.py \
  --serial-port "$KIZZ_PORT" \
  --schedule "$KIZZ_WORKSPACE/manifests/positive-replay.json" \
  --output-dir "$KIZZ_WORKSPACE/reports/positive-final"

.venv/bin/python tools/run_kizz_physical_playback.py \
  --serial-port "$KIZZ_PORT" \
  --schedule "$KIZZ_WORKSPACE/manifests/adversarial-negative-25m.json" \
  --output-dir "$KIZZ_WORKSPACE/reports/negative-25m-final"

.venv/bin/python tools/run_kizz_physical_playback.py \
  --serial-port "$KIZZ_PORT" \
  --schedule "$KIZZ_WORKSPACE/manifests/unseen-guard.json" \
  --output-dir "$KIZZ_WORKSPACE/reports/unseen-guard-final"
```

The runner counts the platform-level `Kizz wake detected on-device:` event as
end-to-end acceptance and records detector, compact, ordered, load, heap, queue,
drop, overflow, crash, and reboot telemetry. Positive schedules are mandatory:
rejection-only timing never exercises the expensive final path.

## Promotion contract

A candidate is flash-ready only when all of these are bound to one provenance
record:

1. corpus, model, threshold, firmware binary, schedule, and report hashes;
2. full frozen validation and recorded-regression recall;
3. INT8 conversion limits and zero decision mismatches;
4. exact-hardware startup checks;
5. post-flash positive replay recall;
6. an exact before/after negative schedule;
7. a still-untouched negative guard;
8. accepted-path latency, duty, arena, heap, queue, drop, and overflow evidence.

V15 satisfies those gates for wake-path replay. It does not yet establish
multi-human recall, normal STT/command-handler coexistence, or comfortable
internal-heap margin; those remain product-level qualification work.
