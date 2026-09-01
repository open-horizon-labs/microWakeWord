# Kizz Control three-stage cascade v9

This is the shortest reproducible path we know for the **Kizz Control** wake
phrase. It preserves the useful teacher/student work without replaying every
rejected experiment.

```text
16 kHz microphone audio
        |
        v
ordered INT8 detector, continuous, permissive threshold
        |
        | candidate plus 2.2 s pre-roll and 390 ms post-roll
        v
compact INT8 DS-CNN verifier, candidate only
        |
        | about 5.2% of detector candidates
        v
independent ordered-state verifier, candidate only
        |
        v
accept or reject "Kizz Control"
```

The detector and ordered verifier are frozen, provenance-bound students from
the earlier distillation lineage. V9 retrains the middle verifier with the
actual StackChan microphone channel. Retraining all three networks together
was not the winning approach.

## Measured reference result

| Evidence | Result |
| --- | ---: |
| Detector validation recall | 51/52 (98.08%) |
| Fresh StackChan-channel detector/compact recall | 12/12 |
| Locked continuous-negative exposure | 100.47 hours |
| Detector candidates | 19,105 |
| Candidates forwarded by compact verifier | 991 (5.19%) |
| Full-cascade false wakes | 39 |
| Observed false wakes/hour | 0.388 (about one per 2.6 hours) |
| One-sided 95% upper bound | 0.507/hour |

The maintainer accepted `0.388` false wakes/hour as the practical operating
point. The stricter formal gate—one-sided 95% upper bound at or below
`0.1/hour`—did **not** pass. Both facts are recorded. Host replay also does not
establish ESP32-S3 latency, memory, thermal, power, or soak behavior.

The exact models are in
[`reference-cascade-v9`](reference-cascade-v9/README.md). Their hashes and the
reference metrics are machine-readable in
[`control-cascade-v9.yaml`](control-cascade-v9.yaml).

## Storage and installation

The completed workspace occupied about 65 GB: 30 GB of public source material,
22 GB of fresh locked qualification audio, and several more gigabytes of
features, candidate datasets, traces, checkpoints, and reports. Start with at
least **100 GB free**. Use local SSD or directly attached NVMe for feature
arrays and training. Network storage is better suited to immutable archives and
backups than large random-access feature files.

Create the environment in the durable checkout, not in `/tmp`:

```sh
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

python3.12 -m venv .alignment-venv
.alignment-venv/bin/pip install --upgrade pip
.alignment-venv/bin/pip install -r requirements-kizz-teacher.txt

python3.12 -m venv .kokoro-venv
.kokoro-venv/bin/pip install --upgrade pip
.kokoro-venv/bin/pip install 'kokoro>=0.9.4' soundfile 'misaki[en]'
```

Install `ffmpeg` and `espeak-ng` with the host operating system's package
manager. The separate environments keep TensorFlow training, Torch alignment,
and Kokoro synthesis from forcing one incompatible dependency solve.

Choose the durable data workspace before downloading anything:

```sh
export KIZZ_WORKSPACE=/Volumes/Training/kizz-control-cascade-v9
```

Download the three public inputs from their official OpenSLR locations. The
commands are resumable; the freeze stages reject an archive whose official MD5
does not match, and the RIR stage also checks its pinned SHA-256:

```sh
mkdir -p "$KIZZ_WORKSPACE/public/archives" "$KIZZ_WORKSPACE/public"

curl -fL --retry 5 -C - \
  -o "$KIZZ_WORKSPACE/public/archives/musan.tar.gz" \
  https://www.openslr.org/resources/17/musan.tar.gz
curl -fL --retry 5 -C - \
  -o "$KIZZ_WORKSPACE/public/archives/rirs_noises.zip" \
  https://www.openslr.org/resources/28/rirs_noises.zip
curl -fL --retry 5 -C - \
  -o "$KIZZ_WORKSPACE/public/archives/train-clean-360.tar.gz" \
  https://www.openslr.org/resources/12/train-clean-360.tar.gz

tar -xzf "$KIZZ_WORKSPACE/public/archives/musan.tar.gz" \
  -C "$KIZZ_WORKSPACE/public"
unzip -q -o "$KIZZ_WORKSPACE/public/archives/rirs_noises.zip" \
  -d "$KIZZ_WORKSPACE/public"
tar -xzf "$KIZZ_WORKSPACE/public/archives/train-clean-360.tar.gz" \
  -C "$KIZZ_WORKSPACE/public"
```

This produces the default roots `public/musan`, `public/RIRS_NOISES`, and
`public/LibriSpeech/train-clean-360`. Their URLs, licenses, and expected
checksums are also recorded under `public_inputs` in the machine recipe.

Set the external credential and specialist-runtime locations:

```sh
export KIZZ_VOICE_ENV="$HOME/.config/open-horizon-labs/voice.env"
export KIZZ_KOKORO_PYTHON="$PWD/.kokoro-venv/bin/python"
export KIZZ_ALIGNMENT_PYTHON="$PWD/.alignment-venv/bin/python"
```

`KIZZ_VOICE_ENV` is a local dotenv file. Paid synthesis requires
`ASSEMBLYAI_API_KEY`, `DEEPGRAM_API_KEY`, and either `ELEVEN_LABS_API_KEY` or
`ELEVENLABS_API_KEY`. Do not commit that file. The runner refuses paid stages
unless `--allow-paid` is also present.

The recipe accepts environment variables or `--set NAME=VALUE` for MUSAN,
room responses, LibriSpeech, and device corpora. Inspect the resolved graph
before spending money or compute:

```sh
.venv/bin/python tools/run_kizz_control_cascade_recipe.py plan
```

## Run and resume

The runner executes argv arrays directly—never shell strings—and records the
command plus SHA-256 of every stage's evidence. It skips a stage only while its
recipe, command, outputs, and evidence hashes still match.

Generate and qualify four source families in parallel:

```sh
.venv/bin/python tools/run_kizz_control_cascade_recipe.py preflight \
  --stage curate_pronunciations --allow-paid

.venv/bin/python tools/run_kizz_control_cascade_recipe.py run \
  --stage curate_pronunciations --jobs 4 --allow-paid
```

`macOS say` is deliberately absent. It was useful for plumbing but added little
independent acoustic evidence beside AssemblyAI, Deepgram, ElevenLabs, and
Kokoro.

Build the public candidate corpus around events from the frozen detector:

```sh
.venv/bin/python tools/run_kizz_control_cascade_recipe.py run \
  --stage build_candidate_dataset --jobs 4 --allow-paid
```

After collecting the device corpora below, run through packaging:

```sh
.venv/bin/python tools/run_kizz_control_cascade_recipe.py preflight \
  --stage package_firmware_handoff --allow-paid

.venv/bin/python tools/run_kizz_control_cascade_recipe.py run \
  --stage package_firmware_handoff --jobs 8 --allow-paid
```

The eight locked negative shards run concurrently and checkpoint complete audio
files. Their merge rejects missing, duplicate, or drifted evidence. Use
`status` at any time; use `--force` only to replace a valid result:

```sh
.venv/bin/python tools/run_kizz_control_cascade_recipe.py status
```

Paid synthesis is blocked unless `--allow-paid` is present. Provider generation
is resumable and does not regenerate completed descriptors.

## Collect the StackChan microphone splits

Device captures are immutable inputs because playback level, placement,
firmware, microphone gain, and hardware cannot be reproduced by a blind build
command. The tools automate capture; a person or hardware-capable agent must
establish and record the fixture.

Start the independent enrollment service on a LAN-reachable host:

```sh
.venv/bin/python tools/run_enrollment_service.py \
  --corpus "$KIZZ_WORKSPACE/device/train-vol060" \
  --host 0.0.0.0 --port 8091 \
  --public-base-url http://TRAINING_HOST_IP:8091
```

Capture each provider through StackChan at volume `0.60`, then in a separate
session at `0.80`. The reference selected nine train examples per provider per
session and retained 54 captures after quality gates:

```sh
for provider in assemblyai deepgram elevenlabs kokoro; do
  .venv/bin/python tools/capture_kizz_control_adaptation_replays.py \
    --aligned-manifest "$KIZZ_WORKSPACE/manifests/source-pronunciation-curated-aligned.json" \
    --qualification-evidence "$KIZZ_WORKSPACE/device/fresh-test/qualification-evidence.json" \
    --corpus "$KIZZ_WORKSPACE/device/train-vol060" \
    --selection "$KIZZ_WORKSPACE/device/train-vol060/selection.json" \
    --service-url http://TRAINING_HOST_IP:8091 \
    --device-id stackchan-DEVICE_ID --device-profile stackchan-v0.2 \
    --provider "$provider" --per-provider 9 --volume 0.60
done
```

Repeat into `train-vol080` with `--volume 0.80`. Audit rather than hand-edit
rejected rows:

```sh
.venv/bin/python tools/audit_kizz_control_adaptation_replays.py \
  --corpus "$KIZZ_WORKSPACE/device/train-vol060/device-corpus.json" \
  --selection "$KIZZ_WORKSPACE/device/train-vol060/selection.json" \
  --qualification-evidence "$KIZZ_WORKSPACE/device/fresh-test/qualification-evidence.json" \
  --output "$KIZZ_WORKSPACE/device/train-vol060/quality.json" \
  --expected-split train
```

Collect validation separately with
`capture_kizz_control_adaptation_validation_replays.py`, audit it with
`audit_kizz_control_adaptation_validation_replays.py`, and compose repeated
attempts with `compose_kizz_control_adaptation_validation_replays.py`. The
threshold tool accepts validation captures only.

Reserve unused pronunciation-accepted test sources with
`prepare_kizz_fresh_device_qualification_inventory.py`. Capture them with
`capture_kizz_control_device_replays.py` and export immutable evidence with
`export_kizz_device_replays.py`. The finalizer refuses to open test audio until
the compact threshold report exists.

Record the exact source and selection hashes, StackChan target/revision,
firmware hash, device ID, distance, playback level, room/session, microphone
gain, and every attempt including misses. Keep train, validation, and test in
physically separate sessions. Never train from validation/test, and never tune
on the locked 100-hour corpus.

## Compact verifier contract

The middle model sees only 260-frame windows emitted by the frozen detector,
not arbitrary clips.

| Layer group | Channels / operation |
| --- | --- |
| Stem | 5x5 stride-2 Conv2D, 24 channels, ReLU6 |
| DS block 1 | 3x3 depthwise stride 2, 1x1 to 32 |
| DS block 2 | 3x3 depthwise stride 2, 1x1 to 48 |
| DS block 3 | 3x3 depthwise stride 2, 1x1 to 64 |
| DS block 4 | 3x3 depthwise stride 2, 1x1 to 96 |
| Head | flatten plus one logit |

It has 15,793 parameters and about 2.80 million MACs per candidate. Training
uses seed `1056`, 6,000 steps, 75% detector-triggered negatives, proportional
negative sampling, strong augmentation, ReLU6, and training-only INT8 fake
quantization/noise. Conversion emits fully INT8 TFLite and checks validation
equivalence. Device validation freezes the deployed threshold at `0.0`; test
and continuous audio cannot retune it.

## Firmware handoff and ESP-NN

`package_firmware_handoff` emits three models, metadata, thresholds, reports,
and `cascade.json`. It intentionally records
`physical_hardware_qualified: false`.

StackChan firmware uses fixed C execution schedules, static arenas, and ESP-NN
kernels. The topology is fixed while weights and quantization parameters come
from the bound model. This preserves Espressif SIMD; generic generated C can
lose more kernel performance than it saves in framework overhead.

Promotion requires the exact firmware artifact to pass:

1. all three startup AOT/reference equivalence checks;
2. logged model hashes and thresholds matching `cascade.json`;
3. fresh physical speaker-replay recall;
4. compact, ordered, and total candidate p99 latency;
5. detector/cascade duty leaving time for UI, Wi-Fi, STT, and commands;
6. arena, heap, queue, ring, and audio-drop telemetry;
7. at least 30 minutes of soak without reboot, lockout, or thermal trouble.

Host simulation proves geometry, decisions, sharding, and false-wake rate. It
cannot replace these hardware checks.

## Failed approaches and retained lessons

### One model could not do both jobs

The detector needed a permissive threshold for recall. Raising it reduced false
candidates but removed real device wakes. Keep it biased toward recall and use
later stages for rejection.

### Clean-source thresholds did not survive the microphone channel

The v6 compact host threshold rejected all 13 physical calibration attempts.
Lowering it restored 12/12 recall but forwarded 95.9% of candidates and yielded
2.18 false wakes/hour. Device-channel training at two playback levels reduced
forwarding to 5.19% while retaining 12/12 fresh recall.

### Rejection-only timing hid the expensive path

Early measurements exercised compact failures but not the ordered stage.
Performance tests must include candidates that pass compact verification.

### The old int16/xwide path starved the application

The old detector/verifier combination left StackChan effectively CPU-locked.
The compact ReLU6 INT8 graph, static arena reuse, table-based feature
conversion, one CPU-speed lock per verifier run, and sparse ordered invocation
are parts of the solution.

### More synthetic wake clips were not the answer

The useful clean source set was 598 renders: 190 AssemblyAI, 120 Deepgram, 96
ElevenLabs, and 192 Kokoro. Volume comes from randomized room response, gain,
placement, timing, and background overlays plus broad public negatives—not
thousands of near-duplicate wake phrases. Preserve clean originals and derive
variants reproducibly.

### Split and provenance mistakes made scores look better

Voice identities are assigned before synthesis. Device sessions are split
physically. Selection reads validation only; fresh test opens after freeze,
once. Locked audio is not mined until explicitly consumed and replaced by a new
lock.

### Temporary scripts are not a recipe

The experiment accumulated important tools in `/tmp`. They are now in the
durable checkout with tests. Keep code in Git and bulk data under
`KIZZ_WORKSPACE`; do not depend on temporary scripts or shell history.

## Exact reproduction versus a new run

The checked-in outer models reproduce the frozen detector and ordered verifier;
the compact reference reproduces the accepted v9 handoff. A new run reproduces
the **method and gates**, not necessarily the compact bytes, unless every WAV,
quality decision, device capture, public archive, package version, and input
hash matches the original lineage.

Never substitute the reference metrics for a new run's reports. Evidence hashes
and the package boundary are designed to expose that shortcut.
