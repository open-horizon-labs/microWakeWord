# Device corpus enrollment

The enrollment service runs on the LAN, outside UHC. Configure each device with
its full trainer WebSocket URL; never infer it from the production voice
endpoint.

## Start and exercise without hardware

Replace these illustrative IDs, profile, phrase, and workspace with your own.

Initialize `device-corpus.json` before starting the service. Schema 2 registers
each speaker once and assigns that identity to one split:

```json
{
  "schema_version": 2,
  "corpus_id": "my-device-corpus-v1",
  "device_profiles": {
    "my_microphone_profile_v1": {
      "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "sample_format": "s16le",
        "frontend": "my_frontend",
        "gain_profile": "default",
        "preprocessing": {}
      }
    }
  },
  "speakers": {
    "speaker-a": {
      "kind": "human",
      "age_group": "adult",
      "split": "train",
      "identity_verified": true
    }
  },
  "captures": []
}
```

Do not create a second ID for the same person to move a recording into another
split. The enrollment service rejects unregistered speakers and split changes.

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

Queue a bounded attempt through the trainer's HTTP control API:

```sh
curl -X POST http://trainer-host:8091/v1/captures \
  -H 'content-type: application/json' \
  -d '{
    "capture_id":"speaker-a-mic-train-001",
    "device_id":"simulated-mic-1",
    "device_profile":"my_microphone_profile_v1",
    "phrase":"My Wake Phrase",
    "pronunciation":"primary_reading",
    "truth":"positive",
    "source":"human",
    "speaker_id":"speaker-a",
    "session_id":"speaker-a-mic-train",
    "split":"train",
    "duration_ms":2000,
    "conditions":{"distance_cm":100,"room":"kitchen"}
  }'
```

The service commands only the addressed device. The device records the window
regardless of its provisional wake decision, then posts signed-16 PCM to the
command's `upload_url`. `detected` records the old model's outcome; it never
filters a capture. Set `--public-base-url` to an address devices can reach. It
is explicit because the operator may use a loopback control URL while the
microphone is elsewhere on the LAN.

## Device protocol

On `GET /v1/device`, a microphone device sends:

```json
{
  "type": "hello",
  "device_id": "mic-1",
  "device_profile": "my_microphone_profile_v1",
  "firmware_sha": "18433e0",
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "sample_format": "s16le",
    "frontend": "m5unified_mic",
    "gain_profile": "default",
    "preprocessing": {"dc_block": true}
  }
}
```

`device_id` identifies an instance. `device_profile` identifies an acoustic
domain shared by equivalent hardware and preprocessing. The
[`device-profiles.json`](../device-profiles.json) catalog records microphone,
corpus, and enrollment status.

Evaluate one model across microphone-equipped profiles before splitting by
device. Catalog presence, enrollment support, and collected corpus are separate
statuses. Kizz is a reference path, not a framework limit.

The server sends `training_capture` with an absolute `upload_url`. The device
posts one `application/octet-stream` body there with `X-Device-ID` and a boolean
`X-Detected` header. Attempts are 0.5–5 seconds and at most 160,000 bytes. The
older WebSocket upload messages remain accepted for existing clients; new
embedded clients should keep the WebSocket for control and use HTTP for bulk
audio.

Devices that cannot buffer or upload the body in one request may add
`X-Audio-Offset` and `X-Audio-Total` to send ordered segments. Repeating an
acknowledged segment is safe when its bytes match. A pending segmented upload
survives control-WebSocket reconnects and expires after five minutes without an
audio segment. The server snapshots the device profile and firmware when it
queues the capture, so the HTTP upload does not depend on the control socket
remaining connected.

`POST /v1/wake-config` can also carry an optional `audio_preprocessing` object.
The service passes these scalar frontend settings to the addressed device
without assigning hardware-specific meaning. A device must validate, persist,
and report applied settings in its acknowledgement and subsequent `hello`.
Changing preprocessing creates a different acoustic `device_profile`; never
append captures made under the new settings to the old profile.

## Corpus guarantees

`device-corpus.json` requires:

- registered, immutable audio profiles;
- unique capture IDs and content hashes;
- mono 16 kHz signed-16 PCM WAVs;
- explicit truth and train/validation/test assignment;
- explicit capture source (`human`, `synthetic_playback`, `ambient`, or
  `simulated`);
- registered speaker provenance, age cohort, and immutable split assignment;
- no speaker or session crossing splits;
- both provisional detector hits and misses.

Phrase-bearing captures may include a measured span:

```json
"phrase_span": {"start_ms": 160, "end_ms": 960}
```

When present, feature building uses the phrase plus 250 ms of surrounding audio.
This keeps a wake attempt inside the training example even when it occurs in the
middle of a longer recording. The span describes the intended phrase, regardless
of whether the provisional detector fired; it may come from human annotation or
an alignment service. The original WAV and its hash remain unchanged.

Apply reviewed spans in one validated update:

```sh
python tools/apply_phrase_spans.py \
  --corpus work/device-corpus \
  --spans work/reviewed-phrase-spans.json
```

The same spans can anchor the synthetic quality mask described in
[Usage](USAGE.md#4-build-features-and-train). This lets the
recorded corpus prevent generated speech with implausible timing or truncation
risk from entering feature generation.

Production feature building accepts live human and synthetic-playback positives
and hard negatives in the training split. Validation and test require human
speech; ambient-negative splits require ambient recordings. Simulation remains
available for pipeline tests but is excluded from production features.

Build device features and include them in training:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus
python tools/build_device_corpus_features.py \
  --corpus work/device-corpus --output work/device-features
python tools/write_recipe_training_config.py \
  --workspace work/my-wake-word --train-dir work/my-wake-word/trained \
  --device-features-dir work/device-features \
  --device-truncation-strategy random \
  --output work/my-wake-word/training_parameters.yaml
```

The builder preserves manifest splits and writes aligned derivatives under the
feature output directory. Evaluation reports truth, phrase, pronunciation,
profile, speaker, session, and provisional detector outcome. Random temporal
sampling keeps wake phrases that are not aligned to a recording edge in the
training pass. Prefer measured phrase spans for long captures; edge truncation
should be used only when the capture protocol itself guarantees alignment.

For a release gate, add `--qualification` to
`tools/evaluate_device_corpus_model.py`. It accepts only a complete test split
with positive, hard-negative, ambient-negative, and registered human evidence.
`--split all` cannot qualify a model.

To reduce manual review while investigating wake false positives, correlate
quarantined observations with UHC's retained STT telemetry:

```sh
uv run --python 3.12 python tools/analyze_wake_observations.py \
  --corpus work/device-corpus \
  --output work/wake-observation-triage.json
```

The report records provider transcripts, timing, and a weak label. It never
promotes audio into `device-corpus.json`; human review is still required before
promotion into a hard negative.
