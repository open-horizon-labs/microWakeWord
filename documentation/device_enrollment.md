# Device corpus enrollment

The enrollment service runs on the LAN, outside UHC. Configure each device with
its full trainer WebSocket URL; never infer it from the production voice
endpoint.

## Start and exercise without hardware

Replace these illustrative IDs, profile, phrase, and workspace with your own.

```sh
python tools/run_enrollment_service.py --corpus work/device-corpus --port 8091

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
    "speaker_id":"speaker-a",
    "session_id":"speaker-a-mic-train",
    "split":"train",
    "duration_ms":2000,
    "conditions":{"distance_cm":100,"room":"kitchen"}
  }'
```

The service commands only the addressed device. The device records the window
regardless of its provisional wake decision, then sends a `training_sample`
header and signed-16 PCM. `detected` records the old model's outcome; it never
filters a capture.

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

The server sends `training_capture`; the device returns `training_sample` and
one binary PCM message. Attempts are 0.5–5 seconds and at most 160,000 bytes.

## Corpus guarantees

`device-corpus.json` requires:

- registered, immutable audio profiles;
- unique capture IDs and content hashes;
- mono 16 kHz signed-16 PCM WAVs;
- explicit truth and train/validation/test assignment;
- no speaker or session crossing splits;
- both provisional detector hits and misses.

Build device features and include them in training:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus
python tools/build_device_corpus_features.py \
  --corpus work/device-corpus --output work/device-features
python tools/write_recipe_training_config.py \
  --workspace work/my-wake-word --train-dir work/my-wake-word/trained \
  --device-features-dir work/device-features \
  --output work/my-wake-word/training_parameters.yaml
```

The builder preserves manifest splits. Evaluation reports truth, phrase,
pronunciation, profile, and provisional detector outcome.
