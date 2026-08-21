# Device corpus enrollment

The enrollment service is a standalone LAN service. It is not part of UHC and
its endpoint is never inferred from a production voice endpoint. A device is
given the full trainer WebSocket URL explicitly when it enters enrollment mode.

## Start and exercise without hardware

```sh
python tools/run_enrollment_service.py --corpus work/device-corpus --port 8091

python tools/simulate_enrollment_device.py \
  --endpoint ws://trainer-host:8091/v1/device \
  --device-id simulated-stackchan-1 \
  --device-profile m5stack_stackchan_k151_cores3_v1 \
  --no-detected
```

Queue one bounded attempt through the trainer's separate HTTP control API:

```sh
curl -X POST http://trainer-host:8091/v1/captures \
  -H 'content-type: application/json' \
  -d '{
    "capture_id":"speaker-a-stackchan-train-001",
    "device_id":"simulated-stackchan-1",
    "device_profile":"m5stack_stackchan_k151_cores3_v1",
    "phrase":"Hi-Fi Kizz",
    "pronunciation":"hi_fi",
    "truth":"positive",
    "speaker_id":"speaker-a",
    "session_id":"speaker-a-stackchan-train",
    "split":"train",
    "duration_ms":2000,
    "conditions":{"distance_cm":100,"room":"kitchen"}
  }'
```

The service commands only the addressed device. The device records the bounded
window independently of its provisional wake decision, then sends a
`training_sample` header and raw signed-16 PCM. `detected` records what the old
model did; it never decides whether the attempt enters the corpus.

## Device protocol

On `GET /v1/device`, a microphone device sends:

```json
{
  "type": "hello",
  "device_id": "kizz-1",
  "device_profile": "m5stack_stackchan_k151_cores3_v1",
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

`device_id` addresses an instance. `device_profile` identifies an acoustic
domain shared by equivalent hardware and preprocessing. The repository-level
`device-profiles.json` inventory classifies every current controller target.
Seven have onboard microphones: Waveshare Dial, Frame, RLCD, M5Stack Tough,
M5StickS3, StopWatch, and Kizz/StackChan. M5Stack Dial v1.1 and the plain
AtomS3 JoyStick target do not.

Those seven profiles are candidates for evaluating one shared model before
held-out evidence warrants a device-specific model. A catalog entry does not
claim that enrollment firmware or a real corpus exists: those are separate,
explicit statuses. At present only Kizz has the enrollment firmware path, and
no target is marked as having a collected corpus.

The server sends `training_capture`; the device responds with
`training_sample` followed by one binary PCM message. Attempts are bounded to
0.5–5 seconds and 160,000 PCM bytes.

## Corpus guarantees

`device-corpus.json` is the durable contract. Validation requires:

- registered, immutable audio profiles;
- unique capture IDs and content hashes;
- mono 16 kHz signed-16 PCM WAVs;
- explicit truth and train/validation/test assignment;
- no speaker or session crossing splits;
- both provisional detector hits and misses.

Build real-device feature archives and include them in training:

```sh
python tools/validate_device_corpus.py --corpus work/device-corpus
python tools/build_device_corpus_features.py \
  --corpus work/device-corpus --output work/device-features
python tools/write_recipe_training_config.py \
  --workspace work/kizz --train-dir work/kizz/trained \
  --device-features-dir work/device-features \
  --output work/kizz/training_parameters.yaml
```

The builder consumes the manifest's predetermined splits without randomizing
them. Evaluation reports truth, phrase, pronunciation, device profile, and
whether the provisional source detector hit or missed.
