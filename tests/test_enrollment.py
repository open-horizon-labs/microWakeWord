import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from microwakeword.device_corpus import validate_device_corpus
from microwakeword.enrollment import EnrollmentService
from microwakeword.false_wake import promote_false_wake


class SimulatedDeviceEnrollmentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.corpus = Path(self.temporary.name)
        (self.corpus / "device-corpus.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "corpus_id": "test-device-v1",
                    "device_profiles": {
                        "m5stack_stackchan_k151_cores3_v1": {
                            "audio": {
                                "sample_rate": 16000,
                                "channels": 1,
                                "sample_format": "s16le",
                                "frontend": "m5unified_mic",
                                "gain_profile": "default",
                                "preprocessing": {"dc_block": True},
                            }
                        }
                    },
                    "speakers": {
                        "speaker-a": {
                            "kind": "synthetic",
                            "age_group": "unknown",
                            "split": "train",
                        },
                        "speaker-b": {
                            "kind": "synthetic",
                            "age_group": "unknown",
                            "split": "test",
                        },
                        "ambient-room": {
                            "kind": "ambient",
                            "age_group": "not_applicable",
                            "split": "train",
                        },
                    },
                    "captures": [],
                }
            )
        )
        self.service = EnrollmentService(
            self.corpus, "http://trainer.test:8091"
        )
        self.client = TestClient(TestServer(self.service.application()))
        await self.client.start_server()
        self.device = await self.client.ws_connect("/v1/device")
        await self.device.send_json(
            {
                "type": "hello",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_stackchan_k151_cores3_v1",
                "firmware_sha": "simulated",
                "audio": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "sample_format": "s16le",
                    "frontend": "m5unified_mic",
                    "gain_profile": "default",
                    "preprocessing": {"dc_block": True},
                },
            }
        )
        ready = await self.device.receive_json()
        self.assertEqual(ready["type"], "ready")

    async def asyncTearDown(self):
        await self.device.close()
        await self.client.close()
        self.temporary.cleanup()

    async def test_detector_miss_is_retained_end_to_end(self):
        response = await self.client.post(
            "/v1/captures",
            json={
                "capture_id": "missed-positive",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_stackchan_k151_cores3_v1",
                "phrase": "Hi-Fi Kizz",
                "pronunciation": "hi_fi",
                "truth": "positive",
                "source": "simulated",
                "speaker_id": "speaker-a",
                "session_id": "session-a",
                "split": "train",
                "duration_ms": 2000,
                "conditions": {"distance_cm": 100},
            },
        )
        self.assertEqual(response.status, 202)
        command = await self.device.receive_json()
        self.assertEqual(
            command,
            {
                "type": "training_capture",
                "capture_id": "missed-positive",
                "duration_ms": 2000,
                "upload_url": (
                    "http://trainer.test:8091/v1/captures/"
                    "missed-positive/audio"
                ),
            },
        )
        pcm = b"\0\0" * 32000
        await self.device.send_json(
            {
                "type": "training_sample",
                "capture_id": "missed-positive",
                "bytes": len(pcm),
                "detected": False,
            }
        )
        await self.device.send_bytes(pcm[:4096])
        first_ack = await self.device.receive_json()
        self.assertEqual(first_ack["type"], "training_chunk")
        self.assertEqual(first_ack["received_bytes"], 4096)
        await self.device.send_bytes(pcm[4096:])
        second_ack = await self.device.receive_json()
        self.assertEqual(second_ack["type"], "training_chunk")
        self.assertEqual(second_ack["received_bytes"], len(pcm))
        await self.device.send_json(
            {"type": "training_sample_end", "capture_id": "missed-positive"}
        )
        stored = await self.device.receive_json()
        self.assertEqual(stored["type"], "stored")

        manifest = validate_device_corpus(self.corpus)
        attempt = manifest["captures"][0]
        self.assertFalse(attempt["detected"])
        self.assertEqual(attempt["device_profile"], "m5stack_stackchan_k151_cores3_v1")

    async def test_empty_false_wake_is_quarantined_not_added_to_corpus(self):
        pcm = b"\0\0" * 16000
        await self.device.send_json(
            {
                "type": "false_wake_observation",
                "observation_id": "false-wake-001",
                "bytes": len(pcm),
                "wake_probability": 0.74,
                "wake_cutoff": 0.70,
                "wake_to_timeout_ms": 6000,
                "sliding_window": 1,
                "command_speech_frames": 0,
                "command_silence_frames": 120,
            }
        )
        await self.device.send_bytes(pcm)
        chunk = await self.device.receive_json()
        self.assertEqual(chunk["type"], "false_wake_chunk")
        self.assertEqual(chunk["observation_id"], "false-wake-001")
        await self.device.send_json(
            {"type": "false_wake_observation_end", "observation_id": "false-wake-001"}
        )
        stored = await self.device.receive_json()
        self.assertEqual(stored["type"], "false_wake_stored")

        self.assertEqual(validate_device_corpus(self.corpus)["captures"], [])
        metadata = json.loads(
            (self.corpus / "observations" / "false-wakes" / "false-wake-001.json").read_text()
        )
        self.assertEqual(metadata["kind"], "false_wake_no_command")
        self.assertEqual(metadata["wake_cutoff"], 0.70)

    async def test_false_wake_rejects_audio_that_exceeds_header(self):
        await self.device.send_json(
            {
                "type": "false_wake_observation",
                "observation_id": "false-wake-overflow",
                "bytes": 320,
                "wake_probability": 0.74,
                "wake_cutoff": 0.70,
                "wake_to_timeout_ms": 6000,
            }
        )
        await self.device.send_bytes(b"\0\0" * 161)
        error = await self.device.receive_json()
        self.assertEqual(error["type"], "error")
        self.assertIn("exceeds", error["message"])
        self.assertFalse((self.corpus / "observations").exists())

    async def test_command_wake_is_quarantined_separately_from_false_wakes(self):
        pcm = b"\x01\x00" * 8000
        await self.device.send_json(
            {
                "type": "wake_observation",
                "observation_id": "wake-command-001",
                "bytes": len(pcm),
                "wake_probability": 0.91,
                "wake_cutoff": 0.70,
                "wake_to_timeout_ms": 420,
                "outcome": "command_speech",
                "verification_mode": "shadow_all",
                "c_rms_dbfs": -42.5,
                "c_pass": True,
            }
        )
        await self.device.send_bytes(pcm)
        chunk = await self.device.receive_json()
        self.assertEqual(chunk["type"], "wake_observation_chunk")
        await self.device.send_json(
            {"type": "wake_observation_end", "observation_id": "wake-command-001"}
        )
        stored = await self.device.receive_json()
        self.assertEqual(stored["type"], "wake_observation_stored")
        metadata = json.loads(
            (self.corpus / "observations" / "wakes" / "wake-command-001.json").read_text()
        )
        self.assertEqual(metadata["kind"], "wake_observation")
        self.assertEqual(metadata["outcome"], "command_speech")
        self.assertEqual(validate_device_corpus(self.corpus)["captures"], [])

    async def test_false_wake_requires_explicit_review_for_promotion(self):
        pcm = b"\0\0" * 16000
        await self.device.send_json(
            {
                "type": "false_wake_observation",
                "observation_id": "false-wake-promote",
                "bytes": len(pcm),
                "wake_probability": 0.74,
                "wake_cutoff": 0.70,
                "wake_to_timeout_ms": 6000,
            }
        )
        await self.device.send_bytes(pcm)
        await self.device.receive_json()
        await self.device.send_json(
            {"type": "false_wake_observation_end", "observation_id": "false-wake-promote"}
        )
        await self.device.receive_json()
        self.assertEqual(validate_device_corpus(self.corpus)["captures"], [])

        entry = promote_false_wake(
            self.corpus,
            "false-wake-promote",
            reviewer="muness",
            split="train",
            speaker_id="ambient-room",
            session_id="false-wake-review-session",
            reason="listened: ambient room noise, no command speech",
        )
        self.assertEqual(entry["truth"], "hard_negative")
        self.assertEqual(validate_device_corpus(self.corpus)["captures"][0]["capture_id"], entry["capture_id"])
        metadata = json.loads(
            (self.corpus / "observations" / "false-wakes" / "false-wake-promote.json").read_text()
        )
        self.assertEqual(metadata["promoted_capture_id"], entry["capture_id"])

    async def test_detector_miss_can_be_uploaded_over_http(self):
        response = await self.client.post(
            "/v1/captures",
            json={
                "capture_id": "http-missed-positive",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_stackchan_k151_cores3_v1",
                "phrase": "Hi-Fi Kizz",
                "pronunciation": "hi_fi",
                "truth": "positive",
                "source": "simulated",
                "speaker_id": "speaker-a",
                "session_id": "session-http",
                "split": "train",
                "duration_ms": 2000,
                "conditions": {"distance_cm": 100},
            },
        )
        self.assertEqual(response.status, 202)
        command = await self.device.receive_json()
        self.assertEqual(
            command["upload_url"],
            "http://trainer.test:8091/v1/captures/http-missed-positive/audio",
        )

        pcm = b"\0\0" * 32000
        uploaded = await self.client.post(
            "/v1/captures/http-missed-positive/audio",
            data=pcm,
            headers={
                "Content-Type": "application/octet-stream",
                "X-Device-ID": "sim-kizz-1",
                "X-Detected": "false",
            },
        )
        self.assertEqual(uploaded.status, 200)
        self.assertEqual((await uploaded.json())["state"], "stored")
        attempt = validate_device_corpus(self.corpus)["captures"][0]
        self.assertEqual(attempt["capture_id"], "http-missed-positive")
        self.assertFalse(attempt["detected"])

    async def test_segmented_http_upload_is_resumable_and_retains_miss(self):
        response = await self.client.post(
            "/v1/captures",
            json={
                "capture_id": "segmented-missed-positive",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_stackchan_k151_cores3_v1",
                "phrase": "Hi-Fi Kizz",
                "pronunciation": "hi_fi",
                "truth": "positive",
                "source": "simulated",
                "speaker_id": "speaker-a",
                "session_id": "session-segmented-http",
                "split": "train",
                "duration_ms": 500,
                "conditions": {"distance_cm": 100},
            },
        )
        self.assertEqual(response.status, 202)
        await self.device.receive_json()

        pcm = bytes(range(256)) * 62 + bytes(range(128))
        self.assertEqual(len(pcm), 16000)
        common_headers = {
            "Content-Type": "application/octet-stream",
            "X-Device-ID": "sim-kizz-1",
            "X-Detected": "false",
            "X-Audio-Total": str(len(pcm)),
        }
        first = pcm[:2048]
        uploaded = await self.client.post(
            "/v1/captures/segmented-missed-positive/audio",
            data=first,
            headers={**common_headers, "X-Audio-Offset": "0"},
        )
        self.assertEqual(uploaded.status, 200)
        self.assertEqual((await uploaded.json())["state"], "receiving")

        retried = await self.client.post(
            "/v1/captures/segmented-missed-positive/audio",
            data=first,
            headers={**common_headers, "X-Audio-Offset": "0"},
        )
        self.assertEqual(retried.status, 200)
        self.assertEqual((await retried.json())["received_bytes"], len(first))

        for offset in range(len(first), len(pcm), 2048):
            segment = pcm[offset : offset + 2048]
            uploaded = await self.client.post(
                "/v1/captures/segmented-missed-positive/audio",
                data=segment,
                headers={**common_headers, "X-Audio-Offset": str(offset)},
            )
            self.assertEqual(uploaded.status, 200)
        self.assertEqual((await uploaded.json())["state"], "stored")
        attempt = validate_device_corpus(self.corpus)["captures"][0]
        self.assertEqual(attempt["capture_id"], "segmented-missed-positive")
        self.assertFalse(attempt["detected"])

    async def test_segmented_upload_survives_device_socket_reconnect(self):
        response = await self.client.post(
            "/v1/captures",
            json={
                "capture_id": "disconnected-segmented-positive",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_stackchan_k151_cores3_v1",
                "phrase": "Hi-Fi Kizz",
                "pronunciation": "hi_fi",
                "truth": "positive",
                "source": "simulated",
                "speaker_id": "speaker-a",
                "session_id": "session-disconnected-http",
                "split": "train",
                "duration_ms": 500,
                "conditions": {"distance_cm": 100},
            },
        )
        self.assertEqual(response.status, 202)
        await self.device.receive_json()

        pcm = bytes(range(256)) * 62 + bytes(range(128))
        first = pcm[:2048]
        common_headers = {
            "Content-Type": "application/octet-stream",
            "X-Device-ID": "sim-kizz-1",
            "X-Detected": "false",
            "X-Audio-Total": str(len(pcm)),
        }
        uploaded = await self.client.post(
            "/v1/captures/disconnected-segmented-positive/audio",
            data=first,
            headers={**common_headers, "X-Audio-Offset": "0"},
        )
        self.assertEqual(uploaded.status, 200)

        await self.device.close()
        await asyncio.sleep(0.01)

        uploaded = await self.client.post(
            "/v1/captures/disconnected-segmented-positive/audio",
            data=pcm[len(first) :],
            headers={**common_headers, "X-Audio-Offset": str(len(first))},
        )
        self.assertEqual(uploaded.status, 200)
        self.assertEqual((await uploaded.json())["state"], "stored")
        attempt = validate_device_corpus(self.corpus)["captures"][0]
        self.assertEqual(attempt["capture_id"], "disconnected-segmented-positive")
        self.assertFalse(attempt["detected"])

    async def test_endpoint_routes_to_explicit_device_profile(self):
        response = await self.client.post(
            "/v1/captures",
            json={
                "capture_id": "wrong-profile",
                "device_id": "sim-kizz-1",
                "device_profile": "m5stack_dial_v1",
                "phrase": "Hi-Fi Kizz",
                "truth": "positive",
                "source": "simulated",
                "speaker_id": "speaker-b",
                "session_id": "session-b",
                "split": "test",
                "duration_ms": 2000,
            },
        )
        self.assertEqual(response.status, 409)

    async def test_one_device_cannot_be_given_overlapping_capture_windows(self):
        request = {
            "capture_id": "first",
            "device_id": "sim-kizz-1",
            "device_profile": "m5stack_stackchan_k151_cores3_v1",
            "phrase": "Hi-Fi Kizz",
            "truth": "positive",
            "source": "simulated",
            "speaker_id": "speaker-b",
            "session_id": "session-b",
            "split": "test",
            "duration_ms": 2000,
        }
        first = await self.client.post("/v1/captures", json=request)
        self.assertEqual(first.status, 202)
        request["capture_id"] = "second"
        second = await self.client.post("/v1/captures", json=request)
        self.assertEqual(second.status, 409)

    async def test_wake_config_is_routed_and_reported(self):
        response = await self.client.post(
            "/v1/wake-config",
            json={
                "device_id": "sim-kizz-1",
                "probability_cutoff": 0.74,
                "sliding_window": 1,
                "end_silence_ms": 3000,
                "max_utterance_ms": 12000,
                "diagnostics_enabled": True,
                "audio_preprocessing": {"m5unified_magnification": 8},
            },
        )
        self.assertEqual(response.status, 202)
        command = await self.device.receive_json()
        self.assertEqual(
            command,
            {
                "type": "wake_config",
                "probability_cutoff": 0.74,
                "sliding_window": 1,
                "end_silence_ms": 3000,
                "max_utterance_ms": 12000,
                "diagnostics_enabled": True,
                "audio_preprocessing": {"m5unified_magnification": 8},
            },
        )
        await self.device.send_json(
            {
                "type": "wake_config_applied",
                "probability_cutoff": 0.74,
                "sliding_window": 1,
                "end_silence_ms": 3000,
                "max_utterance_ms": 12000,
                "diagnostics_enabled": True,
                "audio_preprocessing": {"m5unified_magnification": 8},
            }
        )
        await asyncio.sleep(0)
        devices = await self.client.get("/v1/devices")
        payload = await devices.json()
        self.assertEqual(
            payload["devices"][0]["wake_config"],
            {
                "probability_cutoff": 0.74,
                "sliding_window": 1,
                "end_silence_ms": 3000,
                "max_utterance_ms": 12000,
                "diagnostics_enabled": True,
                "audio_preprocessing": {"m5unified_magnification": 8},
            },
        )

        await self.device.send_json(
            {
                "type": "voice_telemetry",
                "event": "endpoint",
                "reason": "deepgram_flux",
                "turn_id": 7,
                "wake_to_commit_ms": 1840,
                "command_ms": 1510,
                "trailing_silence_ms": 270,
                "audio_bytes": 57344,
                "speech_frames": 41,
                "silence_frames": 9,
                "end_silence_ms": 800,
                "max_utterance_ms": 12000,
            }
        )
        await asyncio.sleep(0)
        devices = await self.client.get("/v1/devices")
        telemetry = (await devices.json())["devices"][0]["voice_telemetry"]
        self.assertEqual(telemetry[-1]["reason"], "deepgram_flux")
        self.assertEqual(telemetry[-1]["trailing_silence_ms"], 270)


if __name__ == "__main__":
    unittest.main()
