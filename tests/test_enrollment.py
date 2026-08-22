import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from microwakeword.device_corpus import validate_device_corpus
from microwakeword.enrollment import EnrollmentService


class SimulatedDeviceEnrollmentTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.corpus = Path(self.temporary.name)
        self.client = TestClient(
            TestServer(EnrollmentService(self.corpus).application())
        )
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
        self.assertEqual(
            attempt["device_profile"], "m5stack_stackchan_k151_cores3_v1"
        )

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
