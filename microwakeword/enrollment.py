"""Standalone LAN enrollment and calibration service for microphone devices."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
import time
import wave

from aiohttp import WSMsgType, web

from microwakeword.device_corpus import (
    CAPTURE_SOURCES,
    MANIFEST_NAME,
    SPLITS,
    TRUTHS,
    validate_device_corpus,
)


def valid_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 96
        and all(character.isalnum() or character in "-_.:" for character in value)
    )


def validate_audio_profile(audio: object) -> dict:
    if not isinstance(audio, dict):
        raise ValueError("hello requires an audio profile")
    required = {"sample_rate": 16000, "channels": 1, "sample_format": "s16le"}
    for key, expected in required.items():
        if audio.get(key) != expected:
            raise ValueError(f"audio {key} must be {expected}")
    for key in ("frontend", "gain_profile"):
        if not valid_token(audio.get(key)):
            raise ValueError(f"audio {key} is invalid")
    if not isinstance(audio.get("preprocessing", {}), dict):
        raise ValueError("audio preprocessing must be an object")
    return audio


def validate_capture_request(request: object) -> dict:
    if not isinstance(request, dict):
        raise ValueError("capture request must be an object")
    for key in (
        "capture_id",
        "device_id",
        "device_profile",
        "speaker_id",
        "session_id",
    ):
        if not valid_token(request.get(key)):
            raise ValueError(f"{key} is invalid")
    phrase = request.get("phrase")
    if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 160:
        raise ValueError("phrase is invalid")
    if request.get("truth") not in TRUTHS:
        raise ValueError("truth is invalid")
    if request.get("source") not in CAPTURE_SOURCES:
        raise ValueError("source is invalid")
    if request.get("split") not in SPLITS:
        raise ValueError("split is invalid")
    duration = request.get("duration_ms")
    if not isinstance(duration, int) or not 500 <= duration <= 5000:
        raise ValueError("duration_ms must be between 500 and 5000")
    pronunciation = request.get("pronunciation")
    if pronunciation is not None and not valid_token(pronunciation):
        raise ValueError("pronunciation is invalid")
    if not isinstance(request.get("conditions", {}), dict):
        raise ValueError("conditions must be an object")
    return request


def validate_wake_config_request(request: object) -> dict:
    if not isinstance(request, dict):
        raise ValueError("wake config request must be an object")
    if not valid_token(request.get("device_id")):
        raise ValueError("device_id is invalid")
    cutoff = request.get("probability_cutoff")
    window = request.get("sliding_window")
    end_silence_ms = request.get("end_silence_ms")
    max_utterance_ms = request.get("max_utterance_ms")
    diagnostics_enabled = request.get("diagnostics_enabled")
    audio_preprocessing = request.get("audio_preprocessing", {})
    if not isinstance(cutoff, (int, float)) or not 0.10 <= cutoff <= 0.99:
        raise ValueError("probability_cutoff must be between 0.10 and 0.99")
    if not isinstance(window, int) or not 1 <= window <= 20:
        raise ValueError("sliding_window must be between 1 and 20")
    if not isinstance(end_silence_ms, int) or not 300 <= end_silence_ms <= 5000:
        raise ValueError("end_silence_ms must be between 300 and 5000")
    if not isinstance(max_utterance_ms, int) or not 3000 <= max_utterance_ms <= 20000:
        raise ValueError("max_utterance_ms must be between 3000 and 20000")
    if not isinstance(diagnostics_enabled, bool):
        raise ValueError("diagnostics_enabled must be a boolean")
    if (
        not isinstance(audio_preprocessing, dict)
        or len(audio_preprocessing) > 16
        or any(
            not valid_token(key)
            or isinstance(value, (dict, list))
            or not isinstance(value, (str, int, float, bool, type(None)))
            for key, value in audio_preprocessing.items()
        )
    ):
        raise ValueError("audio_preprocessing must contain scalar frontend settings")
    return request


@dataclass
class Device:
    websocket: web.WebSocketResponse
    profile: str
    audio: dict
    firmware_sha: str | None
    wake_config: dict | None = None
    voice_telemetry: list[dict] = field(default_factory=list)


@dataclass
class PendingCapture:
    request: dict
    detected: bool | None = None
    byte_count: int | None = None
    audio: bytearray = field(default_factory=bytearray)
    queued_at: float = 0.0


class EnrollmentService:
    def __init__(self, corpus: Path):
        self.corpus = corpus
        self.devices: dict[str, Device] = {}
        self.pending: dict[str, PendingCapture] = {}
        self.recent_errors: list[dict] = []
        self.lock = asyncio.Lock()

    def application(self) -> web.Application:
        app = web.Application(client_max_size=200_000)
        app.router.add_get("/v1/device", self.device_socket)
        app.router.add_post("/v1/captures", self.enqueue_capture)
        app.router.add_get("/v1/devices", self.list_devices)
        app.router.add_get("/v1/status", self.status)
        app.router.add_post("/v1/wake-config", self.configure_wake)
        return app

    async def status(self, _request: web.Request) -> web.Response:
        async with self.lock:
            pending = {
                capture_id: {
                    "device_id": value.request["device_id"],
                    "declared_bytes": value.byte_count,
                    "received_bytes": len(value.audio),
                }
                for capture_id, value in self.pending.items()
            }
            errors = list(self.recent_errors[-20:])
        return web.json_response({"pending": pending, "recent_errors": errors})

    async def list_devices(self, _request: web.Request) -> web.Response:
        async with self.lock:
            devices = [
                {
                    "device_id": key,
                    "device_profile": value.profile,
                    "audio": value.audio,
                    "wake_config": value.wake_config,
                    "voice_telemetry": value.voice_telemetry[-20:],
                }
                for key, value in sorted(self.devices.items())
            ]
        return web.json_response({"devices": devices})

    async def configure_wake(self, http_request: web.Request) -> web.Response:
        try:
            request = validate_wake_config_request(await http_request.json())
        except (ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        async with self.lock:
            device = self.devices.get(request["device_id"])
            if device is None:
                return web.json_response(
                    {"error": "addressed device is not connected"}, status=503
                )
            try:
                command = {
                    "type": "wake_config",
                    "probability_cutoff": request["probability_cutoff"],
                    "sliding_window": request["sliding_window"],
                    "end_silence_ms": request["end_silence_ms"],
                    "max_utterance_ms": request["max_utterance_ms"],
                    "diagnostics_enabled": request["diagnostics_enabled"],
                }
                if request.get("audio_preprocessing"):
                    command["audio_preprocessing"] = request["audio_preprocessing"]
                await device.websocket.send_json(command)
            except ConnectionError:
                return web.json_response(
                    {"error": "addressed device disconnected"}, status=503
                )
        return web.json_response(
            {
                "device_id": request["device_id"],
                "state": "queued",
                "probability_cutoff": request["probability_cutoff"],
                "sliding_window": request["sliding_window"],
                "end_silence_ms": request["end_silence_ms"],
                "max_utterance_ms": request["max_utterance_ms"],
                "diagnostics_enabled": request["diagnostics_enabled"],
                "audio_preprocessing": request.get("audio_preprocessing", {}),
            },
            status=202,
        )

    async def enqueue_capture(self, http_request: web.Request) -> web.Response:
        try:
            request = validate_capture_request(await http_request.json())
        except (ValueError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        async with self.lock:
            now = time.monotonic()
            for capture_id in [
                key
                for key, pending in self.pending.items()
                if now - pending.queued_at > 10.0
            ]:
                self.pending.pop(capture_id, None)
            device = self.devices.get(request["device_id"])
            if device is None:
                return web.json_response(
                    {"error": "addressed device is not connected"}, status=503
                )
            if device.profile != request["device_profile"]:
                return web.json_response(
                    {"error": "connected device profile does not match"}, status=409
                )
            capture_id = request["capture_id"]
            if capture_id in self.pending or self._capture_exists(capture_id):
                return web.json_response(
                    {"error": "capture_id already exists"}, status=409
                )
            if any(
                pending.request["device_id"] == request["device_id"]
                for pending in self.pending.values()
            ):
                return web.json_response(
                    {"error": "addressed device already has a pending capture"},
                    status=409,
                )
            try:
                self._validate_split_assignment(request)
            except ValueError as error:
                return web.json_response({"error": str(error)}, status=409)
            self.pending[capture_id] = PendingCapture(
                request=dict(request), queued_at=now
            )
            try:
                await device.websocket.send_json(
                    {
                        "type": "training_capture",
                        "capture_id": capture_id,
                        "duration_ms": request["duration_ms"],
                    }
                )
            except ConnectionError:
                self.pending.pop(capture_id, None)
                return web.json_response(
                    {"error": "addressed device disconnected"}, status=503
                )
        return web.json_response(
            {"capture_id": capture_id, "state": "queued"}, status=202
        )

    async def device_socket(self, request: web.Request) -> web.WebSocketResponse:
        websocket = web.WebSocketResponse(max_msg_size=200_000, heartbeat=20)
        await websocket.prepare(request)
        device_id: str | None = None
        try:
            async for message in websocket:
                if message.type == WSMsgType.TEXT:
                    event = json.loads(message.data)
                    if event.get("type") == "hello":
                        device_id = await self._register(websocket, event)
                        await websocket.send_json(
                            {"type": "ready", "device_id": device_id}
                        )
                    elif event.get("type") == "training_sample":
                        await self._sample_header(device_id, event)
                    elif event.get("type") == "training_sample_end":
                        await self._sample_end(device_id, event)
                    elif event.get("type") == "training_error":
                        await self._training_error(device_id, event)
                    elif event.get("type") == "wake_config_applied":
                        await self._wake_config_applied(device_id, event)
                    elif event.get("type") == "wake_config_error":
                        pass
                    elif event.get("type") == "voice_telemetry":
                        await self._voice_telemetry(device_id, event)
                    else:
                        await websocket.send_json(
                            {"type": "error", "message": "unknown event"}
                        )
                elif message.type == WSMsgType.BINARY:
                    capture_id, received_bytes = await self._sample_audio(
                        device_id, bytes(message.data)
                    )
                    await websocket.send_json(
                        {
                            "type": "training_chunk",
                            "capture_id": capture_id,
                            "received_bytes": received_bytes,
                        }
                    )
        except (ValueError, json.JSONDecodeError) as error:
            async with self.lock:
                self.recent_errors.append(
                    {
                        "device_id": device_id,
                        "message": str(error),
                        "received_at": time.time(),
                    }
                )
                del self.recent_errors[:-20]
            await websocket.send_json({"type": "error", "message": str(error)})
        finally:
            if device_id is not None:
                async with self.lock:
                    current = self.devices.get(device_id)
                    if current is not None and current.websocket is websocket:
                        self.devices.pop(device_id, None)
                    for capture_id in [
                        key
                        for key, value in self.pending.items()
                        if value.request["device_id"] == device_id
                    ]:
                        self.pending.pop(capture_id, None)
        return websocket

    async def _register(self, websocket: web.WebSocketResponse, event: dict) -> str:
        device_id = event.get("device_id")
        profile = event.get("device_profile")
        if not valid_token(device_id) or not valid_token(profile):
            raise ValueError("hello has invalid device identity")
        audio = validate_audio_profile(event.get("audio"))
        firmware_sha = event.get("firmware_sha")
        if firmware_sha is not None and not valid_token(firmware_sha):
            raise ValueError("firmware_sha is invalid")
        async with self.lock:
            self.devices[device_id] = Device(websocket, profile, audio, firmware_sha)
        return device_id

    async def _wake_config_applied(
        self, device_id: str | None, event: dict
    ) -> None:
        if device_id is None:
            raise ValueError("device must send hello before wake config status")
        config = validate_wake_config_request(
            {
                "device_id": device_id,
                "probability_cutoff": event.get("probability_cutoff"),
                "sliding_window": event.get("sliding_window"),
                "end_silence_ms": event.get("end_silence_ms"),
                "max_utterance_ms": event.get("max_utterance_ms"),
                "diagnostics_enabled": event.get("diagnostics_enabled"),
                "audio_preprocessing": event.get("audio_preprocessing", {}),
            }
        )
        async with self.lock:
            device = self.devices.get(device_id)
            if device is not None:
                device.wake_config = {
                    "probability_cutoff": config["probability_cutoff"],
                    "sliding_window": config["sliding_window"],
                    "end_silence_ms": config["end_silence_ms"],
                    "max_utterance_ms": config["max_utterance_ms"],
                    "diagnostics_enabled": config["diagnostics_enabled"],
                    "audio_preprocessing": config.get("audio_preprocessing", {}),
                }

    async def _voice_telemetry(self, device_id: str | None, event: dict) -> None:
        if device_id is None:
            raise ValueError("device must send hello before voice telemetry")
        if event.get("event") != "endpoint" or event.get("reason") not in {
            "deepgram_flux", "vad_silence", "max_duration", "no_command"
        }:
            raise ValueError("voice telemetry event is invalid")
        integer_fields = (
            "turn_id", "wake_to_commit_ms", "command_ms", "trailing_silence_ms",
            "audio_bytes", "speech_frames", "silence_frames", "end_silence_ms",
            "max_utterance_ms",
        )
        if any(not isinstance(event.get(key), int) or event[key] < 0
               for key in integer_fields):
            raise ValueError("voice telemetry metrics are invalid")
        sample = {key: event[key] for key in ("event", "reason", *integer_fields)}
        sample["received_at"] = time.time()
        async with self.lock:
            device = self.devices.get(device_id)
            if device is not None:
                device.voice_telemetry.append(sample)
                del device.voice_telemetry[:-100]

    async def _sample_header(self, device_id: str | None, event: dict) -> None:
        if device_id is None:
            raise ValueError("device must send hello before a sample")
        capture_id = event.get("capture_id")
        byte_count = event.get("bytes")
        detected = event.get("detected")
        if not valid_token(capture_id) or not isinstance(detected, bool):
            raise ValueError("training_sample header is invalid")
        if not isinstance(byte_count, int) or not 0 < byte_count <= 160_000:
            raise ValueError("training_sample byte count is invalid")
        async with self.lock:
            pending = self.pending.get(capture_id)
            if pending is None or pending.request["device_id"] != device_id:
                raise ValueError("training_sample does not match a pending capture")
            pending.detected = detected
            pending.byte_count = byte_count

    async def _training_error(self, device_id: str | None, event: dict) -> None:
        capture_id = event.get("capture_id")
        if device_id is None or not valid_token(capture_id):
            raise ValueError("training_error is invalid")
        async with self.lock:
            pending = self.pending.get(capture_id)
            if pending is None or pending.request["device_id"] != device_id:
                raise ValueError("training_error does not match a pending capture")
            self.pending.pop(capture_id, None)

    async def _sample_audio(
        self, device_id: str | None, pcm: bytes
    ) -> tuple[str, int]:
        async with self.lock:
            matches = [
                (capture_id, pending)
                for capture_id, pending in self.pending.items()
                if pending.request["device_id"] == device_id
                and pending.byte_count is not None
            ]
            if len(matches) != 1:
                raise ValueError(
                    "binary audio does not have exactly one pending header"
                )
            capture_id, pending = matches[0]
            if not pcm or len(pcm) % 2:
                raise ValueError("binary audio chunk is invalid")
            if len(pending.audio) + len(pcm) > pending.byte_count:
                raise ValueError("binary audio exceeds its declared length")
            pending.audio.extend(pcm)
            return capture_id, len(pending.audio)

    async def _sample_end(self, device_id: str | None, event: dict) -> None:
        capture_id = event.get("capture_id")
        if device_id is None or not valid_token(capture_id):
            raise ValueError("training_sample_end is invalid")
        async with self.lock:
            pending = self.pending.get(capture_id)
            if (
                pending is None
                or pending.request["device_id"] != device_id
                or pending.byte_count is None
            ):
                raise ValueError("training_sample_end does not match a sample")
            if len(pending.audio) != pending.byte_count:
                raise ValueError("binary audio length does not match its header")
            device = self.devices.get(device_id)
            if device is None:
                raise ValueError("device disconnected before capture completed")
            self._persist(capture_id, pending, device, bytes(pending.audio))
            self.pending.pop(capture_id, None)
            await device.websocket.send_json(
                {"type": "stored", "capture_id": capture_id}
            )

    def _capture_exists(self, capture_id: str) -> bool:
        manifest_path = self.corpus / MANIFEST_NAME
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text())
        return any(
            item.get("capture_id") == capture_id
            for item in manifest.get("captures", [])
        )

    def _validate_split_assignment(self, request: dict) -> None:
        manifest_path = self.corpus / MANIFEST_NAME
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        for capture in manifest.get("captures", []):
            if (
                capture["speaker_id"] == request["speaker_id"]
                and capture["split"] != request["split"]
            ):
                raise ValueError("speaker would cross corpus splits")
            if (
                capture["session_id"] == request["session_id"]
                and capture["split"] != request["split"]
            ):
                raise ValueError("session would cross corpus splits")
            if (
                capture["device_id"] == request["device_id"]
                and capture["device_profile"] != request["device_profile"]
            ):
                raise ValueError("device would cross device profiles")

    def _persist(
        self, capture_id: str, pending: PendingCapture, device: Device, pcm: bytes
    ) -> None:
        audio_dir = self.corpus / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        relative = Path("audio") / f"{capture_id}.wav"
        destination = self.corpus / relative
        temporary_wav = destination.with_suffix(".wav.tmp")
        with wave.open(str(temporary_wav), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm)
        temporary_wav.replace(destination)

        manifest_path = self.corpus / MANIFEST_NAME
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = {
                "schema_version": 1,
                "corpus_id": "hiphi-device-v1",
                "device_profiles": {},
                "captures": [],
            }
        profile = {"audio": device.audio}
        existing = manifest["device_profiles"].get(device.profile)
        if existing is not None and existing != profile:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"device profile {device.profile} changed its audio contract"
            )
        manifest["device_profiles"][device.profile] = profile
        request = pending.request
        manifest["captures"].append(
            {
                "capture_id": capture_id,
                "path": str(relative),
                "truth": request["truth"],
                "source": request["source"],
                "phrase": request["phrase"],
                "pronunciation": request.get("pronunciation"),
                "speaker_id": request["speaker_id"],
                "session_id": request["session_id"],
                "split": request["split"],
                "detected": pending.detected,
                "samples": len(pcm) // 2,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "device_id": request["device_id"],
                "device_profile": request["device_profile"],
                "firmware_sha": device.firmware_sha,
                "conditions": request.get("conditions", {}),
            }
        )
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        prior_manifest = manifest_path.read_bytes() if manifest_path.exists() else None
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary_manifest.replace(manifest_path)
        try:
            validate_device_corpus(self.corpus)
        except Exception:
            destination.unlink(missing_ok=True)
            if prior_manifest is None:
                manifest_path.unlink(missing_ok=True)
            else:
                manifest_path.write_bytes(prior_manifest)
            raise
