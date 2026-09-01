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
    request = dict(request)
    if not valid_token(request.get("device_id")):
        raise ValueError("device_id is invalid")
    cutoff = request.get("probability_cutoff")
    window = request.get("sliding_window")
    end_silence_ms = request.get("end_silence_ms")
    max_utterance_ms = request.get("max_utterance_ms")
    diagnostics_enabled = request.get("diagnostics_enabled")
    audio_preprocessing = request.get("audio_preprocessing", {})
    verification_mode = request.get("verification_mode", "shadow_all")
    c_min_rms_dbfs = request.get("c_min_rms_dbfs", -60.0)
    c_max_clip_percent = request.get("c_max_clip_percent", 25)
    capture_all_wakes = request.get("capture_all_wakes", True)
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
    if verification_mode not in {
        "off", "c_only", "b_only", "c_then_b", "b_then_a_uncertain",
        "c_then_b_then_a", "shadow_all",
    }:
        raise ValueError("verification_mode is invalid")
    if not isinstance(c_min_rms_dbfs, (int, float)) or not -80 <= c_min_rms_dbfs <= -10:
        raise ValueError("c_min_rms_dbfs must be between -80 and -10")
    if not isinstance(c_max_clip_percent, int) or not 0 <= c_max_clip_percent <= 100:
        raise ValueError("c_max_clip_percent must be between 0 and 100")
    if not isinstance(capture_all_wakes, bool):
        raise ValueError("capture_all_wakes must be a boolean")
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
    request.setdefault("verification_mode", verification_mode)
    request.setdefault("c_min_rms_dbfs", c_min_rms_dbfs)
    request.setdefault("c_max_clip_percent", c_max_clip_percent)
    request.setdefault("capture_all_wakes", capture_all_wakes)
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
    device_profile: str
    audio_profile: dict
    firmware_sha: str | None
    detected: bool | None = None
    byte_count: int | None = None
    audio: bytearray = field(default_factory=bytearray)
    queued_at: float = 0.0
    last_activity_at: float = 0.0


@dataclass
class PendingFalseWake:
    """An unreviewed wake-without-command observation.

    These deliberately live outside device-corpus.json.  A false wake is useful
    evidence, but it must be listened to and assigned a split before it can
    become a training negative.
    """

    observation_id: str
    device_id: str
    device_profile: str
    audio_profile: dict
    firmware_sha: str | None
    metadata: dict
    byte_count: int
    directory: str = "false-wakes"
    observation_kind: str = "false_wake_no_command"
    audio: bytearray = field(default_factory=bytearray)
    received_at: float = field(default_factory=time.time)


class EnrollmentService:
    PENDING_CAPTURE_TIMEOUT_SECONDS = 300.0

    def __init__(self, corpus: Path, public_base_url: str | None = None):
        self.corpus = corpus
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.devices: dict[str, Device] = {}
        self.pending: dict[str, PendingCapture] = {}
        self.pending_false_wakes: dict[str, PendingFalseWake] = {}
        self.recent_errors: list[dict] = []
        self.lock = asyncio.Lock()

    def application(self) -> web.Application:
        app = web.Application(client_max_size=200_000)
        app.router.add_get("/v1/device", self.device_socket)
        app.router.add_post("/v1/captures", self.enqueue_capture)
        app.router.add_post(
            "/v1/captures/{capture_id}/audio", self.upload_capture_audio
        )
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
        return web.json_response(
            {
                "pending": pending,
                "pending_false_wakes": {
                    key: {
                        "device_id": value.device_id,
                        "declared_bytes": value.byte_count,
                        "received_bytes": len(value.audio),
                    }
                    for key, value in self.pending_false_wakes.items()
                },
                "recent_errors": errors,
            }
        )

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
            submitted = await http_request.json()
            request = validate_wake_config_request(submitted)
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
                if any(
                    key in submitted
                    for key in (
                        "verification_mode",
                        "c_min_rms_dbfs",
                        "c_max_clip_percent",
                        "capture_all_wakes",
                    )
                ):
                    command.update(
                        {
                            "verification_mode": request["verification_mode"],
                            "c_min_rms_dbfs": request["c_min_rms_dbfs"],
                            "c_max_clip_percent": request["c_max_clip_percent"],
                            "capture_all_wakes": request["capture_all_wakes"],
                        }
                    )
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
                if now - pending.last_activity_at
                > self.PENDING_CAPTURE_TIMEOUT_SECONDS
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
                request=dict(request),
                device_profile=device.profile,
                audio_profile=dict(device.audio),
                firmware_sha=device.firmware_sha,
                queued_at=now,
                last_activity_at=now,
            )
            try:
                public_base_url = self.public_base_url or (
                    f"{http_request.scheme}://{http_request.host}"
                )
                await device.websocket.send_json(
                    {
                        "type": "training_capture",
                        "capture_id": capture_id,
                        "duration_ms": request["duration_ms"],
                        "upload_url": (
                            f"{public_base_url}/v1/captures/{capture_id}/audio"
                        ),
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

    async def upload_capture_audio(
        self, http_request: web.Request
    ) -> web.Response:
        capture_id = http_request.match_info.get("capture_id")
        device_id = http_request.headers.get("X-Device-ID")
        detected_header = http_request.headers.get("X-Detected")
        offset_header = http_request.headers.get("X-Audio-Offset")
        total_header = http_request.headers.get("X-Audio-Total")
        if not valid_token(capture_id) or not valid_token(device_id):
            return web.json_response({"error": "invalid capture identity"}, status=400)
        if detected_header not in {"true", "false"}:
            return web.json_response({"error": "X-Detected must be true or false"}, status=400)
        if (offset_header is None) != (total_header is None):
            return web.json_response(
                {"error": "X-Audio-Offset and X-Audio-Total must be sent together"},
                status=400,
            )
        segmented = offset_header is not None
        try:
            offset = int(offset_header) if segmented else 0
            total = int(total_header) if segmented else 0
        except ValueError:
            return web.json_response({"error": "audio range is invalid"}, status=400)
        pcm = await http_request.read()
        if not pcm or len(pcm) % 2:
            return web.json_response({"error": "PCM body is invalid"}, status=400)

        async with self.lock:
            pending = self.pending.get(capture_id)
            if pending is None or pending.request["device_id"] != device_id:
                return web.json_response({"error": "capture is not pending"}, status=409)
            expected_bytes = pending.request["duration_ms"] * 16000 * 2 // 1000
            if segmented:
                if total != expected_bytes or offset < 0 or offset % 2:
                    return web.json_response(
                        {"error": "audio range does not match capture"}, status=409
                    )
                detected = detected_header == "true"
                if pending.detected is not None and pending.detected != detected:
                    return web.json_response(
                        {"error": "X-Detected changed during upload"}, status=409
                    )
                pending.detected = detected
                pending.byte_count = expected_bytes
                received = len(pending.audio)
                end = offset + len(pcm)
                if offset > received or end > expected_bytes:
                    return web.json_response(
                        {
                            "error": "audio segment is out of sequence",
                            "received_bytes": received,
                        },
                        status=409,
                    )
                if offset < received:
                    if end > received or bytes(pending.audio[offset:end]) != pcm:
                        return web.json_response(
                            {"error": "audio retry differs from retained segment"},
                            status=409,
                        )
                else:
                    pending.audio.extend(pcm)
                    received = len(pending.audio)
                pending.last_activity_at = time.monotonic()
                if received == expected_bytes:
                    self._persist(capture_id, pending, bytes(pending.audio))
                    self.pending.pop(capture_id, None)
                    state = "stored"
                else:
                    state = "receiving"
                return web.json_response(
                    {
                        "capture_id": capture_id,
                        "state": state,
                        "received_bytes": received,
                        "expected_bytes": expected_bytes,
                    }
                )
            if len(pcm) != expected_bytes:
                return web.json_response(
                    {
                        "error": "PCM length does not match capture duration",
                        "expected_bytes": expected_bytes,
                        "received_bytes": len(pcm),
                    },
                    status=409,
                )
            pending.detected = detected_header == "true"
            pending.byte_count = len(pcm)
            pending.audio = bytearray(pcm)
            pending.last_activity_at = time.monotonic()
            self._persist(capture_id, pending, pcm)
            self.pending.pop(capture_id, None)
        return web.json_response(
            {"capture_id": capture_id, "state": "stored", "bytes": len(pcm)}
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
                    elif event.get("type") == "false_wake_observation":
                        await self._false_wake_header(device_id, event)
                    elif event.get("type") == "false_wake_observation_end":
                        await self._false_wake_end(device_id, event)
                    elif event.get("type") == "wake_observation":
                        await self._false_wake_header(device_id, event)
                    elif event.get("type") == "wake_observation_end":
                        await self._false_wake_end(device_id, event)
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
                    capture_id, received_bytes, kind = await self._sample_audio(
                        device_id, bytes(message.data)
                    )
                    await websocket.send_json(
                        {
                            "type": f"{kind}_chunk",
                            "capture_id" if kind == "training" else "observation_id": capture_id,
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

    async def _wake_config_applied(self, device_id: str | None, event: dict) -> None:
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
                "verification_mode": event.get("verification_mode", "shadow_all"),
                "c_min_rms_dbfs": event.get("c_min_rms_dbfs", -60.0),
                "c_max_clip_percent": event.get("c_max_clip_percent", 25),
                "capture_all_wakes": event.get("capture_all_wakes", True),
                "audio_preprocessing": event.get("audio_preprocessing", {}),
            }
        )
        verification_present = any(
            key in event
            for key in (
                "verification_mode",
                "c_min_rms_dbfs",
                "c_max_clip_percent",
                "capture_all_wakes",
            )
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
                if verification_present:
                    device.wake_config.update(
                        {
                            "verification_mode": config["verification_mode"],
                            "c_min_rms_dbfs": config["c_min_rms_dbfs"],
                            "c_max_clip_percent": config["c_max_clip_percent"],
                            "capture_all_wakes": config["capture_all_wakes"],
                        }
                    )

    async def _voice_telemetry(self, device_id: str | None, event: dict) -> None:
        if device_id is None:
            raise ValueError("device must send hello before voice telemetry")
        if event.get("event") != "endpoint" or event.get("reason") not in {
            "deepgram_flux",
            "vad_silence",
            "max_duration",
            "no_command",
        }:
            raise ValueError("voice telemetry event is invalid")
        integer_fields = (
            "turn_id",
            "wake_to_commit_ms",
            "command_ms",
            "trailing_silence_ms",
            "audio_bytes",
            "speech_frames",
            "silence_frames",
            "end_silence_ms",
            "max_utterance_ms",
        )
        if any(
            not isinstance(event.get(key), int) or event[key] < 0
            for key in integer_fields
        ):
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
            pending.last_activity_at = time.monotonic()

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
    ) -> tuple[str, int, str]:
        async with self.lock:
            training_matches = [
                (capture_id, pending)
                for capture_id, pending in self.pending.items()
                if pending.request["device_id"] == device_id
                and pending.byte_count is not None
            ]
            matches: list[tuple[str, object, str]] = [
                (key, value, "training") for key, value in training_matches
            ]
            matches.extend(
                (
                    observation_id,
                    pending,
                    "false_wake" if pending.directory == "false-wakes" else "wake_observation",
                )
                for observation_id, pending in self.pending_false_wakes.items()
                if pending.device_id == device_id
            )
            if len(matches) != 1:
                raise ValueError(
                    "binary audio does not have exactly one pending header"
                )
            capture_id, pending, kind = matches[0]
            if not pcm or len(pcm) % 2:
                raise ValueError("binary audio chunk is invalid")
            if len(pending.audio) + len(pcm) > pending.byte_count:
                raise ValueError("binary audio exceeds its declared length")
            pending.audio.extend(pcm)
            pending.last_activity_at = time.monotonic()
            return capture_id, len(pending.audio), kind

    async def _false_wake_header(self, device_id: str | None, event: dict) -> None:
        if device_id is None:
            raise ValueError("device must send hello before an observation")
        observation_id = event.get("observation_id")
        byte_count = event.get("bytes")
        event_type = event.get("type")
        generic = event_type == "wake_observation"
        if event_type not in {"false_wake_observation", "wake_observation"}:
            raise ValueError("wake observation header is invalid")
        if not valid_token(observation_id) or not isinstance(byte_count, int):
            raise ValueError("wake observation header is invalid")
        if not 0 < byte_count <= 320_000 or byte_count % 2:
            raise ValueError("wake observation byte count is invalid")
        required_numbers = ("wake_probability", "wake_cutoff", "wake_to_timeout_ms")
        if any(not isinstance(event.get(key), (int, float)) for key in required_numbers):
            raise ValueError("wake observation metrics are invalid")
        if not 0 <= event["wake_probability"] <= 1 or not 0.10 <= event["wake_cutoff"] <= 0.99:
            raise ValueError("wake observation score is invalid")
        if not 0 <= event["wake_to_timeout_ms"] <= 15_000:
            raise ValueError("wake observation timeout is invalid")
        outcome = event.get("outcome", "no_command")
        if not valid_token(outcome):
            raise ValueError("wake observation outcome is invalid")
        async with self.lock:
            device = self.devices.get(device_id)
            if device is None:
                raise ValueError("device disconnected before observation")
            existing_pending = self.pending_false_wakes.get(observation_id)
            if existing_pending is not None:
                if (
                    existing_pending.device_id != device_id
                    or existing_pending.byte_count != byte_count
                ):
                    raise ValueError("false_wake_observation_id already exists")
                # Firmware retries the same observation after a dropped socket.
                # Discard any partial payload so the next audio stream starts
                # from byte zero while preserving the idempotency boundary.
                self.pending_false_wakes.pop(observation_id, None)
            if self._observation_exists(observation_id):
                raise ValueError("false_wake_observation_id already exists")
            if any(value.device_id == device_id for value in self.pending_false_wakes.values()):
                raise ValueError("device already has a pending false wake observation")
            metadata = {
                key: event[key]
                for key in event
                if key not in {"type", "observation_id", "bytes"}
                and isinstance(event[key], (str, int, float, bool))
            }
            self.pending_false_wakes[observation_id] = PendingFalseWake(
                observation_id, device_id, device.profile, dict(device.audio),
                device.firmware_sha, metadata, byte_count,
                "wakes" if generic else "false-wakes",
                "wake_observation" if generic else "false_wake_no_command",
            )

    async def _false_wake_end(self, device_id: str | None, event: dict) -> None:
        observation_id = event.get("observation_id")
        if device_id is None or not valid_token(observation_id):
            raise ValueError("wake observation end is invalid")
        async with self.lock:
            pending = self.pending_false_wakes.get(observation_id)
            if pending is None or pending.device_id != device_id:
                raise ValueError("wake observation end does not match an observation")
            if len(pending.audio) != pending.byte_count:
                raise ValueError("false wake audio length does not match its header")
            self._persist_false_wake(pending)
            self.pending_false_wakes.pop(observation_id, None)
            device = self.devices.get(device_id)
            if device is not None:
                await device.websocket.send_json(
                    {
                        "type": "false_wake_stored"
                        if pending.directory == "false-wakes"
                        else "wake_observation_stored",
                        "observation_id": observation_id,
                    }
                )

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
            self._persist(capture_id, pending, bytes(pending.audio))
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

    def _observation_exists(self, observation_id: str) -> bool:
        return any(
            (self.corpus / "observations" / directory / f"{observation_id}.json").exists()
            for directory in ("false-wakes", "wakes")
        )

    def _persist_false_wake(self, pending: PendingFalseWake) -> None:
        directory = self.corpus / "observations" / pending.directory
        directory.mkdir(parents=True, exist_ok=True)
        wav_path = directory / f"{pending.observation_id}.wav"
        with wave.open(str(wav_path.with_suffix(".wav.tmp")), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(bytes(pending.audio))
        wav_path.with_suffix(".wav.tmp").replace(wav_path)
        metadata = {
            "observation_id": pending.observation_id,
            "kind": pending.observation_kind,
            "path": str(wav_path.relative_to(self.corpus)),
            "samples": len(pending.audio) // 2,
            "sha256": hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            "device_id": pending.device_id,
            "device_profile": pending.device_profile,
            "firmware_sha": pending.firmware_sha,
            "audio": pending.audio_profile,
            "received_at": pending.received_at,
            **pending.metadata,
        }
        (directory / f"{pending.observation_id}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )

    def _validate_split_assignment(self, request: dict) -> None:
        manifest_path = self.corpus / MANIFEST_NAME
        if not manifest_path.exists():
            raise ValueError(
                "initialize the corpus and register speakers before capture"
            )
        manifest = json.loads(manifest_path.read_text())
        speaker = manifest.get("speakers", {}).get(request["speaker_id"])
        if speaker is None:
            raise ValueError("speaker is not registered in the corpus")
        if speaker.get("split") != request["split"]:
            raise ValueError("capture split differs from registered speaker split")
        expected_kind = {
            "human": "human",
            "synthetic_playback": "synthetic",
            "ambient": "ambient",
        }.get(request["source"])
        if expected_kind and speaker.get("kind") != expected_kind:
            raise ValueError("capture source differs from registered speaker kind")
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

    def _persist(self, capture_id: str, pending: PendingCapture, pcm: bytes) -> None:
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
                "schema_version": 2,
                "corpus_id": "hiphi-device-v1",
                "device_profiles": {},
                "speakers": {},
                "captures": [],
            }
        profile = {"audio": pending.audio_profile}
        existing = manifest["device_profiles"].get(pending.device_profile)
        if existing is not None and existing != profile:
            destination.unlink(missing_ok=True)
            raise ValueError(
                f"device profile {pending.device_profile} changed its audio contract"
            )
        manifest["device_profiles"][pending.device_profile] = profile
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
                "firmware_sha": pending.firmware_sha,
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
