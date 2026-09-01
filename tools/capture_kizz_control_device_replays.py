#!/usr/bin/env python3
"""Replay reserved Kizz Control evidence through a connected StackChan mic.

The source clips remain excluded from training.  This tool asks the independent
enrollment endpoint for a bounded device capture, plays one reserved clip, and
waits until the service has persisted the resulting target-channel recording.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence


PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")

# The deployed verifier window contains 2.20 seconds of pre-trigger context.
# Captures with a shorter lead produce zero-padded positive windows that cannot
# occur once the firmware has been listening continuously.  Keep a small margin
# for capture/enrollment scheduling jitter.
MIN_CONTINUOUS_PREROLL_SECONDS = 2.30
DEFAULT_CONTINUOUS_PREROLL_SECONDS = 2.50

SELECTION_SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "provider_voice_round_robin_v1"


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body is not None else {},
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError(f"{url}: expected a JSON object")
    return value


def _capture_ids(corpus: Path) -> set[str]:
    manifest_path = corpus / "device-corpus.json"
    if not manifest_path.is_file():
        return set()
    manifest = json.loads(manifest_path.read_text())
    return {str(item["capture_id"]) for item in manifest.get("captures", [])}


def _wait_for_capture(
    corpus: Path,
    capture_id: str,
    timeout: float,
    *,
    service_url: str,
) -> None:
    deadline = time.monotonic() + timeout
    missing_since: float | None = None
    while time.monotonic() < deadline:
        if capture_id in _capture_ids(corpus):
            return
        status = _json_request(service_url.rstrip("/") + "/v1/status")
        if capture_id in status.get("pending", {}):
            missing_since = None
        elif missing_since is None:
            missing_since = time.monotonic()
        elif time.monotonic() - missing_since >= 2.0:
            raise RuntimeError(
                f"device removed pending capture without persisting it: {capture_id}"
            )
        time.sleep(0.2)
    raise TimeoutError(f"device capture did not persist: {capture_id}")


def _wait_for_pending_clear(
    corpus: Path,
    capture_id: str,
    *,
    service_url: str,
    timeout: float = 15.0,
) -> None:
    """Wait until a failed device attempt is safe to enqueue again."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if capture_id in _capture_ids(corpus):
            return
        status = _json_request(service_url.rstrip("/") + "/v1/status")
        if capture_id not in status.get("pending", {}):
            return
        time.sleep(0.2)
    # Recheck both durable and transient state at the deadline. The device can
    # finish its final segment between the loop condition and this decision.
    if capture_id in _capture_ids(corpus):
        return
    status = _json_request(service_url.rstrip("/") + "/v1/status")
    if capture_id not in status.get("pending", {}):
        return
    raise TimeoutError(f"failed capture remained pending: {capture_id}")


def _capture_id(row: dict[str, Any]) -> str:
    provider = re.sub(r"[^a-z0-9]+", "-", str(row["provider"]).casefold()).strip("-")
    return f"kc-replay-{provider}-{str(row['descriptor_sha256'])[:16]}"


def _speaker_id(row: dict[str, Any]) -> str:
    provider = re.sub(r"[^a-z0-9]+", "-", str(row["provider"]).casefold()).strip("-")
    voice = re.sub(
        r"[^a-z0-9]+",
        "-",
        str(row.get("voice_id") or row.get("voice") or "unknown").casefold(),
    ).strip("-")
    return f"replay-{provider}-{voice}"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_manifest_sha256(manifest: Path) -> str:
    return _sha256_file(manifest)


def _reserved_rows(manifest: Path, provider: str) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text())
    return [
        dict(row)
        for row in payload.get("examples", [])
        if row.get("provider") == provider
        and int(row.get("label", -1)) == 1
        and row.get("reserved_evidence_role") == "target_channel_positive"
        and row.get("training_eligible") is False
    ]


def _stratified_provider_rows(
    rows: Sequence[dict[str, Any]], *, provider: str, per_provider: int
) -> list[dict[str, Any]]:
    """Select one clip per voice per round, with deterministic tie-breaking."""
    by_voice: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        voice = str(row.get("voice_id") or row.get("voice") or "")
        audio_hash = str(row.get("audio_sha256") or "")
        if not voice or not audio_hash:
            raise ValueError(f"{provider} reserved row lacks voice or audio hash")
        by_voice.setdefault(voice, []).append(dict(row))
    for voice in by_voice:
        by_voice[voice].sort(
            key=lambda row: (
                str(row.get("render_text") or ""),
                str(row.get("source_id") or ""),
                str(row.get("descriptor_sha256") or ""),
            )
        )
    selected: list[dict[str, Any]] = []
    positions = {voice: 0 for voice in sorted(by_voice)}
    used_audio: set[str] = set()
    while len(selected) < per_provider:
        made_progress = False
        for voice in sorted(by_voice):
            candidates = by_voice[voice]
            while positions[voice] < len(candidates):
                row = candidates[positions[voice]]
                positions[voice] += 1
                audio_hash = str(row["audio_sha256"])
                if audio_hash in used_audio:
                    continue
                used_audio.add(audio_hash)
                selected.append(row)
                made_progress = True
                break
            if len(selected) >= per_provider:
                break
        if not made_progress:
            break
    if len(selected) < per_provider:
        raise ValueError(
            f"provider {provider} has only {len(selected)} unique stratified reserved clips; "
            f"needs {per_provider}"
        )
    return selected


def _validate_selected_rows(rows: Sequence[dict[str, Any]]) -> None:
    hashes = [str(row.get("audio_sha256") or "") for row in rows]
    if any(not value for value in hashes):
        raise ValueError("selected evidence row lacks audio_sha256")
    if len(hashes) != len(set(hashes)):
        raise ValueError("selected evidence contains duplicate audio hashes")


def _selection_payload(
    manifest: Path,
    providers: Sequence[str],
    per_provider: int,
    rows: Sequence[dict[str, Any]],
    *,
    existing_capture_ids: Sequence[str],
    pronunciation_audit: Path | None = None,
) -> dict[str, Any]:
    _validate_selected_rows(rows)
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_algorithm": SELECTION_ALGORITHM,
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": _source_manifest_sha256(manifest),
        "providers": list(providers),
        "per_provider": per_provider,
        "selected_count": len(rows),
        "selected_audio_sha256": [str(row["audio_sha256"]) for row in rows],
        "selected_examples": [dict(row) for row in rows],
        "preserved_existing_v1_capture_ids": sorted(str(value) for value in existing_capture_ids),
        "locked_before_teacher_scoring": True,
    }
    if pronunciation_audit is not None:
        payload["source_pronunciation_audit"] = {
            "path": str(pronunciation_audit.resolve()),
            "sha256": _sha256_file(pronunciation_audit),
        }
    payload["selection_sha256"] = hashlib.sha256(
        _canonical_json({key: value for key, value in payload.items() if key != "selection_sha256"})
    ).hexdigest()
    return payload


def _write_locked_selection(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=path.name, dir=path.parent, delete=False
    ) as temporary:
        temporary.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _ensure_capture_corpus(
    corpus: Path,
    rows: Sequence[dict[str, Any]],
    *,
    device_profile: str,
    audio_profile: dict[str, Any],
) -> None:
    """Initialize the independently registered replay corpus before capture."""
    manifest_path = corpus / "device-corpus.json"
    speakers = {
        _speaker_id(row): {
            "kind": "synthetic",
            "age_group": "unknown",
            "split": "test",
            "provider": row["provider"],
            "voice": row.get("voice"),
            "voice_id": row.get("voice_id"),
        }
        for row in rows
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        missing = sorted(set(speakers) - set(manifest.get("speakers", {})))
        if missing:
            raise ValueError(f"capture corpus lacks locked replay speakers: {missing}")
        if manifest.get("device_profiles", {}).get(device_profile, {}).get("audio") != audio_profile:
            raise ValueError("capture corpus device audio profile differs from connected StackChan")
        return
    corpus.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "corpus_id": "kizz-control-voice-stratified-device-replays-v2",
        "device_profiles": {device_profile: {"audio": audio_profile}},
        "speakers": speakers,
        "captures": [],
    }
    _write_locked_selection(manifest_path, payload)


def _load_locked_selection(
    path: Path,
    *,
    manifest: Path,
    providers: Sequence[str],
    per_provider: int,
    pronunciation_audit: Path | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    expected = hashlib.sha256(
        _canonical_json(
            {key: value for key, value in payload.items() if key != "selection_sha256"}
        )
    ).hexdigest()
    if payload.get("selection_sha256") != expected:
        raise ValueError(f"selected evidence manifest is corrupted: {path}")
    if payload.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise ValueError(f"unsupported selected evidence manifest: {path}")
    if payload.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise ValueError(f"selected evidence algorithm mismatch: {path}")
    if payload.get("source_manifest_sha256") != _source_manifest_sha256(manifest):
        raise ValueError("selected evidence is stale: source manifest changed")
    if payload.get("providers") != list(providers) or payload.get("per_provider") != per_provider:
        raise ValueError("selected evidence is locked to a different provider/count contract")
    if pronunciation_audit is not None and payload.get("source_pronunciation_audit", {}).get(
        "sha256"
    ) != _sha256_file(pronunciation_audit):
        raise ValueError("selected evidence is locked to a different pronunciation audit")
    rows = [dict(row) for row in payload.get("selected_examples", [])]
    _validate_selected_rows(rows)
    return rows


def lock_selected_evidence(
    manifest: Path,
    selection_path: Path,
    providers: Sequence[str],
    *,
    per_provider: int,
    existing_capture_ids: Sequence[str] = (),
    pronunciation_audit: Path | None = None,
) -> list[dict[str, Any]]:
    """Create or reuse an immutable, pre-scoring source-evidence selection."""
    if selection_path.exists():
        return _load_locked_selection(
            selection_path,
            manifest=manifest,
            providers=providers,
            per_provider=per_provider,
            pronunciation_audit=pronunciation_audit,
        )
    rows: list[dict[str, Any]] = []
    used_audio: set[str] = set()
    for provider in providers:
        selected = _stratified_provider_rows(
            _reserved_rows(manifest, provider),
            provider=provider,
            per_provider=per_provider,
        )
        for row in selected:
            audio_hash = str(row["audio_sha256"])
            if audio_hash in used_audio:
                raise ValueError("reserved evidence duplicates audio across providers")
            used_audio.add(audio_hash)
            rows.append(row)
    payload = _selection_payload(
        manifest,
        providers,
        per_provider,
        rows,
        existing_capture_ids=existing_capture_ids,
        pronunciation_audit=pronunciation_audit,
    )
    _write_locked_selection(selection_path, payload)
    return rows


def replay_rows(
    manifest: Path,
    providers: Sequence[str],
    *,
    per_provider: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    used_audio: set[str] = set()
    for provider in providers:
        selected = _stratified_provider_rows(
            _reserved_rows(manifest, provider),
            provider=provider,
            per_provider=per_provider,
        )
        for row in selected:
            audio_hash = str(row["audio_sha256"])
            if audio_hash in used_audio:
                raise ValueError("reserved evidence duplicates audio across providers")
            used_audio.add(audio_hash)
            rows.append(row)
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8091")
    parser.add_argument("--device-id", default="kizz-1")
    parser.add_argument(
        "--device-profile",
        default="m5stack_stackchan_k151_cores3_room_scale_v2",
    )
    parser.add_argument(
        "--provider",
        action="append",
        choices=PROVIDERS,
    )
    parser.add_argument("--per-provider", type=int, default=6)
    parser.add_argument(
        "--selected-evidence-manifest",
        type=Path,
        help="locked source selection; defaults to <corpus>/selected-evidence-v1.json",
    )
    parser.add_argument("--source-pronunciation-audit", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument(
        "--lead-seconds", type=float, default=DEFAULT_CONTINUOUS_PREROLL_SECONDS
    )
    parser.add_argument("--volume", type=float, default=0.55)
    # The current firmware deliberately resumes a segmented upload one KiB at a
    # time.  On a busy 2.4 GHz network a five-second capture can therefore take
    # well over a minute to persist even though every segment is making
    # progress.  Keep the CLI timeout above the device/server capture timeout;
    # treating normal back-pressure as a failed recording loses evidence and
    # can cause the next request to collide with an upload still in flight.
    parser.add_argument("--persist-timeout", type=float, default=180.0)
    parser.add_argument("--inter-capture-seconds", type=float, default=1.0)
    parser.add_argument(
        "--capture-attempts",
        type=int,
        default=3,
        help="bounded retries for a device/network upload failure",
    )
    args = parser.parse_args(argv)
    if (
        args.per_provider < 1
        or not 500 <= args.duration_ms <= 5000
        or not MIN_CONTINUOUS_PREROLL_SECONDS <= args.lead_seconds <= 3
        or not 0 < args.volume <= 1
        or args.persist_timeout <= 0
        or args.inter_capture_seconds < 0
        or not 1 <= args.capture_attempts <= 10
    ):
        parser.error("invalid replay timing, count, or volume")
    providers = tuple(args.provider or PROVIDERS)
    devices = _json_request(args.service_url.rstrip("/") + "/v1/devices").get(
        "devices", []
    )
    connected = next(
        (
            item
            for item in devices
            if item.get("device_id") == args.device_id
            and item.get("device_profile") == args.device_profile
        ),
        None,
    )
    if connected is None:
        parser.error("the requested StackChan is not connected to enrollment")
    selection_path = args.selected_evidence_manifest or (
        args.corpus / "selected-evidence-v1.json"
    )
    rows = lock_selected_evidence(
        args.source_manifest,
        selection_path,
        providers,
        per_provider=args.per_provider,
        existing_capture_ids=_capture_ids(args.corpus),
        pronunciation_audit=args.source_pronunciation_audit,
    )
    audit = json.loads(args.source_pronunciation_audit.read_text())
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("qualified") is not True
        or audit.get("locked_before_device_capture") is not True
        or audit.get("source_manifest_sha256") != _source_manifest_sha256(args.source_manifest)
    ):
        raise ValueError("source-pronunciation audit is not a qualified pre-capture gate")
    if set(audit.get("reserved_audio_sha256", [])) != {
        str(row["audio_sha256"]) for row in rows
    }:
        raise ValueError("selected evidence differs from pronunciation-qualified reserved audio")
    _ensure_capture_corpus(
        args.corpus,
        rows,
        device_profile=args.device_profile,
        audio_profile=dict(connected["audio"]),
    )
    existing = _capture_ids(args.corpus)
    completed = skipped = 0
    for row in rows:
        capture_id = _capture_id(row)
        if capture_id in existing:
            skipped += 1
            continue
        provider = str(row["provider"])
        request = {
            "capture_id": capture_id,
            "device_id": args.device_id,
            "device_profile": args.device_profile,
            "speaker_id": _speaker_id(row),
            "session_id": f"kc-replay-{provider}-voice-stratified-test-v2",
            "phrase": "Kizz Control",
            "pronunciation": "canonical",
            "truth": "positive",
            "source": "synthetic_playback",
            "split": "test",
            "duration_ms": args.duration_ms,
            "conditions": {
                "evidence_role": "reserved_target_channel_positive",
                "source_provider": provider,
                "source_voice": row.get("voice"),
                "source_audio_sha256": row.get("audio_sha256"),
                "source_descriptor_sha256": row.get("descriptor_sha256"),
                "render_text": row.get("render_text"),
                "playback_volume": args.volume,
                "lead_seconds": args.lead_seconds,
            },
        }
        for attempt in range(1, args.capture_attempts + 1):
            try:
                queued = _json_request(
                    args.service_url.rstrip("/") + "/v1/captures", request
                )
                if queued.get("state") != "queued":
                    raise RuntimeError(f"capture was not queued: {queued}")
                time.sleep(args.lead_seconds)
                subprocess.run(
                    ["afplay", "-v", str(args.volume), str(Path(row["path"]))],
                    check=True,
                    timeout=10,
                )
                _wait_for_capture(
                    args.corpus,
                    capture_id,
                    args.persist_timeout,
                    service_url=args.service_url,
                )
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                failure: Exception = RuntimeError(
                    f"capture enqueue failed: {error.code}: {detail}"
                )
            except (RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
                failure = error
            if capture_id in _capture_ids(args.corpus):
                break
            if attempt >= args.capture_attempts:
                raise RuntimeError(
                    f"capture failed after {attempt} attempts: {capture_id}: {failure}"
                ) from failure
            _wait_for_pending_clear(
                args.corpus,
                capture_id,
                service_url=args.service_url,
                timeout=args.persist_timeout,
            )
            print(
                json.dumps(
                    {
                        "capture_id": capture_id,
                        "retry": attempt + 1,
                        "reason": str(failure),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(min(5.0, float(2 ** (attempt - 1))))
        existing.add(capture_id)
        completed += 1
        print(
            json.dumps(
                {
                    "capture_id": capture_id,
                    "provider": provider,
                    "render_text": row.get("render_text"),
                    "completed": completed,
                    "total": len(rows) - skipped,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        # Persistence happens in the HTTP handler just before the final segment
        # response reaches the device.  Give its upload task time to clear the
        # in-flight flag before asking it to accept another directed capture.
        time.sleep(args.inter_capture_seconds)
    print(
        json.dumps(
            {"completed": completed, "skipped_existing": skipped, "selected": len(rows)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
