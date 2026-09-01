#!/usr/bin/env python3
"""Capture train-voice Kizz Control renders through the connected StackChan.

This corpus is teacher-adaptation input, never qualification evidence. Source
rows must already have passed the pinned phoneme alignment gate, must belong to
the train split, and must be provider/voice-disjoint from the locked target-
device qualification corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
from pathlib import Path
from typing import Any, Sequence

try:
    from tools.capture_kizz_control_device_replays import (
        PROVIDERS,
        _canonical_json,
        _capture_id,
        _capture_ids,
        _json_request,
        _sha256_file,
        _speaker_id,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from capture_kizz_control_device_replays import (  # type: ignore[no-redef]
        PROVIDERS,
        _canonical_json,
        _capture_id,
        _capture_ids,
        _json_request,
        _sha256_file,
        _speaker_id,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )


SELECTION_ALGORITHM = "provider_train_voice_round_robin_v1"
EVIDENCE_ROLE = "teacher_adaptation_target_channel_positive"


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"manifest contains no examples: {path}")
    return [dict(row) for row in rows]


def _provider_voice(row: dict[str, Any]) -> tuple[str, str]:
    conditions = row.get("conditions", {})
    provider = str(row.get("provider") or conditions.get("source_provider") or "").lower()
    voice = str(row.get("voice") or conditions.get("source_voice") or "").lower()
    if provider not in PROVIDERS or not voice:
        raise ValueError("row lacks an approved provider/voice identity")
    return provider, voice


def select_rows(
    aligned_manifest: Path,
    qualification_evidence: Path,
    *,
    providers: Sequence[str],
    per_provider: int,
) -> list[dict[str, Any]]:
    heldout_voices = {_provider_voice(row) for row in _rows(qualification_evidence)}
    eligible = []
    for row in _rows(aligned_manifest):
        if (
            int(row.get("label", -1)) != 1
            or row.get("split") != "train"
            or row.get("target_id") != "kizz-control"
            or row.get("training_eligible") is not True
            or row.get("alignment", {})
            .get("pronunciation_decision", {})
            .get("accepted")
            is not True
        ):
            continue
        if _provider_voice(row) in heldout_voices:
            raise ValueError("train source provider/voice overlaps qualification evidence")
        eligible.append(row)

    selected: list[dict[str, Any]] = []
    for provider in providers:
        by_voice: dict[str, list[dict[str, Any]]] = {}
        for row in eligible:
            row_provider, voice = _provider_voice(row)
            if row_provider == provider:
                by_voice.setdefault(voice, []).append(row)
        for rows in by_voice.values():
            rows.sort(
                key=lambda row: (
                    str(row.get("render_text", "")),
                    str(row.get("audio_sha256", "")),
                )
            )
        voices = sorted(by_voice)
        available = sum(len(rows) for rows in by_voice.values())
        if available < per_provider:
            raise ValueError(
                f"provider {provider} has only {available} eligible train renders; "
                f"needs {per_provider}"
            )
        # Exhaust one render per voice before taking a second render from any
        # voice.  This keeps voice diversity primary while allowing a larger
        # target-channel corpus than the number of available TTS voices.
        positions = {voice: 0 for voice in voices}
        provider_rows: list[dict[str, Any]] = []
        while len(provider_rows) < per_provider:
            made_progress = False
            for voice in voices:
                position = positions[voice]
                if position >= len(by_voice[voice]):
                    continue
                provider_rows.append(by_voice[voice][position])
                positions[voice] += 1
                made_progress = True
                if len(provider_rows) == per_provider:
                    break
            if not made_progress:
                raise ValueError(f"provider {provider} selection stalled")
        selected.extend(provider_rows)
    hashes = [str(row.get("audio_sha256", "")) for row in selected]
    if any(not value for value in hashes) or len(hashes) != len(set(hashes)):
        raise ValueError("selected adaptation sources have missing/duplicate audio")
    return selected


def _selection_payload(
    aligned_manifest: Path,
    qualification_evidence: Path,
    rows: Sequence[dict[str, Any]],
    *,
    providers: Sequence[str],
    per_provider: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kizz_control_teacher_adaptation_device_replay_selection",
        "selection_algorithm": SELECTION_ALGORITHM,
        "aligned_manifest": str(aligned_manifest.resolve()),
        "aligned_manifest_sha256": _sha256_file(aligned_manifest),
        "qualification_evidence": str(qualification_evidence.resolve()),
        "qualification_evidence_sha256": _sha256_file(qualification_evidence),
        "providers": list(providers),
        "per_provider": per_provider,
        "selected_count": len(rows),
        "selected_examples": list(rows),
        "locked_before_teacher_adaptation": True,
    }
    payload["selection_sha256"] = hashlib.sha256(
        _canonical_json(payload)
    ).hexdigest()
    return payload


def lock_selection(
    path: Path,
    aligned_manifest: Path,
    qualification_evidence: Path,
    *,
    providers: Sequence[str],
    per_provider: int,
) -> list[dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text())
        declared = payload.pop("selection_sha256", None)
        actual = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if declared != actual:
            raise ValueError("adaptation replay selection hash mismatch")
        if (
            payload.get("selection_algorithm") != SELECTION_ALGORITHM
            or payload.get("aligned_manifest_sha256") != _sha256_file(aligned_manifest)
            or payload.get("qualification_evidence_sha256")
            != _sha256_file(qualification_evidence)
            or payload.get("providers") != list(providers)
            or payload.get("per_provider") != per_provider
        ):
            raise ValueError("locked adaptation replay selection contract changed")
        return [dict(row) for row in payload.get("selected_examples", [])]
    rows = select_rows(
        aligned_manifest,
        qualification_evidence,
        providers=providers,
        per_provider=per_provider,
    )
    _write_locked_selection(
        path,
        _selection_payload(
            aligned_manifest,
            qualification_evidence,
            rows,
            providers=providers,
            per_provider=per_provider,
        ),
    )
    return rows


def _ensure_corpus(
    corpus: Path,
    rows: Sequence[dict[str, Any]],
    *,
    device_profile: str,
    audio_profile: dict[str, Any],
) -> None:
    manifest_path = corpus / "device-corpus.json"
    speakers = {
        _speaker_id(row): {
            "kind": "synthetic",
            "age_group": "unknown",
            "split": "train",
            "provider": row["provider"],
            "voice": row.get("voice"),
            "voice_id": row.get("voice_id"),
        }
        for row in rows
    }
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        if set(payload.get("speakers", {})) != set(speakers):
            raise ValueError("device adaptation corpus speaker contract changed")
        return
    corpus.mkdir(parents=True, exist_ok=True)
    _write_locked_selection(
        manifest_path,
        {
            "schema_version": 2,
            "corpus_id": "kizz-control-teacher-adaptation-device-replays-v1",
            "device_profiles": {device_profile: {"audio": audio_profile}},
            "speakers": speakers,
            "captures": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--qualification-evidence", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8091")
    parser.add_argument("--device-id", default="kizz-1")
    parser.add_argument(
        "--device-profile", default="m5stack_stackchan_k151_cores3_room_scale_v2"
    )
    parser.add_argument("--provider", action="append", choices=PROVIDERS)
    parser.add_argument("--per-provider", type=int, default=4)
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--lead-seconds", type=float, default=0.55)
    parser.add_argument("--volume", type=float, default=0.45)
    parser.add_argument("--persist-timeout", type=float, default=180.0)
    parser.add_argument("--capture-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    if (
        args.per_provider < 1
        or not 500 <= args.duration_ms <= 5000
        or not 0 <= args.lead_seconds <= 2
        or not 0 < args.volume <= 1
        or args.persist_timeout <= 0
        or not 1 <= args.capture_attempts <= 10
    ):
        parser.error("invalid replay capture settings")
    providers = tuple(args.provider or PROVIDERS)
    devices = _json_request(args.service_url.rstrip("/") + "/v1/devices").get(
        "devices", []
    )
    connected = next(
        (
            row
            for row in devices
            if row.get("device_id") == args.device_id
            and row.get("device_profile") == args.device_profile
        ),
        None,
    )
    if connected is None:
        parser.error("the requested StackChan is not connected to enrollment")
    rows = lock_selection(
        args.selection,
        args.aligned_manifest,
        args.qualification_evidence,
        providers=providers,
        per_provider=args.per_provider,
    )
    _ensure_corpus(
        args.corpus,
        rows,
        device_profile=args.device_profile,
        audio_profile=dict(connected["audio"]),
    )
    existing = _capture_ids(args.corpus)
    completed = skipped = 0
    for row in rows:
        capture_id = "adapt-" + _capture_id(row)
        if capture_id in existing:
            skipped += 1
            continue
        provider, voice = _provider_voice(row)
        request = {
            "capture_id": capture_id,
            "device_id": args.device_id,
            "device_profile": args.device_profile,
            "speaker_id": _speaker_id(row),
            "session_id": f"kc-adaptation-{provider}-{voice}-v1",
            "phrase": "Kizz Control",
            "pronunciation": "canonical",
            "truth": "positive",
            "source": "synthetic_playback",
            "split": "train",
            "duration_ms": args.duration_ms,
            "conditions": {
                "evidence_role": EVIDENCE_ROLE,
                "source_provider": provider,
                "source_voice": voice,
                "source_audio_sha256": row["audio_sha256"],
                "source_descriptor_sha256": row["descriptor_sha256"],
                "render_text": row.get("render_text"),
                "playback_volume": args.volume,
                "lead_seconds": args.lead_seconds,
                "selection_algorithm": SELECTION_ALGORITHM,
            },
        }
        failure: Exception | None = None
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
                failure = None
                break
            except urllib.error.HTTPError as error:
                failure = RuntimeError(
                    f"capture enqueue failed: {error.code}: "
                    + error.read().decode(errors="replace")
                )
            except (RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
                failure = error
            if capture_id in _capture_ids(args.corpus):
                failure = None
                break
            if attempt < args.capture_attempts:
                _wait_for_pending_clear(
                    args.corpus,
                    capture_id,
                    service_url=args.service_url,
                    timeout=args.persist_timeout,
                )
        if failure is not None:
            raise RuntimeError(f"capture failed: {capture_id}: {failure}") from failure
        existing.add(capture_id)
        completed += 1
        print(
            json.dumps(
                {
                    "capture_id": capture_id,
                    "provider": provider,
                    "voice": voice,
                    "completed": completed,
                    "remaining": len(rows) - skipped - completed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(1.0)
    print(
        json.dumps(
            {"completed": completed, "skipped": skipped, "selected": len(rows)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
