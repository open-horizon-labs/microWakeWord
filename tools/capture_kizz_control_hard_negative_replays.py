#!/usr/bin/env python3
"""Capture provenance-locked Kizz Control hard negatives through StackChan.

The source manifest is locked before capture and may contain synthetic near-
phrase playback (``truth=hard_negative``) and ambient playback
(``truth=ambient_negative``).  Every source hash is verified before audio is
played.  Captures are train-only and are intended for detector-conditioned
verifier mining; validation and test guards belong in separate corpora.
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
        DEFAULT_CONTINUOUS_PREROLL_SECONDS,
        MIN_CONTINUOUS_PREROLL_SECONDS,
        _canonical_json,
        _capture_ids,
        _json_request,
        _sha256_file,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from capture_kizz_control_device_replays import (  # type: ignore[no-redef]
        DEFAULT_CONTINUOUS_PREROLL_SECONDS,
        MIN_CONTINUOUS_PREROLL_SECONDS,
        _canonical_json,
        _capture_ids,
        _json_request,
        _sha256_file,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )


TRUTHS = {"hard_negative", "ambient_negative"}
SOURCES = {"synthetic_playback", "ambient"}
SOURCE_KINDS = {"synthetic_playback": "synthetic", "ambient": "ambient"}
SELECTION_ALGORITHM = "physical_hard_negative_replay_matrix_v1"


def _valid_token(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 96
        and all(character.isalnum() or character in "-_.:" for character in value)
    )


def load_sources(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    raw_rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("source manifest requires nonempty examples")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            raise ValueError(f"example[{index}] must be an object")
        row = dict(raw)
        source_id = row.get("source_id")
        speaker_id = row.get("speaker_id")
        if not _valid_token(source_id) or not _valid_token(speaker_id):
            raise ValueError(f"example[{index}] has an invalid source/speaker id")
        if source_id in seen_ids:
            raise ValueError(f"duplicate source_id: {source_id}")
        truth = row.get("truth")
        source = row.get("source")
        if truth not in TRUTHS or source not in SOURCES:
            raise ValueError(f"{source_id}: invalid truth/source")
        if truth == "ambient_negative" and source != "ambient":
            raise ValueError(f"{source_id}: ambient_negative requires source=ambient")
        phrase = row.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip() or len(phrase) > 160:
            raise ValueError(f"{source_id}: invalid phrase/description")
        raw_audio = row.get("path")
        if not isinstance(raw_audio, str) or not raw_audio:
            raise ValueError(f"{source_id}: missing audio path")
        audio_path = Path(raw_audio)
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise ValueError(f"{source_id}: audio is missing: {audio_path}")
        declared_hash = row.get("audio_sha256")
        observed_hash = _sha256_file(audio_path)
        if declared_hash != observed_hash:
            raise ValueError(f"{source_id}: audio SHA-256 differs from manifest")
        if observed_hash in seen_hashes:
            raise ValueError(f"{source_id}: duplicate source audio")
        seen_ids.add(str(source_id))
        seen_hashes.add(observed_hash)
        row["path"] = str(audio_path)
        rows.append(row)
    return rows


def _capture_id(row: dict[str, Any], volume: float, repeat: int) -> str:
    material = _canonical_json(
        {
            "source_id": row["source_id"],
            "audio_sha256": row["audio_sha256"],
            "volume": volume,
            "repeat": repeat,
            "algorithm": SELECTION_ALGORITHM,
        }
    )
    return "hardneg-" + hashlib.sha256(material).hexdigest()[:20]


def _lock_selection(
    path: Path,
    source_manifest: Path,
    rows: Sequence[dict[str, Any]],
    *,
    volumes: Sequence[float],
    repeats: int,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kizz_control_physical_hard_negative_replay_selection",
        "selection_algorithm": SELECTION_ALGORITHM,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "sources": list(rows),
        "volumes": list(volumes),
        "repeats": repeats,
        "locked_before_capture": True,
    }
    payload["selection_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise ValueError("locked hard-negative replay selection changed")
        return
    _write_locked_selection(path, payload)


def _ensure_corpus(
    corpus: Path,
    rows: Sequence[dict[str, Any]],
    *,
    device_profile: str,
    audio_profile: dict[str, Any],
) -> None:
    manifest_path = corpus / "device-corpus.json"
    speakers = {
        str(row["speaker_id"]): {
            "kind": SOURCE_KINDS[str(row["source"])],
            "age_group": (
                "not_applicable" if row["source"] == "ambient" else "unknown"
            ),
            "split": "train",
            "provider": row.get("provider", "physical-replay"),
            "voice": row.get("voice"),
        }
        for row in rows
    }
    expected_profile = {"audio": audio_profile}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("speakers") != speakers:
            raise ValueError("hard-negative corpus speaker contract changed")
        if manifest.get("device_profiles", {}).get(device_profile) != expected_profile:
            raise ValueError("hard-negative corpus device profile changed")
        return
    corpus.mkdir(parents=True, exist_ok=True)
    _write_locked_selection(
        manifest_path,
        {
            "schema_version": 2,
            "corpus_id": "kizz-control-physical-hard-negatives-v1",
            "device_profiles": {device_profile: expected_profile},
            "speakers": speakers,
            "captures": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--service-url", default="http://127.0.0.1:8091")
    parser.add_argument("--device-id", default="kizz-1")
    parser.add_argument(
        "--device-profile",
        default="m5stack_stackchan_k151_cores3_room_scale_v2",
    )
    parser.add_argument("--volume", action="append", type=float)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument(
        "--lead-seconds", type=float, default=DEFAULT_CONTINUOUS_PREROLL_SECONDS
    )
    parser.add_argument("--persist-timeout", type=float, default=180.0)
    parser.add_argument("--inter-capture-seconds", type=float, default=1.0)
    parser.add_argument("--capture-attempts", type=int, default=3)
    args = parser.parse_args(argv)
    volumes = tuple(args.volume or (0.35, 0.45, 0.55))
    if (
        not volumes
        or any(not 0 < value <= 1 for value in volumes)
        or not 1 <= args.repeats <= 20
        or not 500 <= args.duration_ms <= 5000
        or not MIN_CONTINUOUS_PREROLL_SECONDS <= args.lead_seconds <= 3
        or args.persist_timeout <= 0
        or args.inter_capture_seconds < 0
        or not 1 <= args.capture_attempts <= 10
    ):
        parser.error("invalid replay matrix or capture timing")
    rows = load_sources(args.source_manifest)
    service_url = args.service_url.rstrip("/")
    devices = _json_request(service_url + "/v1/devices").get("devices", [])
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
    selection = args.selection or args.corpus / "hard-negative-selection-v1.json"
    _lock_selection(
        selection,
        args.source_manifest,
        rows,
        volumes=volumes,
        repeats=args.repeats,
    )
    _ensure_corpus(
        args.corpus,
        rows,
        device_profile=args.device_profile,
        audio_profile=dict(connected["audio"]),
    )
    existing = _capture_ids(args.corpus)
    planned = [
        (row, volume, repeat)
        for row in rows
        for volume in volumes
        for repeat in range(1, args.repeats + 1)
    ]
    completed = skipped = 0
    for row, volume, repeat in planned:
        capture_id = _capture_id(row, volume, repeat)
        if capture_id in existing:
            skipped += 1
            continue
        request = {
            "capture_id": capture_id,
            "device_id": args.device_id,
            "device_profile": args.device_profile,
            "speaker_id": row["speaker_id"],
            "session_id": "kc-hardneg-physical-v11",
            "phrase": row["phrase"],
            "truth": row["truth"],
            "source": row["source"],
            "split": "train",
            "duration_ms": args.duration_ms,
            "conditions": {
                "evidence_role": "physical_playback_hard_negative_training",
                "source_id": row["source_id"],
                "source_audio_sha256": row["audio_sha256"],
                "playback_volume": volume,
                "lead_seconds": args.lead_seconds,
                "repeat": repeat,
                "selection_algorithm": SELECTION_ALGORITHM,
                **dict(row.get("conditions", {})),
            },
        }
        failure: Exception | None = None
        for attempt in range(1, args.capture_attempts + 1):
            try:
                queued = _json_request(service_url + "/v1/captures", request)
                if queued.get("state") != "queued":
                    raise RuntimeError(f"capture was not queued: {queued}")
                time.sleep(args.lead_seconds)
                subprocess.run(
                    ["afplay", "-v", str(volume), str(row["path"])],
                    check=True,
                    timeout=15,
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
                    "source_id": row["source_id"],
                    "volume": volume,
                    "repeat": repeat,
                    "completed": completed,
                    "remaining": len(planned) - skipped - completed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        time.sleep(args.inter_capture_seconds)
    print(
        json.dumps(
            {"completed": completed, "skipped": skipped, "planned": len(planned)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
