#!/usr/bin/env python3
"""Capture the held-out target-channel Kizz Control adaptation validation set.

This is deliberately a separate pipeline from the train adaptation replay
tool.  It selects one accepted, phone-aligned validation render for each
provider voice represented by the locked target-device qualification set,
then records those renders through the independent enrollment endpoint.
Validation captures are never training rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.error
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from tools.capture_kizz_control_device_replays import (
        PROVIDERS,
        _canonical_json,
        _json_request,
        _sha256_file,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from capture_kizz_control_device_replays import (  # type: ignore[no-redef]
        PROVIDERS,
        _canonical_json,
        _json_request,
        _sha256_file,
        _wait_for_capture,
        _wait_for_pending_clear,
        _write_locked_selection,
    )


SELECTION_ALGORITHM = "provider_voice_aligned_validation_round_robin_v1"
EVIDENCE_ROLE = "teacher_adaptation_target_channel_validation_positive"
CORPUS_ID = "kizz-control-teacher-adaptation-validation-device-replays-v1"


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


def _hashes(row: dict[str, Any]) -> set[str]:
    conditions = row.get("conditions", {})
    values = {
        row.get("audio_sha256"),
        row.get("sha256"),
        row.get("source_audio_sha256"),
        conditions.get("source_audio_sha256"),
    }
    return {str(value) for value in values if value}


def _identities(row: dict[str, Any]) -> set[tuple[str, str]]:
    conditions = row.get("conditions", {})
    identities: set[tuple[str, str]] = set()
    for field in (
        "source_id",
        "provenance_id",
        "parent_source_id",
        "parent_id",
        "speaker_id",
        "session_id",
    ):
        value = row.get(field)
        if value:
            identities.add((field, str(value)))
    for field in ("source_descriptor_sha256", "source_id", "parent_source_id"):
        value = conditions.get(field)
        if value:
            identities.add((field, str(value)))
    return identities


def _provider_voice_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    voices: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        provider, voice = _provider_voice(row)
        voices[provider].add(voice)
    return {provider: len(voices.get(provider, set())) for provider in PROVIDERS}


def _evidence_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"qualification evidence contains no examples: {path}")
    return [dict(row) for row in rows]


def _corpus_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("captures") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"device corpus contains no captures: {path}")
    return [dict(row) for row in rows]


def _selection_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("selected_examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"locked selection contains no selected_examples: {path}")
    return [dict(row) for row in rows]


def _assert_disjoint(
    candidates: Iterable[dict[str, Any]],
    excluded: Iterable[dict[str, Any]],
    *,
    label: str,
) -> None:
    excluded_voices = {_provider_voice(row) for row in excluded}
    excluded_hashes = set().union(*(_hashes(row) for row in excluded)) if excluded else set()
    excluded_ids = set().union(*(_identities(row) for row in excluded)) if excluded else set()
    for row in candidates:
        provider_voice = _provider_voice(row)
        if provider_voice in excluded_voices:
            raise ValueError(f"{label} provider/voice overlap: {provider_voice}")
        overlap = _hashes(row) & excluded_hashes
        if overlap:
            raise ValueError(f"{label} audio/source hash overlap: {sorted(overlap)}")
        overlap_ids = _identities(row) & excluded_ids
        if overlap_ids:
            raise ValueError(f"{label} provenance overlap: {sorted(overlap_ids)}")


def _is_accepted_validation_row(row: dict[str, Any]) -> bool:
    alignment = row.get("alignment") or {}
    decision = alignment.get("pronunciation_decision") or {}
    return (
        int(row.get("label", -1)) == 1
        and row.get("split") == "validation"
        and row.get("target_id") == "kizz-control"
        and row.get("training_eligible") is True
        and decision.get("accepted") is True
        and "forced_alignment" in str(alignment.get("method", ""))
        and bool(row.get("target_phones"))
        and bool(row.get("audio_sha256"))
        and bool(row.get("path"))
    )


def _eligible_validation_rows(aligned_manifest: Path) -> list[dict[str, Any]]:
    return [row for row in _rows(aligned_manifest) if _is_accepted_validation_row(row)]


def select_rows(
    aligned_manifest: Path,
    qualification_evidence: Path,
    train_corpus: Path,
    train_selection: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Select one deterministic validation source per expected provider voice."""
    target_rows = _evidence_rows(qualification_evidence)

    train_rows = _corpus_rows(train_corpus) + _selection_rows(train_selection)
    aligned_rows = _rows(aligned_manifest)
    candidates = [row for row in aligned_rows if _is_accepted_validation_row(row)]
    expected = _provider_voice_counts(candidates)
    if any(expected[provider] == 0 for provider in PROVIDERS):
        raise ValueError(f"eligible validation inventory must contain all providers: {expected}")
    _assert_disjoint(candidates, target_rows, label="target qualification")
    _assert_disjoint(candidates, train_rows, label="train device corpus/selection")

    by_voice: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_voice[_provider_voice(row)].append(row)
    selected: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        voices = sorted(voice for (item_provider, voice) in by_voice if item_provider == provider)
        wanted = expected[provider]
        if len(voices) != wanted:
            raise ValueError(f"validation provider {provider} inventory count drifted")
        for voice in voices:
            options = sorted(
                by_voice[(provider, voice)],
                key=lambda row: (
                    str(row.get("render_text", "")),
                    str(row.get("audio_sha256", "")),
                    str(row.get("source_id", "")),
                ),
            )
            selected.append(options[0])
    hashes = [str(row["audio_sha256"]) for row in selected]
    if len(hashes) != len(set(hashes)):
        raise ValueError("validation sources contain duplicate audio hashes")
    return selected, expected


def _selection_payload(
    aligned_manifest: Path,
    qualification_evidence: Path,
    train_corpus: Path,
    train_selection: Path,
    rows: Sequence[dict[str, Any]],
    expected: dict[str, int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "kizz_control_teacher_adaptation_validation_device_replay_selection",
        "selection_algorithm": SELECTION_ALGORITHM,
        "aligned_manifest": str(aligned_manifest.resolve()),
        "aligned_manifest_sha256": _sha256_file(aligned_manifest),
        "qualification_evidence": str(qualification_evidence.resolve()),
        "qualification_evidence_sha256": _sha256_file(qualification_evidence),
        "train_corpus": str(train_corpus.resolve()),
        "train_corpus_sha256": _sha256_file(train_corpus),
        "train_selection": str(train_selection.resolve()),
        "train_selection_sha256": _sha256_file(train_selection),
        "providers": list(PROVIDERS),
        "expected_voice_counts": expected,
        "selected_count": len(rows),
        "selected_examples": [dict(row) for row in rows],
        "locked_before_device_capture": True,
    }
    payload["selection_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return payload


def lock_selection(
    path: Path,
    aligned_manifest: Path,
    qualification_evidence: Path,
    train_corpus: Path,
    train_selection: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if path.is_file():
        payload = json.loads(path.read_text())
        declared = payload.pop("selection_sha256", None)
        if declared != hashlib.sha256(_canonical_json(payload)).hexdigest():
            raise ValueError("validation replay selection hash mismatch")
        eligible = _eligible_validation_rows(aligned_manifest)
        expected = _provider_voice_counts(eligible)
        if any(expected[provider] == 0 for provider in PROVIDERS):
            raise ValueError(f"eligible validation inventory must contain all providers: {expected}")
        required = {
            "selection_algorithm": SELECTION_ALGORITHM,
            "aligned_manifest_sha256": _sha256_file(aligned_manifest),
            "qualification_evidence_sha256": _sha256_file(qualification_evidence),
            "train_corpus_sha256": _sha256_file(train_corpus),
            "train_selection_sha256": _sha256_file(train_selection),
            "expected_voice_counts": expected,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise ValueError("locked validation replay selection contract changed")
        rows = [dict(row) for row in payload.get("selected_examples", [])]
        if payload.get("selected_count") != len(rows):
            raise ValueError("locked validation replay selection count drifted")
        return rows, expected
    rows, expected = select_rows(
        aligned_manifest, qualification_evidence, train_corpus, train_selection
    )
    _write_locked_selection(
        path,
        _selection_payload(
            aligned_manifest,
            qualification_evidence,
            train_corpus,
            train_selection,
            rows,
            expected,
        ),
    )
    return rows, expected


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def _capture_id(row: dict[str, Any]) -> str:
    provider, _ = _provider_voice(row)
    return f"kc-adaptation-validation-{_safe_id(provider)}-{str(row['descriptor_sha256'])[:16]}"


def _speaker_id(row: dict[str, Any]) -> str:
    provider, voice = _provider_voice(row)
    return f"replay-validation-{_safe_id(provider)}-{_safe_id(str(row.get('voice_id') or voice))}"


def build_capture_request(
    row: dict[str, Any], *, device_id: str, device_profile: str, duration_ms: int, volume: float, lead_seconds: float
) -> dict[str, Any]:
    provider, voice = _provider_voice(row)
    return {
        "capture_id": _capture_id(row),
        "device_id": device_id,
        "device_profile": device_profile,
        "speaker_id": _speaker_id(row),
        "session_id": f"kc-adaptation-validation-{provider}-{voice}-v1",
        "phrase": "Kizz Control",
        "pronunciation": "canonical",
        "truth": "positive",
        "source": "synthetic_playback",
        "split": "validation",
        "duration_ms": duration_ms,
        "conditions": {
            "evidence_role": EVIDENCE_ROLE,
            "source_provider": provider,
            "source_voice": voice,
            "source_audio_sha256": row["audio_sha256"],
            "source_descriptor_sha256": row["descriptor_sha256"],
            "render_text": row.get("render_text"),
            "playback_volume": volume,
            "lead_seconds": lead_seconds,
            "selection_algorithm": SELECTION_ALGORITHM,
        },
    }


def _capture_ids(corpus: Path) -> set[str]:
    if not (corpus / "device-corpus.json").is_file():
        return set()
    payload = json.loads((corpus / "device-corpus.json").read_text())
    return {str(row["capture_id"]) for row in payload.get("captures", [])}


def _ensure_corpus(corpus: Path, rows: Sequence[dict[str, Any]], *, device_profile: str, audio_profile: dict[str, Any]) -> None:
    path = corpus / "device-corpus.json"
    speakers = {
        _speaker_id(row): {
            "kind": "synthetic",
            "age_group": "unknown",
            "split": "validation",
            "provider": row["provider"],
            "voice": row.get("voice"),
            "voice_id": row.get("voice_id"),
        }
        for row in rows
    }
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("corpus_id") != CORPUS_ID:
            raise ValueError("existing corpus is not the validation replay corpus")
        if set(payload.get("speakers", {})) != set(speakers):
            raise ValueError("validation replay corpus speaker contract changed")
        return
    corpus.mkdir(parents=True, exist_ok=True)
    _write_locked_selection(
        path,
        {
            "schema_version": 2,
            "corpus_id": CORPUS_ID,
            "device_profiles": {device_profile: {"audio": audio_profile}},
            "speakers": speakers,
            "captures": [],
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--qualification-evidence", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--service-url", default="http://127.0.0.1:8091")
    parser.add_argument("--device-id", default="kizz-1")
    parser.add_argument("--device-profile", default="m5stack_stackchan_k151_cores3_room_scale_v2")
    parser.add_argument("--duration-ms", type=int, default=5000)
    parser.add_argument("--lead-seconds", type=float, default=0.55)
    parser.add_argument("--volume", type=float, default=0.45)
    parser.add_argument("--persist-timeout", type=float, default=180.0)
    parser.add_argument("--capture-attempts", type=int, default=3)
    parser.add_argument("--inter-capture-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    if not 500 <= args.duration_ms <= 5000 or not 0 < args.volume <= 1 or args.lead_seconds < 0 or args.persist_timeout <= 0 or not 1 <= args.capture_attempts <= 10 or args.inter_capture_seconds < 0:
        parser.error("invalid validation replay settings")
    devices = _json_request(args.service_url.rstrip("/") + "/v1/devices").get("devices", [])
    connected = next((item for item in devices if item.get("device_id") == args.device_id and item.get("device_profile") == args.device_profile), None)
    if connected is None:
        parser.error("the requested StackChan is not connected to enrollment")
    rows, expected = lock_selection(args.selection, args.aligned_manifest, args.qualification_evidence, args.train_corpus, args.train_selection)
    if len(rows) != sum(expected.values()):
        raise ValueError("locked validation selection does not realize expected voice counts")
    _ensure_corpus(args.corpus, rows, device_profile=args.device_profile, audio_profile=dict(connected["audio"]))
    existing = _capture_ids(args.corpus)
    completed = skipped = 0
    for row in rows:
        capture_id = _capture_id(row)
        if capture_id in existing:
            skipped += 1
            continue
        request = build_capture_request(row, device_id=args.device_id, device_profile=args.device_profile, duration_ms=args.duration_ms, volume=args.volume, lead_seconds=args.lead_seconds)
        failure: Exception | None = None
        for attempt in range(1, args.capture_attempts + 1):
            try:
                queued = _json_request(args.service_url.rstrip("/") + "/v1/captures", request)
                if queued.get("state") != "queued":
                    raise RuntimeError(f"capture was not queued: {queued}")
                time.sleep(args.lead_seconds)
                subprocess.run(["afplay", "-v", str(args.volume), str(Path(row["path"]))], check=True, timeout=10)
                _wait_for_capture(args.corpus, capture_id, args.persist_timeout, service_url=args.service_url)
                failure = None
                break
            except urllib.error.HTTPError as error:
                failure = RuntimeError(f"capture enqueue failed: {error.code}: " + error.read().decode(errors="replace"))
            except (RuntimeError, TimeoutError, subprocess.SubprocessError) as error:
                failure = error
            if capture_id in _capture_ids(args.corpus):
                failure = None
                break
            if attempt < args.capture_attempts:
                _wait_for_pending_clear(args.corpus, capture_id, service_url=args.service_url, timeout=args.persist_timeout)
        if failure is not None:
            raise RuntimeError(f"capture failed: {capture_id}: {failure}") from failure
        existing.add(capture_id)
        completed += 1
        print(json.dumps({"capture_id": capture_id, "completed": completed, "remaining": len(rows) - skipped - completed}, sort_keys=True), flush=True)
        time.sleep(args.inter_capture_seconds)
    print(json.dumps({"completed": completed, "skipped_existing": skipped, "selected": len(rows), "expected_voice_counts": expected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
