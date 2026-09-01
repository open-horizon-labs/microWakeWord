#!/usr/bin/env python3
"""Compose the first qualified Kizz Control validation replay per source.

This is a quality-only composer.  It does not score audio with a wake model and
does not copy audio: selected capture paths remain absolute paths into the
attempt corpus that produced them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


QUALITY_KIND = "kizz_control_teacher_adaptation_device_replay_quality"
QUALITY_SCOPE = "validation_only_target_channel_positive_quality"
COMPOSITION_ALGORITHM = "first_acoustically_qualified_attempt_in_declared_order_v1"
CORPUS_ID = "kizz-control-teacher-adaptation-validation-device-replays-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _rows(payload: dict[str, Any], key: str, path: Path) -> list[dict[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path}: expected a list of {key}")
    return [dict(row) for row in value]


def _provider_voice(row: dict[str, Any]) -> tuple[str, str]:
    conditions = row.get("conditions") or {}
    provider = str(row.get("provider") or conditions.get("source_provider") or "").casefold()
    voice = str(row.get("voice") or conditions.get("source_voice") or "").casefold()
    if not provider or not voice:
        raise ValueError("capture/source row lacks provider and voice")
    return provider, voice


def _source_hash(row: dict[str, Any]) -> str:
    conditions = row.get("conditions") or {}
    value = row.get("audio_sha256") or row.get("source_audio_sha256") or conditions.get("source_audio_sha256")
    if not value:
        raise ValueError("selection row lacks source audio hash")
    return str(value)


def _selection_hash(selection: dict[str, Any], path: Path) -> str:
    declared = selection.get("selection_sha256")
    actual = hashlib.sha256(_canonical({key: value for key, value in selection.items() if key != "selection_sha256"})).hexdigest()
    if declared != actual:
        raise ValueError(f"{path}: selection_sha256 mismatch")
    return str(declared)


def _resolved_capture_path(corpus_dir: Path, row: dict[str, Any]) -> Path:
    value = Path(str(row.get("path", "")))
    return value if value.is_absolute() else (corpus_dir / value).resolve()


def _validate_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str, str]:
    payload = _json(path)
    selected = _rows(payload, "selected_examples", path)
    selection_sha = _selection_hash(payload, path)
    if payload.get("selection_algorithm") != "provider_voice_aligned_validation_round_robin_v1":
        raise ValueError(f"{path}: unexpected selection algorithm")
    seen: set[tuple[tuple[str, str], str]] = set()
    source_hashes: set[str] = set()
    voices: set[tuple[str, str]] = set()
    for row in selected:
        identity = (_provider_voice(row), _source_hash(row))
        if identity in seen or _provider_voice(row) in voices or _source_hash(row) in source_hashes:
            raise ValueError(f"{path}: duplicate locked provider/voice/source identity")
        seen.add(identity)
        voices.add(_provider_voice(row))
        source_hashes.add(_source_hash(row))
        source_path = Path(str(row.get("path", "")))
        if not source_path.is_absolute():
            source_path = (path.parent / source_path).resolve()
        if not source_path.is_file() or _sha256(source_path) != _source_hash(row):
            raise ValueError(f"{path}: locked source audio hash drift for {_provider_voice(row)}")
    return payload, selected, selection_sha, _sha256(path)


def _attempt_spec(spec: str) -> tuple[Path, Path]:
    if "=" not in spec:
        raise ValueError("--attempt must be CORPUS_DIR=QUALITY_REPORT")
    corpus_text, report_text = spec.split("=", 1)
    if not corpus_text or not report_text:
        raise ValueError("--attempt must be CORPUS_DIR=QUALITY_REPORT")
    return Path(corpus_text).expanduser().resolve(), Path(report_text).expanduser().resolve()


def _validate_attempt(
    corpus_dir: Path,
    report_path: Path,
    selection: dict[str, Any],
    selected: list[dict[str, Any]],
    selection_file_sha: str,
) -> tuple[dict[str, Any], dict[tuple[tuple[str, str], str], tuple[dict[str, Any], dict[str, Any]]]]:
    manifest_path = corpus_dir / "device-corpus.json"
    if not manifest_path.is_file():
        raise ValueError(f"{corpus_dir}: missing device-corpus.json")
    corpus = _json(manifest_path)
    report = _json(report_path)
    if corpus.get("corpus_id") != CORPUS_ID:
        raise ValueError(f"{manifest_path}: corpus_id mismatch")
    if report.get("kind") != QUALITY_KIND or report.get("gate_scope") != QUALITY_SCOPE:
        raise ValueError(f"{report_path}: unexpected validation quality kind/scope")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{report_path}: missing inputs")
    if inputs.get("corpus_sha256") != _sha256(manifest_path):
        raise ValueError(f"{report_path}: stale corpus manifest hash")
    if inputs.get("selection_sha256") != selection_file_sha:
        raise ValueError(f"{report_path}: selection hash mismatch")
    evidence_sha = selection.get("qualification_evidence_sha256")
    if not evidence_sha or inputs.get("qualification_evidence_sha256") != evidence_sha:
        raise ValueError(f"{report_path}: selection evidence hash mismatch")
    expected = {(_provider_voice(row), _source_hash(row)): row for row in selected}
    captures = _rows(corpus, "captures", manifest_path)
    by_capture: dict[str, dict[str, Any]] = {}
    for capture in captures:
        capture_id = str(capture.get("capture_id", ""))
        if not capture_id or capture_id in by_capture:
            raise ValueError(f"{manifest_path}: duplicate capture identity")
        path = _resolved_capture_path(corpus_dir, capture)
        declared_hash = str(capture.get("sha256", ""))
        if not path.is_file() or not declared_hash or _sha256(path) != declared_hash:
            raise ValueError(f"{manifest_path}: capture/file hash drift for {capture_id}")
        by_capture[capture_id] = capture
    results = _rows(report, "results", report_path)
    identities: dict[tuple[tuple[str, str], str], tuple[dict[str, Any], dict[str, Any]]] = {}
    for result in results:
        identity = ((_provider_voice(result), str(result.get("source_audio_sha256", ""))))
        if identity in identities:
            raise ValueError(f"{report_path}: duplicate provider/voice/source result")
        if identity not in expected:
            raise ValueError(f"{report_path}: source/provider/voice mismatch")
        capture_id = str(result.get("capture_id", ""))
        capture = by_capture.get(capture_id)
        if capture is None or result.get("audio_sha256") != capture.get("sha256"):
            raise ValueError(f"{report_path}: result capture/hash mismatch")
        if _provider_voice(capture) != _provider_voice(result):
            raise ValueError(f"{report_path}: capture/provider/voice mismatch")
        if _source_hash(capture) != identity[1]:
            raise ValueError(f"{report_path}: capture/source hash mismatch")
        if result.get("source_audio_sha256") != _source_hash(expected[identity]):
            raise ValueError(f"{report_path}: source hash mismatch")
        identities[identity] = (capture, result)
    if set(identities) != set(expected):
        raise ValueError(f"{report_path}: results do not exactly realize locked selection")
    if len(captures) != len(results):
        raise ValueError(f"{report_path}: corpus captures/results count mismatch")
    return corpus, identities


def compose(selection_path: Path, attempts: Sequence[tuple[Path, Path]], output_dir: Path) -> dict[str, Any]:
    if not attempts:
        raise ValueError("at least one --attempt is required")
    selection, selected, selection_content_sha, selection_file_sha = _validate_selection(selection_path.resolve())
    validated: list[tuple[Path, Path, dict[str, Any], dict[tuple[tuple[str, str], str], tuple[dict[str, Any], dict[str, Any]]]]] = []
    for corpus, report in attempts:
        validated.append((corpus.resolve(), report.resolve(), *_validate_attempt(corpus.resolve(), report.resolve(), selection, selected, selection_file_sha)))
    chosen: dict[tuple[tuple[str, str], str], tuple[int, Path, Path, dict[str, Any], dict[str, Any]]] = {}
    for index, (corpus_dir, report_path, corpus, identities) in enumerate(validated):
        for identity, (capture, result) in identities.items():
            if result.get("qualified") is True and identity not in chosen:
                chosen[identity] = (index, corpus_dir, report_path, capture, result)
    missing = [identity for identity in ((_provider_voice(row), _source_hash(row)) for row in selected) if identity not in chosen]
    if missing:
        raise ValueError(f"no qualified attempt for locked identities: {missing}")
    baseline = validated[0][2]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    captures: list[dict[str, Any]] = []
    selected_metrics: list[dict[str, Any]] = []
    for row in selected:
        identity = (_provider_voice(row), _source_hash(row))
        index, corpus_dir, report_path, capture, result = chosen[identity]
        item = dict(capture)
        item["path"] = str(_resolved_capture_path(corpus_dir, capture))
        item["conditions"] = dict(item.get("conditions") or {})
        item["conditions"]["selected_attempt_index"] = index
        captures.append(item)
        selected_metrics.append({
            "capture_id": capture["capture_id"],
            "provider": identity[0][0],
            "voice": identity[0][1],
            "source_audio_sha256": identity[1],
            "selected_attempt_index": index,
            "attempt_corpus_sha256": _sha256(corpus_dir / "device-corpus.json"),
            "attempt_quality_report_sha256": _sha256(report_path),
            "result": result,
        })
    payload = {
        "schema_version": 2,
        "corpus_id": baseline["corpus_id"],
        "device_profiles": baseline.get("device_profiles", {}),
        "speakers": baseline.get("speakers", {}),
        "captures": captures,
        "composition": {
            "algorithm": COMPOSITION_ALGORITHM,
            "selection": str(selection_path.resolve()),
            "selection_sha256": selection_file_sha,
            "selection_content_sha256": selection_content_sha,
            "selection_evidence_sha256": selection["qualification_evidence_sha256"],
            "attempts": [
                {"order": index, "corpus": str(corpus), "corpus_sha256": _sha256(corpus / "device-corpus.json"), "quality_report": str(report), "quality_report_sha256": _sha256(report)}
                for index, (corpus, report, _, _) in enumerate(validated)
            ],
            "selected": selected_metrics,
        },
    }
    target = output_dir / "device-corpus.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--attempt", action="append", required=True, metavar="CORPUS_DIR=QUALITY_REPORT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    attempts = [_attempt_spec(value) for value in args.attempt]
    payload = compose(args.selection, attempts, args.output)
    print(json.dumps({"captures": len(payload["captures"]), "output": str((args.output / "device-corpus.json").resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
