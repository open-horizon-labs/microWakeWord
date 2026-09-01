#!/usr/bin/env python3
"""Audit the held-out Kizz Control adaptation validation replay corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

from microwakeword.kizz_phoneme_teacher import TARGET_SAMPLE_RATE, sha256_file
try:
    from tools.capture_kizz_control_adaptation_validation_replays import (
        CORPUS_ID,
        EVIDENCE_ROLE,
        PROVIDERS,
        SELECTION_ALGORITHM,
        _canonical_json,
        _hashes,
        _identities,
        _provider_voice,
        _provider_voice_counts,
    )
except ModuleNotFoundError:  # direct ``python tools/...`` execution
    from capture_kizz_control_adaptation_validation_replays import (  # type: ignore[no-redef]
        CORPUS_ID,
        EVIDENCE_ROLE,
        PROVIDERS,
        SELECTION_ALGORITHM,
        _canonical_json,
        _hashes,
        _identities,
        _provider_voice,
        _provider_voice_counts,
    )


def _audio(path: Path) -> np.ndarray:
    values, rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if int(rate) != TARGET_SAMPLE_RATE:
        raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz: {path}")
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values


def _envelope(values: np.ndarray, frame_samples: int = 320) -> np.ndarray:
    padded = np.pad(values, (0, (-len(values)) % frame_samples))
    frames = padded.reshape(-1, frame_samples)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def _best_match(capture: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    captured = _envelope(capture)
    reference = _envelope(source)
    if len(reference) > len(captured) or float(np.std(reference)) <= 1e-8:
        return -1.0, 0.0
    best = (-1.0, 0)
    for start in range(len(captured) - len(reference) + 1):
        candidate = captured[start : start + len(reference)]
        if float(np.std(candidate)) <= 1e-8:
            continue
        correlation = float(np.corrcoef(candidate, reference)[0, 1])
        if correlation > best[0]:
            best = correlation, start
    return best[0], best[1] * 0.020


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rows(payload: dict[str, Any], key: str, path: Path) -> list[dict[str, Any]]:
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"{path} contains no {key}")
    return [dict(row) for row in rows]


def _source_path(row: dict[str, Any], selection_payload: dict[str, Any]) -> Path:
    value = Path(str(row.get("path", "")))
    if value.is_absolute():
        return value
    manifest = Path(str(selection_payload["aligned_manifest"]))
    return manifest.parent / value


def _capture_path(capture: dict[str, Any], corpus: Path) -> Path:
    value = Path(str(capture.get("path", "")))
    root = corpus if corpus.is_dir() else corpus.parent
    return value if value.is_absolute() else root / value


def _selection_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    declared = payload.get("selection_sha256")
    actual = hashlib.sha256(
        _canonical_json({key: value for key, value in payload.items() if key != "selection_sha256"})
    ).hexdigest()
    if declared != actual:
        raise ValueError("validation replay selection hash mismatch")
    if payload.get("selection_algorithm") != SELECTION_ALGORITHM:
        raise ValueError("validation replay selection algorithm mismatch")
    return payload


def _all_excluded(rows: Iterable[dict[str, Any]]) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]:
    hashes: set[str] = set()
    voices: set[tuple[str, str]] = set()
    identities: set[tuple[str, str]] = set()
    for row in rows:
        hashes |= _hashes(row)
        voices.add(_provider_voice(row))
        identities |= _identities(row)
    return hashes, voices, identities


def audit(
    corpus: Path,
    selection: Path,
    qualification_evidence: Path,
    train_corpus: Path,
    train_selection: Path,
    *,
    min_correlation: float = 0.50,
    min_rms_dbfs: float = -50.0,
    max_rms_dbfs: float = -10.0,
    max_clip_percent: float = 0.10,
    min_lag_seconds: float = 0.20,
    max_lag_seconds: float = 3.25,
    minimum_declared_lead_seconds: float = 2.30,
    lead_tolerance_seconds: float = 0.50,
) -> dict[str, Any]:
    corpus_payload = _load_json(corpus / "device-corpus.json") if corpus.is_dir() else _load_json(corpus)
    selection_payload = _selection_payload(selection)
    selected = _rows(selection_payload, "selected_examples", selection)
    target_rows = _rows(_load_json(qualification_evidence), "examples", qualification_evidence)
    train_rows = _rows(_load_json(train_corpus), "captures", train_corpus) + _rows(_load_json(train_selection), "selected_examples", train_selection)
    expected = {str(key): int(value) for key, value in selection_payload.get("expected_voice_counts", {}).items()}
    reasons: list[str] = []
    selected_expected = _provider_voice_counts(selected)
    if len({str(row.get("audio_sha256", "")) for row in selected}) != len(selected):
        reasons.append("locked_validation_selection_has_duplicate_audio_hash")
    if any(selected_expected[provider] == 0 for provider in PROVIDERS):
        reasons.append("locked_validation_selection_missing_provider")
    if expected != selected_expected:
        reasons.append("expected_voice_counts_not_bound_to_locked_validation_selection")
    if corpus_payload.get("corpus_id") != CORPUS_ID:
        reasons.append("corpus_id_invalid")
    captures = _rows(corpus_payload, "captures", corpus / "device-corpus.json")
    selected_by_hash = {str(row.get("audio_sha256")): row for row in selected}
    target_hashes, target_voices, target_ids = _all_excluded(target_rows)
    train_hashes, train_voices, train_ids = _all_excluded(train_rows)
    for source in selected:
        source_overlap = _hashes(source) & (target_hashes | train_hashes)
        source_id_overlap = _identities(source) & (target_ids | train_ids)
        if _provider_voice(source) in target_voices:
            reasons.append("selected_source_qualification_voice_overlap")
        if _provider_voice(source) in train_voices:
            reasons.append("selected_source_train_voice_overlap")
        if source_overlap:
            reasons.append("selected_source_hash_overlap")
        if source_id_overlap:
            reasons.append("selected_source_provenance_overlap")
    provider_counts: Counter[str] = Counter()
    provider_voices: dict[str, set[str]] = defaultdict(set)
    results: list[dict[str, Any]] = []
    captured_source_hashes: set[str] = set()
    capture_ids: set[str] = set()
    for capture in captures:
        conditions = capture.get("conditions") or {}
        provider_voice = _provider_voice(capture)
        provider, voice = provider_voice
        source_hash = str(conditions.get("source_audio_sha256", ""))
        source = selected_by_hash.get(source_hash)
        path = _capture_path(capture, corpus)
        row_reasons: list[str] = []
        if capture.get("capture_id") in capture_ids:
            row_reasons.append("duplicate_capture_id")
        capture_ids.add(str(capture.get("capture_id")))
        if capture.get("truth") != "positive" or capture.get("split") != "validation" or conditions.get("evidence_role") != EVIDENCE_ROLE:
            row_reasons.append("capture_role_or_split_invalid")
        if provider_voice in target_voices:
            row_reasons.append("qualification_voice_overlap")
        if provider_voice in train_voices:
            row_reasons.append("train_voice_overlap")
        capture_hash = str(capture.get("sha256", ""))
        if not path.is_file() or not capture_hash or sha256_file(path) != capture_hash:
            row_reasons.append("capture_audio_hash_drift")
        capture_identity_overlap = _identities(capture) & (target_ids | train_ids)
        if capture_identity_overlap:
            row_reasons.append("capture_provenance_overlap")
        if _hashes(capture) & (target_hashes | train_hashes):
            row_reasons.append("capture_hash_overlap")
        correlation = lag = rms = clip = None
        source_path: Path | None = None
        if source is None:
            row_reasons.append("source_not_in_locked_selection")
        else:
            source_path = _source_path(source, selection_payload)
            if not source_path.is_file() or sha256_file(source_path) != source_hash:
                row_reasons.append("selected_source_hash_drift")
            if _provider_voice(source) != provider_voice:
                row_reasons.append("source_provider_voice_mismatch")
            if path.is_file() and source_path.is_file():
                captured = _audio(path)
                source_audio = _audio(source_path)
                correlation, lag = _best_match(captured, source_audio)
                rms = 20.0 * math.log10(max(float(np.sqrt(np.mean(captured * captured))), 1e-12))
                clip = 100.0 * float(np.mean(np.abs(captured) >= 0.999))
                if correlation < min_correlation:
                    row_reasons.append("source_capture_correlation_below_minimum")
                if not min_lag_seconds <= lag <= max_lag_seconds:
                    row_reasons.append("playback_lag_outside_contract")
                declared_lead = conditions.get("lead_seconds")
                if not isinstance(declared_lead, (int, float)):
                    row_reasons.append("declared_lead_missing_or_invalid")
                else:
                    if float(declared_lead) < minimum_declared_lead_seconds:
                        row_reasons.append(
                            "declared_lead_below_continuous_preroll"
                        )
                    if (
                        abs(lag - float(declared_lead))
                        > lead_tolerance_seconds
                    ):
                        row_reasons.append(
                            "playback_lag_differs_from_declared_lead"
                        )
                if not min_rms_dbfs <= rms <= max_rms_dbfs:
                    row_reasons.append("capture_rms_outside_contract")
                if clip > max_clip_percent:
                    row_reasons.append("capture_clipping_above_maximum")
        provider_counts[provider] += 1
        provider_voices[provider].add(voice)
        captured_source_hashes.add(source_hash)
        results.append({"capture_id": capture.get("capture_id"), "audio_sha256": capture_hash, "source_audio_sha256": source_hash, "provider": provider, "voice": voice, "envelope_correlation": correlation, "playback_lag_seconds": lag, "rms_dbfs": rms, "clip_percent": clip, "qualified": not row_reasons, "failure_reasons": row_reasons})

    selected_hashes = set(selected_by_hash)
    if captured_source_hashes != selected_hashes:
        reasons.append("captures_do_not_exactly_realize_locked_selection")
    if len(captures) != len(selected):
        reasons.append("capture_count_does_not_match_selection")
    if provider_counts != Counter({provider: expected.get(provider, 0) for provider in PROVIDERS}):
        reasons.append("provider_counts_do_not_match_locked_voice_counts")
    if any(provider_voices.get(provider, set()) != {voice for item_provider, voice in (_provider_voice(row) for row in selected) if item_provider == provider} for provider in PROVIDERS):
        reasons.append("provider_voice_sets_do_not_match_locked_selection")
    return {
        "schema_version": 1,
        "kind": "kizz_control_teacher_adaptation_device_replay_quality",
        "gate_scope": "validation_only_target_channel_positive_quality",
        "expected_split": "validation",
        "qualified": not reasons and all(row["qualified"] for row in results),
        "inputs": {"corpus": str(corpus.resolve()), "corpus_sha256": sha256_file(corpus / "device-corpus.json") if corpus.is_dir() else sha256_file(corpus), "selection": str(selection.resolve()), "selection_sha256": sha256_file(selection), "qualification_evidence": str(qualification_evidence.resolve()), "qualification_evidence_sha256": sha256_file(qualification_evidence), "train_corpus": str(train_corpus.resolve()), "train_corpus_sha256": sha256_file(train_corpus), "train_selection": str(train_selection.resolve()), "train_selection_sha256": sha256_file(train_selection)},
        "limits": {
            "min_correlation": min_correlation,
            "min_rms_dbfs": min_rms_dbfs,
            "max_rms_dbfs": max_rms_dbfs,
            "max_clip_percent": max_clip_percent,
            "min_lag_seconds": min_lag_seconds,
            "max_lag_seconds": max_lag_seconds,
            "minimum_declared_lead_seconds": minimum_declared_lead_seconds,
            "lead_tolerance_seconds": lead_tolerance_seconds,
        },
        "expected_voice_counts": expected,
        "counts": {"captures": len(captures), "selected": len(selected), "providers": dict(sorted(provider_counts.items())), "voices": {key: sorted(value) for key, value in sorted(provider_voices.items())}},
        "results": sorted(results, key=lambda row: str(row["capture_id"])),
        "failure_reasons": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--qualification-evidence", type=Path, required=True)
    parser.add_argument("--train-corpus", type=Path, required=True)
    parser.add_argument("--train-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.corpus.resolve(), args.selection.resolve(), args.qualification_evidence.resolve(), args.train_corpus.resolve(), args.train_selection.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"qualified": report["qualified"], "counts": report["counts"], "expected_voice_counts": report["expected_voice_counts"]}, sort_keys=True))
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
