#!/usr/bin/env python3
"""Qualify split-bound StackChan replays before model adaptation/evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf

from microwakeword.kizz_phoneme_teacher import TARGET_SAMPLE_RATE, sha256_file


PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")
EVIDENCE_ROLES = {
    "train": "teacher_adaptation_target_channel_positive",
    "validation": "teacher_adaptation_target_channel_validation_positive",
}


def _examples(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"manifest contains no examples: {path}")
    return [dict(row) for row in rows]


def _provider_voice(row: dict) -> tuple[str, str]:
    conditions = row.get("conditions", {})
    provider = str(row.get("provider") or conditions.get("source_provider") or "").lower()
    voice = str(row.get("voice") or conditions.get("source_voice") or "").lower()
    if provider not in PROVIDERS or not voice:
        raise ValueError("row lacks approved provider/voice metadata")
    return provider, voice


def _mono(path: Path) -> np.ndarray:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    values = np.asarray(values, dtype=np.float32)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        raise ValueError(f"audio must be {TARGET_SAMPLE_RATE} Hz: {path}")
    if values.ndim == 2:
        values = values.mean(axis=1, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"audio is empty or non-finite: {path}")
    return values


def _rms_envelope(values: np.ndarray, frame_samples: int = 320) -> np.ndarray:
    padded = np.pad(values, (0, (-len(values)) % frame_samples))
    frames = padded.reshape(-1, frame_samples)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def _best_envelope_match(capture: np.ndarray, source: np.ndarray) -> tuple[float, float]:
    captured = _rms_envelope(capture)
    reference = _rms_envelope(source)
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


def audit(
    corpus: Path,
    selection: Path,
    qualification_evidence: Path,
    *,
    min_correlation: float = 0.75,
    min_rms_dbfs: float = -50.0,
    max_rms_dbfs: float = -10.0,
    max_clip_percent: float = 0.10,
    min_lag_seconds: float = 0.20,
    max_lag_seconds: float = 1.50,
    expected_split: str = "train",
) -> dict:
    if expected_split not in EVIDENCE_ROLES:
        raise ValueError(f"unsupported expected split: {expected_split}")
    expected_role = EVIDENCE_ROLES[expected_split]
    corpus_payload = json.loads(corpus.read_text())
    selection_payload = json.loads(selection.read_text())
    captures = [dict(row) for row in corpus_payload.get("captures", [])]
    selected = [dict(row) for row in selection_payload.get("selected_examples", [])]
    selected_by_hash = {str(row.get("audio_sha256", "")): row for row in selected}
    selected_counts: Counter[str] = Counter()
    selected_voices: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        provider, voice = _provider_voice(row)
        selected_counts[provider] += 1
        selected_voices[provider].add(voice)
    heldout_voices = {_provider_voice(row) for row in _examples(qualification_evidence)}
    reasons = []
    results = []
    provider_counts: Counter[str] = Counter()
    provider_voices: dict[str, set[str]] = defaultdict(set)
    captured_source_hashes = set()
    for capture in captures:
        conditions = capture.get("conditions", {})
        provider_voice = _provider_voice(capture)
        provider, voice = provider_voice
        source_hash = str(conditions.get("source_audio_sha256", ""))
        source = selected_by_hash.get(source_hash)
        audio_path = Path(str(capture.get("path", "")))
        if not audio_path.is_absolute():
            audio_path = corpus.parent / audio_path
        row_reasons = []
        if (
            capture.get("truth") != "positive"
            or capture.get("split") != expected_split
            or conditions.get("evidence_role") != expected_role
        ):
            row_reasons.append("capture_role_or_split_invalid")
        if provider_voice in heldout_voices:
            row_reasons.append("qualification_voice_overlap")
        if source is None:
            row_reasons.append("source_not_in_locked_selection")
        if not audio_path.is_file() or sha256_file(audio_path) != capture.get("sha256"):
            row_reasons.append("capture_audio_hash_drift")
        correlation = lag_seconds = rms_dbfs = clip_percent = None
        if source is not None and audio_path.is_file():
            source_path = Path(str(source["path"]))
            if (
                not source_path.is_file()
                or sha256_file(source_path) != source_hash
                or _provider_voice(source) != provider_voice
            ):
                row_reasons.append("selected_source_provenance_drift")
            else:
                captured = _mono(audio_path)
                source_audio = _mono(source_path)
                correlation, lag_seconds = _best_envelope_match(captured, source_audio)
                rms_dbfs = 20.0 * math.log10(
                    max(float(np.sqrt(np.mean(captured * captured))), 1e-12)
                )
                clip_percent = 100.0 * float(np.mean(np.abs(captured) >= 0.999))
                if correlation < min_correlation:
                    row_reasons.append("source_capture_correlation_below_minimum")
                if not min_lag_seconds <= lag_seconds <= max_lag_seconds:
                    row_reasons.append("playback_lag_outside_contract")
                if not min_rms_dbfs <= rms_dbfs <= max_rms_dbfs:
                    row_reasons.append("capture_rms_outside_contract")
                if clip_percent > max_clip_percent:
                    row_reasons.append("capture_clipping_above_maximum")
        provider_counts[provider] += 1
        provider_voices[provider].add(voice)
        captured_source_hashes.add(source_hash)
        reasons.extend(f"{capture.get('capture_id')}:{reason}" for reason in row_reasons)
        results.append(
            {
                "capture_id": capture.get("capture_id"),
                "audio_sha256": capture.get("sha256"),
                "source_audio_sha256": source_hash,
                "provider": provider,
                "voice": voice,
                "envelope_correlation": correlation,
                "playback_lag_seconds": lag_seconds,
                "rms_dbfs": rms_dbfs,
                "clip_percent": clip_percent,
                "qualified": not row_reasons,
                "failure_reasons": row_reasons,
            }
        )
    selected_hashes = set(selected_by_hash)
    if captured_source_hashes != selected_hashes:
        reasons.append("captures_do_not_exactly_realize_locked_selection")
    if provider_counts != selected_counts:
        reasons.append("provider_counts_do_not_match_locked_selection")
    if any(provider_voices[provider] != selected_voices[provider] for provider in PROVIDERS):
        reasons.append("provider_voices_do_not_match_locked_selection")
    return {
        "schema_version": 1,
        "kind": "kizz_control_teacher_adaptation_device_replay_quality",
        "gate_scope": f"{expected_split}_only_target_channel_positive_quality",
        "expected_split": expected_split,
        "expected_evidence_role": expected_role,
        "qualified": not reasons,
        "inputs": {
            "corpus": str(corpus.resolve()),
            "corpus_sha256": sha256_file(corpus),
            "selection": str(selection.resolve()),
            "selection_sha256": sha256_file(selection),
            "qualification_evidence": str(qualification_evidence.resolve()),
            "qualification_evidence_sha256": sha256_file(qualification_evidence),
        },
        "limits": {
            "min_correlation": min_correlation,
            "min_rms_dbfs": min_rms_dbfs,
            "max_rms_dbfs": max_rms_dbfs,
            "max_clip_percent": max_clip_percent,
            "min_lag_seconds": min_lag_seconds,
            "max_lag_seconds": max_lag_seconds,
        },
        "counts": {
            "captures": len(captures),
            "providers": dict(sorted(provider_counts.items())),
            "voices": {
                key: sorted(value) for key, value in sorted(provider_voices.items())
            },
        },
        "results": sorted(results, key=lambda row: str(row["capture_id"])),
        "failure_reasons": reasons,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--qualification-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-split",
        choices=sorted(EVIDENCE_ROLES),
        default="train",
    )
    args = parser.parse_args(argv)
    report = audit(
        args.corpus.resolve(),
        args.selection.resolve(),
        args.qualification_evidence.resolve(),
        expected_split=args.expected_split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"qualified": report["qualified"], "counts": report["counts"]},
            sort_keys=True,
        )
    )
    return 0 if report["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
