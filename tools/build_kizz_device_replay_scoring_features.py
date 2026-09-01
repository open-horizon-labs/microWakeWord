#!/usr/bin/env python3
"""Build test-only scoring windows from locked StackChan replay captures.

Unlike the ordered-state training feature builder, this scorer does not invent
phone timings.  It centers the complete reserved source utterance at the lag
measured by source/capture envelope correlation and emits only model inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.audit_kizz_control_adaptation_replays import _best_envelope_match, _mono
from tools.build_kizz_aligned_teacher_features_v3 import (
    SAMPLE_RATE,
    frontend,
    load_audio,
    place_phrase_context,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def build(
    corpus: Path,
    selection: Path,
    pronunciation_audit: Path,
    output: Path,
    *,
    minimum_correlation: float = 0.75,
) -> dict[str, Any]:
    corpus = corpus.expanduser().resolve()
    selection = selection.expanduser().resolve()
    pronunciation_audit = pronunciation_audit.expanduser().resolve()
    output = output.expanduser().resolve()
    corpus_manifest = corpus / "device-corpus.json"
    corpus_payload = _load(corpus_manifest)
    selection_payload = _load(selection)
    audit = _load(pronunciation_audit)
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("qualified") is not True
        or audit.get("locked_before_device_capture") is not True
    ):
        raise ValueError("source pronunciation audit is not qualified and pre-capture locked")
    accepted = {
        str(row.get("source_id"))
        for row in audit.get("results", [])
        if row.get("accepted") is True
    }
    selected = selection_payload.get("selected_examples")
    captures = corpus_payload.get("captures")
    if not isinstance(selected, list) or not isinstance(captures, list):
        raise ValueError("selection or device corpus rows are missing")
    selected_by_hash = {
        str(row.get("audio_sha256")): dict(row) for row in selected
    }
    if len(selected_by_hash) != len(selected):
        raise ValueError("locked selection has missing or duplicate audio hashes")
    capture_by_hash = {
        str(row.get("conditions", {}).get("source_audio_sha256")): dict(row)
        for row in captures
    }
    if set(capture_by_hash) != set(selected_by_hash):
        raise ValueError("device captures do not exactly cover the locked selection")

    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for source_hash, source in sorted(
        selected_by_hash.items(), key=lambda item: str(item[1].get("source_id"))
    ):
        source_id = str(source.get("source_id", ""))
        if source_id not in accepted:
            raise ValueError(f"locked source is not pronunciation accepted: {source_id}")
        source_path = Path(str(source.get("path", ""))).expanduser().resolve()
        capture = capture_by_hash[source_hash]
        capture_path = Path(str(capture.get("path", "")))
        if not capture_path.is_absolute():
            capture_path = corpus / capture_path
        capture_path = capture_path.resolve()
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source_hash
            or not capture_path.is_file()
            or sha256_file(capture_path) != capture.get("sha256")
        ):
            raise ValueError("source or capture audio binding drift")
        correlation, measured_lag = _best_envelope_match(
            _mono(capture_path), _mono(source_path)
        )
        lead_seconds = float(capture.get("conditions", {}).get("lead_seconds", 0.0))
        correlation_qualified = (
            correlation >= minimum_correlation and 0.20 <= measured_lag <= 1.50
        )
        if correlation_qualified:
            lag = measured_lag
            timing_basis = "complete_source_utterance_envelope_alignment"
        elif 0.20 <= lead_seconds <= 1.50:
            lag = lead_seconds
            timing_basis = "commanded_capture_lead_fallback"
        else:
            raise ValueError(f"{capture['capture_id']}: no bounded scoring-window timing")
        source_duration = len(load_audio(source_path)) / SAMPLE_RATE
        capture_audio = load_audio(capture_path)
        context, translation = place_phrase_context(
            capture_audio,
            (lag, lag + source_duration),
            desired_phrase_center_s=None,
        )
        values = np.asarray(frontend(context), dtype=np.float32)
        feature_index = len(features)
        features.append(values)
        rows.append(
            {
                "source_id": f"device-qualification:{capture['capture_id']}",
                "feature_index": feature_index,
                "feature_sha256": feature_sha256(values),
                "split": "test",
                "label": 1,
                "capture_id": capture["capture_id"],
                "capture_audio_sha256": capture["sha256"],
                "capture_path": str(capture_path),
                "parent_source_id": source_id,
                "source_audio_sha256": source_hash,
                "provider": source.get("provider"),
                "voice_id": source.get("voice_id"),
                "envelope_correlation": correlation,
                "measured_playback_lag_seconds": measured_lag,
                "scoring_window_lag_seconds": lag,
                "correlation_qualified": correlation_qualified,
                "source_duration_seconds": source_duration,
                "context_translation_seconds": translation,
                "timing_basis": timing_basis,
            }
        )
    values = np.asarray(features, dtype=np.float32)
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features.npy"
    np.save(features_path, values, allow_pickle=False)
    report = {
        "schema_version": 1,
        "kind": "kizz_control_locked_target_device_scoring_features",
        "gate_scope": "test_only_model_scoring_without_phone_timing_claim",
        "training_eligible": False,
        "deployment_qualification": False,
        "bindings": {
            "device_corpus": {
                "path": str(corpus_manifest),
                "sha256": sha256_file(corpus_manifest),
            },
            "selection": {"path": str(selection), "sha256": sha256_file(selection)},
            "pronunciation_audit": {
                "path": str(pronunciation_audit),
                "sha256": sha256_file(pronunciation_audit),
            },
        },
        "array_sha256": {features_path.name: sha256_file(features_path)},
        "counts": {
            "selected": len(selected),
            "materialized": len(rows),
            "correlation_qualified": sum(
                bool(row["correlation_qualified"]) for row in rows
            ),
            "commanded_lead_fallback": sum(
                not bool(row["correlation_qualified"]) for row in rows
            ),
        },
        "input_shape": [260, 40],
        "examples": rows,
        "limitations": [
            "does not claim MMS phone alignment acceptance",
            "complete source utterance is the phrase-window timing proxy",
            "weak/out-of-contract envelope matches use the recorded capture lead",
        ],
    }
    manifest_path = output / "source-manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--pronunciation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-correlation", type=float, default=0.75)
    args = parser.parse_args(argv)
    report = build(
        args.corpus,
        args.selection,
        args.pronunciation_audit,
        args.output,
        minimum_correlation=args.minimum_correlation,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
