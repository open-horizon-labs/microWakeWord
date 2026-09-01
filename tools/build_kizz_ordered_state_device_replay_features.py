#!/usr/bin/env python3
"""Build ordered-state features for locked target-device replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.ordered_state import OrderedStateTopology
from microwakeword.wake_phrase import KIZZ_CONTROL
from tools.audit_kizz_control_adaptation_replays import _best_envelope_match, _mono
from tools.extend_kizz_detector_student_cache_with_device_replays import (
    _device_example,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path, key: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get(key) if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: missing {key}")
    return [dict(row) for row in rows]


def build(
    corpus: Path,
    selection: Path,
    aligned_manifest: Path,
    output: Path,
    *,
    minimum_correlation: float = 0.75,
    allow_consumed_commanded_lead_fallback: bool = False,
) -> dict[str, Any]:
    corpus_manifest = corpus / "device-corpus.json"
    captures = _rows(corpus_manifest, "captures")
    selected = _rows(selection, "selected_examples")
    aligned_payload = json.loads(aligned_manifest.read_text())
    aligned = _rows(aligned_manifest, "examples")
    consumed_timing_only = (
        aligned_payload.get("gate_scope") == "consumed_development_timing_only"
        and aligned_payload.get("pronunciation_qualified") is False
        and aligned_payload.get("deployment_qualification") is False
    )
    if allow_consumed_commanded_lead_fallback and not consumed_timing_only:
        raise ValueError(
            "commanded-lead fallback is restricted to consumed rejected timing evidence"
        )
    selected_by_hash = {
        str(row.get("audio_sha256", "")): row for row in selected
    }
    aligned_by_source = {str(row.get("source_id", "")): row for row in aligned}
    if len(selected_by_hash) != len(selected):
        raise ValueError("selection contains missing or duplicate source hashes")

    topology = OrderedStateTopology(KIZZ_CONTROL.phones, 1)
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    results: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for capture in captures:
        conditions = capture.get("conditions", {})
        source_hash = str(conditions.get("source_audio_sha256", ""))
        source = selected_by_hash.get(source_hash)
        if source is None:
            raise ValueError(f"capture source is not in locked selection: {source_hash}")
        aligned_row = aligned_by_source.get(str(source.get("source_id", "")))
        if aligned_row is None:
            excluded.append(
                {
                    "capture_id": str(capture.get("capture_id", "")),
                    "reason": "selected_source_has_no_qualified_phone_alignment",
                }
            )
            continue
        source_path = Path(str(source.get("path", ""))).resolve()
        capture_path = Path(str(capture.get("path", "")))
        if not capture_path.is_absolute():
            capture_path = corpus / capture_path
        capture_path = capture_path.resolve()
        if (
            not source_path.is_file()
            or sha256_file(source_path) != source_hash
            or not capture_path.is_file()
            or sha256_file(capture_path) != capture.get("sha256")
            or aligned_row.get("audio_sha256") != source_hash
        ):
            raise ValueError("source, capture, or aligned audio binding drifted")
        correlation, lag = _best_envelope_match(
            _mono(capture_path), _mono(source_path)
        )
        timing_basis = "source_capture_envelope_alignment"
        if correlation < minimum_correlation or lag <= 0:
            lead = float(conditions.get("lead_seconds", 0.0))
            if allow_consumed_commanded_lead_fallback and 0.20 <= lead <= 1.50:
                lag = lead
                timing_basis = "consumed_commanded_capture_lead_fallback"
            else:
                excluded.append(
                    {
                        "capture_id": str(capture.get("capture_id", "")),
                        "reason": "source_capture_alignment_quality_failed",
                    }
                )
                continue
        device_row = dict(aligned_row)
        device_row.update(
            {
                "source_id": f"device-qualification:{capture['capture_id']}",
                "path": str(capture_path),
                "audio_sha256": str(capture["sha256"]),
                "split": "test",
                "provider": conditions.get("source_provider"),
                "voice": conditions.get("source_voice"),
                "phrase_span": {
                    "start_s": float(aligned_row["phrase_span"]["start_s"]) + lag,
                    "end_s": float(aligned_row["phrase_span"]["end_s"]) + lag,
                },
                "phone_spans": [
                    {
                        "phone": span["phone"],
                        "start_s": float(span["start_s"]) + lag,
                        "end_s": float(span["end_s"]) + lag,
                    }
                    for span in aligned_row["phone_spans"]
                ],
            }
        )
        feature, target = _device_example(
            device_row,
            topology,
            desired_phrase_center_s=None,
            gain_db=0.0,
        )
        feature_rows.append(feature)
        target_rows.append(target)
        results.append(
            {
                "capture_id": capture["capture_id"],
                "source_id": aligned_row["source_id"],
                "provider": conditions.get("source_provider"),
                "voice": conditions.get("source_voice"),
                "audio_sha256": capture["sha256"],
                "path": str(capture_path),
                "source_audio_sha256": source_hash,
                "phrase_span": device_row["phrase_span"],
                "phone_spans": device_row["phone_spans"],
                "envelope_correlation": correlation,
                "playback_lag_seconds": lag,
                "timing_basis": timing_basis,
                "pronunciation_qualified": not consumed_timing_only,
                "runtime_detected": capture.get("detected"),
            }
        )
    if not results:
        raise ValueError("no target-device replay has a qualified phone alignment")
    output.mkdir(parents=True, exist_ok=True)
    features_path = output / "features.npy"
    targets_path = output / "targets.npy"
    np.save(features_path, np.asarray(feature_rows, dtype=np.float32))
    np.save(targets_path, np.asarray(target_rows, dtype=np.int32))
    report = {
        "schema_version": 1,
        "kind": "kizz_control_ordered_state_target_device_replay_features",
        "gate_scope": "locked_test_only_target_channel_positive_features",
        "training_eligible": False,
        "pronunciation_qualified": not consumed_timing_only,
        "inputs": {
            "corpus": {"path": str(corpus_manifest), "sha256": sha256_file(corpus_manifest)},
            "selection": {"path": str(selection.resolve()), "sha256": sha256_file(selection)},
            "aligned_manifest": {
                "path": str(aligned_manifest.resolve()),
                "sha256": sha256_file(aligned_manifest),
            },
        },
        "outputs": {
            "features": {"path": str(features_path), "sha256": sha256_file(features_path)},
            "targets": {"path": str(targets_path), "sha256": sha256_file(targets_path)},
        },
        "counts": {
            "selected": len(selected),
            "captures": len(captures),
            "materialized": len(results),
            "excluded": len(excluded),
        },
        "results": results,
        "excluded": excluded,
    }
    (output / "feature-provenance.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--aligned-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-correlation", type=float, default=0.75)
    parser.add_argument(
        "--allow-consumed-commanded-lead-fallback", action="store_true"
    )
    args = parser.parse_args(argv)
    report = build(
        args.corpus.resolve(),
        args.selection.resolve(),
        args.aligned_manifest.resolve(),
        args.output.resolve(),
        minimum_correlation=args.minimum_correlation,
        allow_consumed_commanded_lead_fallback=(
            args.allow_consumed_commanded_lead_fallback
        ),
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
