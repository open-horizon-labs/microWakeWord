#!/usr/bin/env python3
"""Retain forced phone timings from consumed, pronunciation-rejected positives.

This is deliberately not a pronunciation qualification path.  It exists only
after locked test evidence has been consumed for development, so a rejected
recording can provide approximate frame supervision for hard-positive
adaptation without being relabeled as clean validation evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from microwakeword.wake_phrase import get_wake_phrase
from tools.build_kizz_phone_alignment_v3 import (
    MmsAligner,
    SAMPLE_RATE,
    _crop_for_alignment,
    _load_audio,
    phone_spans_from_token_spans,
    pronunciation_decision,
    sha256_file,
)


def build(
    manifest: Path,
    output: Path,
    source_ids: Sequence[str],
    *,
    aligner: Any | None = None,
    phrase_id: str = "kizz-control",
) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError("alignment input has no examples")
    requested = set(source_ids)
    if not requested:
        raise ValueError("at least one consumed source ID is required")
    by_id = {str(row.get("source_id")): dict(row) for row in rows}
    if requested - set(by_id):
        raise ValueError(f"consumed source IDs are missing: {sorted(requested-set(by_id))}")
    phrase = get_wake_phrase(phrase_id)
    runner = aligner or MmsAligner(device="cpu", phrase_spec=phrase)

    selected = []
    audit = []
    for source_id in sorted(requested):
        row = by_id[source_id]
        path = Path(str(row.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != row.get("audio_sha256"):
            raise ValueError(f"consumed source audio binding drift: {source_id}")
        samples, _ = _load_audio(path)
        cropped, crop_offset = _crop_for_alignment(row, samples)
        measured = runner.score_and_align(cropped)
        decision = pronunciation_decision(
            measured["candidate_nll"],
            minimum_margin_per_token=0.0,
            maximum_canonical_nll_per_token=3.5,
            phrase_spec=phrase,
            token_lengths=runner.token_lengths,
        )
        if decision["accepted"]:
            raise ValueError(
                f"{source_id}: use the normal qualified alignment path for accepted audio"
            )
        phrase_span, phone_spans = phone_spans_from_token_spans(
            measured["token_spans"],
            waveform_samples=len(cropped),
            emission_frames=measured["emission_frames"],
            crop_offset_seconds=crop_offset,
            phones=phrase.phones,
        )
        retained = dict(row)
        retained.update(
            {
                "phrase_span": phrase_span,
                "phone_spans": phone_spans,
                "training_eligible": False,
                "alignment": {
                    "method": "mms_character_ctc_rejected_timing_only",
                    "timing_source": str(runner.checkpoint),
                    "model_sha256": runner.checkpoint_sha256,
                    "transcript": phrase.ctc_transcript,
                    "token_spans": measured["token_spans"],
                    "pronunciation_decision": decision,
                    "pronunciation_qualified": False,
                    "consumed_development_evidence": True,
                },
            }
        )
        selected.append(retained)
        audit.append(
            {
                "source_id": source_id,
                "audio_sha256": row["audio_sha256"],
                "accepted": False,
                "reasons": decision["reasons"],
                "canonical_margin_per_token": decision[
                    "canonical_margin_per_token"
                ],
                "timings_retained": True,
            }
        )
    report = {
        "schema_version": 1,
        "kind": "kizz_control_consumed_rejected_alignment_timings",
        "gate_scope": "consumed_development_timing_only",
        "training_eligible": False,
        "deployment_qualification": False,
        "pronunciation_qualified": False,
        "source_manifest": str(manifest.resolve()),
        "source_manifest_sha256": sha256_file(manifest),
        "aligner": runner.metadata,
        "counts": {"requested": len(requested), "timings_retained": len(selected)},
        "examples": selected,
        "audit": audit,
        "limitations": [
            "pronunciation-rejected evidence must not enter clean validation",
            "timings are approximate frame supervision for consumed hard positives only",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append", required=True)
    parser.add_argument("--phrase-id", default="kizz-control")
    args = parser.parse_args(argv)
    report = build(
        args.manifest.resolve(),
        args.output.resolve(),
        args.source_id,
        phrase_id=args.phrase_id,
    )
    print(json.dumps(report["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
