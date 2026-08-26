#!/usr/bin/env python3
"""Export target-device adaptation validation rows for student selection."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Sequence

from microwakeword.kizz_phoneme_teacher import sha256_file


APPROVED_PROVIDERS = ("assemblyai", "deepgram", "elevenlabs", "kokoro")


def export(manifest_path: Path, output: Path) -> dict:
    payload = json.loads(manifest_path.read_text())
    rows = [
        dict(row)
        for row in payload.get("examples", [])
        if row.get("split") == "validation"
        and int(row.get("label", -1)) == 1
        and row.get("source_group") == "device_channel_positive"
    ]
    counts = Counter(str(row.get("provider")) for row in rows)
    if len(rows) != 12 or set(counts) != set(APPROVED_PROVIDERS):
        raise ValueError("adaptation manifest lacks the locked 12-row device validation contract")
    examples = []
    for row in sorted(rows, key=lambda item: (item["provider"], item["audio_sha256"])):
        path = Path(row["path"]).resolve()
        if not path.is_file() or sha256_file(path) != row.get("audio_sha256"):
            raise ValueError(f"device validation audio drifted: {path}")
        examples.append(
            {
                **row,
                "path": str(path),
                "split": "validation",
                "label": 1,
                "training_eligible": False,
                "locked_deployment_anchor": True,
                "evidence_role": "student_checkpoint_device_validation_positive",
            }
        )
    report = {
        "schema_version": 1,
        "kind": "kizz_student_device_validation_evidence",
        "locked_before_student_checkpoint_selection": True,
        "training_eligible": False,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "counts": {"total": len(examples), "providers": dict(sorted(counts.items()))},
        "examples": examples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = export(args.adaptation_manifest, args.output)
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
