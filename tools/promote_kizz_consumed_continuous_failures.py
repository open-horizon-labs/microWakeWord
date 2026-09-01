#!/usr/bin/env python3
"""Promote consumed continuous-negative failures into development mining assets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import _atomic_json, _binding, sha256_file


REPORT_KIND = "kizz_control_int8_continuous_negative_cascade_v1"


def _object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolved_binding(value: Mapping[str, Any], anchor: Path, label: str) -> Path:
    raw = value.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = anchor / path
    path = path.resolve()
    if not path.is_file() or value.get("sha256") != sha256_file(path):
        raise ValueError(f"{label} hash drift")
    if "bytes" in value and value.get("bytes") != path.stat().st_size:
        raise ValueError(f"{label} byte-size drift")
    return path


def promote(
    lock_path: Path,
    report_paths: Sequence[Path],
    output: Path,
    source_group: str | None = None,
    selection: str = "accepted",
) -> dict[str, Any]:
    source_group = source_group.strip() if source_group is not None else None
    if source_group == "":
        raise ValueError("source group must be non-empty")
    if selection not in {"accepted", "all"}:
        raise ValueError("selection must be 'accepted' or 'all'")
    lock_path = lock_path.expanduser().resolve()
    lock = _object(lock_path, "locked manifest")
    if lock.get("schema_version") != 2 or lock.get("locked_before_scoring") is not True:
        raise ValueError("locked manifest contract drift")
    examples = lock.get("examples")
    if not isinstance(examples, list):
        raise ValueError("locked manifest examples are missing")
    by_source: dict[str, dict[str, Any]] = {}
    for row in examples:
        if not isinstance(row, dict) or not isinstance(row.get("source_id"), str):
            raise ValueError("locked manifest example contract drift")
        source_id = str(row["source_id"])
        if source_id in by_source:
            raise ValueError("duplicate locked source identity")
        by_source[source_id] = dict(row)

    selected: dict[str, tuple[dict[str, Any], Path, int]] = {}
    report_bindings = []
    report_binding_by_path: dict[Path, dict[str, Any]] = {}
    for raw_report_path in report_paths:
        report_path = raw_report_path.expanduser().resolve()
        report = _object(report_path, "continuous-cascade report")
        shard = report.get("shard")
        bindings = report.get("bindings")
        if (
            report.get("schema_version") != 1
            or report.get("kind") != REPORT_KIND
            or not isinstance(shard, Mapping)
            or shard.get("complete") is not True
            or not isinstance(bindings, Mapping)
            or not isinstance(bindings.get("locked_manifest"), Mapping)
        ):
            raise ValueError("continuous-cascade report contract drift")
        bound_lock = _resolved_binding(
            bindings["locked_manifest"], report_path.parent, "report locked manifest"
        )
        if bound_lock != lock_path:
            raise ValueError("report binds a different locked manifest")
        report_binding = _binding(report_path)
        report_bindings.append(report_binding)
        report_binding_by_path[report_path] = report_binding
        files = report.get("files")
        if not isinstance(files, list):
            raise ValueError("continuous-cascade report files are missing")
        for evidence in files:
            if not isinstance(evidence, dict):
                raise ValueError("continuous-cascade file evidence contract drift")
            accepted = evidence.get("accepted_false_wakes")
            if not isinstance(accepted, int) or accepted < 0:
                raise ValueError("accepted-false-wake evidence is invalid")
            if accepted == 0 and selection == "accepted":
                continue
            source_id = evidence.get("source_id")
            if source_id not in by_source or source_id in selected:
                raise ValueError("accepted source identity is missing or duplicated")
            source = by_source[str(source_id)]
            if (
                Path(str(evidence.get("path"))).resolve()
                != Path(str(source.get("path"))).resolve()
                or evidence.get("audio_sha256") != source.get("audio_sha256")
                or abs(float(evidence.get("duration_seconds")) - float(source.get("duration_seconds")))
                > 1e-9
            ):
                raise ValueError("accepted source evidence disagrees with locked manifest")
            selected[str(source_id)] = (source, report_path, accepted)
    if not selected:
        raise ValueError("reports contain no selected continuous-negative sources")
    if selection == "all" and set(selected) != set(by_source):
        raise ValueError("all-source promotion requires complete locked-manifest evidence")

    promoted = []
    for source_id in sorted(selected):
        source, report_path, accepted = selected[source_id]
        row = dict(source)
        row.update(
            {
                "original_locked_split": source.get("split"),
                "split": "train",
                "label": 0,
                "training_eligible": True,
                "locked_holdout": False,
                "locked_deployment_anchor": False,
                "semantic_label": "non_wake",
                "source_group": source_group
                or f"consumed_continuous_{source.get('category', 'negative')}",
                "speaker_id": source.get(
                    "speaker_id", source.get("source_group_id", source_id)
                ),
                "session_id": source.get("session_id", source_id),
                "ancestry_id": source.get("ancestry_id", source_id),
                "consumed_continuous_evidence": {
                    "accepted_false_wakes": accepted,
                    "report": report_binding_by_path[report_path],
                },
            }
        )
        promoted.append(row)

    payload = {
        "schema_version": 1,
        "kind": "kizz_consumed_continuous_negative_development_assets",
        "selection_policy": {
            "source_lock_was_consumed_before_promotion": True,
            "selection": selection,
            "accepted_false_wakes_required": selection == "accepted",
            "development_only": True,
            "future_qualification_lock_must_be_disjoint": True,
            "promoted_source_group": source_group,
        },
        "inputs": {
            "consumed_locked_manifest": _binding(lock_path),
            "continuous_cascade_reports": report_bindings,
        },
        "counts": {
            "source_files": len(promoted),
            "accepted_false_wakes": sum(
                int(row["consumed_continuous_evidence"]["accepted_false_wakes"])
                for row in promoted
            ),
            "exposure_seconds": sum(float(row["duration_seconds"]) for row in promoted),
        },
        "examples": promoted,
    }
    output = output.expanduser().resolve()
    _atomic_json(output, payload)
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        **payload["counts"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-group")
    parser.add_argument("--selection", choices=("accepted", "all"), default="accepted")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            promote(
                args.locked_manifest,
                args.report,
                args.output,
                source_group=args.source_group,
                selection=args.selection,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
