#!/usr/bin/env python3
"""Quarantine rejected C1 renders and repair the reserved replay panel.

The independent pronunciation audit is evidence, not a requirement that every
TTS render be usable.  This stage preserves the immutable source manifest,
marks rejected positives training-ineligible, and deterministically replaces
failed reserved anchors with accepted test clips.  Replacements prefer the
same provider and voice so the original provider/voice panel remains intact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_kizz_control_source_pronunciations import (
    AUDITED_PROVIDERS,
    RUNTIME_RESERVED_PROVIDERS,
)
from tools.generate_kizz_control_c1_corpus import (
    RESERVED_REPLAYS_PER_PROVIDER,
    corpus_mix_report,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_examples(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("examples"), list):
        raise ValueError(f"{path}: expected an examples manifest")
    rows = [dict(row) for row in payload["examples"]]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every example must be an object")
    return payload, rows


def _rank(seed: int, provider: str, voice: str, row: dict[str, Any]) -> bytes:
    return hashlib.sha256(
        (
            f"{seed}\0{provider}\0{voice}\0{row.get('audio_sha256')}\0"
            f"{row.get('source_id')}"
        ).encode()
    ).digest()


def _apply_provider_balance_cap(
    rows: list[dict[str, Any]], *, seed: int, maximum_share: float = 0.35
) -> list[dict[str, Any]]:
    """Exclude deterministic excess positives until every provider clears the cap."""
    excluded: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        while True:
            eligible = [
                row
                for row in rows
                if row.get("split") == split
                and int(row.get("label", -1)) == 1
                and row.get("training_eligible") is True
            ]
            if not eligible:
                break
            counts = Counter(str(row.get("provider", "")) for row in eligible)
            provider, count = max(counts.items(), key=lambda item: (item[1], item[0]))
            if count / len(eligible) <= maximum_share:
                break
            candidates = sorted(
                (row for row in eligible if row.get("provider") == provider),
                key=lambda row: _rank(
                    seed, provider, str(row.get("voice", "")), row
                ),
                reverse=True,
            )
            if not candidates:
                raise ValueError(f"{split}: cannot enforce provider balance cap")
            row = candidates[0]
            row["training_eligible"] = False
            row["exclusion_reason"] = "provider_balance_cap"
            excluded.append(
                {
                    "source_id": row.get("source_id"),
                    "split": split,
                    "provider": provider,
                    "audio_sha256": row.get("audio_sha256"),
                }
            )
    return excluded


def curate(
    source_manifest: Path,
    pronunciation_audit: Path,
    output: Path,
    quarantine_output: Path,
    report_output: Path,
    *,
    seed: int = 231,
    excluded_positive_providers: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source, rows = _load_examples(source_manifest)
    audit = json.loads(pronunciation_audit.read_text(encoding="utf-8"))
    if (
        audit.get("gate_scope") != "independent_source_pronunciation_qc"
        or audit.get("source_manifest_sha256") != sha256_file(source_manifest)
        or set(audit.get("scope", {}).get("splits", []))
        != {"train", "validation", "test"}
        or audit.get("scope", {}).get("gate_mode") != "all"
    ):
        raise ValueError("pronunciation audit is not the complete source-bound audit")

    audited = audit.get("results", [])
    if not isinstance(audited, list):
        raise ValueError("pronunciation audit results must be a list")
    audit_by_id = {str(item.get("source_id", "")): item for item in audited}
    if (
        not audit_by_id
        or "" in audit_by_id
        or len(audit_by_id) != len(audited)
    ):
        raise ValueError("pronunciation audit has missing or duplicate source identities")
    expected_ids = {
        str(row.get("source_id", ""))
        for row in rows
        if int(row.get("label", -1)) == 1
        and row.get("provider") in AUDITED_PROVIDERS
        and row.get("split") in {"train", "validation", "test"}
    }
    if expected_ids != set(audit_by_id):
        raise ValueError("pronunciation audit does not cover every C1 positive")

    rejected_rows: list[dict[str, Any]] = []
    failed_reserved: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result = audit_by_id.get(str(row.get("source_id", "")))
        if result is None:
            continue
        if result.get("audio_sha256") != row.get("audio_sha256"):
            raise ValueError(f"pronunciation audio hash drift: {row.get('source_id')}")
        row["pronunciation_qc"] = {
            "accepted": result.get("accepted") is True,
            "phones": str(result.get("phones", "")),
            "model": str(audit.get("model", {}).get("name", "")),
            "report_sha256": sha256_file(pronunciation_audit),
        }
        if result.get("accepted") is True:
            continue
        if row.get("reserved_evidence_role") == "target_channel_positive":
            failed_reserved[str(row.get("provider"))].append(dict(row))
        row.pop("reserved_evidence_role", None)
        row["training_eligible"] = False
        row["exclusion_reason"] = "pronunciation_qc_rejected"
        quarantined = dict(row)
        quarantined["quarantine_reason"] = "pronunciation_qc_rejected"
        quarantined["source_was_reserved"] = bool(
            result.get("reserved") is True
        )
        rejected_rows.append(quarantined)

    source_audit_exclusions: list[dict[str, Any]] = []
    excluded_provider_set = set(excluded_positive_providers)
    for row in rows:
        if (
            int(row.get("label", -1)) == 1
            and row.get("provider") in excluded_provider_set
            and row.get("training_eligible") is True
        ):
            row["training_eligible"] = False
            row["exclusion_reason"] = "source_audit_only_provider"
            source_audit_exclusions.append(
                {
                    "source_id": row.get("source_id"),
                    "provider": row.get("provider"),
                    "split": row.get("split"),
                    "audio_sha256": row.get("audio_sha256"),
                }
            )

    replacements: list[dict[str, Any]] = []
    for provider in RUNTIME_RESERVED_PROVIDERS:
        retained = [
            row
            for row in rows
            if row.get("provider") == provider
            and row.get("reserved_evidence_role") == "target_channel_positive"
        ]
        candidates = [
            row
            for row in rows
            if row.get("provider") == provider
            and row.get("split") == "test"
            and int(row.get("label", -1)) == 1
            and row.get("pronunciation_qc", {}).get("accepted") is True
            and row.get("reserved_evidence_role") != "target_channel_positive"
        ]
        selected_ids: set[str] = set()
        retained_voice_counts = Counter(str(row.get("voice", "")) for row in retained)
        failures = sorted(
            failed_reserved.get(provider, []),
            key=lambda row: (str(row.get("voice", "")), str(row.get("source_id", ""))),
        )
        for failed in failures:
            voice = str(failed.get("voice", ""))
            same_voice = sorted(
                (
                    row
                    for row in candidates
                    if str(row.get("voice", "")) == voice
                    and str(row.get("source_id", "")) not in selected_ids
                ),
                key=lambda row: _rank(seed, provider, voice, row),
            )
            pool = same_voice or sorted(
                (
                    row
                    for row in candidates
                    if str(row.get("source_id", "")) not in selected_ids
                ),
                key=lambda row: (
                    retained_voice_counts[str(row.get("voice", ""))],
                    _rank(seed, provider, str(row.get("voice", "")), row),
                ),
            )
            if not pool:
                raise ValueError(f"{provider}: no accepted replacement for failed reserve")
            replacement = pool[0]
            selected_ids.add(str(replacement["source_id"]))
            retained_voice_counts[str(replacement.get("voice", ""))] += 1
            replacements.append(
                {
                    "provider": provider,
                    "failed_source_id": failed.get("source_id"),
                    "failed_voice": voice,
                    "replacement_source_id": replacement.get("source_id"),
                    "replacement_voice": replacement.get("voice"),
                    "same_voice": replacement.get("voice") == voice,
                }
            )

        needed = RESERVED_REPLAYS_PER_PROVIDER - len(retained) - len(selected_ids)
        if needed < 0:
            raise ValueError(f"{provider}: too many retained reserved anchors")
        extras = sorted(
            (
                row
                for row in candidates
                if str(row.get("source_id", "")) not in selected_ids
            ),
            key=lambda row: (
                retained_voice_counts[str(row.get("voice", ""))],
                _rank(seed, provider, str(row.get("voice", "")), row),
            ),
        )
        if len(extras) < needed:
            raise ValueError(f"{provider}: insufficient accepted reserved candidates")
        selected_ids.update(str(row["source_id"]) for row in extras[:needed])

        for row in rows:
            if str(row.get("source_id", "")) not in selected_ids:
                continue
            row["reserved_evidence_role"] = "target_channel_positive"
            row["training_eligible"] = False
            row["exclusion_reason"] = "reserved_for_device_replay"

    declared_modeled_providers = source.get("positive_provider_policy", {}).get(
        "expected_positive_providers"
    )
    if declared_modeled_providers is None:
        modeled_providers = set(AUDITED_PROVIDERS) - excluded_provider_set
    elif (
        not isinstance(declared_modeled_providers, list)
        or not declared_modeled_providers
        or any(
            not isinstance(provider, str) or provider not in AUDITED_PROVIDERS
            for provider in declared_modeled_providers
        )
    ):
        raise ValueError("source positive provider policy is malformed")
    else:
        modeled_providers = set(declared_modeled_providers) - excluded_provider_set
    balance_exclusions = _apply_provider_balance_cap(rows, seed=seed)
    mix = corpus_mix_report(
        rows, expected_positive_providers=sorted(modeled_providers)
    )
    if not mix["qualified"]:
        raise ValueError(f"curated source mix does not qualify: {mix['violations']}")

    counts = Counter(
        (
            str(row.get("split")),
            str(row.get("provider")),
            int(row.get("label", -1)),
            row.get("training_eligible") is True,
        )
        for row in rows
    )
    audit_sha = sha256_file(pronunciation_audit)
    curated = dict(source)
    curated.update(
        {
            "schema_version": 3,
            "recipe": "kizz_control_c1_pronunciation_curated",
            "inputs": {
                "source_manifest": str(source_manifest.resolve()),
                "source_manifest_sha256": sha256_file(source_manifest),
                "pronunciation_audit": str(pronunciation_audit.resolve()),
                "pronunciation_audit_sha256": audit_sha,
            },
            "counts": {
                f"{split}:{provider}:{label}:{str(eligible).lower()}": count
                for (split, provider, label, eligible), count in sorted(counts.items())
            },
            "source_mix_contract": mix,
            "positive_provider_policy": {
                "mode": "explicit_pronunciation_curated_subset",
                "expected_positive_providers": sorted(modeled_providers),
                "source_audit_only_providers": sorted(excluded_provider_set),
            },
            "examples": sorted(
                rows,
                key=lambda row: (
                    str(row.get("split", "")),
                    str(row.get("provider", "")),
                    str(row.get("source_id", "")),
                ),
            ),
        }
    )
    quarantine = {
        "schema_version": 2,
        "recipe": "kizz_control_c1_pronunciation_quarantine",
        "inputs": curated["inputs"],
        "counts": {
            "quarantined": len(rejected_rows),
            "reserved_quarantined": sum(
                row["source_was_reserved"] for row in rejected_rows
            ),
        },
        "examples": sorted(rejected_rows, key=lambda row: str(row.get("source_id", ""))),
    }
    reserved = [
        row
        for row in rows
        if row.get("reserved_evidence_role") == "target_channel_positive"
    ]
    report = {
        "schema_version": 1,
        "kind": "kizz_control_c1_pronunciation_curation",
        "qualified": len(reserved) == 24 and len(replacements) == sum(
            len(value) for value in failed_reserved.values()
        ),
        "seed": seed,
        "inputs": curated["inputs"],
        "counts": {
            "source_examples": len(rows),
            "audited_positives": len(audited),
            "accepted_positives": sum(item.get("accepted") is True for item in audited),
            "quarantined_positives": len(rejected_rows),
            "provider_balance_exclusions": len(balance_exclusions),
            "source_audit_only_exclusions": len(source_audit_exclusions),
            "reserved_anchors": len(reserved),
            "reserved_replacements": len(replacements),
        },
        "reserved_provider_contract": {
            provider: {
                "count": sum(row.get("provider") == provider for row in reserved),
                "voices": sorted(
                    {str(row.get("voice", "")) for row in reserved if row.get("provider") == provider}
                ),
            }
            for provider in RUNTIME_RESERVED_PROVIDERS
        },
        "replacements": replacements,
        "provider_balance_exclusions": balance_exclusions,
        "source_audit_only_exclusions": source_audit_exclusions,
    }
    if not report["qualified"]:
        raise ValueError("pronunciation curation did not repair the reserved panel")

    for path, payload in (
        (output, curated),
        (quarantine_output, quarantine),
        (report_output, report),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return curated, quarantine, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--pronunciation-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quarantine-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument(
        "--exclude-positive-provider",
        action="append",
        default=[],
        help="retain a provider for audit but exclude all of its positives from modeling",
    )
    args = parser.parse_args(argv)
    curated, quarantine, report = curate(
        args.source_manifest,
        args.pronunciation_audit,
        args.output,
        args.quarantine_output,
        args.report_output,
        seed=args.seed,
        excluded_positive_providers=args.exclude_positive_provider,
    )
    print(
        json.dumps(
            {
                "examples": len(curated["examples"]),
                "quarantined": quarantine["counts"]["quarantined"],
                "reserved_replacements": report["counts"]["reserved_replacements"],
                "qualified": report["qualified"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
