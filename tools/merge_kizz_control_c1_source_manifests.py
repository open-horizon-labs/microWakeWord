#!/usr/bin/env python3
"""Merge independently-rendered Kizz C1 provider manifests without losing lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from tools.generate_kizz_control_c1_corpus import corpus_mix_report


def _balance_positive_provider_share(
    rows: list[dict[str, Any]], providers: Sequence[str], maximum: float = 0.35
) -> list[dict[str, str]]:
    """Withhold the minimum deterministic rows needed to satisfy the mix cap."""
    exclusions: list[dict[str, str]] = []
    for split in ("train", "validation", "test"):
        while True:
            eligible = [
                row
                for row in rows
                if row.get("training_eligible") is True
                and int(row.get("label", -1)) == 1
                and row.get("split") == split
            ]
            counts = {
                provider: sum(row.get("provider") == provider for row in eligible)
                for provider in providers
            }
            if not eligible or max(counts.values(), default=0) / len(eligible) <= maximum:
                break
            provider = max(counts, key=lambda name: (counts[name], name))
            candidates = sorted(
                (row for row in eligible if row.get("provider") == provider),
                key=lambda row: str(row.get("descriptor_sha256")),
                reverse=True,
            )
            row = candidates[0]
            row["training_eligible"] = False
            row["exclusion_reason"] = "deterministic_provider_balance_cap"
            exclusions.append(
                {
                    "source_id": str(row.get("source_id")),
                    "provider": provider,
                    "split": split,
                }
            )
    return exclusions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def merge(
    manifests: Sequence[Path], output: Path, *, expected_providers: Sequence[str]
) -> dict[str, Any]:
    providers = tuple(sorted(set(expected_providers)))
    if not manifests or not providers:
        raise ValueError("at least one manifest and expected provider are required")
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for raw_path in manifests:
        path = raw_path.expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list) or not all(
            isinstance(row, dict) for row in examples
        ):
            raise ValueError(f"{path}: expected an examples manifest")
        rows.extend(dict(row) for row in examples)
        bindings.append({"path": str(path), "sha256": _sha256(path)})

    seen_descriptor: set[str] = set()
    seen_path: set[str] = set()
    seen_audio: set[str] = set()
    for row in rows:
        provider = str(row.get("provider", ""))
        descriptor = str(row.get("descriptor_sha256", ""))
        audio_hash = str(row.get("audio_sha256", ""))
        path = str(Path(str(row.get("path", ""))).expanduser().resolve())
        if provider not in providers:
            raise ValueError(f"unexpected provider in source manifest: {provider}")
        if not descriptor or descriptor in seen_descriptor:
            raise ValueError("missing or duplicate source descriptor")
        if not audio_hash or audio_hash in seen_audio:
            raise ValueError("missing or duplicate source audio hash")
        if not Path(path).is_file() or path in seen_path:
            raise ValueError("missing or duplicate source audio path")
        row["path"] = path
        seen_descriptor.add(descriptor)
        seen_audio.add(audio_hash)
        seen_path.add(path)

    rows.sort(
        key=lambda row: (
            str(row.get("split")),
            -int(row.get("label", -1)),
            str(row.get("provider")),
            str(row.get("voice")),
            str(row.get("source_id")),
        )
    )
    balance_exclusions = _balance_positive_provider_share(rows, providers)
    mix = corpus_mix_report(rows, expected_positive_providers=providers)
    if not mix["qualified"]:
        raise ValueError(f"merged source mix does not qualify: {mix['violations']}")
    payload = {
        "schema_version": 2,
        "recipe": "kizz_control_c1_independent_provider_merge_v1",
        "positive_provider_policy": {
            "expected_positive_providers": list(providers),
            "macos_say_training_excluded": "macos-say" not in providers,
        },
        "input_bindings": sorted(bindings, key=lambda item: item["path"]),
        "deterministic_provider_balance_exclusions": balance_exclusions,
        "source_mix_contract": mix,
        "examples": rows,
    }
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite merged manifest: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--expected-provider", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = merge(
        args.manifest,
        args.output,
        expected_providers=args.expected_provider,
    )
    print(
        json.dumps(
            {
                "examples": len(payload["examples"]),
                "providers": payload["positive_provider_policy"][
                    "expected_positive_providers"
                ],
                "qualified": payload["source_mix_contract"]["qualified"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
