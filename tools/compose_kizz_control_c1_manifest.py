#!/usr/bin/env python3
"""Compose Kizz Control C1 with phrase-independent frozen negatives.

Only eligible C1 source positives/collisions enter. Frozen negatives retain
their train/validation/test role: train rows may augment or train a model, while
non-training ESC-50 rows remain clean evaluation evidence. Locked deployment or
continuous-stream anchors never enter. A second manifest contains an equal-count
deterministic train-only overlay pool, preventing the largest archived source
from dominating positive augmentation by row count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microwakeword.kizz_data_contract import sha256_file
from tools.generate_kizz_control_c1_corpus import (
    causal_negative_decision,
    corpus_mix_report,
)


REUSABLE_NEGATIVE_GROUPS = frozenset(
    ("public_speech", "music", "background_noise", "device_collision")
)
OVERLAY_GROUP_MAP = {
    "public_speech": "speech",
    "device_collision": "speech",
    "music": "music",
    "background_noise": "background_noise",
    "kizz_control_phonetic_collision": "collision",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("examples"), list):
        raise ValueError(f"{path}: expected an examples manifest")
    return value


def _stable_sample(
    rows: Sequence[dict[str, Any]], count: int, seed: int, group: str
) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{seed}\0{group}\0{row.get('audio_sha256')}\0{row.get('source_id')}".encode()
        ).digest(),
    )
    return ranked[: min(count, len(ranked))]


def reusable_frozen_negatives(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep split-aware reusable negatives without admitting locked anchors."""
    selected = []
    for raw in rows:
        if (
            int(raw.get("label", -1)) != 0
            or raw.get("source_group") not in REUSABLE_NEGATIVE_GROUPS
            or raw.get("locked_deployment_anchor") is True
            or raw.get("split") not in {"train", "validation", "test"}
        ):
            continue
        row = dict(raw)
        if row["split"] == "train" and row.get("training_eligible") is not True:
            raise ValueError("train negative must be explicitly training eligible")
        if row["split"] != "train" and row.get("training_eligible") is not False:
            raise ValueError("evaluation negative must be explicitly non-training")
        selected.append(row)
    return selected


def compose(
    source_manifest: Path,
    frozen_manifest: Path,
    output: Path,
    overlay_output: Path,
    *,
    overlay_per_family: int = 120,
    seed: int = 231,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _load(source_manifest)
    frozen = _load(frozen_manifest)
    source_rows = [
        dict(row)
        for row in source["examples"]
        if row.get("training_eligible") is True
    ]
    impossible_source_negatives = [
        {
            "source_id": row.get("source_id"),
            "render_text": row.get("render_text"),
            "reason": causal_negative_decision(str(row.get("render_text", "")))[
                "reason"
            ],
        }
        for row in source["examples"]
        if int(row.get("label", -1)) == 0
        and not causal_negative_decision(str(row.get("render_text", "")))[
            "qualified"
        ]
    ]
    for row in source_rows:
        if int(row.get("label", -1)) == 0 and not causal_negative_decision(
            str(row.get("render_text", ""))
        )["qualified"]:
            raise ValueError(
                "causally unlearnable suffix-extension negative reached composition"
            )
    provider_policy = source.get("positive_provider_policy", {})
    expected_positive_providers = provider_policy.get(
        "expected_positive_providers"
    )
    mix = corpus_mix_report(
        source["examples"],
        **(
            {"expected_positive_providers": expected_positive_providers}
            if expected_positive_providers is not None
            else {}
        ),
    )
    if not mix["qualified"]:
        raise ValueError(f"C1 source mix does not qualify: {mix['violations']}")
    frozen_rows = reusable_frozen_negatives(frozen["examples"])
    rows = source_rows + frozen_rows
    seen_paths: set[str] = set()
    seen_audio: set[str] = set()
    for index, row in enumerate(rows):
        path = str(Path(row["path"]).resolve())
        audio_hash = str(row.get("audio_sha256", ""))
        if not audio_hash or audio_hash in seen_audio:
            raise ValueError(f"duplicate or missing audio hash at composed row {index}")
        if path in seen_paths:
            raise ValueError(f"duplicate path at composed row {index}")
        lowered = {part.casefold() for part in Path(path).parts}
        if "false-wakes" in lowered or "observations" in lowered:
            raise ValueError("quarantined observation path cannot enter training")
        seen_audio.add(audio_hash)
        seen_paths.add(path)
        row["path"] = path
    rows.sort(
        key=lambda row: (
            row["split"],
            -int(row["label"]),
            row["source_group"],
            row["source_id"],
        )
    )
    counts = Counter(
        (str(row["split"]), int(row["label"]), str(row["source_group"]))
        for row in rows
    )
    payload = {
        "schema_version": 2,
        "recipe": "kizz_control_c1_clean_source_plus_frozen_negatives",
        "inputs": {
            "source_manifest": str(source_manifest.resolve()),
            "source_manifest_sha256": sha256_file(source_manifest),
            "frozen_manifest": str(frozen_manifest.resolve()),
            "frozen_manifest_sha256": sha256_file(frozen_manifest),
        },
        "source_mix_contract": mix,
        "causal_negative_exclusions": impossible_source_negatives,
        "counts": [
            {"split": split, "label": label, "source_group": group, "count": count}
            for (split, label, group), count in sorted(counts.items())
        ],
        "examples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train" or int(row.get("label", -1)) != 0:
            continue
        family = OVERLAY_GROUP_MAP.get(str(row.get("source_group")))
        if family is not None:
            candidates[family].append(row)
    required = {"speech", "music", "background_noise", "collision"}
    if set(candidates) != required:
        raise ValueError(
            f"balanced overlay inventory is missing families: {sorted(required-set(candidates))}"
        )
    selected = []
    for family in sorted(required):
        sample = _stable_sample(candidates[family], overlay_per_family, seed, family)
        if len(sample) < overlay_per_family:
            raise ValueError(
                f"overlay family {family} has {len(sample)} rows, needs {overlay_per_family}"
            )
        for row in sample:
            item = dict(row)
            item["overlay_family"] = family
            selected.append(item)
    overlay = {
        "schema_version": 2,
        "recipe": "kizz_control_c1_balanced_overlay_inventory",
        "source_manifest": str(output.resolve()),
        "source_manifest_sha256": sha256_file(output),
        "seed": seed,
        "per_family": overlay_per_family,
        "family_counts": dict(Counter(row["overlay_family"] for row in selected)),
        "examples": selected,
    }
    overlay_output.parent.mkdir(parents=True, exist_ok=True)
    overlay_output.write_text(json.dumps(overlay, indent=2, sort_keys=True) + "\n")
    return payload, overlay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay-output", type=Path, required=True)
    parser.add_argument("--overlay-per-family", type=int, default=120)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args(argv)
    if args.overlay_per_family < 1:
        parser.error("--overlay-per-family must be positive")
    payload, overlay = compose(
        args.source_manifest,
        args.frozen_manifest,
        args.output,
        args.overlay_output,
        overlay_per_family=args.overlay_per_family,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "examples": len(payload["examples"]),
                "overlay_examples": len(overlay["examples"]),
                "overlay_family_counts": overlay["family_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
