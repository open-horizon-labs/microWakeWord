"""Fail-closed identity and audio contracts for Kizz qualification evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

LINEAGE_KEYS = (
    "provenance_id",
    "ancestry_id",
    "parent_id",
    "parent_source_id",
    "source_id",
)
PARTITION_KEYS = ("speaker_id", "voice_id", "session_id")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_identity(key: str, value: object) -> str:
    text = str(value)
    for prefix in ("audio-sha256:", "audio_sha256:", "source-audio-sha256:"):
        if text.startswith(prefix):
            return "audio:" + text.removeprefix(prefix)
    if key in ("audio_sha256", "source_audio_sha256"):
        return "audio:" + text
    return f"{key}:{text}"


def identity_aliases(
    row: Mapping[str, Any], *, include_partition_identity: bool = False
) -> frozenset[str]:
    """Return every stable alias that can reveal cross-group leakage."""
    aliases = {
        _normalized_identity(key, row[key])
        for key in ("audio_sha256", "source_audio_sha256", *LINEAGE_KEYS)
        if row.get(key)
    }
    if include_partition_identity:
        aliases.update(
            _normalized_identity(key, row[key])
            for key in PARTITION_KEYS
            if row.get(key)
        )
    return frozenset(aliases)


def group_identity_overlaps(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_partition_identity: bool = False,
) -> list[dict[str, Any]]:
    """Return identities owned by more than one named evidence group."""
    owners: dict[str, set[str]] = {}
    for group, rows in groups.items():
        for row in rows:
            for identity in identity_aliases(
                row, include_partition_identity=include_partition_identity
            ):
                owners.setdefault(identity, set()).add(group)
    return [
        {"identity": identity, "groups": sorted(group_names)}
        for identity, group_names in sorted(owners.items())
        if len(group_names) > 1
    ]


def require_disjoint_groups(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    include_partition_identity: bool = False,
) -> None:
    overlaps = group_identity_overlaps(
        groups, include_partition_identity=include_partition_identity
    )
    if overlaps:
        first = overlaps[0]
        raise ValueError(
            "qualification evidence groups overlap: "
            f"{first['identity']} in {', '.join(first['groups'])}"
        )


def validate_audio_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    group: str,
    require_locked_anchor: bool = False,
) -> dict[str, Any]:
    """Verify declared hashes against each immutable qualification audio file."""
    hashes = set()
    source_ids = set()
    for index, row in enumerate(rows):
        source_id = row.get("source_id") or row.get("observation_id") or row.get("id")
        if not source_id:
            raise ValueError(f"{group} row {index} has no stable source ID")
        source_id = str(source_id)
        if source_id in source_ids:
            raise ValueError(f"{group} contains duplicate source ID: {source_id}")
        source_ids.add(source_id)
        declared = row.get("audio_sha256") or row.get("source_audio_sha256")
        if not declared:
            raise ValueError(f"{group} row {source_id} has no declared audio SHA-256")
        declared = str(declared)
        if len(declared) != 64 or any(
            character not in "0123456789abcdef" for character in declared
        ):
            raise ValueError(f"{group} row {source_id} has invalid audio SHA-256")
        path_value = row.get("path") or row.get("audio_path") or row.get("recording")
        if not path_value:
            raise ValueError(f"{group} row {source_id} has no audio path")
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != declared:
            raise ValueError(
                f"{group} row {source_id} audio hash mismatch: "
                f"declared {declared}, got {actual}"
            )
        if declared in hashes:
            raise ValueError(f"{group} contains duplicate audio SHA-256: {declared}")
        hashes.add(declared)
        if require_locked_anchor and (
            row.get("locked_deployment_anchor") is not True
            or row.get("training_eligible") is not False
            or int(row.get("label", -1)) != 0
        ):
            raise ValueError(
                f"{group} row {source_id} is not a locked, training-ineligible negative"
            )
    return {
        "group": group,
        "count": len(rows),
        "unique_audio_sha256": len(hashes),
        "unique_source_ids": len(source_ids),
    }


__all__ = [
    "LINEAGE_KEYS",
    "PARTITION_KEYS",
    "group_identity_overlaps",
    "identity_aliases",
    "require_disjoint_groups",
    "sha256_file",
    "validate_audio_rows",
]
