#!/usr/bin/env python3
"""Reserve a deterministic multisource positive inventory before model scoring."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PURPOSE = "fresh_target_channel_positive_candidate_inventory"
SELECTION_POLICY = "deterministic_provider_then_voice_round_robin_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved}: manifest root must be an object")
    return resolved, payload


def _binding(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _manifest_rows(payload: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    rows = payload.get("examples")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: source manifest requires an examples list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: every source example must be an object")
    return [dict(row) for row in rows]


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _relevant_hashes(value: Any) -> set[str]:
    """Collect audio/source/parent hashes from an examples/captures subtree."""
    hashes: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if _is_sha256(child) and (
                key == "sha256"
                or key.endswith("_sha256")
                and any(token in key for token in ("audio", "source", "parent"))
            ):
                hashes.add(str(child).lower())
            hashes.update(_relevant_hashes(child))
    elif isinstance(value, list):
        for child in value:
            hashes.update(_relevant_hashes(child))
    return hashes


def _evidence_subtrees(payload: Mapping[str, Any]) -> Iterable[Any]:
    for key in ("examples", "captures"):
        value = payload.get(key)
        if isinstance(value, list):
            yield value


def _source_audio_hash(row: Mapping[str, Any], context: str) -> str:
    value = row.get("audio_sha256")
    if not _is_sha256(value):
        raise ValueError(f"{context}: audio_sha256 must be a lowercaseable SHA-256")
    return str(value).lower()


def _source_id(row: Mapping[str, Any], context: str) -> str:
    value = row.get("source_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: source_id must be a nonempty string")
    return value


def _qualified_identities(payload: Mapping[str, Any], path: Path) -> set[tuple[str, str]]:
    identities: set[tuple[str, str]] = set()
    for subtree in _evidence_subtrees(payload):
        for raw in subtree:
            if not isinstance(raw, Mapping):
                continue
            if "source_id" not in raw or "audio_sha256" not in raw:
                continue
            context = f"{path}/{raw.get('source_id', '<missing>')}"
            identities.add((_source_id(raw, context), _source_audio_hash(raw, context)))
    if not identities:
        raise ValueError(f"{path}: qualified source manifest has no source/audio identities")
    return identities


def _provider(row: Mapping[str, Any], context: str) -> str:
    value = row.get("provider")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: provider must be a nonempty string")
    return value.strip()


def _is_macos_say(provider: str) -> bool:
    normalized = "".join(character for character in provider.lower() if character.isalnum())
    return normalized in {"say", "macossay", "applesay"}


def _voice(row: Mapping[str, Any], provider: str, context: str) -> str:
    for key in ("voice_id", "provider_voice_id", "voice", "speaker_id"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return f"{provider}:{value.strip()}"
    raise ValueError(f"{context}: source requires a voice identity")


def _audio_path(row: Mapping[str, Any], manifest_path: Path, context: str) -> Path:
    raw = row.get("path")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{context}: path must be a nonempty string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _stable_candidate_key(row: Mapping[str, Any], provider: str, voice: str) -> tuple[str, str]:
    identity = {
        "provider": provider,
        "voice": voice,
        "source_id": row["source_id"],
        "audio_sha256": row["audio_sha256"],
    }
    return hashlib.sha256(_canonical_bytes(identity)).hexdigest(), str(row["source_id"])


def _balanced_select(
    rows: Sequence[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        buckets[str(row["provider"])][str(row["inventory_voice_id"])].append(row)
    for provider, voices in buckets.items():
        for voice, candidates in voices.items():
            candidates.sort(key=lambda row: _stable_candidate_key(row, provider, voice))

    providers = sorted(buckets)
    voice_orders = {provider: sorted(buckets[provider]) for provider in providers}
    voice_positions = {provider: 0 for provider in providers}
    candidate_positions = {
        (provider, voice): 0
        for provider in providers
        for voice in voice_orders[provider]
    }
    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for provider in providers:
            voices = voice_orders[provider]
            for _ in range(len(voices)):
                voice_index = voice_positions[provider] % len(voices)
                voice_positions[provider] += 1
                voice = voices[voice_index]
                candidate_index = candidate_positions[(provider, voice)]
                if candidate_index >= len(buckets[provider][voice]):
                    continue
                selected.append(copy.deepcopy(buckets[provider][voice][candidate_index]))
                candidate_positions[(provider, voice)] += 1
                progressed = True
                break
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def _atomic_no_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8"))
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        # Some removable macOS volumes support atomic rename but can block
        # indefinitely on hard-link creation.  Recheck immediately before the
        # rename so the normal no-clobber contract is preserved for this
        # single-writer artifact pipeline.
        if path.exists():
            raise FileExistsError(path)
        os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(
    source_manifests: Sequence[Path],
    exclude_manifests: Sequence[Path],
    qualified_source_manifest: Path,
    output: Path,
    *,
    count: int = 32,
    minimum_providers: int = 3,
    minimum_voices: int = 8,
    allow_unconsumed_training_eligible: bool = False,
) -> dict[str, Any]:
    if count < 1 or minimum_providers < 1 or minimum_voices < 1:
        raise ValueError("count, minimum providers, and minimum voices must be positive")
    if minimum_providers > count or minimum_voices > count:
        raise ValueError("minimum providers and voices cannot exceed count")
    if not source_manifests:
        raise ValueError("at least one source manifest is required")
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)

    qualified_path, qualified_payload = _load_manifest(qualified_source_manifest)
    qualified = _qualified_identities(qualified_payload, qualified_path)

    exclusion_bindings: list[dict[str, Any]] = []
    excluded_hashes: set[str] = set()
    for raw_path in sorted(exclude_manifests, key=lambda value: str(value.expanduser().resolve())):
        path, payload = _load_manifest(raw_path)
        exclusion_bindings.append(_binding(path))
        for subtree in _evidence_subtrees(payload):
            excluded_hashes.update(_relevant_hashes(subtree))

    source_bindings: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    identity_hashes: dict[str, str] = {}
    audio_identities: dict[str, str] = {}
    exact_seen: set[tuple[str, str]] = set()
    for raw_path in sorted(source_manifests, key=lambda value: str(value.expanduser().resolve())):
        manifest_path, payload = _load_manifest(raw_path)
        source_bindings.append(_binding(manifest_path))
        for row_index, row in enumerate(_manifest_rows(payload, manifest_path)):
            if row.get("label") not in (1, True):
                continue
            if row.get("split") != "test":
                continue
            if (
                not allow_unconsumed_training_eligible
                and row.get("training_eligible") is not False
            ):
                continue
            context = f"{manifest_path}/examples/{row_index}"
            provider = _provider(row, context)
            if _is_macos_say(provider):
                continue
            source_id = _source_id(row, context)
            audio_hash = _source_audio_hash(row, context)
            identity = (source_id, audio_hash)
            if identity not in qualified:
                continue
            previous_hash = identity_hashes.setdefault(source_id, audio_hash)
            if previous_hash != audio_hash:
                raise ValueError(f"source_id {source_id!r} maps to conflicting audio hashes")
            previous_id = audio_identities.setdefault(audio_hash, source_id)
            if previous_id != source_id:
                raise ValueError(
                    f"audio hash {audio_hash} maps to conflicting source IDs: "
                    f"{previous_id!r} and {source_id!r}"
                )
            if identity in exact_seen:
                continue
            exact_seen.add(identity)
            if _relevant_hashes(row) & excluded_hashes:
                continue
            path = _audio_path(row, manifest_path, context)
            observed_hash = sha256_file(path)
            if observed_hash != audio_hash:
                raise ValueError(
                    f"{context}: source audio hash drift: expected {audio_hash}, got {observed_hash}"
                )
            voice = _voice(row, provider, context)
            candidate = copy.deepcopy(row)
            candidate["provider"] = provider
            candidate["audio_sha256"] = audio_hash
            candidate["inventory_voice_id"] = voice
            candidates.append(candidate)

    available_providers = {str(row["provider"]) for row in candidates}
    available_voices = {str(row["inventory_voice_id"]) for row in candidates}
    if len(candidates) < count:
        raise ValueError(f"only {len(candidates)} qualified unconsumed candidates; need {count}")
    if len(available_providers) < minimum_providers:
        raise ValueError(
            f"only {len(available_providers)} qualified providers; need {minimum_providers}"
        )
    if len(available_voices) < minimum_voices:
        raise ValueError(f"only {len(available_voices)} qualified voices; need {minimum_voices}")

    selected = _balanced_select(candidates, count)
    selected_providers = Counter(str(row["provider"]) for row in selected)
    selected_voices = Counter(str(row["inventory_voice_id"]) for row in selected)
    selected_provider_voices = {
        provider: {
            str(row["inventory_voice_id"])
            for row in selected
            if str(row["provider"]) == provider
        }
        for provider in selected_providers
    }
    if len(selected) != count:
        raise ValueError(f"balanced selection produced only {len(selected)} candidates")
    if len(selected_providers) < minimum_providers:
        raise ValueError("balanced selection does not realize the minimum provider count")
    if len(selected_voices) < minimum_voices:
        raise ValueError("balanced selection does not realize the minimum voice count")

    for index, row in enumerate(selected):
        row["candidate_inventory_selection_index"] = index
        row["reserved_evidence_role"] = "target_channel_positive"
        row["evidence_status"] = "reserved"
        row["exclusion_reason"] = "reserved_for_fresh_final_device_qualification"
        row["locked_before_scoring"] = True
        row["training_eligible"] = False

    payload: dict[str, Any] = {
        "schema_version": 1,
        "corpus_id": "kizz-control-reserved-multisource-qualification-inventory-v1",
        "purpose": PURPOSE,
        "locked_before_scoring": True,
        "training_eligible": False,
        "inputs": {
            "source_manifests": source_bindings,
            "exclude_manifests": exclusion_bindings,
            "qualified_source_manifest": _binding(qualified_path),
        },
        "selection_policy": {
            "name": SELECTION_POLICY,
            "requested_count": count,
            "minimum_providers": minimum_providers,
            "minimum_voices": minimum_voices,
            "candidate_requirements": {
                "label": 1,
                "split": "test",
                "source_training_eligible": (
                    "either_but_not_present_in_any_exclusion_manifest"
                    if allow_unconsumed_training_eligible
                    else False
                ),
                "reserved_output_training_eligible": False,
                "macos_say_excluded": True,
                "qualified_identity": "exact_source_id_and_audio_sha256",
            },
            "exclusion": "recursive_audio_source_parent_hash_intersection",
            "balance_order": ["provider", "voice"],
            "candidate_order": "sha256_of_provider_voice_source_id_audio_sha256",
            "model_scores_read_or_used": False,
        },
        "counts": {
            "qualified_unconsumed_available": len(candidates),
            "selected": len(selected),
            "providers": len(selected_providers),
            "voices": len(selected_voices),
        },
        "provider_counts": dict(sorted(selected_providers.items())),
        "voice_counts": dict(sorted(selected_voices.items())),
        "reserved_evidence_contract": {
            "role": "target_channel_positive",
            "status": "reserved",
            "locked_before_scoring": True,
            "training_eligible": False,
            "total_count": len(selected),
            "providers": {
                provider: {
                    "count": count,
                    "minimum_voices": len(selected_provider_voices[provider]),
                }
                for provider, count in sorted(selected_providers.items())
            },
        },
        "examples": selected,
    }
    _atomic_no_replace_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, action="append", required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--qualified-source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--minimum-providers", type=int, default=3)
    parser.add_argument("--minimum-voices", type=int, default=8)
    parser.add_argument(
        "--allow-unconsumed-training-eligible",
        action="store_true",
        help=(
            "Allow qualified test-voice sources marked training-eligible only when "
            "their recursive source/audio hashes are absent from every exclusion manifest."
        ),
    )
    args = parser.parse_args(argv)
    try:
        payload = prepare(
            args.source_manifest,
            args.exclude_manifest,
            args.qualified_source_manifest,
            args.output,
            count=args.count,
            minimum_providers=args.minimum_providers,
            minimum_voices=args.minimum_voices,
            allow_unconsumed_training_eligible=args.allow_unconsumed_training_eligible,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
