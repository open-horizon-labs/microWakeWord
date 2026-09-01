#!/usr/bin/env python3
"""Freeze an untouched LibriSpeech continuous-negative lock.

This is deliberately a pre-scoring operation.  It validates and hashes the
raw FLAC files, selects whole speakers in a seeded deterministic order, and
binds the resulting holdout to the candidate corpus it must not overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import soundfile as sf


SCHEMA_VERSION = 2
LOCK_SCOPE = "locked_untouched_continuous_negative_corpus"
TARGET_SAMPLE_RATE = 16_000
DEFAULT_MINIMUM_HOURS = 100.0
DEFAULT_MARGIN_HOURS = 0.25
DEFAULT_SUBSET = "train-clean-360"
PROVIDER = "openslr_librispeech_slr12"
LICENSE = "CC BY 4.0"
HASH_FIELDS = (
    "audio_sha256",
    "sha256",
    "source_audio_sha256",
    "parent_source_audio_sha256",
)
LIBRISPEECH_SPEAKER_RE = re.compile(r"^librispeech(?:-mini|-speaker)?:([0-9]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_binding(path: Path, expected_md5: str | None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(block)
            md5.update(block)
    actual_md5 = md5.hexdigest()
    if expected_md5 is not None:
        expected_md5 = expected_md5.strip().lower()
        if not re.fullmatch(r"[0-9a-f]{32}", expected_md5):
            raise ValueError("source_archive_md5 must be 32 lowercase hexadecimal characters")
        if actual_md5 != expected_md5:
            raise ValueError(
                f"source archive MD5 mismatch: expected {expected_md5}, got {actual_md5}"
            )
    return {
        "path": str(path),
        "sha256": sha256.hexdigest(),
        "bytes": path.stat().st_size,
        "md5": actual_md5,
        "expected_md5": expected_md5,
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(_canonical_bytes(value))
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _binding(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _candidate_rows(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = _binding(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"candidate corpus is not readable JSON: {error}") from error
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError("candidate corpus requires nonempty object examples")
    return binding, payload


def _excluded_lock(path: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    binding = _binding(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"excluded continuous lock is not readable JSON: {error}") from error
    rows = payload.get("examples") if isinstance(payload, dict) else None
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("gate_scope") != LOCK_SCOPE
        or payload.get("locked_before_scoring") is not True
        or payload.get("training_eligible") is not False
        or not isinstance(rows, list)
        or not rows
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("excluded continuous lock has an unsupported contract")
    identities = {"speaker": set(), "audio": set(), "ancestry": set()}
    for index, row in enumerate(rows):
        speaker = row.get("speaker_id")
        ancestry = row.get("ancestry_id")
        audio = row.get("audio_sha256")
        if not all(isinstance(value, str) and value for value in (speaker, ancestry, audio)):
            raise ValueError(f"excluded continuous lock example {index} lacks identity")
        identities["speaker"].add(_canonical_speaker_identity(speaker))
        identities["ancestry"].add(ancestry)
        identities["audio"].add(audio)
    return binding, identities


def _canonical_speaker_identity(value: str) -> str:
    """Normalize historical LibriSpeech speaker prefixes before overlap checks."""
    match = LIBRISPEECH_SPEAKER_RE.fullmatch(value)
    if match is None:
        return value
    return f"librispeech-speaker:{match.group(1)}"


def _candidate_identities(payload: Mapping[str, Any]) -> dict[str, dict[str, set[str]]]:
    rows = payload["examples"]
    result = {
        split: {"speaker": set(), "audio": set(), "ancestry": set()}
        for split in ("train", "validation", "test")
    }
    for index, row in enumerate(rows):
        split = row.get("split")
        if split not in result:
            raise ValueError(f"candidate example {index} has unsupported split: {split!r}")
        speaker = row.get("speaker_id")
        ancestry = row.get("ancestry_id")
        if not isinstance(speaker, str) or not speaker:
            raise ValueError(f"candidate example {index} lacks speaker_id")
        if not isinstance(ancestry, str) or not ancestry:
            raise ValueError(f"candidate example {index} lacks ancestry_id")
        hashes = {
            value
            for field in HASH_FIELDS
            if isinstance((value := row.get(field)), str) and value
        }
        if not hashes:
            raise ValueError(f"candidate example {index} lacks an audio hash")
        result[split]["speaker"].add(_canonical_speaker_identity(speaker))
        result[split]["ancestry"].add(ancestry)
        result[split]["audio"].update(hashes)
    return result


def _source_name(subset: str) -> str:
    return f"OpenSLR SLR12 LibriSpeech {subset}"


def _speaker_order(speaker: str, seed: int, subset: str) -> tuple[bytes, str]:
    namespace = "360" if subset == DEFAULT_SUBSET else subset
    key = f"kizz-control-c1-continuous-librispeech-{namespace}-v1:{seed}:{speaker}"
    return hashlib.sha256(key.encode("utf-8")).digest(), speaker


def _validated_row(root: Path, path: Path, subset: str) -> dict[str, Any]:
    relative = path.relative_to(root)
    if len(relative.parts) != 3 or path.suffix.lower() != ".flac":
        raise ValueError(f"unexpected LibriSpeech {subset} path: {relative}")
    speaker, chapter, filename = relative.parts
    if not speaker.isdigit() or not chapter.isdigit() or not filename.startswith(f"{speaker}-{chapter}-"):
        raise ValueError(f"unexpected LibriSpeech identity path: {relative}")
    digest = sha256_file(path)
    try:
        with sf.SoundFile(path) as audio:
            if audio.format != "FLAC":
                raise ValueError("audio is not FLAC")
            if audio.samplerate != TARGET_SAMPLE_RATE or audio.channels != 1 or len(audio) <= 0:
                raise ValueError("audio is not 16 kHz mono with positive frames")
            expected_frames = len(audio)
            decoded_frames = 0
            for block in audio.blocks(blocksize=65_536, dtype="float32", always_2d=True):
                decoded_frames += len(block)
                if not bool(np.all(np.isfinite(block))):
                    raise ValueError("audio contains non-finite samples")
    except (sf.LibsndfileError, RuntimeError, ValueError) as error:
        raise ValueError(f"LibriSpeech audio contract drift: {path}: {error}") from error
    if decoded_frames != expected_frames:
        raise ValueError(f"LibriSpeech full-decode frame drift: {path}")
    speaker_id = f"librispeech-speaker:{speaker}"
    return {
        "source_id": f"librispeech-{subset}:{path.stem}",
        "path": str(path),
        "sha256": digest,
        "audio_sha256": digest,
        "duration_seconds": expected_frames / TARGET_SAMPLE_RATE,
        "speaker_id": speaker_id,
        "session_id": f"librispeech-chapter:{speaker}:{chapter}",
        "ancestry_id": speaker_id,
        "source_group": "public_speech",
        "semantic_label": "non_wake_public_speech",
        "category": "speech",
        "source": _source_name(subset),
        "provider": PROVIDER,
        "license": LICENSE,
        "split": "test",
        "label": 0,
        "training_eligible": False,
        "locked_holdout": True,
        "locked_deployment_anchor": False,
    }


def _overlap_proof(
    selected: Iterable[Mapping[str, Any]], candidate: Mapping[str, Mapping[str, set[str]]]
) -> dict[str, dict[str, int]]:
    speaker_ids = {str(row["speaker_id"]) for row in selected}
    hashes = {str(row["audio_sha256"]) for row in selected}
    ancestry_ids = {str(row["ancestry_id"]) for row in selected}
    proof: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        proof[split] = {
            "speaker_ids": len(speaker_ids & candidate[split]["speaker"]),
            "audio_sha256": len(hashes & candidate[split]["audio"]),
            "ancestry_ids": len(ancestry_ids & candidate[split]["ancestry"]),
        }
    return proof


def freeze(
    root: Path,
    candidate_corpus: Path,
    output: Path,
    *,
    source_archive: Path | None = None,
    source_archive_md5: str | None = None,
    exclude_locked_manifest: Path | Sequence[Path] | None = None,
    minimum_hours: float = DEFAULT_MINIMUM_HOURS,
    margin_hours: float = DEFAULT_MARGIN_HOURS,
    seed: int = 36012,
    subset: str = DEFAULT_SUBSET,
) -> dict[str, Any]:
    """Validate and freeze a candidate-disjoint, whole-speaker LibriSpeech lock."""
    if not math.isfinite(minimum_hours) or minimum_hours < DEFAULT_MINIMUM_HOURS:
        raise ValueError(f"minimum_hours must be finite and at least {DEFAULT_MINIMUM_HOURS}")
    if not math.isfinite(margin_hours) or margin_hours < 0:
        raise ValueError("margin_hours must be finite and non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not isinstance(subset, str) or re.fullmatch(
        r"train-(?:clean|other)-[0-9]+", subset
    ) is None:
        raise ValueError("subset must name a LibriSpeech train-clean/train-other partition")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    candidate_corpus = candidate_corpus.expanduser().resolve()
    candidate_binding, candidate_payload = _candidate_rows(candidate_corpus)
    candidate = _candidate_identities(candidate_payload)
    if exclude_locked_manifest is None:
        excluded_paths: list[Path] = []
    elif isinstance(exclude_locked_manifest, Path):
        excluded_paths = [exclude_locked_manifest]
    else:
        excluded_paths = list(exclude_locked_manifest)
    excluded_bindings: list[dict[str, Any]] = []
    excluded = {"speaker": set(), "audio": set(), "ancestry": set()}
    for raw_path in excluded_paths:
        binding, identities = _excluded_lock(raw_path.expanduser().resolve())
        excluded_bindings.append(binding)
        for name in excluded:
            excluded[name].update(identities[name])
    if source_archive is None and source_archive_md5 is not None:
        raise ValueError("source_archive_md5 requires source_archive")
    archive_binding = (
        _archive_binding(source_archive, source_archive_md5)
        if source_archive is not None
        else None
    )

    files = sorted(root.rglob("*.flac"))
    if not files:
        raise ValueError(f"LibriSpeech {subset} root contains no FLAC files")
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_source_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for path in files:
        row = _validated_row(root, path, subset)
        if row["source_id"] in seen_source_ids:
            raise ValueError(f"duplicate LibriSpeech source ID: {row['source_id']}")
        if row["audio_sha256"] in seen_hashes:
            raise ValueError(f"duplicate LibriSpeech audio hash: {row['audio_sha256']}")
        seen_source_ids.add(row["source_id"])
        seen_hashes.add(row["audio_sha256"])
        by_speaker[str(row["speaker_id"])].append(row)
    available_by_speaker = {
        speaker: rows
        for speaker, rows in by_speaker.items()
        if _canonical_speaker_identity(speaker) not in excluded["speaker"]
    }
    target_seconds = (minimum_hours + margin_hours) * 3600.0
    selected: list[dict[str, Any]] = []
    exposure_seconds = 0.0
    ordered_speakers = sorted(
        available_by_speaker, key=lambda item: _speaker_order(item, seed, subset)
    )
    for speaker_id in ordered_speakers:
        if exposure_seconds >= target_seconds:
            break
        rows = sorted(
            available_by_speaker[speaker_id], key=lambda row: str(row["source_id"])
        )
        selected.extend(rows)
        exposure_seconds += math.fsum(float(row["duration_seconds"]) for row in rows)
    if exposure_seconds + 1e-9 < target_seconds:
        raise ValueError(
            "LibriSpeech source inventory cannot satisfy the continuous lock target: "
            f"{exposure_seconds / 3600.0:.6f}h < {minimum_hours + margin_hours:.6f}h"
        )

    selected = sorted(selected, key=lambda row: str(row["source_id"]))
    selected_hashes = [str(row["audio_sha256"]) for row in selected]
    selected_speakers = [str(row["speaker_id"]) for row in selected]
    selected_ancestry = [str(row["ancestry_id"]) for row in selected]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise ValueError("continuous lock contains duplicate audio hashes")
    selected_speaker_counts = Counter(selected_speakers)
    if any(
        selected_speaker_counts[speaker_id] != len(available_by_speaker[speaker_id])
        for speaker_id in selected_speaker_counts
    ):
        raise ValueError("continuous lock contains a partial speaker identity")
    if set(selected_speakers) != set(selected_ancestry):
        raise ValueError("continuous lock speaker/ancestry identity contract drift")
    proof = _overlap_proof(selected, candidate)
    if any(value for split in proof.values() for value in split.values()):
        raise ValueError(f"continuous lock overlaps candidate corpus: {proof}")
    selected_identity = {
        "speaker": set(selected_speakers),
        "audio": set(selected_hashes),
        "ancestry": set(selected_ancestry),
    }
    excluded_overlap = {
        name: len(selected_identity[name] & excluded[name])
        for name in ("speaker", "audio", "ancestry")
    }
    if any(excluded_overlap.values()):
        raise ValueError(
            f"continuous lock overlaps excluded continuous lock: {excluded_overlap}"
        )

    exposure_seconds = math.fsum(float(row["duration_seconds"]) for row in selected)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": (
            "kizz_control_librispeech_"
            + subset.replace("-", "_")
            + "_continuous_negative_lock"
        ),
        "gate_scope": LOCK_SCOPE,
        "locked_before_scoring": True,
        "training_eligible": False,
        "source": _source_name(subset),
        "bindings": {
            "candidate_corpus": candidate_binding,
            "source_archive": archive_binding,
            # Keep the singular binding for existing one-lock consumers while
            # recording the complete exclusion chain for third and later locks.
            "excluded_continuous_lock": (
                excluded_bindings[0] if len(excluded_bindings) == 1 else None
            ),
            "excluded_continuous_locks": excluded_bindings,
        },
        "selection_policy": {
            "seed": seed,
            "identity_unit": "whole LibriSpeech speaker",
            "algorithm": (
                "sort speaker identities by sha256('kizz-control-c1-continuous-"
                f"librispeech-{'360' if subset == DEFAULT_SUBSET else subset}-v1:"
                "{seed}:{speaker_id}') bytes then speaker_id; "
                "append every FLAC for each speaker until minimum_hours + margin_hours"
            ),
            "minimum_hours": minimum_hours,
            "margin_hours": margin_hours,
            "target_hours": minimum_hours + margin_hours,
            "audio_contract": "FLAC, exact 16000 Hz, mono, positive full-decode frames, finite samples",
            "excluded_speakers": len(excluded["speaker"]),
            "excluded_lock_count": len(excluded_bindings),
        },
        "overlap_proof": {
            "candidate_corpus_splits": proof,
            "speaker_overlap": 0,
            "audio_sha256_overlap": 0,
            "ancestry_overlap": 0,
            "excluded_continuous_lock": excluded_overlap,
        },
        "counts": {
            "files": len(selected),
            "speakers": len(set(selected_speakers)),
            "exposure_seconds": exposure_seconds,
            "exposure_hours": exposure_seconds / 3600.0,
            "categories": dict(Counter(str(row["category"]) for row in selected)),
            "sources": dict(Counter(str(row["source"]) for row in selected)),
        },
        "examples": selected,
    }
    output = output.expanduser().resolve()
    _atomic_json(output, payload)
    return {"output": str(output), "sha256": sha256_file(output), **payload["counts"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="LibriSpeech subset directory")
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--candidate-corpus", type=Path, required=True, help="existing corpus.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, help="optional SLR12 archive to bind")
    parser.add_argument(
        "--exclude-locked-manifest",
        type=Path,
        action="append",
        help=(
            "prior continuous lock whose speakers/audio/ancestry must be excluded; "
            "repeat to exclude multiple consumed locks"
        ),
    )
    parser.add_argument(
        "--source-archive-md5",
        help="official 32-hex OpenSLR archive MD5 to verify and bind",
    )
    parser.add_argument("--minimum-hours", type=float, default=DEFAULT_MINIMUM_HOURS)
    parser.add_argument("--margin-hours", type=float, default=DEFAULT_MARGIN_HOURS)
    parser.add_argument("--seed", type=int, default=36012)
    args = parser.parse_args(argv)
    print(json.dumps(freeze(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
