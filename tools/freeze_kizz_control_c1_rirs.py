#!/usr/bin/env python3
"""Freeze a deterministic, train-only OpenSLR SLR28 RIR subset for C1."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Sequence
import wave


OPENSLR28_ARCHIVE_MD5 = "e6f48e257286e05de56413b4779d8ffb"
OPENSLR28_ARCHIVE_SHA256 = (
    "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"
)
STRATA = ("real", "smallroom", "mediumroom", "largeroom")
SIMULATED_STRATA = STRATA[1:]
ROOM_PATTERN = re.compile(r"Room[0-9]+\Z")


def file_hashes(path: Path, *algorithms: str) -> dict[str, str]:
    digests = {name: hashlib.new(name) for name in algorithms}
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(block)
    return {name: digest.hexdigest() for name, digest in digests.items()}


def _is_below(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _contains_internal_symlink(path: Path, root: Path) -> bool:
    """Check only components below root; system ancestors may themselves link."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return True
    return False


def _has_noise_token(stem: str) -> bool:
    return re.search(r"(?:^|[_-])noise(?:[_-]|$)", stem, re.IGNORECASE) is not None


def _real_source_identity(stem: str) -> str:
    """Collapse known varying microphone positions into a stable source identity."""
    value = re.sub(r"_imp[0-9]+\Z", "", stem, flags=re.IGNORECASE)
    value = re.sub(r"_(?:near|far)_angl[ab]\Z", "", value, flags=re.IGNORECASE)
    if value.lower().startswith("air_"):
        value = re.sub(r"(?:_[0-9]+)+\Z", "", value)
    return value


def _classify_path(path: Path, root: Path) -> dict[str, str] | None:
    """Return structural metadata, or ``None`` for documented real noise files."""
    if path.suffix.lower() != ".wav":
        raise ValueError(f"RIR discovery contains a non-WAV path: {path}")
    relative = path.relative_to(root)
    parts = relative.parts
    stem = path.stem

    if len(parts) == 2 and parts[0] == "real_rirs_isotropic_noises":
        if _has_noise_token(stem):
            return None
        lower = stem.lower()
        if "_rir_" not in lower and not lower.startswith("air_"):
            raise ValueError(f"unrecognized real impulse-response filename: {path}")
        source_identity = _real_source_identity(stem)
        return {
            "stratum": "real",
            "room_id": source_identity,
            "source_identity": source_identity,
            "source_group_id": f"openslr28:real:{source_identity}",
        }

    if (
        len(parts) == 4
        and parts[0] == "simulated_rirs"
        and parts[1] in SIMULATED_STRATA
        and ROOM_PATTERN.fullmatch(parts[2])
    ):
        room_id = parts[2]
        if _has_noise_token(stem):
            raise ValueError(f"noise file found in simulated RIR tree: {path}")
        if not stem.startswith(f"{room_id}-"):
            raise ValueError(f"simulated RIR filename does not match its room: {path}")
        return {
            "stratum": parts[1],
            "room_id": room_id,
            "source_identity": room_id,
            "source_group_id": f"openslr28:simulated:{parts[1]}:{room_id}",
        }

    raise ValueError(f"WAV is outside the recognized SLR28 RIR layout: {path}")


def _audio_metadata(path: Path) -> dict[str, int | float]:
    try:
        with wave.open(str(path), "rb") as audio:
            channels = audio.getnchannels()
            sample_rate = audio.getframerate()
            frames = audio.getnframes()
            sample_width = audio.getsampwidth()
            compression = audio.getcomptype()
            payload = audio.readframes(frames)
    except wave.Error as error:
        # Some official SLR28 real-room files are multichannel PCM16 WAVEX
        # (format 65534), which Python's stdlib wave reader cannot decode.
        # SoundFile/libsndfile handles that standard container without a
        # conversion that would break the archive-bound audio hash.
        if "unknown format: 65534" not in str(error):
            raise ValueError(f"malformed RIR audio: {path}: {error}") from error
        try:
            import soundfile as sf

            info = sf.info(str(path))
            if info.format != "WAVEX" or info.subtype != "PCM_16":
                raise ValueError("unsupported WAVE_FORMAT_EXTENSIBLE subtype")
            decoded, sample_rate = sf.read(str(path), dtype="int16", always_2d=True)
        except Exception as fallback_error:
            raise ValueError(
                f"malformed RIR audio: {path}: {fallback_error}"
            ) from fallback_error
        channels = int(info.channels)
        frames = int(info.frames)
        sample_width = 2
        compression = "NONE"
        payload = decoded.tobytes()
    except (EOFError, OSError) as error:
        raise ValueError(f"malformed RIR audio: {path}: {error}") from error
    if (
        channels < 1
        or sample_rate < 1
        or frames < 1
        or sample_width < 1
        or compression != "NONE"
    ):
        raise ValueError(f"invalid RIR audio metadata: {path}")
    expected_bytes = frames * channels * sample_width
    if len(payload) != expected_bytes:
        raise ValueError(f"truncated RIR audio: {path}")
    if not any(payload):
        raise ValueError(f"empty/silent RIR audio: {path}")
    return {
        "duration_seconds": frames / sample_rate,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "frames": frames,
        "sample_width_bytes": sample_width,
    }


def inventory_rirs(
    rir_root: Path,
    *,
    discovered_paths: Iterable[Path] | None = None,
) -> list[dict[str, Any]]:
    root_input = Path(rir_root)
    if not root_input.is_dir() or root_input.is_symlink():
        raise ValueError(f"RIR root must be a real directory: {rir_root}")
    input_root = root_input.absolute()
    root = root_input.resolve()
    required = [root / "real_rirs_isotropic_noises"] + [
        root / "simulated_rirs" / stratum for stratum in SIMULATED_STRATA
    ]
    if any(not path.is_dir() or path.is_symlink() for path in required):
        raise ValueError("RIR root lacks the exact real/simulated SLR28 layout")

    raw_paths = (
        list(discovered_paths)
        if discovered_paths is not None
        else [
            path
            for directory in required
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        ]
    )
    normalized: list[Path] = []
    seen_paths: set[Path] = set()
    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        if not candidate.is_file():
            raise ValueError(f"RIR discovery path is not a file: {candidate}")
        absolute = candidate.absolute()
        resolved = candidate.resolve()
        path_root = (
            input_root
            if _is_below(absolute, input_root)
            else root
            if _is_below(absolute, root)
            else None
        )
        if (
            not _is_below(resolved, root)
            or path_root is None
            or _contains_internal_symlink(absolute, path_root)
        ):
            raise ValueError(f"unsafe RIR path: {candidate}")
        if resolved in seen_paths:
            raise ValueError(f"duplicate RIR path: {resolved}")
        seen_paths.add(resolved)
        normalized.append(resolved)

    rows: list[dict[str, Any]] = []
    seen_hashes: dict[str, Path] = {}
    for path in sorted(normalized, key=lambda item: item.relative_to(root).as_posix()):
        classification = _classify_path(path, root)
        if classification is None:
            continue
        metadata = _audio_metadata(path)
        audio_hash = file_hashes(path, "sha256")["sha256"]
        duplicate = seen_hashes.get(audio_hash)
        if duplicate is not None:
            raise ValueError(f"duplicate RIR audio hash: {duplicate} and {path}")
        seen_hashes[audio_hash] = path
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": str(path),
                "relative_path": relative,
                "sha256": audio_hash,
                "audio_sha256": audio_hash,
                **metadata,
                **classification,
                "source_id": f"openslr28:{relative}",
            }
        )
    counts = Counter(str(row["stratum"]) for row in rows)
    missing = [stratum for stratum in STRATA if counts[stratum] == 0]
    if missing:
        raise ValueError(f"RIR inventory lacks strata: {', '.join(missing)}")
    return rows


def _stable_rank(seed: int, *parts: str) -> bytes:
    value = "\0".join((str(seed), *parts)).encode("utf-8")
    return hashlib.sha256(value).digest()


def select_balanced_rirs(
    rows: Sequence[dict[str, Any]], *, per_stratum: int, seed: int
) -> list[dict[str, Any]]:
    if (
        isinstance(per_stratum, bool)
        or not isinstance(per_stratum, int)
        or per_stratum < 1
    ):
        raise ValueError("per-stratum count must be a positive integer")
    selected: list[dict[str, Any]] = []
    for stratum in STRATA:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get("stratum") == stratum:
                groups[str(row["source_group_id"])].append(dict(row))
        available = sum(len(values) for values in groups.values())
        if available < per_stratum:
            raise ValueError(
                f"RIR stratum {stratum} has {available} responses; "
                f"requires {per_stratum}"
            )
        group_order = sorted(
            groups,
            key=lambda group_id: (_stable_rank(seed, stratum, group_id), group_id),
        )
        ordered_groups = {
            group_id: sorted(
                groups[group_id],
                key=lambda row: (
                    _stable_rank(seed, stratum, group_id, str(row["source_id"])),
                    str(row["source_id"]),
                ),
            )
            for group_id in group_order
        }
        depth = 0
        stratum_rows: list[dict[str, Any]] = []
        while len(stratum_rows) < per_stratum:
            added = False
            for group_id in group_order:
                values = ordered_groups[group_id]
                if depth < len(values):
                    stratum_rows.append(values[depth])
                    added = True
                    if len(stratum_rows) == per_stratum:
                        break
            if not added:
                raise ValueError(f"could not complete balanced selection for {stratum}")
            depth += 1
        if stratum in SIMULATED_STRATA:
            expected_rooms = min(per_stratum, len(groups))
            actual_rooms = len({row["source_group_id"] for row in stratum_rows})
            if actual_rooms != expected_rooms:
                raise ValueError(f"simulated {stratum} selection is not room-balanced")
        selected.extend(stratum_rows)
    order = {stratum: index for index, stratum in enumerate(STRATA)}
    return sorted(
        selected,
        key=lambda row: (
            order[str(row["stratum"])],
            str(row["source_group_id"]),
            str(row["source_id"]),
        ),
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_outputs(
    rir_root: Path, archive: Path, output: Path, report: Path
) -> None:
    resolved = [Path(value).resolve() for value in (archive, output, report)]
    if len(set(resolved)) != len(resolved):
        raise ValueError("archive, output, and report-output must be distinct paths")
    root = Path(rir_root).resolve()
    for destination in resolved[1:]:
        if destination.exists() and destination.is_dir():
            raise ValueError(f"output path is a directory: {destination}")
        if _is_below(destination, root):
            raise ValueError(
                f"output may not modify the extracted RIR tree: {destination}"
            )


def freeze(
    rir_root: Path,
    archive: Path,
    output: Path,
    report_output: Path,
    *,
    per_stratum: int,
    seed: int,
    expected_archive_md5: str = OPENSLR28_ARCHIVE_MD5,
    expected_archive_sha256: str = OPENSLR28_ARCHIVE_SHA256,
    discovered_paths: Iterable[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    archive = Path(archive)
    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"SLR28 archive must be a real file: {archive}")
    if archive.suffix.lower() != ".zip":
        raise ValueError(f"SLR28 archive must be a ZIP file: {archive}")
    if not re.fullmatch(r"[0-9a-f]{32}", expected_archive_md5):
        raise ValueError("expected archive MD5 must be lowercase hexadecimal")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_archive_sha256):
        raise ValueError("expected archive SHA-256 must be lowercase hexadecimal")
    _validate_outputs(rir_root, archive, output, report_output)
    archive_hashes = file_hashes(archive, "md5", "sha256")
    if archive_hashes["md5"] != expected_archive_md5:
        raise ValueError("SLR28 archive MD5 mismatch")
    if archive_hashes["sha256"] != expected_archive_sha256:
        raise ValueError("SLR28 archive SHA-256 mismatch")

    inventory = inventory_rirs(rir_root, discovered_paths=discovered_paths)
    selected = select_balanced_rirs(inventory, per_stratum=per_stratum, seed=seed)
    paths = [str(row["path"]) for row in selected]
    hashes = [str(row["sha256"]) for row in selected]
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise ValueError("selected RIR manifest contains duplicate paths or hashes")

    archive_path = str(archive.resolve())
    examples = []
    for row in selected:
        item = dict(row)
        audio_path = Path(item["path"])
        if _has_noise_token(audio_path.stem):
            raise ValueError(f"noise reached selected RIR manifest: {item['path']}")
        if file_hashes(audio_path, "sha256")["sha256"] != item["sha256"]:
            raise ValueError(f"selected RIR changed during freeze: {audio_path}")
        live_metadata = _audio_metadata(audio_path)
        if any(live_metadata[key] != item[key] for key in live_metadata):
            raise ValueError(
                f"selected RIR metadata changed during freeze: {audio_path}"
            )
        item.update(
            {
                "split": "train",
                "training_eligible": True,
                "semantic_label": "room_impulse_response",
                "source": "OpenSLR SLR28",
                "archive_path": archive_path,
                "archive_md5": archive_hashes["md5"],
                "archive_sha256": archive_hashes["sha256"],
                "provenance_id": (
                    f"archive-sha256:{archive_hashes['sha256']}#{item['relative_path']}"
                ),
            }
        )
        examples.append(item)

    by_stratum = Counter(str(row["stratum"]) for row in examples)
    expected_counts = {stratum: per_stratum for stratum in STRATA}
    if dict(by_stratum) != expected_counts:
        raise ValueError("selected RIR manifest is not exactly balanced")
    unique_groups = {
        stratum: len(
            {
                str(row["source_group_id"])
                for row in examples
                if row["stratum"] == stratum
            }
        )
        for stratum in STRATA
    }
    provenance = {
        "dataset": "OpenSLR SLR28 RIRS_NOISES",
        "url": "https://www.openslr.org/28/",
        "archive_path": archive_path,
        "archive_md5": archive_hashes["md5"],
        "archive_sha256": archive_hashes["sha256"],
    }
    manifest = {
        "schema_version": 1,
        "kind": "kizz_control_c1_train_rir_manifest",
        "frozen": True,
        "split": "train",
        "training_only": True,
        "training_eligible": True,
        "selection": {
            "algorithm": "seeded-source-group-round-robin-v1",
            "seed": seed,
            "per_stratum": per_stratum,
            "strata": list(STRATA),
        },
        "inputs": {
            "rir_root": str(Path(rir_root).resolve()),
            "archive": provenance,
        },
        "counts": {
            "inventory": len(inventory),
            "selected": len(examples),
            "by_stratum": expected_counts,
            "unique_source_groups_by_stratum": unique_groups,
        },
        "examples": examples,
    }
    manifest_bytes = _json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    report = {
        "schema_version": 1,
        "kind": "kizz_control_c1_train_rir_freeze_report",
        "qualified": True,
        "manifest_path": str(Path(output).resolve()),
        "manifest_sha256": manifest_sha256,
        "archive": provenance,
        "selection": manifest["selection"],
        "validation": {
            "archive_hashes_match": True,
            "all_audio_live_hashes": True,
            "all_audio_nonempty": True,
            "all_examples_train_only": True,
            "noise_examples": 0,
            "duplicate_paths": 0,
            "duplicate_hashes": 0,
            "selected_by_stratum": expected_counts,
            "unique_source_groups_by_stratum": unique_groups,
        },
    }
    _atomic_write(Path(output).resolve(), manifest_bytes)
    _atomic_write(Path(report_output).resolve(), _json_bytes(report))
    return manifest, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, required=True)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args(argv)
    manifest, report = freeze(
        args.rir_root,
        args.archive,
        args.output,
        args.report_output,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "selected": manifest["counts"]["selected"],
                "by_stratum": manifest["counts"]["by_stratum"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
