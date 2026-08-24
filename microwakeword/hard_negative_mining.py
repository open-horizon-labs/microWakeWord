"""Deterministic, feature-native hard-negative mining for wake-word models.

Source RaggedMmaps are opened read-only and processed one item at a time. Scan
artifacts are shard-scoped; ``merge_shards`` produces the single corpus consumed
by training while reapplying global quotas.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from mmap_ninja.ragged import RaggedMmap

from microwakeword.inference import Model


EXCLUDED_COMPONENTS = {
    "dinner_party_eval",
    "evaluation",
    "testing_ambient",
    "validation_ambient",
    "observations",
    "false-wakes",
}
SPLITS = ("training", "validation", "testing")
SCORE_BANDS = (0.0, 0.5, 0.7, 0.8, 0.9, 1.01)
CONTEXT_FRAMES = 200
FEATURE_BINS = 40
MINER_ALGORITHM_VERSION = 2


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_layout(path: Path) -> str:
    """Hash RaggedMmap coordinates and shapes separately from feature bytes."""
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.name != "data.ninja"
    )
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def config_hash(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _excluded(path: Path) -> bool:
    components = {part.lower() for part in path.parts}
    return bool(components & EXCLUDED_COMPONENTS) or any(
        "evaluation" in part.lower() for part in path.parts
    )


def discover_archives(roots: Sequence[Path]) -> list[dict]:
    """Discover eligible nested RaggedMmaps and preserve their source splits."""
    discovered: list[dict] = []
    for root in sorted(Path(path).resolve() for path in roots):
        if _excluded(root):
            continue
        candidates = (
            [root] if root.name.endswith("_mmap") else sorted(root.rglob("*_mmap"))
        )
        for archive in candidates:
            if not archive.is_dir() or _excluded(archive):
                continue
            split = next((part for part in archive.parts[::-1] if part in SPLITS), None)
            if split is None:
                raise ValueError(f"eligible archive has no training split: {archive}")
            required = ("data.ninja", "starts", "ends", "shapes")
            if not all((archive / name).exists() for name in required):
                raise ValueError(f"corrupt RaggedMmap metadata: {archive}")
            discovered.append(
                {
                    "path": str(archive),
                    "split": split,
                    "archive": archive.name,
                    "source_hash": sha256_file(archive / "data.ninja"),
                    "source_layout_hash": sha256_layout(archive),
                }
            )
    return sorted(discovered, key=lambda item: (item["split"], item["path"]))


def score_band(score: float, bands: Sequence[float] = SCORE_BANDS) -> str:
    values = tuple(float(value) for value in bands)
    if (
        len(values) < 2
        or values[0] != 0.0
        or values[-1] < 1.0
        or any(left >= right for left, right in zip(values, values[1:]))
    ):
        raise ValueError("score bands must increase from 0 through at least 1")
    for low, high in zip(values, values[1:]):
        if low <= score < high:
            return f"{low:g}-{high:g}"
    return f"{values[-2]:g}-{values[-1]:g}"


def effective_score_band_quota(
    per_source_quota: int,
    cutoff: float,
    requested: int | None,
    bands: Sequence[float] = SCORE_BANDS,
) -> int:
    """Return a per-band quota whose aggregate can fill the source quota."""
    if requested is not None:
        if requested <= 0:
            raise ValueError("score_band_quota must be positive")
        return requested
    active = sum(high > cutoff for _low, high in zip(bands, bands[1:]))
    if not active:
        raise ValueError("cutoff leaves no active score bands")
    return math.ceil(per_source_quota / active)


def local_maxima(scores: Sequence[float], cutoff: float) -> list[int]:
    result = []
    for index, score in enumerate(scores):
        if score < cutoff:
            continue
        left = scores[index - 1] if index else float("-inf")
        right = scores[index + 1] if index + 1 < len(scores) else float("-inf")
        if score >= left and score >= right:
            result.append(index)
    return result


def prediction_coordinates(
    model: object,
    prediction_index: int,
    stream_frames: int,
    context_frames: int,
) -> tuple[int, int]:
    """Map a Model prediction to the source frames that produced it."""
    input_frames = int(getattr(model, "input_feature_slices"))
    effective_stride = int(getattr(model, "stride"))
    end = min(int(stream_frames), input_frames + prediction_index * effective_stride)
    return max(0, end - context_frames), end


def temporal_nms(candidates: Sequence[dict], min_distance_frames: int) -> list[dict]:
    kept: list[dict] = []
    ordered = sorted(
        candidates,
        key=lambda item: (-item["score"], item["end_frame"], item["model"]),
    )
    for candidate in ordered:
        if all(
            abs(candidate["end_frame"] - prior["end_frame"]) >= min_distance_frames
            for prior in kept
        ):
            kept.append(candidate)
    return sorted(kept, key=lambda item: item["end_frame"])


def shard_artifact_root(output: Path, shard_index: int, shard_count: int) -> Path:
    return Path(output) / "shards" / f"{shard_index:05d}-of-{shard_count:05d}"


def _stable_random_key(seed: int, record: dict) -> str:
    material = (
        f"{seed}:{record['source_hash']}:{record['item_index']}:"
        f"{record['start_frame']}:{record['end_frame']}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def deterministic_reserve_starts(
    seed: int,
    source_hash: str,
    item_index: int,
    available_starts: int,
    interval_frames: int,
) -> list[int]:
    """Choose one seeded start within each interval of an item's timeline."""
    if available_starts <= 0 or interval_frames <= 0:
        raise ValueError("reserve window dimensions must be positive")
    starts = []
    for bucket_start in range(0, available_starts, interval_frames):
        bucket_end = min(available_starts, bucket_start + interval_frames)
        material = (
            f"{seed}:{source_hash}:{item_index}:{bucket_start}:{bucket_end}"
        ).encode()
        offset = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
        starts.append(bucket_start + offset % (bucket_end - bucket_start))
    return starts


def _record_key(record: dict) -> tuple:
    return (
        record["source_path"],
        int(record["item_index"]),
        int(record["start_frame"]),
        int(record["end_frame"]),
    )


def _window(
    spectrogram: np.ndarray, start: int, end: int, context_frames: int
) -> np.ndarray:
    data = np.asarray(spectrogram[start:end])
    if data.ndim != 2 or data.shape[1] != FEATURE_BINS:
        raise ValueError(f"unexpected selected window shape: {data.shape}")
    if len(data) > context_frames:
        data = data[-context_frames:]
    if len(data) < context_frames:
        padded = np.zeros((context_frames, FEATURE_BINS), dtype=data.dtype)
        if len(data):
            padded[-len(data) :] = data
        data = padded
    return np.asarray(data)


@dataclass(frozen=True)
class Candidate:
    source_path: str
    source_hash: str
    source_layout_hash: str
    archive: str
    item_index: int
    start_frame: int
    end_frame: int
    split: str
    scores: dict[str, float]
    model_shas: dict[str, str]
    score: float
    reason: str
    seed: int
    shard: str
    config_hash: str
    shard_config_hash: str

    def record(self) -> dict:
        return asdict(self)


def _model_path_items(
    model_paths: Sequence[str | tuple[str, Path]],
) -> list[tuple[str, Path]]:
    result = []
    names = set()
    for item in model_paths:
        if isinstance(item, tuple):
            name, path = item
        else:
            if "=" not in item:
                raise ValueError(f"model must be NAME=PATH: {item}")
            name, path = item.split("=", 1)
        if not name or name in names or not Path(path).is_file():
            raise ValueError(f"model must have a unique NAME and existing PATH: {item}")
        names.add(name)
        result.append((name, Path(path).resolve()))
    if not result:
        raise ValueError("at least one model is required")
    return result


def _checkpoint_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as target:
            json.dump(payload, target, sort_keys=True, indent=2)
            target.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _load_checkpoint(path: Path, expected_hash: str) -> dict:
    if not path.exists():
        return {
            "config_hash": expected_hash,
            "completed": [],
            "records": [],
            "reserve": [],
        }
    payload = json.loads(path.read_text())
    if payload.get("config_hash") != expected_hash:
        raise ValueError("checkpoint configuration does not match this shard")
    return payload


def _checkpoint_payload(
    run_hash: str,
    completed: set[tuple[str, int]],
    selected: list[dict],
    reserve: list[dict],
) -> dict:
    return {
        "config_hash": run_hash,
        "completed": [list(key) for key in sorted(completed)],
        "records": selected,
        "reserve": reserve,
    }


def _validate_spectrogram(
    archive: dict, item_index: int, item: np.ndarray
) -> np.ndarray:
    spectrogram = np.asarray(item)
    if spectrogram.ndim != 2 or spectrogram.shape[1] != FEATURE_BINS:
        raise ValueError(
            f"unexpected spectrogram shape in {archive['path']}[{item_index}]: "
            f"{spectrogram.shape}"
        )
    return spectrogram


def _scores_at_end(
    score_points: dict[str, list[dict]], end_frame: int
) -> dict[str, float]:
    return {
        name: (
            min(points, key=lambda point: abs(point["end_frame"] - end_frame))["score"]
            if points
            else 0.0
        )
        for name, points in score_points.items()
    }


def _restore_quota_counts(selected: Sequence[dict]) -> tuple[dict, dict, dict]:
    sources: dict[str, int] = {}
    items: dict[tuple[str, int], int] = {}
    bands: dict[tuple[str, str], int] = {}
    for record in selected:
        source_hash = record["source_hash"]
        sources[source_hash] = sources.get(source_hash, 0) + 1
        item_key = (source_hash, int(record["item_index"]))
        items[item_key] = items.get(item_key, 0) + 1
        band = record["reason"].split(":", 1)[1]
        bands[(source_hash, band)] = bands.get((source_hash, band), 0) + 1
    return sources, items, bands


def _select_with_global_quotas(
    candidates: Sequence[dict],
    per_source_quota: int,
    per_item_quota: int,
    band_quota: int,
) -> list[dict]:
    selected: list[dict] = []
    sources: dict[str, int] = {}
    items: dict[tuple[str, int], int] = {}
    bands: dict[tuple[str, str], int] = {}
    for record in sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            item["source_path"],
            int(item["item_index"]),
            int(item["end_frame"]),
        ),
    ):
        source = record["source_hash"]
        item_key = (source, int(record["item_index"]))
        band = record["reason"].split(":", 1)[1]
        if (
            sources.get(source, 0) >= per_source_quota
            or items.get(item_key, 0) >= per_item_quota
            or bands.get((source, band), 0) >= band_quota
        ):
            continue
        selected.append(record)
        sources[source] = sources.get(source, 0) + 1
        items[item_key] = items.get(item_key, 0) + 1
        bands[(source, band)] = bands.get((source, band), 0) + 1
    return selected


def _write_records(
    records: Sequence[dict],
    output: Path,
    role: str,
    context_frames: int,
    replacement_token: str,
) -> int:
    count = 0
    for split in SPLITS:
        split_records = [record for record in records if record["split"] == split]
        if not split_records:
            continue
        destination = Path(output) / role / split / "wakeword_mmap"
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_destination = destination
        backup = None
        if destination.exists():
            suffix = 0
            backup = destination.with_name(
                f"{destination.name}.previous-{replacement_token[:8]}"
            )
            while backup.exists():
                suffix += 1
                backup = destination.with_name(
                    f"{destination.name}.previous-{replacement_token[:8]}-{suffix}"
                )
            write_destination = destination.with_name(
                f"{destination.name}.rewrite-{replacement_token[:8]}"
            )
            while write_destination.exists():
                suffix += 1
                write_destination = destination.with_name(
                    f"{destination.name}.rewrite-{replacement_token[:8]}-{suffix}"
                )

        def generator(items=split_records):
            loaded_path = None
            loaded_mmap = None
            for record in items:
                if record["source_path"] != loaded_path:
                    loaded_path = record["source_path"]
                    loaded_mmap = RaggedMmap(loaded_path)
                source = np.asarray(loaded_mmap[int(record["item_index"])])
                yield _window(
                    source,
                    int(record["start_frame"]),
                    int(record["end_frame"]),
                    context_frames,
                )

        RaggedMmap.from_generator(
            str(write_destination), generator(), batch_size=100, verbose=False
        )
        if backup is not None:
            os.replace(destination, backup)
            os.replace(write_destination, destination)
        count += len(split_records)
    return count


def _write_manifest(
    path: Path, selected: Sequence[dict], reserve: Sequence[dict]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as target:
        for record in selected:
            target.write(json.dumps(record, sort_keys=True) + "\n")
        for record in reserve:
            target.write(json.dumps(record, sort_keys=True) + "\n")


def mine(
    roots: Sequence[Path],
    model_paths: Sequence[str | tuple[str, Path]],
    output: Path,
    *,
    cutoff: float = 0.5,
    context_frames: int = CONTEXT_FRAMES,
    stride: int | None = None,
    nms_frames: int = 220,
    per_source_quota: int = 128,
    per_item_quota: int = 4,
    score_band_quota: int | None = None,
    reserve_fraction: float = 0.15,
    seed: int = 231,
    shard_index: int = 0,
    shard_count: int = 1,
    max_items: int | None = None,
    checkpoint: Path | None = None,
    checkpoint_interval: int = 1000,
    required_model_shas: dict[str, str] | None = None,
    model_factory: Callable[[Path, int | None], object] = Model,
) -> dict:
    """Mine one deterministic shard into a shard-scoped artifact directory."""
    if not 0.1 <= reserve_fraction <= 0.2:
        raise ValueError("reserve_fraction must be between 0.1 and 0.2")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard selection")
    if context_frames <= 0 or per_source_quota <= 0 or per_item_quota <= 0:
        raise ValueError("context and quotas must be positive")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    if max_items is not None and max_items < 0:
        raise ValueError("max_items cannot be negative")

    models = _model_path_items(model_paths)
    archives = discover_archives(roots)
    band_quota = effective_score_band_quota(per_source_quota, cutoff, score_band_quota)
    model_shas = {name: sha256_file(path) for name, path in models}
    required_model_shas = required_model_shas or {}
    unknown_required = sorted(set(required_model_shas) - set(model_shas))
    if unknown_required:
        raise ValueError(
            f"required SHA names are not configured models: {unknown_required}"
        )
    for name, expected in required_model_shas.items():
        if len(expected) != 64 or any(
            character not in "0123456789abcdef" for character in expected
        ):
            raise ValueError(f"required SHA for {name} is not lowercase SHA-256")
        if model_shas[name] != expected:
            raise ValueError(
                f"model {name} SHA-256 mismatch: expected {expected}, "
                f"got {model_shas[name]}"
            )
    corpus_config = {
        "algorithm_version": MINER_ALGORITHM_VERSION,
        "archives": archives,
        "models": [(name, str(path), model_shas[name]) for name, path in models],
        "required_model_shas": required_model_shas,
        "cutoff": cutoff,
        "context_frames": context_frames,
        "stride": stride,
        "nms_frames": nms_frames,
        "per_source_quota": per_source_quota,
        "per_item_quota": per_item_quota,
        "score_band_quota": band_quota,
        "reserve_fraction": reserve_fraction,
        "seed": seed,
    }
    corpus_hash = config_hash(corpus_config)
    shard_config = {
        "corpus_hash": corpus_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    run_hash = config_hash(shard_config)
    artifact_root = shard_artifact_root(output, shard_index, shard_count)
    checkpoint = checkpoint or (artifact_root / "mining-checkpoint.json")
    state = _load_checkpoint(checkpoint, run_hash)
    completed = {
        (str(path), int(item_index)) for path, item_index in state.get("completed", [])
    }
    selected = list(state.get("records", []))
    reserve_pool = list(state.get("reserve", []))
    sources, items, bands = _restore_quota_counts(selected)
    models_by_name = [(name, model_factory(path, stride)) for name, path in models]
    reserve_capacity = max(
        1,
        math.ceil(
            per_source_quota
            * max(1, len(archives))
            * reserve_fraction
            / (1 - reserve_fraction)
        ),
    )

    global_index = 0
    scored_this_run = 0
    stopped_at_limit = False
    for archive in archives:
        mmap = RaggedMmap(archive["path"])
        for item_index in range(len(mmap)):
            assigned = global_index % shard_count == shard_index
            global_index += 1
            if not assigned:
                continue
            identity = (archive["path"], item_index)
            if identity in completed:
                continue
            if max_items is not None and scored_this_run >= max_items:
                stopped_at_limit = True
                break

            spectrogram = _validate_spectrogram(archive, item_index, mmap[item_index])
            all_candidates: list[dict] = []
            score_points: dict[str, list[dict]] = {}
            for name, model in models_by_name:
                model.reset_states()
                scores = [
                    float(value) for value in model.predict_spectrogram(spectrogram)
                ]
                points = []
                for prediction_index, value in enumerate(scores):
                    start_frame, end_frame = prediction_coordinates(
                        model,
                        prediction_index,
                        len(spectrogram),
                        context_frames,
                    )
                    points.append(
                        {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                            "score": value,
                        }
                    )
                score_points[name] = points
                for prediction_index in local_maxima(scores, cutoff):
                    all_candidates.append(
                        {
                            **points[prediction_index],
                            "model": name,
                        }
                    )

            merged = []
            for candidate in temporal_nms(all_candidates, nms_frames):
                nearby = [
                    item
                    for item in all_candidates
                    if abs(item["end_frame"] - candidate["end_frame"]) < nms_frames
                ]
                merged.append(
                    {
                        **candidate,
                        "scores": {
                            name: max(
                                (
                                    item["score"]
                                    for item in nearby
                                    if item["model"] == name
                                ),
                                default=0.0,
                            )
                            for name, _model in models_by_name
                        },
                    }
                )

            for candidate in merged:
                band = score_band(candidate["score"])
                source = archive["source_hash"]
                item_key = (source, item_index)
                if (
                    sources.get(source, 0) >= per_source_quota
                    or items.get(item_key, 0) >= per_item_quota
                    or bands.get((source, band), 0) >= band_quota
                ):
                    continue
                selected.append(
                    Candidate(
                        archive["path"],
                        source,
                        archive["source_layout_hash"],
                        archive["archive"],
                        item_index,
                        int(candidate["start_frame"]),
                        int(candidate["end_frame"]),
                        archive["split"],
                        candidate["scores"],
                        model_shas,
                        float(candidate["score"]),
                        f"high_score:{band}",
                        seed,
                        f"{shard_index}/{shard_count}",
                        corpus_hash,
                        run_hash,
                    ).record()
                )
                sources[source] = sources.get(source, 0) + 1
                items[item_key] = items.get(item_key, 0) + 1
                bands[(source, band)] = bands.get((source, band), 0) + 1

            random_starts = deterministic_reserve_starts(
                seed,
                archive["source_hash"],
                item_index,
                max(1, len(spectrogram) - context_frames + 1),
                nms_frames,
            )
            selected_keys = {_record_key(record) for record in selected}
            for start_frame in random_starts:
                end_frame = min(len(spectrogram), start_frame + context_frames)
                scores = _scores_at_end(score_points, end_frame)
                record = Candidate(
                    archive["path"],
                    archive["source_hash"],
                    archive["source_layout_hash"],
                    archive["archive"],
                    item_index,
                    int(start_frame),
                    int(end_frame),
                    archive["split"],
                    scores,
                    model_shas,
                    max(scores.values(), default=0.0),
                    "random_reserve",
                    seed,
                    f"{shard_index}/{shard_count}",
                    corpus_hash,
                    run_hash,
                ).record()
                if _record_key(record) not in selected_keys:
                    reserve_pool.append(record)
            reserve_pool = sorted(
                reserve_pool, key=lambda record: _stable_random_key(seed, record)
            )[:reserve_capacity]
            completed.add(identity)
            scored_this_run += 1
            if scored_this_run % checkpoint_interval == 0:
                _checkpoint_write(
                    checkpoint,
                    _checkpoint_payload(run_hash, completed, selected, reserve_pool),
                )
        if stopped_at_limit:
            break

    _checkpoint_write(
        checkpoint,
        _checkpoint_payload(run_hash, completed, selected, reserve_pool),
    )
    reserve_target = math.ceil(
        len(selected) * reserve_fraction / (1 - reserve_fraction)
    )
    reserve_output = sorted(
        reserve_pool, key=lambda record: _stable_random_key(seed, record)
    )[:reserve_target]
    selected_count = _write_records(
        selected, artifact_root, "mined", context_frames, run_hash
    )
    reserve_count = _write_records(
        reserve_output, artifact_root, "random_reserve", context_frames, run_hash
    )
    manifest = artifact_root / "mining-manifest.jsonl"
    _write_manifest(manifest, selected, reserve_output)
    metadata = {
        "schema_version": 1,
        "config_hash": corpus_hash,
        "shard_config_hash": run_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "scan_complete": not stopped_at_limit,
        "completed_items": len(completed),
        "scored_this_run": scored_this_run,
        "selected": selected_count,
        "random_reserve": reserve_count,
        "checkpoint_interval": checkpoint_interval,
        "config": corpus_config,
    }
    _checkpoint_write(artifact_root / "mining-metadata.json", metadata)
    return {**metadata, "artifact_root": str(artifact_root), "manifest": str(manifest)}


def merge_shards(
    output: Path,
    shard_count: int,
    *,
    allow_incomplete: bool = False,
) -> dict:
    """Merge shard manifests into one globally quota-controlled corpus."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    metadata = []
    records = []
    for shard_index in range(shard_count):
        root = shard_artifact_root(output, shard_index, shard_count)
        metadata_path = root / "mining-metadata.json"
        manifest_path = root / "mining-manifest.jsonl"
        if not metadata_path.exists() or not manifest_path.exists():
            raise ValueError(f"missing shard artifact: {root}")
        item = json.loads(metadata_path.read_text())
        if (
            item.get("shard_index") != shard_index
            or item.get("shard_count") != shard_count
        ):
            raise ValueError(f"shard metadata identity mismatch: {root}")
        if not allow_incomplete and not item.get("scan_complete"):
            raise ValueError(f"shard scan is incomplete: {root}")
        metadata.append(item)
        records.extend(
            json.loads(line)
            for line in manifest_path.read_text().splitlines()
            if line.strip()
        )
    hashes = {item["config_hash"] for item in metadata}
    if len(hashes) != 1:
        raise ValueError("shards do not share one corpus configuration")
    corpus_hash = hashes.pop()
    config = metadata[0]["config"]
    if any(item["config"] != config for item in metadata[1:]):
        raise ValueError("shard corpus configurations differ")
    for record in records:
        if record.get("config_hash") != corpus_hash:
            raise ValueError("manifest record configuration does not match shard")

    unique = {}
    for record in records:
        key = (_record_key(record), record["reason"])
        previous = unique.get(key)
        if previous is None or float(record["score"]) > float(previous["score"]):
            unique[key] = record
    selected_candidates = [
        record
        for record in unique.values()
        if record["reason"].startswith("high_score:")
    ]
    selected = _select_with_global_quotas(
        selected_candidates,
        int(config["per_source_quota"]),
        int(config["per_item_quota"]),
        int(config["score_band_quota"]),
    )
    selected_keys = {_record_key(record) for record in selected}
    reserve_candidates = [
        record
        for record in unique.values()
        if record["reason"] == "random_reserve"
        and _record_key(record) not in selected_keys
    ]
    reserve_target = math.ceil(
        len(selected)
        * float(config["reserve_fraction"])
        / (1 - float(config["reserve_fraction"]))
    )
    reserve = sorted(
        reserve_candidates,
        key=lambda record: _stable_random_key(int(config["seed"]), record),
    )[:reserve_target]
    selected_count = _write_records(
        selected,
        Path(output),
        "mined",
        int(config["context_frames"]),
        corpus_hash,
    )
    reserve_count = _write_records(
        reserve,
        Path(output),
        "random_reserve",
        int(config["context_frames"]),
        corpus_hash,
    )
    manifest = Path(output) / "mining-manifest.jsonl"
    _write_manifest(manifest, selected, reserve)
    merged_metadata = {
        "schema_version": 1,
        "config_hash": corpus_hash,
        "shard_count": shard_count,
        "allow_incomplete": allow_incomplete,
        "selected": selected_count,
        "random_reserve": reserve_count,
        "source_shards": [item["shard_config_hash"] for item in metadata],
        "config": config,
    }
    _checkpoint_write(Path(output) / "mining-metadata.json", merged_metadata)
    return {**merged_metadata, "manifest": str(manifest)}


def inventory(roots: Sequence[Path]) -> list[dict]:
    """Return discovery metadata without opening or scoring archive items."""
    return discover_archives(roots)
