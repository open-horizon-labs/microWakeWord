#!/usr/bin/env python3
"""Qualify a v3 Kizz teacher at a validation-only operating point.

This evaluator is deliberately read-only with respect to training artifacts.  It
scores fixed frontend features, chooses the threshold from validation positives
and validation negatives only, then applies that frozen threshold to aligned
held-out positives, localized natural positives, and the complete quarantined
false-wake set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from microwakeword.kizz_evaluation_contract import (
    require_disjoint_groups,
    validate_audio_rows,
)
from microwakeword.kizz_teacher import FEATURE_BINS, INPUT_FRAMES, build_teacher
from microwakeword.ordered_state import (
    KIZZ_PHONES,
    OrderedStateTopology,
    ordered_state_duration_score_numpy,
)
from tools.build_kizz_aligned_teacher_features_v3 import (
    frontend,
    load_audio,
    place_phrase_context,
)

SCHEMA_VERSION = 1
FEATURE_STEP_SECONDS = 0.01
FALSE_WAKE_EXPECTED_COUNT = 62


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(path: Path) -> str:
    return sha256_file(path)


def _rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("examples")
    if rows is None and isinstance(payload, dict):
        rows = payload.get("observations")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(
            f"{path}: expected a JSON list, examples list, or observations list"
        )
    return rows


def _stable_identity(row: Mapping[str, Any], *, fallback: str) -> str:
    """Return a content/provenance identity, never a display name alone."""
    for key in ("audio_sha256", "source_audio_sha256"):
        value = row.get(key)
        if value:
            # Audio hashes are the same identity even when one manifest calls
            # the field source_audio_sha256 and another calls it audio_sha256.
            return f"audio_sha256:{value}"
    for key in ("provenance_id", "parent_id", "parent_source_id", "source_id"):
        value = row.get(key)
        if value:
            return f"{key}:{value}"
    return f"fallback:{fallback}"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return _sha256_bytes(array.tobytes())


def dedupe_positive_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Dedupe positives by stable source/audio identity, deterministically.

    Returns ``(kept, duplicates)``.  Duplicate records are retained in the
    report for auditability but are never allowed to increase recall.
    """
    kept: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for index, raw in enumerate(records):
        record = dict(raw)
        identity = _stable_identity(record, fallback=str(index))
        record["stable_identity"] = identity
        if identity in kept:
            duplicates.append(
                {
                    "stable_identity": identity,
                    "duplicate_source_id": record.get("source_id"),
                    "kept_source_id": kept[identity].get("source_id"),
                }
            )
        else:
            kept[identity] = record
    return [kept[key] for key in sorted(kept)], duplicates


def score_logits(
    model: Any,
    features: np.ndarray,
    *,
    topology: OrderedStateTopology,
    minimum_path_frames: int,
    maximum_path_frames: int,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or tuple(values.shape[1:]) != (INPUT_FRAMES, FEATURE_BINS):
        raise ValueError("features must have shape [N, 260, 40]")
    scores: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        logits = np.asarray(
            model.predict(values[start : start + batch_size], verbose=0)
        )
        scores.append(
            ordered_state_duration_score_numpy(
                logits,
                topology,
                minimum_path_frames=minimum_path_frames,
                maximum_path_frames=maximum_path_frames,
            )
        )
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


def _load_fixed_features(path: Path) -> np.ndarray:
    values = np.load(path, mmap_mode="r")
    if values.ndim != 3 or tuple(values.shape[1:]) != (INPUT_FRAMES, FEATURE_BINS):
        raise ValueError(f"{path}: expected [N, 260, 40] features")
    return values


def score_fixed_source(
    model: Any,
    path: Path,
    *,
    topology: OrderedStateTopology,
    minimum_path_frames: int,
    maximum_path_frames: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = _load_fixed_features(path)
    scores = score_logits(
        model,
        values,
        topology=topology,
        minimum_path_frames=minimum_path_frames,
        maximum_path_frames=maximum_path_frames,
        batch_size=batch_size,
    )
    return scores, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "item_count": len(values),
        "exposure_seconds": float(len(values) * INPUT_FRAMES * FEATURE_STEP_SECONDS),
    }


def choose_validation_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    negative_exposure_seconds: float,
    *,
    min_recall: float,
    max_faph: float,
) -> dict[str, Any]:
    """Select the highest validation recall satisfying the validation FAPH gate."""
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if not len(positive) or not len(negative):
        raise ValueError("validation positive and negative scores are required")
    if negative_exposure_seconds <= 0:
        raise ValueError("negative exposure must be positive")
    candidates = []
    for threshold in np.unique(np.concatenate((positive, negative))):
        recall = float(np.mean(positive >= threshold))
        false_accepts = int(np.sum(negative >= threshold))
        faph = false_accepts / (negative_exposure_seconds / 3600.0)
        if recall >= min_recall and faph <= max_faph:
            candidates.append((recall, -faph, float(threshold), false_accepts))
    if not candidates:
        best_threshold = float(np.quantile(positive, 1.0 - min_recall, method="lower"))
        return {
            "qualified": False,
            "threshold": None,
            "selection_scope": "validation_only",
            "threshold_at_recall_floor": best_threshold,
            "recall_at_recall_floor": float(np.mean(positive >= best_threshold)),
            "false_accepts_at_recall_floor": int(np.sum(negative >= best_threshold)),
            "faph_at_recall_floor": float(
                np.sum(negative >= best_threshold)
                / (negative_exposure_seconds / 3600.0)
            ),
            "candidate_count": 0,
        }
    recall, neg_faph, threshold, false_accepts = max(candidates)
    return {
        "qualified": True,
        "threshold": threshold,
        "positive_recall": recall,
        "faph": -neg_faph,
        "false_accepts": false_accepts,
        "candidate_count": len(candidates),
        "selection_scope": "validation_only",
    }


def _summary(scores: Sequence[float], threshold: float | None) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    result: dict[str, Any] = {"count": len(values)}
    if len(values):
        result.update(
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            median=float(np.median(values)),
        )
        if threshold is not None:
            result["accepted"] = int(np.sum(values >= threshold))
    return result


def _duration_args(args: argparse.Namespace) -> tuple[int, int]:
    minimum = int(args.minimum_path_frames)
    maximum = int(args.maximum_path_frames)
    topology_minimum = len(KIZZ_PHONES) * int(args.states_per_phone)
    if minimum < topology_minimum or maximum < minimum:
        raise ValueError("duration bounds do not fit the selected ordered topology")
    return minimum, maximum


def _feature_provenance(path: Path, split: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples")
    if not isinstance(examples, list):
        raise TypeError("feature provenance must contain examples")
    manifests = payload.get("positive_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise TypeError("feature provenance must declare its positive manifests")
    parents = {}
    for manifest in manifests:
        manifest_path = Path(manifest["path"]).resolve()
        if sha256_file(manifest_path) != manifest.get("sha256"):
            raise ValueError(f"stale positive provenance manifest: {manifest_path}")
        for parent in _rows(manifest_path):
            source_id = parent.get("source_id")
            if source_id:
                parents[str(source_id)] = dict(parent)
    enriched = []
    for row in examples:
        if row.get("split") != split or row.get("augmentation") is not None:
            continue
        parent_id = row.get("parent_source_id")
        parent = parents.get(str(parent_id))
        if parent is None:
            raise ValueError(f"feature provenance parent is missing: {parent_id}")
        if row.get("source_audio_sha256") != parent.get("audio_sha256"):
            raise ValueError(
                f"feature provenance parent audio hash differs: {parent_id}"
            )
        enriched.append(
            {
                **parent,
                "feature_source_id": row.get("source_id"),
                "feature_record": dict(row),
            }
        )
    return enriched


def _localized_features(path: Path, phrase_span: Mapping[str, Any]) -> np.ndarray:
    def number(name: str) -> float:
        if phrase_span.get(f"{name}_s") is not None:
            return float(phrase_span[f"{name}_s"])
        return float(phrase_span[f"{name}_ms"]) / 1000.0

    context, _ = place_phrase_context(
        load_audio(path), (number("start"), number("end"))
    )
    return frontend(context)[None, ...]


def _false_wake_scores(
    model: Any,
    cache_manifest: Path,
    anchor_manifest: Path,
    *,
    topology: OrderedStateTopology,
    minimum_path_frames: int,
    maximum_path_frames: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
    observations: dict[str, dict[str, Any]] = {}
    for split_name, split in payload.get("splits", {}).items():
        feature_path = Path(split["path"])
        values = _load_fixed_features(feature_path)
        expected = sum(int(item["window_count"]) for item in split["observations"])
        if expected != len(values):
            raise ValueError(f"false-wake {split_name}: manifest/window count mismatch")
        scores = score_logits(
            model,
            values,
            topology=topology,
            minimum_path_frames=minimum_path_frames,
            maximum_path_frames=maximum_path_frames,
            batch_size=batch_size,
        )
        cursor = 0
        for item in split["observations"]:
            count = int(item["window_count"])
            identity = f"audio_sha256:{item['audio_sha256']}"
            if identity in observations:
                raise ValueError(f"duplicate false-wake observation: {identity}")
            window_scores = scores[cursor : cursor + count]
            cursor += count
            observations[identity] = {
                "stable_identity": identity,
                "observation_id": item["observation_id"],
                "audio_sha256": item["audio_sha256"],
                "split": split_name,
                "window_count": count,
                "maximum_score": float(np.max(window_scores)),
                "feature_path": str(feature_path.resolve()),
                "feature_sha256": sha256_file(feature_path),
            }
    if len(observations) != FALSE_WAKE_EXPECTED_COUNT:
        raise ValueError(
            f"false-wake cache must cover all {FALSE_WAKE_EXPECTED_COUNT} observations; "
            f"found {len(observations)}"
        )
    anchor_meta = validate_false_wake_anchor_contract(
        payload, observations, anchor_manifest
    )
    return [observations[key] for key in sorted(observations)], {
        "manifest": str(cache_manifest.resolve()),
        "manifest_sha256": _json_sha256(cache_manifest),
        "observation_count": len(observations),
        **anchor_meta,
    }


def validate_false_wake_anchor_contract(
    cache_payload: Mapping[str, Any],
    observations: Mapping[str, Any],
    anchor_manifest: Path,
    *,
    expected_count: int = FALSE_WAKE_EXPECTED_COUNT,
) -> dict[str, Any]:
    """Bind a feature cache to the explicit immutable anchor manifest."""
    anchor_manifest = anchor_manifest.resolve()
    if not anchor_manifest.is_file():
        raise FileNotFoundError(anchor_manifest)
    declared_path = cache_payload.get("manifest")
    if not declared_path:
        raise ValueError("false-wake cache does not declare its anchor manifest")
    cache_source_manifest = Path(declared_path).resolve()
    if not cache_source_manifest.is_file():
        raise FileNotFoundError(cache_source_manifest)
    cache_source_sha = _json_sha256(cache_source_manifest)
    if cache_payload.get("manifest_sha256") != cache_source_sha:
        raise ValueError("false-wake cache anchor manifest hash is stale")
    cache_source_rows = _rows(cache_source_manifest)
    cache_source_identities = {
        f"audio_sha256:{row['audio_sha256']}"
        for row in cache_source_rows
        if row.get("audio_sha256")
    }
    if cache_source_identities != set(observations):
        raise ValueError("false-wake cache differs from its declared source manifest")
    actual_manifest_sha = _json_sha256(anchor_manifest)
    anchor_rows = _rows(anchor_manifest)
    if len(anchor_rows) != expected_count:
        raise ValueError(
            f"frozen false-wake anchor manifest must contain {expected_count} rows"
        )
    audio_contract = validate_audio_rows(
        anchor_rows,
        group="false_wake_anchor",
        require_locked_anchor=True,
    )
    anchor_identities = {f"audio_sha256:{row['audio_sha256']}" for row in anchor_rows}
    if anchor_identities != set(observations):
        raise ValueError("false-wake feature cache does not cover the frozen anchors")
    return {
        "cache_source_manifest": str(cache_source_manifest),
        "cache_source_manifest_sha256": cache_source_sha,
        "anchor_manifest": str(anchor_manifest),
        "anchor_manifest_sha256": actual_manifest_sha,
        "anchor_audio_contract": audio_contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-positive-features", type=Path, required=True)
    parser.add_argument("--validation-positive-provenance", type=Path, required=True)
    parser.add_argument(
        "--validation-negative-source",
        action="append",
        required=True,
        metavar="ID=PATH",
    )
    parser.add_argument("--heldout-positive-features", type=Path, required=True)
    parser.add_argument("--heldout-positive-provenance", type=Path, required=True)
    parser.add_argument("--localized-positive-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-cache-manifest", type=Path, required=True)
    parser.add_argument("--false-wake-anchor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states-per-phone", type=int, default=2)
    parser.add_argument("--minimum-path-frames", type=int, default=24)
    parser.add_argument("--maximum-path-frames", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--max-faph", type=float, default=0.10)
    args = parser.parse_args(argv)
    if (
        args.states_per_phone < 1
        or args.batch_size < 1
        or not 0 < args.min_recall <= 1
        or args.max_faph < 0
    ):
        parser.error("invalid model, batch, or operating-point limits")
    try:
        minimum, maximum = _duration_args(args)
        sources = []
        for raw in args.validation_negative_source:
            if "=" not in raw:
                raise ValueError("validation negative source must be ID=PATH")
            source_id, raw_path = raw.split("=", 1)
            path = Path(raw_path).resolve()
            if not source_id or not path.is_file():
                raise ValueError(f"invalid validation negative source: {raw}")
            sources.append((source_id, path))

        validation_rows = _feature_provenance(
            args.validation_positive_provenance, "validation"
        )
        validation_features = _load_fixed_features(args.validation_positive_features)
        if len(validation_rows) != len(validation_features):
            raise ValueError("validation feature/provenance counts differ")
        aligned_rows = _feature_provenance(args.heldout_positive_provenance, "test")
        aligned_features = _load_fixed_features(args.heldout_positive_features)
        if len(aligned_rows) != len(aligned_features):
            raise ValueError("held-out feature/provenance counts differ")
        natural_rows = [
            row
            for row in _rows(args.localized_positive_manifest)
            if int(row.get("label", 1)) == 1
        ]
        anchor_rows = _rows(args.false_wake_anchor_manifest)
        evidence_contracts = {
            "validation_positive": validate_audio_rows(
                validation_rows, group="validation_positive"
            ),
            "heldout_aligned_positive": validate_audio_rows(
                aligned_rows, group="heldout_aligned_positive"
            ),
            "natural_positive": validate_audio_rows(
                natural_rows, group="natural_positive"
            ),
            "false_wake_anchor": validate_audio_rows(
                anchor_rows,
                group="false_wake_anchor",
                require_locked_anchor=True,
            ),
        }
        evidence_groups = {
            "validation_positive": validation_rows,
            "heldout_aligned_positive": aligned_rows,
            "natural_positive": natural_rows,
            "false_wake_anchor": anchor_rows,
        }
        require_disjoint_groups(evidence_groups)
        require_disjoint_groups(
            {
                "validation": validation_rows,
                "heldout": aligned_rows + natural_rows + anchor_rows,
            },
            include_partition_identity=True,
        )

        topology = OrderedStateTopology(KIZZ_PHONES, args.states_per_phone)
        model = build_teacher(topology=topology)
        model.load_weights(args.model)
        validation_positive, _validation_positive_meta = score_fixed_source(
            model,
            args.validation_positive_features,
            topology=topology,
            minimum_path_frames=minimum,
            maximum_path_frames=maximum,
            batch_size=args.batch_size,
        )
        validation_negative = []
        negative_meta = []
        for source_id, path in sources:
            scores, meta = score_fixed_source(
                model,
                path,
                topology=topology,
                minimum_path_frames=minimum,
                maximum_path_frames=maximum,
                batch_size=args.batch_size,
            )
            validation_negative.extend(scores)
            meta["id"] = source_id
            negative_meta.append(meta)
        negative_exposure = float(
            sum(item["exposure_seconds"] for item in negative_meta)
        )
        point = choose_validation_threshold(
            validation_positive,
            validation_negative,
            negative_exposure,
            min_recall=args.min_recall,
            max_faph=args.max_faph,
        )
        threshold = point["threshold"]

        aligned_records = [
            {
                "source_id": row.get("parent_source_id", row.get("source_id")),
                "audio_sha256": row.get("audio_sha256")
                or row.get("source_audio_sha256"),
                "provenance": row,
                "score": float(score),
            }
            for row, score in zip(
                aligned_rows,
                score_logits(
                    model,
                    aligned_features,
                    topology=topology,
                    minimum_path_frames=minimum,
                    maximum_path_frames=maximum,
                    batch_size=args.batch_size,
                ),
            )
        ]
        natural_records = []
        natural_exclusions = []
        for row in natural_rows:
            path = Path(row["path"]).resolve()
            if not isinstance(row.get("phrase_span"), Mapping):
                natural_exclusions.append(
                    {
                        "source_id": row.get("source_id", str(path)),
                        "audio_sha256": row.get("audio_sha256"),
                        "reason": "missing_phrase_span",
                    }
                )
                continue
            feature = _localized_features(path, row["phrase_span"])
            score = score_logits(
                model,
                feature,
                topology=topology,
                minimum_path_frames=minimum,
                maximum_path_frames=maximum,
                batch_size=1,
            )[0]
            audio_hash = row.get("audio_sha256") or sha256_file(path)
            natural_records.append(
                {
                    "source_id": row.get("source_id", str(path)),
                    "audio_sha256": audio_hash,
                    "path": str(path),
                    "provenance": row,
                    "score": float(score),
                }
            )
        positives, duplicates = dedupe_positive_records(
            aligned_records + natural_records
        )
        false_wakes, false_wake_meta = _false_wake_scores(
            model,
            args.false_wake_cache_manifest,
            args.false_wake_anchor_manifest,
            topology=topology,
            minimum_path_frames=minimum,
            maximum_path_frames=maximum,
            batch_size=args.batch_size,
        )
        false_accepts = (
            None
            if threshold is None
            else [item for item in false_wakes if item["maximum_score"] >= threshold]
        )
        positive_accepts = (
            None
            if threshold is None
            else [item for item in positives if item["score"] >= threshold]
        )
        heldout_recall = (
            None
            if threshold is None or not positives
            else len(positive_accepts) / len(positives)
        )
        reasons = []
        if not point["qualified"]:
            reasons.append("validation_operating_point_not_qualified")
        if len(positives) == 0:
            reasons.append("no_deduplicated_heldout_positive_opportunities")
        if natural_exclusions:
            reasons.append("unscorable_natural_positive_anchor")
        if heldout_recall is not None and heldout_recall < args.min_recall:
            reasons.append("heldout_positive_recall_below_minimum")
        if false_accepts is not None and false_accepts:
            reasons.append("quarantined_false_wake_accepted")
        result = {
            "schema_version": SCHEMA_VERSION,
            "gate_scope": "teacher_clip_and_anchor_prequalification",
            "qualified": not reasons,
            "reasons": reasons,
            "model": str(args.model.resolve()),
            "model_sha256": sha256_file(args.model),
            "provenance": {
                "qualifier_tool_sha256": sha256_file(Path(__file__).resolve()),
            },
            "topology": {
                "phones": list(KIZZ_PHONES),
                "states_per_phone": args.states_per_phone,
            },
            "duration_bounds": {
                "minimum_path_frames": minimum,
                "maximum_path_frames": maximum,
                "minimum_seconds": minimum * 0.03,
                "maximum_seconds": maximum * 0.03,
            },
            "validation": {
                "positive": _summary(validation_positive, point["threshold"]),
                "negative": _summary(validation_negative, point["threshold"]),
                "negative_exposure_seconds": negative_exposure,
                "negative_sources": negative_meta,
                "operating_point": point,
                "positive_provenance": {
                    "path": str(args.validation_positive_provenance.resolve()),
                    "sha256": sha256_file(args.validation_positive_provenance),
                },
            },
            "heldout_positives": {
                "count": len(positives),
                "recall": heldout_recall,
                "accepted": None if positive_accepts is None else len(positive_accepts),
                "duplicates": duplicates,
                "localized_exclusions": natural_exclusions,
                "records": positives,
                "aligned_provenance": {
                    "path": str(args.heldout_positive_provenance.resolve()),
                    "sha256": sha256_file(args.heldout_positive_provenance),
                },
            },
            "false_wakes": {
                **false_wake_meta,
                "accepted_count": None if false_accepts is None else len(false_accepts),
                "accepted_observations": false_accepts,
                "records": false_wakes,
            },
            "evidence_contracts": evidence_contracts,
            "read_only": True,
            "training_data_modified": False,
        }
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["qualified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
