#!/usr/bin/env python3
"""Mine detector-triggered verifier negatives from a provenance manifest.

Only label-zero train rows explicitly marked eligible are read by default.  An
optional development-validation pass may also read unlocked validation rows;
test and deployment-anchor rows are never read.  Existing candidate feature
hashes are deduplicated, so this can extend a candidate corpus incrementally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.mine_kizz_librispeech_hard_negatives import (
    FEATURE_BINS,
    WINDOW_FRAMES,
    _atomic_json,
    _binding,
    _feature_hash,
    _mine_file,
    sha256_file,
)
from tools.simulate_kizz_int8_cascade import load_firmware_artifact
from tools.trace_kizz_ordered_state_detector import (
    _threshold_from_report,
    _validate_artifact,
)
from tools.train_kizz_candidate_verifier import load_verified_dataset
from tools.evaluate_kizz_int8_continuous_cascade import TFLiteRuntime


def _load_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _eligible_rows(
    manifest: Mapping[str, Any], *, include_validation: bool = False,
    validation_only: bool = False,
) -> list[dict[str, Any]]:
    if validation_only and not include_validation:
        raise ValueError("validation_only requires include_validation")
    raw = manifest.get("examples")
    if not isinstance(raw, list):
        raise ValueError("source manifest must contain examples")
    rows = []
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("source manifest examples must be objects")
        split = item.get("split")
        if split not in ({"train", "validation"} if include_validation else {"train"}):
            continue
        if validation_only and split != "validation":
            continue
        if item.get("label") != 0:
            continue
        if item.get("locked_holdout") or item.get("locked_deployment_anchor"):
            continue
        if split == "train" and item.get("training_eligible") is False:
            continue
        path = Path(str(item.get("path", ""))).resolve()
        if not path.is_file():
            raise ValueError(f"training source is missing: {path}")
        row = dict(item)
        row["path"] = str(path)
        rows.append(row)
    if not rows:
        raise ValueError("source manifest has no eligible development negatives")
    return sorted(rows, key=lambda item: str(item.get("source_id", item["path"])))


def _effective_split(source: Mapping[str, Any], holdout_fraction: float) -> str:
    split = str(source["split"])
    if split != "train" or holdout_fraction == 0:
        return split
    identity = str(
        source.get("ancestry_id")
        or source.get("session_id")
        or source.get("speaker_id")
        or source.get("source_id")
        or source.get("audio_sha256")
        or source.get("sha256")
        or source["path"]
    )
    bucket = int.from_bytes(
        hashlib.sha256(identity.encode("utf-8")).digest()[:8], "big"
    ) / 2**64
    return "validation" if bucket < holdout_fraction else "train"


def _base_selection(
    rows: Sequence[Mapping[str, Any]], excluded_providers: Sequence[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    excluded = {value.strip().lower() for value in excluded_providers if value.strip()}
    kept: list[int] = []
    removed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if str(row.get("provider", "")).strip().lower() in excluded:
            removed.append(dict(row))
        else:
            kept.append(index)
    if not kept:
        raise ValueError("base-provider exclusion removed every candidate")
    return np.asarray(kept, dtype=np.int64), removed


def _copy_relative_binding_files(
    value: Any, *, source_root: Path, output_root: Path
) -> list[str]:
    """Preserve locally bound sidecars when deriving an immutable dataset."""
    copied: list[str] = []
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            raw_path = item.get("path")
            expected_sha256 = item.get("sha256")
            if isinstance(raw_path, str) and isinstance(expected_sha256, str):
                relative = Path(raw_path)
                if not relative.is_absolute():
                    source = (source_root / relative).resolve()
                    target = (output_root / relative).resolve()
                    if not source.is_relative_to(source_root):
                        raise ValueError(f"relative binding escapes source dataset: {raw_path}")
                    if not target.is_relative_to(output_root):
                        raise ValueError(f"relative binding escapes output dataset: {raw_path}")
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    if sha256_file(source) != expected_sha256:
                        raise ValueError(f"relative binding hash drift: {source}")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    copied.append(relative.as_posix())
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return sorted(set(copied))


def mine(
    base_dataset: Path,
    base_corpus_sha256: str,
    source_manifest: Path,
    detector_metadata: Path,
    detector_model: Path,
    detector_threshold_report: Path,
    output: Path,
    top_k_per_file: int,
    include_validation: bool = False,
    validation_only: bool = False,
    train_development_holdout_fraction: float = 0.0,
    exclude_base_provider: Sequence[str] = (),
) -> dict[str, Any]:
    if top_k_per_file < 1:
        raise ValueError("top_k_per_file must be positive")
    if not 0.0 <= train_development_holdout_fraction <= 0.5:
        raise ValueError("train_development_holdout_fraction must be within [0,0.5]")
    base = load_verified_dataset(
        base_dataset, expected_corpus_sha256=base_corpus_sha256
    )
    detector_metadata = detector_metadata.resolve()
    detector_model = detector_model.resolve()
    _, topology, detector_contract = _validate_artifact(
        detector_metadata, detector_model
    )
    threshold, threshold_provenance = _threshold_from_report(
        detector_threshold_report, topology
    )
    detector = load_firmware_artifact(detector_metadata, "detector")
    runtime = TFLiteRuntime(detector_model, detector)
    source_manifest = source_manifest.resolve()
    sources = _eligible_rows(
        _load_object(source_manifest, "source manifest"),
        include_validation=include_validation,
        validation_only=validation_only,
    )

    base_indexes, excluded_base_rows = _base_selection(
        base.rows, exclude_base_provider
    )
    rows = [dict(base.rows[int(index)]) for index in base_indexes]
    existing_hashes = {
        str(row.get("candidate_feature_sha256", row.get("feature_sha256")))
        for row in rows
        if row.get("candidate_feature_sha256") or row.get("feature_sha256")
    }
    new_features: list[np.ndarray] = []
    new_scores: list[float] = []
    new_feature_frames: list[int] = []
    new_score_frames: list[int] = []
    source_ledger = []
    total_seconds = 0.0
    total_hops = 0
    duplicates = 0
    new_candidates_by_split = {"train": 0, "validation": 0}
    sources_by_split = {"train": 0, "validation": 0}
    exposure_by_split = {"train": 0.0, "validation": 0.0}

    for source in sources:
        original_split = str(source["split"])
        split = _effective_split(source, train_development_holdout_fraction)
        sources_by_split[split] += 1
        path = Path(source["path"])
        audio_hash = sha256_file(path)
        declared_hash = source.get("audio_sha256", source.get("sha256"))
        if declared_hash != audio_hash:
            raise ValueError(f"source hash drift: {path}")
        with sf.SoundFile(path) as audio:
            if audio.samplerate <= 0 or audio.channels <= 0:
                raise ValueError(f"audio contract drift: {path}")
            duration = len(audio) / audio.samplerate
        if abs(duration - float(source.get("duration_seconds", duration))) > 0.01:
            raise ValueError(f"source duration drift: {path}")
        total_seconds += duration
        exposure_by_split[split] += duration
        candidates, frame_count, hop_count = _mine_file(
            path,
            path.parent,
            runtime,
            topology,
            detector_contract,
            threshold,
            top_k=top_k_per_file,
        )
        total_hops += hop_count
        source_id = str(source.get("source_id", f"audio-sha256:{audio_hash}"))
        appended = 0
        for ordinal, (score, trigger, feature) in enumerate(candidates):
            feature16 = feature.astype(np.float16)
            feature_hash = _feature_hash(feature16)
            if feature_hash in existing_hashes:
                duplicates += 1
                continue
            existing_hashes.add(feature_hash)
            suffix = sha256_file(path)[:10] + f"{trigger:08x}{ordinal:02x}"
            candidate_id = f"{source_id}::manifest-hard-negative::{suffix}"
            rows.append(
                {
                    "source_id": candidate_id,
                    "candidate_id": candidate_id,
                    "parent_source_id": source_id,
                    "source_parent_source_id": source_id,
                    "speaker_id": str(source.get("speaker_id", source_id)),
                    "session_id": str(source.get("session_id", source_id)),
                    "ancestry_id": str(source.get("ancestry_id", source_id)),
                    "audio_sha256": audio_hash,
                    "source_audio_sha256": audio_hash,
                    "parent_source_audio_sha256": audio_hash,
                    "source_group": str(source.get("source_group", "negative_audio")),
                    "semantic_label": str(source.get("semantic_label", "non_wake")),
                    "provider": str(source.get("source", "manifest_audio")),
                    "split": split,
                    "source_manifest_split": original_split,
                    "label": 0,
                    "duration_seconds": duration,
                    "detector_conditioned": True,
                    "detector_score": score,
                    "detector_feature_frame_index": trigger,
                    "detector_score_frame_index": trigger,
                    "detector_event_ordinal": ordinal,
                    "detector_event": {
                        "feature_frame_index": trigger,
                        "feature_time_seconds": trigger * 0.01,
                        "score": score,
                        "score_frame_index": trigger,
                    },
                    "window": {
                        "requested_start_frame": trigger - 259,
                        "requested_stop_frame_exclusive": trigger + 1,
                        "source_start_frame": max(0, trigger - 259),
                        "source_stop_frame_exclusive": trigger + 1,
                        "left_padding_frames": max(0, 259 - trigger),
                        "right_padding_frames": 0,
                    },
                    "candidate_feature_sha256": feature_hash,
                    "feature_sha256": feature_hash,
                }
            )
            new_features.append(feature16)
            new_scores.append(score)
            new_feature_frames.append(trigger)
            new_score_frames.append(trigger)
            appended += 1
            new_candidates_by_split[split] += 1
        source_ledger.append(
            {
                "source_id": source_id,
                "path": str(path),
                "audio_sha256": audio_hash,
                "duration_seconds": duration,
                "split": split,
                "source_manifest_split": original_split,
                "candidate_count": len(candidates),
                "appended_candidate_count": appended,
                "frontend_feature_frames": frame_count,
                "detector_hops": hop_count,
            }
        )

    for index, row in enumerate(rows):
        row["feature_index"] = index
    features = np.concatenate(
        [
            np.asarray(base.features[base_indexes], dtype=np.float16),
            np.asarray(new_features, dtype=np.float16).reshape(
                -1, WINDOW_FRAMES, FEATURE_BINS
            ),
        ]
    )
    labels = np.concatenate(
        [
            np.asarray(base.labels[base_indexes], dtype=np.int8),
            np.zeros(len(new_features), dtype=np.int8),
        ]
    )
    scores = np.concatenate(
        [
            np.asarray(base.detector_scores[base_indexes], dtype=np.float32),
            np.asarray(new_scores, dtype=np.float32),
        ]
    )
    old_feature_frames = np.load(base.root / "detector_feature_frames.npy")
    old_score_frames = np.load(base.root / "detector_score_frames.npy")
    feature_frames = np.concatenate(
        [
            old_feature_frames[base_indexes].astype(np.int32),
            np.asarray(new_feature_frames, dtype=np.int32),
        ]
    )
    score_frames = np.concatenate(
        [
            old_score_frames[base_indexes].astype(np.int32),
            np.asarray(new_score_frames, dtype=np.int32),
        ]
    )

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _copy_relative_binding_files(
        base.corpus, source_root=base.root, output_root=output
    )
    arrays = {
        "features.npy": features,
        "labels.npy": labels,
        "detector_scores.npy": scores,
        "detector_feature_frames.npy": feature_frames,
        "detector_score_frames.npy": score_frames,
    }
    for name, values in arrays.items():
        np.save(output / name, values, allow_pickle=False)
    ledger_path = output / "manifest-hard-negative-sources.json"
    _atomic_json(
        ledger_path,
        {
            "schema_version": 1,
            "kind": "kizz_training_manifest_hard_negative_sources",
            "selection_policy": {
                "train_requires_training_eligible": True,
                "validation_is_development_only": True,
                "test_read": False,
                "locked_holdout_read": False,
                "deployment_anchor_read": False,
                "train_development_holdout_fraction": train_development_holdout_fraction,
                "train_holdout_assignment": "sha256(source_identity)_first_u64_fraction",
            },
            "source_manifest": _binding(source_manifest),
            "files": source_ledger,
            "counts": {
                "files": len(sources),
                "candidates": len(new_features),
                "duplicates_skipped": duplicates,
                "exposure_seconds": total_seconds,
                "candidates_by_split": new_candidates_by_split,
                "files_by_split": sources_by_split,
                "exposure_seconds_by_split": exposure_by_split,
            },
            "detector": {
                "artifact": _binding(detector_model),
                "metadata": _binding(detector_metadata),
                "threshold_report": _binding(detector_threshold_report),
            },
        },
    )

    corpus = json.loads(json.dumps(base.corpus))
    corpus["examples"] = rows
    corpus["array_sha256"] = {
        name: sha256_file(output / name) for name in arrays
    }
    corpus.setdefault("bindings", {})["base_candidate_corpus"] = _binding(
        base.corpus_path
    )
    corpus["bindings"]["training_manifest_hard_negative_sources"] = _binding(
        ledger_path
    )
    corpus["hard_negative_selection"].update(
        raw_training_count=int(corpus["hard_negative_selection"]["raw_training_count"])
        + new_candidates_by_split["train"],
        selected_training_count=int(
            corpus["hard_negative_selection"]["selected_training_count"]
        )
        + new_candidates_by_split["train"],
        top_k=top_k_per_file,
    )
    removed_by_split_label: dict[tuple[str, int], int] = {}
    for row in excluded_base_rows:
        key = (str(row["split"]), int(row["label"]))
        removed_by_split_label[key] = removed_by_split_label.get(key, 0) + 1
    for (split, label), count in removed_by_split_label.items():
        field = "selected_negative_candidates" if label == 0 else "selected_positive_candidates"
        corpus["counts"]["by_split"][split][field] -= count
    corpus["counts"]["selected_candidates"] = len(rows)
    corpus["counts"]["selected_negatives"] = int(np.sum(labels == 0))
    corpus["counts"]["selected_positives"] = int(np.sum(labels == 1))
    for split in ("train", "validation"):
        counts = corpus["counts"]["by_split"][split]
        counts["raw_detector_candidates"] += new_candidates_by_split[split]
        counts["raw_negative_candidates"] += new_candidates_by_split[split]
        counts["selected_negative_candidates"] += new_candidates_by_split[split]
        counts["source_examples"] += sources_by_split[split]
        counts["exposure_seconds"] += exposure_by_split[split]
        counts["negative_exposure_seconds"] += exposure_by_split[split]
        counts["raw_candidate_rate_per_second"] = (
            counts["raw_detector_candidates"] / counts["exposure_seconds"]
        )
        counts["raw_candidate_rate_per_hour"] = (
            counts["raw_candidate_rate_per_second"] * 3600
        )
        counts["raw_negative_candidate_rate_per_hour"] = (
            counts["raw_negative_candidates"]
            * 3600
            / counts["negative_exposure_seconds"]
        )
    train_negative_count = sum(
        int(row["split"] == "train" and int(row["label"]) == 0) for row in rows
    )
    heldout_negative_count = sum(
        int(row["split"] in {"validation", "test"} and int(row["label"]) == 0)
        for row in rows
    )
    corpus["hard_negative_selection"]["selected_training_count"] = train_negative_count
    corpus["hard_negative_selection"][
        "heldout_candidates_unfiltered"
    ] = heldout_negative_count
    corpus["base_candidate_exclusions"] = {
        "providers": sorted(
            {value.strip().lower() for value in exclude_base_provider if value.strip()}
        ),
        "excluded_candidate_count": len(excluded_base_rows),
        "excluded_candidate_ids": sorted(
            str(row.get("candidate_id", row.get("source_id")))
            for row in excluded_base_rows
        ),
        "retained_for_diagnostic_provenance_only": True,
    }
    corpus["training_manifest_hard_negative_extension"] = {
        "source_manifest": _binding(source_manifest),
        "source_ledger": _binding(ledger_path),
        "appended_training_negatives": new_candidates_by_split["train"],
        "appended_negatives_by_split": new_candidates_by_split,
        "duplicates_skipped": duplicates,
        "excluded_base_candidates": len(excluded_base_rows),
        "locked_holdout_used_for_training": False,
        "threshold": threshold,
        "threshold_provenance": threshold_provenance,
        "train_development_holdout_fraction": train_development_holdout_fraction,
    }
    _atomic_json(output / "corpus.json", corpus)
    return {
        "output": str(output),
        "corpus_sha256": sha256_file(output / "corpus.json"),
        "source_files": len(sources),
        "source_hours": total_seconds / 3600,
        "appended_training_negatives": new_candidates_by_split["train"],
        "appended_negatives_by_split": new_candidates_by_split,
        "duplicates_skipped": duplicates,
        "rows": len(rows),
        "detector_hops": total_hops,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--base-corpus-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--detector-metadata", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path, required=True)
    parser.add_argument("--detector-threshold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k-per-file", type=int, default=4)
    parser.add_argument(
        "--include-validation",
        action="store_true",
        help="also mine unlocked validation rows for development threshold selection",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="mine only unlocked development-validation rows (requires --include-validation)",
    )
    parser.add_argument(
        "--train-development-holdout-fraction",
        type=float,
        default=0.0,
        help="deterministically relabel this fraction of train source identities as development validation",
    )
    parser.add_argument(
        "--exclude-base-provider",
        action="append",
        default=[],
        help="exclude this provider from inherited candidate rows (repeatable)",
    )
    args = parser.parse_args(argv)
    print(json.dumps(mine(**vars(args)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
