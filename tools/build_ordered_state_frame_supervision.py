#!/usr/bin/env python3
"""Build fixed aligned-frame arrays for the ordered-state trainer.

The input is a JSON manifest (an array or JSONL) whose records contain one
feature matrix and one explicit output-frame timestamp grid.  Phone boundaries
must come from a reviewed forced aligner or from a synthesizer timing record;
this tool never derives phone boundaries from text or divides a phrase into
equal-duration phones.

Example record::

    {
      "source_id": "speaker-01/session-02/clip-03",
      "source_group": "speaker-01/session-02",
      "split": "train",
      "truth": true,
      "duration_s": 2.4,
      "features_path": "features/clip-03.npy",
      "feature_frame_step_seconds": 0.01,
      "target_frame_times_s": [0.03, 0.06, 0.09],
      "alignment": {
        "method": "forced_aligner",
        "timing_source": "alignments/clip-03.json",
        "reviewed": true
      },
      "phrase_span": {"start_s": 0.31, "end_s": 1.92},
      "phone_spans": [
        {"phone": "h", "start_s": 0.31, "end_s": 0.45}
      ]
    }

The target cadence is deliberately a required manifest field.  The default
30 ms cadence is the ordered-state model contract; feature cadence defaults to
the product 10 ms frontend cadence.  A ``source_group`` may occur in only one
split, preventing speaker/session leakage across train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from microwakeword.ordered_state_data import (
    example_from_mapping,
    frame_state_targets,
)

DEFAULT_FEATURE_STEP_SECONDS = 0.01
DEFAULT_TARGET_STEP_SECONDS = 0.03
FEATURE_BINS = 40
SPLITS = frozenset(("train", "validation", "test"))
ALIGNMENT_METHODS = frozenset(
    ("forced_aligner", "reviewed_forced_alignment", "synthesizer")
)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, list):
            records = value
        elif isinstance(value, dict) and "records" in value:
            records = value["records"]
        elif isinstance(value, dict):
            # A one-record JSON manifest is useful for small reproducibility
            # checks and is unambiguous because records always have source_id.
            records = [value]
        else:
            records = None
    if not isinstance(records, list) or not records:
        raise ValueError("manifest must contain a non-empty record list")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("manifest records must be objects")
    return records


def _feature_path(record: Mapping[str, Any], manifest_dir: Path) -> Path:
    raw = record.get("features_path")
    if not raw:
        raise ValueError("features_path is required")
    path = Path(str(raw))
    return path if path.is_absolute() else manifest_dir / path


def _load_features(record: Mapping[str, Any], manifest_dir: Path) -> np.ndarray:
    path = _feature_path(record, manifest_dir)
    if not path.is_file():
        raise ValueError(f"feature file does not exist: {path}")
    try:
        loaded = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load features: {path}") from exc
    if isinstance(loaded, np.lib.npyio.NpzFile):
        try:
            if "features" not in loaded.files:
                raise ValueError("feature npz must contain a features array")
            features = loaded["features"]
        finally:
            loaded.close()
    else:
        features = loaded
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != FEATURE_BINS:
        raise ValueError("features must have shape [time, 40]")
    if not features.shape[0]:
        raise ValueError("features must contain at least one frame")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")
    return features.astype(np.float32, copy=False)


def _validate_alignment_metadata(record: Mapping[str, Any]) -> None:
    alignment = record.get("alignment")
    if not isinstance(alignment, Mapping):
        raise ValueError("alignment timing metadata is required")
    method = alignment.get("method")
    if method not in ALIGNMENT_METHODS:
        raise ValueError("alignment method must be a reviewed aligner or synthesizer")
    timing_source = alignment.get("timing_source")
    if not isinstance(timing_source, str) or not timing_source.strip():
        raise ValueError("alignment timing_source is required")
    if method in {"forced_aligner", "reviewed_forced_alignment"} and not bool(
        alignment.get("reviewed")
    ):
        raise ValueError("forced-aligner timing must be explicitly reviewed")
    if method == "synthesizer" and not alignment.get("timing_record"):
        raise ValueError("synthesizer metadata requires a timing_record")


def _target_times(record: Mapping[str, Any], expected_step: float) -> np.ndarray:
    raw = record.get("target_frame_times_s")
    if not isinstance(raw, list) or not raw:
        raise ValueError("explicit target_frame_times_s are required")
    times = np.asarray([_finite_float(value, "target frame time") for value in raw])
    if np.any(times < 0):
        raise ValueError("target frame times must be non-negative")
    if len(times) > 1 and not np.all(
        np.isclose(np.diff(times), expected_step, rtol=1e-6, atol=1e-7)
    ):
        raise ValueError("target frame times have the wrong cadence")
    return times


def _validate_record(
    record: Mapping[str, Any],
    manifest_dir: Path,
    feature_step: float,
    target_step: float,
) -> tuple[np.ndarray, np.ndarray, float, str, str]:
    source_id = record.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id is required")
    source_group = record.get("source_group")
    if not isinstance(source_group, str) or not source_group.strip():
        raise ValueError("source_group is required for split isolation")
    split = record.get("split")
    if split not in SPLITS:
        raise ValueError("split must be train, validation, or test")
    declared_feature_step = _finite_float(
        record.get("feature_frame_step_seconds"), "feature_frame_step_seconds"
    )
    if not _close(declared_feature_step, feature_step):
        raise ValueError("feature frame cadence is incorrect")
    _validate_alignment_metadata(record)
    features = _load_features(record, manifest_dir)
    duration = _finite_float(record.get("duration_s"), "duration_s")
    times = _target_times(record, target_step)
    if np.any(times > duration):
        raise ValueError("target frame time exceeds duration")
    if len(times) > features.shape[0]:
        raise ValueError("target frame count exceeds feature frame count")
    example = example_from_mapping(record)
    targets = frame_state_targets(example, times)
    if targets is None:
        raise ValueError("positive frame supervision requires measured phone spans")
    weight = _finite_float(record.get("weight", 1.0), "weight")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    return features, targets, weight, source_group, split


def build_frame_supervision(
    records: Iterable[Mapping[str, Any]],
    manifest_dir: Path,
    output_dir: Path,
    *,
    feature_step_seconds: float = DEFAULT_FEATURE_STEP_SECONDS,
    target_step_seconds: float = DEFAULT_TARGET_STEP_SECONDS,
    expected_feature_frames: int | None = None,
    expected_target_frames: int | None = None,
) -> dict[str, Any]:
    """Validate records and write trainer-compatible NumPy arrays."""
    feature_step = _finite_float(feature_step_seconds, "feature cadence")
    target_step = _finite_float(target_step_seconds, "target cadence")
    if feature_step <= 0 or target_step <= 0:
        raise ValueError("cadences must be positive")
    if not _close(target_step, 3 * feature_step):
        raise ValueError("target cadence must be exactly three feature frames")

    if expected_feature_frames is not None and expected_feature_frames < 1:
        raise ValueError("expected feature frame count must be positive")
    if expected_target_frames is not None and expected_target_frames < 1:
        raise ValueError("expected target frame count must be positive")
    records = list(records)
    rows = []
    groups: dict[str, str] = {}
    source_ids: set[str] = set()
    for record in records:
        features, targets, weight, group, split = _validate_record(
            record, manifest_dir, feature_step, target_step
        )
        source_id = str(record["source_id"])
        if source_id in source_ids:
            raise ValueError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        previous_split = groups.setdefault(group, split)
        if previous_split != split:
            raise ValueError(f"source_group leaks across splits: {group}")
        rows.append((features, targets, weight))

    if not rows:
        raise ValueError("ordered-state frame supervision is empty")
    feature_shape = rows[0][0].shape
    target_shape = rows[0][1].shape
    if (
        expected_feature_frames is not None
        and feature_shape[0] != expected_feature_frames
    ):
        raise ValueError("feature frame count is incompatible with the model")
    if expected_target_frames is not None and target_shape[0] != expected_target_frames:
        raise ValueError("target frame count is incompatible with the model")
    if any(features.shape != feature_shape for features, _, _ in rows):
        raise ValueError("all feature matrices must have the same shape")
    if any(targets.shape != target_shape for _, targets, _ in rows):
        raise ValueError("all target grids must have the same length")
    features = np.stack([row[0] for row in rows]).astype(np.float32, copy=False)
    targets = np.stack([row[1] for row in rows]).astype(np.int32, copy=False)
    weights = np.asarray([row[2] for row in rows], dtype=np.float32)
    if not np.any(weights > 0):
        raise ValueError("at least one example must have positive weight")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "targets.npy", targets)
    np.save(output_dir / "weights.npy", weights)
    return {
        "examples": len(rows),
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "feature_frame_step_seconds": feature_step,
        "target_frame_step_seconds": target_step,
        "splits": sorted({str(record["split"]) for record in records}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-frame-step-seconds", type=float, default=0.01)
    parser.add_argument("--target-frame-step-seconds", type=float, default=0.03)
    parser.add_argument("--feature-frame-count", type=int, required=True)
    parser.add_argument("--target-frame-count", type=int, required=True)
    args = parser.parse_args()
    summary = build_frame_supervision(
        _load_manifest(args.manifest),
        args.manifest.parent,
        args.output,
        feature_step_seconds=args.feature_frame_step_seconds,
        target_step_seconds=args.target_frame_step_seconds,
        expected_feature_frames=args.feature_frame_count,
        expected_target_frames=args.target_frame_count,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
