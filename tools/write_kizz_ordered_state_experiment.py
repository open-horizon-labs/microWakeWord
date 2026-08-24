#!/usr/bin/env python3
"""Write leakage-checked binary-control and ordered-state Kizz configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from microwakeword.provenance import sha256_path

try:
    from tools.write_stratified_training_config import stratified_config
except ModuleNotFoundError:  # pragma: no cover
    from write_stratified_training_config import stratified_config

FORBIDDEN_COMPONENTS = {"observations", "false-wakes", "evidence"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_quarantine(path: Path, role: str) -> None:
    components = {
        component.casefold()
        for candidate in (path, path.resolve(strict=False))
        for component in candidate.parts
    }
    forbidden = sorted(components & FORBIDDEN_COMPONENTS)
    if forbidden:
        raise ValueError(f"{role} contains quarantined path components: {forbidden}")


def validate_frozen_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("frozen source manifest requires schema_version 1")
    if manifest.get("threshold_selection_split") != "validation":
        raise ValueError("threshold selection must be frozen to validation")
    if manifest.get("source_disjoint") != {
        "source_id": True,
        "speaker_id": True,
        "session_id": True,
    }:
        raise ValueError("source, speaker, and session disjointness must be explicit")
    exposure = manifest.get("counts", {}).get("exposure_seconds_by_split", {})
    for split in ("validation", "test"):
        if float(exposure.get(split, 0.0)) < 100 * 3600:
            raise ValueError(f"{split} negative exposure is below 100 hours")
    for source in manifest.get("sources", []):
        reject_quarantine(Path(source["path"]), f"frozen {source.get('source_id')}")
    return manifest


def _source(
    path: Path,
    name: str,
    truth: bool,
    group: str | None,
    weight: float,
    *,
    penalty: float = 1.0,
    truncation: str = "random",
    evaluation: bool = True,
    expected_path_sha256: str | None = None,
) -> dict:
    reject_quarantine(path, name)
    source = {
        "features_dir": str(path),
        "source_name": name,
        "truth": truth,
        "group": group,
        "within_group_weight": weight,
        "penalty_weight": penalty,
        "truncation_strategy": truncation,
        "evaluation_enabled": evaluation,
    }
    if expected_path_sha256 is not None:
        source["expected_path_sha256"] = expected_path_sha256
    return source


def experiment_plan(
    workspace: Path,
    train_dir: Path,
    model_family: str,
    frozen_manifest: Path,
    hard_negatives: Path,
    frame_supervision: Path | None = None,
    frame_weight: float = 0.25,
) -> dict:
    if model_family not in {"binary_control", "ordered_state"}:
        raise ValueError("model_family must be binary_control or ordered_state")
    frozen = validate_frozen_manifest(frozen_manifest)
    frame_weight = float(frame_weight)
    if frame_weight < 0:
        raise ValueError("frame_weight must be non-negative")
    if (
        model_family == "ordered_state"
        and frame_weight > 0
        and frame_supervision is None
    ):
        raise ValueError("ordered_state requires measured frame supervision")
    canonical = workspace / "features.v32-canonical-connected"
    device = workspace / "promoted.v32-device-canonical-span-aligned" / "features"
    negative_view = workspace / "ordered-v1-frozen-negative-view"
    required = [
        canonical / "feature-build-manifest.json",
        device / "promoted-audio-feature-build.json",
        hard_negatives / "mining-metadata.json",
    ]
    if any(not path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise ValueError(f"missing experiment provenance: {missing}")
    training_directories = (
        canonical / "positive",
        canonical / "hard_negative",
        device / "positive",
        hard_negatives / "mined",
        hard_negatives / "random_reserve",
        negative_view / "speech",
        negative_view / "non_speech",
    )
    for directory in training_directories:
        if not directory.is_dir():
            raise ValueError(f"missing feature directory: {directory}")
        reject_quarantine(directory, "training feature")
    training_directory_hashes = {
        str(directory): sha256_path(directory) for directory in training_directories
    }

    training_loss: dict = (
        {
            "name": "ordered_state_sequence",
            "self_loop_probability": 0.6,
            "next_state_probability": 0.4,
        }
        if model_family == "ordered_state"
        else {
            "name": "binary_focal_crossentropy",
            "gamma": 2.0,
            "apply_class_balancing": False,
        }
    )
    if model_family == "ordered_state" and frame_weight > 0:
        assert frame_supervision is not None
        reject_quarantine(frame_supervision, "frame supervision")
        for filename in ("features.npy", "targets.npy", "weights.npy"):
            if not (frame_supervision / filename).is_file():
                raise ValueError(f"missing frame supervision array: {filename}")
        frame_hashes = {
            filename: sha256(frame_supervision / filename)
            for filename in ("features.npy", "targets.npy", "weights.npy")
        }
        training_loss.update(
            {
                "frame_weight": frame_weight,
                "frame_supervision": {
                    "directory": str(frame_supervision),
                    "batch_size": 64,
                    "seed": 24102,
                    "expected_files_sha256": frame_hashes,
                },
            }
        )

    overrides = {
        "train_dir": str(train_dir),
        "model_family": model_family,
        "model_parameters": {
            "first_conv_filters": 48,
            "first_conv_kernel_size": 5,
            "stride": 3,
            "pointwise_filters": "96,96,96,96",
            "mixconv_kernel_sizes": "[3],[5],[7],[9]",
        },
        "training_seed": 24101 if model_family == "binary_control" else 24102,
        "deterministic_training": True,
        "training_steps": [20000, 10000],
        "learning_rates": [0.001, 0.0002],
        "batch_size": 128,
        "eval_step_interval": 5000,
        "positive_class_weight": [1, 1],
        "negative_class_weight": [1, 1],
        "training_loss": training_loss,
        "require_binary_validation": True,
        "target_minimization": 0,
        "minimization_metric": "validation_false_positives",
        "maximization_metric": "average_viable_recall",
    }
    sources = [
        _source(
            canonical / "positive",
            "canonical-positive",
            True,
            "canonical_positive",
            1,
            truncation="truncate_start",
            expected_path_sha256=training_directory_hashes[str(canonical / "positive")],
        ),
        _source(
            device / "positive",
            "reviewed-device-positive",
            True,
            "device_positive",
            1,
            truncation="truncate_start",
            expected_path_sha256=training_directory_hashes[str(device / "positive")],
        ),
        _source(
            canonical / "hard_negative",
            "canonical-collision-negative",
            False,
            "collision_negative",
            1,
            expected_path_sha256=training_directory_hashes[
                str(canonical / "hard_negative")
            ],
        ),
        _source(
            hard_negatives / "mined",
            "v19-train-family-hard-intervals",
            False,
            "mined_negative",
            1,
            penalty=2.0,
            evaluation=False,
            expected_path_sha256=training_directory_hashes[
                str(hard_negatives / "mined")
            ],
        ),
        _source(
            hard_negatives / "random_reserve",
            "deterministic-random-negative-reserve",
            False,
            "random_negative",
            1,
            evaluation=False,
            expected_path_sha256=training_directory_hashes[
                str(hard_negatives / "random_reserve")
            ],
        ),
        _source(
            negative_view / "speech",
            "source-disjoint-speech-negative",
            False,
            "ordinary_speech",
            1,
            expected_path_sha256=training_directory_hashes[
                str(negative_view / "speech")
            ],
        ),
        _source(
            negative_view / "non_speech",
            "source-disjoint-nonspeech-negative",
            False,
            "background_negative",
            1,
            expected_path_sha256=training_directory_hashes[
                str(negative_view / "non_speech")
            ],
        ),
    ]
    return {
        "schema_version": 1,
        "experiment": "kizz-ordered-state-v1",
        "model_family": model_family,
        "candidate_matrix": ["binary_control", "ordered_state"],
        "ordered_state_frame_weight": (
            frame_weight if model_family == "ordered_state" else None
        ),
        "sampling_groups": {
            "canonical_positive": 0.30,
            "device_positive": 0.10,
            "collision_negative": 0.15,
            "mined_negative": 0.15,
            "random_negative": 0.10,
            "ordinary_speech": 0.15,
            "background_negative": 0.05,
        },
        "balance_guard": {
            "maximum_negative_sampling_share": 0.60,
            "maximum_negative_weighted_pressure_share": 0.70,
        },
        "frozen_negative_manifest": {
            "path": str(frozen_manifest.resolve()),
            "sha256": sha256(frozen_manifest),
            "inventory_sha256": frozen["inventory_sha256"],
        },
        "input_hashes": {
            **{str(path): sha256(path) for path in required},
            **training_directory_hashes,
        },
        "config_overrides": overrides,
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--model-family",
        choices=("binary_control", "ordered_state"),
        required=True,
    )
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--hard-negatives", type=Path, required=True)
    parser.add_argument("--frame-supervision", type=Path)
    parser.add_argument("--frame-weight", type=float, default=0.25)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    plan = experiment_plan(
        args.workspace,
        args.train_dir,
        args.model_family,
        args.frozen_manifest,
        args.hard_negatives,
        args.frame_supervision,
        args.frame_weight,
    )
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(yaml.safe_dump(plan, sort_keys=False))
    config = stratified_config(args.base_config, args.output_plan)
    config["qualification_manifest"] = plan["frozen_negative_manifest"]
    config["candidate_matrix"] = plan["candidate_matrix"]
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
