#!/usr/bin/env python3
"""Materialize the controlled Kizz v32 scratch/adaptation training configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

try:
    from tools.write_stratified_training_config import stratified_config
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from write_stratified_training_config import stratified_config


V19_CHECKPOINT_100_WEIGHTS_SHA256 = (
    "a7b134dd3eb54d80f4c8141566e3b5d5bb14e556f095f11d64383b22c410f091"
)
QUARANTINED_PATH_COMPONENTS = {"observations", "false-wakes", "evidence"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_quarantined_path(path: Path, role: str) -> None:
    components = {
        component.casefold()
        for candidate in (path, path.resolve(strict=False))
        for component in candidate.parts
    }
    forbidden = sorted(components & QUARANTINED_PATH_COMPONENTS)
    if forbidden:
        raise ValueError(
            f"{role} cannot use quarantined evidence path components: {forbidden}"
        )


def validate_promoted_device_features(features: Path) -> dict:
    feature_manifest = features / "promoted-audio-feature-build.json"
    if not feature_manifest.is_file():
        raise ValueError(
            f"missing promoted device feature manifest: {feature_manifest}"
        )
    manifest = json.loads(feature_manifest.read_text())
    entries = manifest.get("entries", [])
    if not entries or any(
        entry.get("truth") != "positive"
        or entry.get("text") != "Hi-Fi Kizz"
        or entry.get("human_reviewed") is not True
        or entry.get("training_eligible") is not True
        for entry in entries
    ):
        raise ValueError(
            "promoted device features are not canonical reviewed positives"
        )
    splits = {entry.get("split") for entry in entries}
    if "train" not in splits:
        raise ValueError("promoted device features require train entries")
    source_manifest = Path(manifest.get("source_manifest", ""))
    reject_quarantined_path(source_manifest, "promoted device source manifest")
    if not source_manifest.is_file() or sha256(source_manifest) != manifest.get(
        "source_manifest_sha256"
    ):
        raise ValueError("promoted device source manifest provenance mismatch")
    source = json.loads(source_manifest.read_text())
    speakers = {
        split: set(ids) for split, ids in source.get("speaker_ids_by_split", {}).items()
    }
    if speakers.get("train", set()) & speakers.get("test", set()):
        raise ValueError("promoted device train/test speakers overlap")
    return manifest


def experiment_plan(
    workspace: Path,
    feature_manifest: Path,
    mined_features: Path,
    device_features: Path,
    train_dir: Path,
    mode: str,
    initial_weights: Path | None = None,
) -> dict:
    if mode not in {"scratch", "v19-adapt", "candidate-remine"}:
        raise ValueError("mode must be scratch, v19-adapt, or candidate-remine")
    if mode == "scratch" and initial_weights is not None:
        raise ValueError("scratch mode cannot use initial weights")
    if mode == "v19-adapt":
        if initial_weights is None or not initial_weights.is_file():
            raise ValueError("v19-adapt requires checkpoint-100 initial weights")
        if sha256(initial_weights) != V19_CHECKPOINT_100_WEIGHTS_SHA256:
            raise ValueError("initial weights are not deployed v19 checkpoint 100")
    if mode == "candidate-remine" and (
        initial_weights is None or not initial_weights.is_file()
    ):
        raise ValueError("candidate-remine requires candidate initial weights")
    for path, role in (
        (feature_manifest, "canonical features"),
        (mined_features, "mined features"),
        (device_features, "promoted device features"),
    ):
        reject_quarantined_path(path, role)

    negatives = workspace / "negative-datasets"
    overrides = {
        "train_dir": str(train_dir),
        "model_parameters": {
            "first_conv_filters": 48,
            "first_conv_kernel_size": 5,
            "stride": 3,
            "pointwise_filters": "96,96,96,96",
            "mixconv_kernel_sizes": "[5],[7,11],[9,15],[23]",
        },
        "training_seed": {
            "scratch": 23132,
            "v19-adapt": 23133,
            "candidate-remine": 23134,
        }[mode],
        "deterministic_training": True,
        "training_steps": [20000, 10000] if mode == "scratch" else [1000, 500],
        "positive_class_weight": [1, 1],
        "negative_class_weight": [1, 1],
        "learning_rates": (
            [0.001, 0.0002] if mode == "scratch" else [0.000005, 0.000001]
        ),
        "batch_size": 128,
        "eval_step_interval": 1000 if mode == "scratch" else 250,
        "training_loss": {
            "name": "binary_focal_crossentropy",
            "gamma": 2.0,
            "apply_class_balancing": False,
        },
        "require_binary_validation": True,
        "target_minimization": 0,
        "minimization_metric": "validation_false_positives",
        "maximization_metric": "average_viable_recall",
    }
    if initial_weights is not None:
        overrides["initial_weights"] = str(initial_weights)
        overrides["freeze_batch_normalization"] = True

    sources = [
        {
            "feature_build_manifest": str(feature_manifest),
            "class": "positive",
            "feature_split": "training",
            "source_prefix": "v32-canonical-positive",
            "truth": True,
            "group": "canonical_positive",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "truncate_start",
        },
        {
            "features_dir": str(device_features / "positive"),
            "source_name": "promoted-device-canonical-positive",
            "truth": True,
            "group": "device_positive",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "truncate_start",
            "evaluation_enabled": False,
        },
        {
            "feature_build_manifest": str(feature_manifest),
            "class": "hard_negative",
            "feature_split": "training",
            "source_prefix": "v32-collision-negative",
            "truth": False,
            "group": "collision_negative",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
        {
            "features_dir": str(mined_features / "mined"),
            "source_name": (
                "candidate-union-mined-negative"
                if mode == "candidate-remine"
                else "v19-mined-negative"
            ),
            "truth": False,
            "group": "mined_negative",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
        {
            "features_dir": str(mined_features / "random_reserve"),
            "source_name": "deterministic-random-negative",
            "truth": False,
            "group": "random_negative",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
        {
            "features_dir": str(negatives / "speech"),
            "source_name": "ordinary-speech",
            "truth": False,
            "group": "ordinary_speech",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
        {
            "features_dir": str(negatives / "dinner_party"),
            "source_name": "household-speech",
            "truth": False,
            "group": "ordinary_speech",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
        {
            "features_dir": str(negatives / "no_speech"),
            "source_name": "background-only",
            "truth": False,
            "group": "background_negative",
            "within_group_weight": 1,
            "penalty_weight": 1,
            "truncation_strategy": "random",
        },
    ]
    for split in ("validation", "testing"):
        for class_name, truth in (("positive", True), ("hard_negative", False)):
            sources.append(
                {
                    "feature_build_manifest": str(feature_manifest),
                    "class": class_name,
                    "feature_split": split,
                    "source_prefix": f"v32-{split}-{class_name}",
                    "truth": truth,
                    "within_group_weight": 0,
                    "penalty_weight": 1,
                    "truncation_strategy": "split",
                    "evaluation_enabled": True,
                }
            )
    sources.extend(
        [
            {
                "features_dir": str(negatives / "dinner_party_eval"),
                "source_name": "untouched-dinner-party-evaluation",
                "truth": False,
                "within_group_weight": 0,
                "penalty_weight": 1,
                "truncation_strategy": "split",
                "evaluation_enabled": True,
            },
        ]
    )
    return {
        "schema_version": 1,
        "sampling_groups": {
            "canonical_positive": 0.25,
            "device_positive": 0.15,
            "collision_negative": 0.15,
            "mined_negative": 0.15,
            "random_negative": 0.10,
            "ordinary_speech": 0.15,
            "background_negative": 0.05,
        },
        "balance_guard": {
            "maximum_negative_sampling_share": 0.60,
            "maximum_negative_weighted_pressure_share": 0.60,
        },
        "config_overrides": overrides,
        "initialization": (
            {
                "weights": str(initial_weights),
                "weights_sha256": sha256(initial_weights),
            }
            if initial_weights is not None
            else {"weights": None, "weights_sha256": None}
        ),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--mined-features", type=Path, required=True)
    parser.add_argument("--device-features", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("scratch", "v19-adapt", "candidate-remine"),
        required=True,
    )
    parser.add_argument("--initial-weights", type=Path)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    args = parser.parse_args()
    validate_promoted_device_features(args.device_features)
    plan = experiment_plan(
        args.workspace,
        args.feature_manifest,
        args.mined_features,
        args.device_features,
        args.train_dir,
        args.mode,
        args.initial_weights,
    )
    args.output_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_plan.write_text(yaml.safe_dump(plan, sort_keys=False))
    config = stratified_config(args.base_config, args.output_plan)
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(yaml.safe_dump(config, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
