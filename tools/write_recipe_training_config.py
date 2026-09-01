#!/usr/bin/env python3
"""Write the model-training YAML for a prepared Kizz feature workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

DEVICE_TRUNCATION_STRATEGIES = ("random", "truncate_start", "truncate_end")


def feature(
    path: Path,
    weight: float,
    penalty: float,
    truth: bool,
    strategy: str,
    evaluation_enabled: bool = True,
) -> dict:
    result = {
        "features_dir": str(path),
        "sampling_weight": weight,
        "penalty_weight": penalty,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }
    if not evaluation_enabled:
        result["evaluation_enabled"] = False
    return result


def training_config(
    workspace: Path,
    train_dir: Path,
    features_dir: Path | None = None,
    hard_negative_sampling_weight: float = 8.0,
    hard_negative_penalty_weight: float = 4.0,
    device_features_dir: Path | None = None,
    device_only: bool = False,
    device_adaptation: bool = False,
    device_train_only: bool = False,
    device_positive_only: bool = False,
    initial_weights: Path | None = None,
    device_positive_sampling_weight: float = 6.0,
    device_hard_negative_sampling_weight: float = 8.0,
    freeze_feature_extractor: bool = False,
    device_truncation_strategy: str = "random",
    positive_features_dir: Path | None = None,
    hard_negative_features_dir: Path | None = None,
    additional_hard_negative_features_dirs: list[Path] | None = None,
    additional_hard_negative_sampling_weight: float = 2.0,
    positive_sampling_weight: float = 3.0,
) -> dict:
    negatives = workspace / "negative-datasets"
    features = features_dir or workspace / "features"
    positive_features = positive_features_dir or features / "positive"
    hard_negative_features = hard_negative_features_dir or features / "hard_negative"
    additional_hard_negatives = additional_hard_negative_features_dirs or []
    selected_device_modes = sum((device_only, device_adaptation, device_train_only))
    if selected_device_modes > 1:
        raise ValueError("choose one device training mode")
    if selected_device_modes and device_features_dir is None:
        raise ValueError("device training requires --device-features-dir")
    if device_positive_only and not selected_device_modes:
        raise ValueError("--device-positive-only requires a device training mode")
    if device_truncation_strategy not in DEVICE_TRUNCATION_STRATEGIES:
        raise ValueError(
            "device truncation strategy must be one of: "
            + ", ".join(DEVICE_TRUNCATION_STRATEGIES)
        )
    general_evaluation = not device_adaptation
    feature_sources = (
        []
        if device_only
        else [
            feature(
                positive_features,
                positive_sampling_weight,
                1.0,
                True,
                "truncate_start",
                general_evaluation,
            ),
            feature(
                hard_negative_features,
                hard_negative_sampling_weight,
                hard_negative_penalty_weight,
                False,
                "random",
                general_evaluation,
            ),
            feature(
                negatives / "speech", 10.0, 1.5, False, "random", general_evaluation
            ),
            feature(
                negatives / "dinner_party",
                10.0,
                1.5,
                False,
                "random",
                general_evaluation,
            ),
            feature(
                negatives / "no_speech",
                6.0,
                1.0,
                False,
                "random",
                general_evaluation,
            ),
        ]
    )
    if not device_adaptation and not device_only:
        feature_sources.extend(
            [
                feature(hard_negative_features, 0.0, 1.0, False, "split"),
                feature(negatives / "dinner_party_eval", 0.0, 1.0, False, "split"),
            ]
        )
    if not device_only:
        for additional in additional_hard_negatives:
            feature_sources.append(
                feature(
                    additional,
                    additional_hard_negative_sampling_weight,
                    hard_negative_penalty_weight,
                    False,
                    "random",
                    general_evaluation,
                )
            )
            if not device_adaptation:
                feature_sources.append(feature(additional, 0.0, 1.0, False, "split"))
    if device_train_only:
        feature_sources.append(
            feature(
                device_features_dir / "positive",
                device_positive_sampling_weight,
                2.0,
                True,
                device_truncation_strategy,
                False,
            )
        )
        if not device_positive_only:
            feature_sources.append(
                feature(
                    device_features_dir / "hard_negative",
                    device_hard_negative_sampling_weight,
                    4.0,
                    False,
                    device_truncation_strategy,
                    False,
                )
            )
    elif device_features_dir is not None:
        feature_sources.extend(
            [
                feature(
                    device_features_dir / "positive",
                    6.0,
                    2.0,
                    True,
                    device_truncation_strategy,
                ),
                feature(
                    device_features_dir / "hard_negative",
                    8.0,
                    4.0,
                    False,
                    device_truncation_strategy,
                ),
                feature(
                    device_features_dir / "ambient_negative", 4.0, 2.0, False, "split"
                ),
            ]
        )
    if device_only:
        training_steps = [3000, 2000]
        negative_class_weight = [4, 8]
        batch_size = 32
        eval_step_interval = 500
    elif device_train_only and initial_weights is not None:
        training_steps = [1000, 500]
        negative_class_weight = [24, 32]
        batch_size = 128
        eval_step_interval = 500
    elif device_adaptation or device_train_only:
        training_steps = [6000, 4000]
        negative_class_weight = [8, 12]
        batch_size = 64
        eval_step_interval = 500
    else:
        training_steps = [20000, 10000]
        negative_class_weight = [24, 32]
        batch_size = 128
        eval_step_interval = 1000
    config = {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": feature_sources,
        "training_steps": training_steps,
        "positive_class_weight": [1, 1],
        "negative_class_weight": negative_class_weight,
        "learning_rates": (
            [0.00005, 0.00001]
            if device_train_only and initial_weights is not None
            else [0.001, 0.0002]
        ),
        "batch_size": batch_size,
        "time_mask_max_size": [4, 4],
        "time_mask_count": [1, 1],
        "freq_mask_max_size": [3, 3],
        "freq_mask_count": [1, 1],
        "eval_step_interval": eval_step_interval,
        "clip_duration_ms": 2000,
        "target_minimization": 0,
        "minimization_metric": "validation_false_positives",
        "maximization_metric": "average_viable_recall",
    }
    if initial_weights is not None:
        config["initial_weights"] = str(initial_weights)
        config["freeze_batch_normalization"] = True
    if freeze_feature_extractor:
        if initial_weights is None:
            raise ValueError("freezing the feature extractor requires initial weights")
        config["freeze_feature_extractor"] = True
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--positive-features-dir", type=Path)
    parser.add_argument("--hard-negative-features-dir", type=Path)
    parser.add_argument(
        "--additional-hard-negative-features-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--additional-hard-negative-sampling-weight", type=float, default=2.0
    )
    parser.add_argument("--hard-negative-sampling-weight", type=float, default=8.0)
    parser.add_argument("--positive-sampling-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-penalty-weight", type=float, default=4.0)
    parser.add_argument("--device-features-dir", type=Path)
    parser.add_argument(
        "--device-only",
        action="store_true",
        help="Run a bounded device-corpus adaptation pass without general corpora",
    )
    parser.add_argument(
        "--device-adaptation",
        action="store_true",
        help="Train on all sources but select checkpoints only on device evidence",
    )
    parser.add_argument(
        "--device-train-only",
        action="store_true",
        help=(
            "Mix a small device training split into general training while "
            "retaining general-corpus checkpoint selection"
        ),
    )
    parser.add_argument(
        "--device-positive-only",
        action="store_true",
        help=(
            "Use enrolled device positives without claiming a device hard-negative "
            "source; general collision corpora remain active"
        ),
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help="Initialize a new training directory from compatible Keras weights",
    )
    parser.add_argument("--device-positive-sampling-weight", type=float, default=6.0)
    parser.add_argument(
        "--device-hard-negative-sampling-weight", type=float, default=8.0
    )
    parser.add_argument(
        "--freeze-feature-extractor",
        action="store_true",
        help="Fine-tune only the final classifier from compatible initial weights",
    )
    parser.add_argument(
        "--device-truncation-strategy",
        choices=DEVICE_TRUNCATION_STRATEGIES,
        default="random",
        help=(
            "Choose how device recordings longer than the training window are "
            "sampled. Random preserves speech that is not aligned to a clip edge."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        yaml.safe_dump(
            training_config(
                args.workspace,
                args.train_dir,
                args.features_dir,
                args.hard_negative_sampling_weight,
                args.hard_negative_penalty_weight,
                args.device_features_dir,
                args.device_only,
                args.device_adaptation,
                args.device_train_only,
                args.device_positive_only,
                args.initial_weights,
                args.device_positive_sampling_weight,
                args.device_hard_negative_sampling_weight,
                args.freeze_feature_extractor,
                args.device_truncation_strategy,
                args.positive_features_dir,
                args.hard_negative_features_dir,
                args.additional_hard_negative_features_dir,
                args.additional_hard_negative_sampling_weight,
                args.positive_sampling_weight,
            ),
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
