#!/usr/bin/env python3
"""Write the model-training YAML for a prepared Kizz feature workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def feature(
    path: Path, weight: float, penalty: float, truth: bool, strategy: str
) -> dict:
    return {
        "features_dir": str(path),
        "sampling_weight": weight,
        "penalty_weight": penalty,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


def training_config(
    workspace: Path,
    train_dir: Path,
    features_dir: Path | None = None,
    hard_negative_sampling_weight: float = 8.0,
    hard_negative_penalty_weight: float = 4.0,
    device_features_dir: Path | None = None,
) -> dict:
    negatives = workspace / "negative-datasets"
    features = features_dir or workspace / "features"
    feature_sources = [
        feature(features / "positive", 3.0, 1.0, True, "truncate_start"),
        feature(
            features / "hard_negative",
            hard_negative_sampling_weight,
            hard_negative_penalty_weight,
            False,
            "random",
        ),
        feature(negatives / "speech", 10.0, 1.5, False, "random"),
        feature(negatives / "dinner_party", 10.0, 1.5, False, "random"),
        feature(negatives / "no_speech", 6.0, 1.0, False, "random"),
        feature(features / "hard_negative", 0.0, 1.0, False, "split"),
        feature(negatives / "dinner_party_eval", 0.0, 1.0, False, "split"),
    ]
    if device_features_dir is not None:
        feature_sources.extend(
            [
                feature(
                    device_features_dir / "positive", 6.0, 2.0, True, "truncate_start"
                ),
                feature(
                    device_features_dir / "hard_negative", 8.0, 4.0, False, "random"
                ),
                feature(
                    device_features_dir / "ambient_negative", 4.0, 2.0, False, "split"
                ),
                feature(
                    device_features_dir / "hard_negative", 0.0, 1.0, False, "split"
                ),
            ]
        )
    return {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": feature_sources,
        "training_steps": [20000, 10000],
        "positive_class_weight": [1, 1],
        "negative_class_weight": [24, 32],
        "learning_rates": [0.001, 0.0002],
        "batch_size": 128,
        "time_mask_max_size": [4, 4],
        "time_mask_count": [1, 1],
        "freq_mask_max_size": [3, 3],
        "freq_mask_count": [1, 1],
        "eval_step_interval": 1000,
        "clip_duration_ms": 2000,
        "target_minimization": 0.5,
        "minimization_metric": "ambient_false_positives_per_hour",
        "maximization_metric": "average_viable_recall",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path)
    parser.add_argument("--hard-negative-sampling-weight", type=float, default=8.0)
    parser.add_argument("--hard-negative-penalty-weight", type=float, default=4.0)
    parser.add_argument("--device-features-dir", type=Path)
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
            ),
            sort_keys=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
