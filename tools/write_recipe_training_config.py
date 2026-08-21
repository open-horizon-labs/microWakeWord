#!/usr/bin/env python3
"""Write the model-training YAML for a prepared Kizz feature workspace."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def feature(path: Path, weight: float, penalty: float, truth: bool, strategy: str) -> dict:
    return {
        "features_dir": str(path),
        "sampling_weight": weight,
        "penalty_weight": penalty,
        "truth": truth,
        "truncation_strategy": strategy,
        "type": "mmap",
    }


def training_config(workspace: Path, train_dir: Path) -> dict:
    negatives = workspace / "negative-datasets"
    features = workspace / "features"
    return {
        "window_step_ms": 10,
        "train_dir": str(train_dir),
        "features": [
            feature(features / "positive", 3.0, 1.0, True, "truncate_start"),
            feature(features / "hard_negative", 16.0, 8.0, False, "random"),
            feature(negatives / "speech", 10.0, 1.5, False, "random"),
            feature(negatives / "dinner_party", 10.0, 1.5, False, "random"),
            feature(negatives / "no_speech", 6.0, 1.0, False, "random"),
            feature(negatives / "dinner_party_eval", 0.0, 1.0, False, "split"),
        ],
        "training_steps": [20000, 10000],
        "positive_class_weight": [1, 1],
        "negative_class_weight": [24, 32],
        "learning_rates": [0.001, 0.0002],
        "batch_size": 128,
        "time_mask_max_size": [4, 4],
        "time_mask_count": [1, 1],
        "freq_mask_max_size": [3, 3],
        "freq_mask_count": [1, 1],
        "eval_step_interval": 500,
        "clip_duration_ms": 2000,
        "target_minimization": 0.5,
        "minimization_metric": "ambient_false_positives_per_hour",
        "maximization_metric": "average_viable_recall",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        yaml.safe_dump(training_config(args.workspace, args.train_dir), sort_keys=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
