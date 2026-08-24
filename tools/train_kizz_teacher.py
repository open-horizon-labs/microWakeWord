#!/usr/bin/env python3
"""Train a new offline full-context Kizz teacher.

The teacher is deliberately separate from the firmware student. It learns
frame-level Kizz state logits from the training split and writes a provenance
bound checkpoint plus a JSON training report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf

from microwakeword.kizz_teacher import (
    NegativeSource,
    TeacherBatchSequence,
    build_teacher,
    teacher_loss,
)
from microwakeword.kizz_data_contract import sha256_file as balance_sha256_file
from microwakeword.kizz_data_contract import validate_balance_manifest


sha256_file = balance_sha256_file


def parse_source(value: str) -> NegativeSource:
    if "=" not in value:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    source_id, raw_path = value.split("=", 1)
    if not source_id or not raw_path:
        raise argparse.ArgumentTypeError("negative source must be ID=PATH")
    path = Path(raw_path).resolve()
    if not (path.is_dir() or (path.is_file() and path.suffix == ".npy")):
        raise argparse.ArgumentTypeError(
            f"negative source does not exist or is not a .npy cache: {path}"
        )
    return NegativeSource(source_id, path)


def parse_probability(value: str) -> tuple[str, float]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("source probability must be ID=PROBABILITY")
    source_id, raw_probability = value.split("=", 1)
    try:
        probability = float(raw_probability)
    except ValueError as error:
        raise argparse.ArgumentTypeError("source probability must be numeric") from error
    if not source_id or probability < 0:
        raise argparse.ArgumentTypeError("source probability must be non-negative")
    return source_id, probability


def train(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    balance_report = validate_balance_manifest(
        args.balance_manifest,
        args.balance_contract,
    )
    balance_report_path = output / "balance-report.json"
    balance_report_path.write_text(
        json.dumps(balance_report, indent=2, sort_keys=True) + "\n"
    )
    if not balance_report["qualified"]:
        raise ValueError(
            "source-balance contract rejected manifest; see "
            f"{balance_report_path}"
        )
    tf.keras.utils.set_random_seed(args.seed)

    model = build_teacher(hidden_size=args.hidden_size, recurrent_layers=args.recurrent_layers)
    sequence = TeacherBatchSequence(
        args.positive_features,
        args.positive_targets,
        args.negative_source,
        batch_size=args.batch_size,
        seed=args.seed,
        steps_per_epoch=args.steps,
        negative_state=args.negative_state,
        negative_source_weights=[
            args.negative_source_probabilities.get(source.source_id, 1.0)
            for source in args.negative_source
        ],
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    best_loss = float("inf")
    losses = []
    for step in range(args.steps):
        features, batch = sequence[step]
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = teacher_loss(
                logits,
                batch["states"],
                batch["label"],
                frame_weight=args.frame_weight,
                sequence_weight=(
                    args.sequence_weight
                    if (step + 1) % args.sequence_every == 0
                    else 0.0
                ),
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        updates = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        optimizer.apply_gradients(updates)
        loss_value = float(loss.numpy())
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            model.save_weights(output / "best.weights.h5")
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "loss": loss_value,
                        "best_loss": best_loss,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    model.save_weights(output / "last.weights.h5")
    model.save(output / "teacher.keras")
    config = {
        "schema_version": 1,
        "model": "kizz_offline_teacher",
        "input_shape": [260, 40],
        "output_shape": [66, 23],
        "hidden_size": args.hidden_size,
        "recurrent_layers": args.recurrent_layers,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "frame_weight": args.frame_weight,
        "sequence_weight": args.sequence_weight,
        "sequence_every": args.sequence_every,
        "negative_state": args.negative_state,
        "negative_source_probabilities": args.negative_source_probabilities,
        "positive_features": str(args.positive_features.resolve()),
        "positive_features_sha256": sha256_file(args.positive_features),
        "positive_targets": str(args.positive_targets.resolve()),
        "positive_targets_sha256": sha256_file(args.positive_targets),
        "balance_manifest": str(args.balance_manifest.resolve()),
        "balance_manifest_sha256": sha256_file(args.balance_manifest),
        "balance_contract": str(args.balance_contract.resolve()),
        "balance_report": str(balance_report_path),
        "balance_report_sha256": sha256_file(balance_report_path),
        "negative_sources": [
            {"id": source.source_id, "path": str(source.path)}
            for source in args.negative_source
        ],
        "best_loss": best_loss,
        "last_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
    }
    (output / "teacher-training.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-features", type=Path, required=True)
    parser.add_argument("--positive-targets", type=Path, required=True)
    parser.add_argument("--balance-manifest", type=Path, required=True)
    parser.add_argument("--balance-contract", type=Path, required=True)
    parser.add_argument("--negative-source", type=parse_source, action="append", required=True)
    parser.add_argument(
        "--negative-source-probability",
        type=parse_probability,
        action="append",
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--recurrent-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--frame-weight", type=float, default=0.25)
    parser.add_argument("--sequence-weight", type=float, default=0.75)
    parser.add_argument("--sequence-every", type=int, default=10)
    parser.add_argument("--negative-state", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=24103)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.learning_rate <= 0
        or args.frame_weight < 0
        or args.sequence_weight < 0
        or args.sequence_every < 1
    ):
        parser.error("invalid training objective or schedule")
    args.negative_source_probabilities = dict(args.negative_source_probability)
    unknown = set(args.negative_source_probabilities) - {
        source.source_id for source in args.negative_source
    }
    if unknown:
        parser.error(f"probabilities reference unknown sources: {sorted(unknown)}")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
