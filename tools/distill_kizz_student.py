#!/usr/bin/env python3
"""Distill a compact causal Kizz student from a frozen teacher-logit cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import tensorflow as tf

from microwakeword.distillation import distillation_loss
from microwakeword.ordered_state_model import model as build_student


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_teacher_qualification(path: Path, teacher_weights: Path) -> dict:
    report = json.loads(path.read_text())
    if not report.get("qualified"):
        raise ValueError("teacher qualification report does not pass its hard gate")
    expected = report.get("model_sha256")
    actual = sha256_file(teacher_weights)
    if expected != actual:
        raise ValueError(
            "teacher qualification report is for different weights: "
            f"expected {expected}, got {actual}"
        )
    return report


def student_flags() -> SimpleNamespace:
    return SimpleNamespace(
        pointwise_filters="96,96,96,96",
        residual_connection="0,0,0,0",
        repeat_in_block="1,1,1,1",
        mixconv_kernel_sizes="[3], [5], [7], [9]",
        first_conv_filters=48,
        first_conv_kernel_size=5,
        stride=3,
        num_states=23,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-prefix", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--teacher-weight", type=float, default=0.5)
    parser.add_argument(
        "--negative-state",
        type=int,
        default=1,
        help="hard frame state used for negative windows (1=silence)",
    )
    parser.add_argument("--sequence-weight", type=float, default=0.0)
    parser.add_argument(
        "--sequence-teacher-weight",
        type=float,
        default=0.0,
        help="match the teacher's ordered-state completion margin directly",
    )
    parser.add_argument(
        "--sequence-every",
        type=int,
        default=10,
        help="apply the slow sequence objective every N batches",
    )
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--seed", type=int, default=24105)
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args(argv)
    if (
        args.steps < 1
        or args.batch_size < 1
        or args.sequence_every < 1
        or args.negative_state not in (0, 1)
        or args.sequence_teacher_weight < 0
    ):
        parser.error("steps, batch-size, and sequence-every must be positive")

    prefix = args.cache_prefix
    cache_metadata = json.loads(prefix.with_suffix(".json").read_text())
    teacher_weights = Path(cache_metadata["teacher_weights"])
    qualification = require_teacher_qualification(args.teacher_qualification, teacher_weights)
    features = np.load(prefix.with_name("features.npy"), mmap_mode="r")
    targets = np.load(prefix.with_name("targets.npy"), mmap_mode="r")
    labels = np.load(prefix.with_name("labels.npy"), mmap_mode="r")
    teacher_logits = np.load(prefix.with_name("teacher_logits.npy"), mmap_mode="r")
    if (
        features.shape[0] != targets.shape[0]
        or features.shape[0] != labels.shape[0]
        or features.shape[0] != teacher_logits.shape[0]
    ):
        parser.error("cache arrays must contain the same number of samples")
    tf.keras.utils.set_random_seed(args.seed)
    student = build_student(student_flags(), (260, 40), None)
    if args.init_weights is not None:
        student.load_weights(args.init_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    best_loss = float("inf")
    losses = []

    def _train_batch(x, hard, soft, sequence_labels, use_sequence: bool):
        with tf.GradientTape() as tape:
            logits = student(x, training=True)
            loss = distillation_loss(
                logits,
                soft,
                hard,
                temperature=args.temperature,
                hard_weight=args.hard_weight,
                teacher_weight=args.teacher_weight,
                sequence_weight=args.sequence_weight if use_sequence else 0.0,
                sequence_teacher_weight=(
                    args.sequence_teacher_weight if use_sequence else 0.0
                ),
                sequence_labels=sequence_labels,
            )
        gradients = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(gradients, student.trainable_variables))
        return loss

    @tf.function
    def train_batch_frame(x, hard, soft, sequence_labels):
        return _train_batch(x, hard, soft, sequence_labels, False)

    @tf.function
    def train_batch_sequence(x, hard, soft, sequence_labels):
        return _train_batch(x, hard, soft, sequence_labels, True)

    args.output.mkdir(parents=True, exist_ok=True)
    for step in range(args.steps):
        indexes = rng.integers(0, len(features), size=args.batch_size)
        hard_targets = np.asarray(targets[indexes], dtype=np.int32).copy()
        negative = np.asarray(labels[indexes]) < 0.5
        hard_targets[negative, :] = args.negative_state
        batch_args = (
            tf.convert_to_tensor(np.asarray(features[indexes], dtype=np.float32)),
            tf.convert_to_tensor(hard_targets),
            tf.convert_to_tensor(np.asarray(teacher_logits[indexes], dtype=np.float32)),
            tf.convert_to_tensor(np.asarray(labels[indexes], dtype=np.float32)),
        )
        if args.sequence_weight and (step + 1) % args.sequence_every == 0:
            loss = train_batch_sequence(*batch_args)
        else:
            loss = train_batch_frame(*batch_args)
        value = float(loss.numpy())
        losses.append(value)
        if value < best_loss:
            best_loss = value
            student.save_weights(args.output / "best.weights.h5")
        if (step + 1) % args.log_interval == 0 or step == 0:
            print(json.dumps({"step": step + 1, "loss": value, "best_loss": best_loss}), flush=True)

    student.save_weights(args.output / "last.weights.h5")
    student.save(args.output / "student.keras")
    metadata = {
        "schema_version": 1,
        "model": "ordered_state_causal_student_distilled",
        "input_shape": [260, 40],
        "output_shape": [66, 23],
        "cache_prefix": str(prefix.resolve()),
        "cache_files_sha256": {
            name: sha256_file(prefix.with_name(name))
            for name in ("features.npy", "targets.npy", "labels.npy", "teacher_logits.npy")
        },
        "teacher_qualification": str(args.teacher_qualification.resolve()),
        "teacher_qualification_sha256": sha256_file(args.teacher_qualification),
        "teacher_qualification_threshold": qualification["operating_point"]["threshold"],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "hard_weight": args.hard_weight,
        "teacher_weight": args.teacher_weight,
        "negative_state": args.negative_state,
        "sequence_weight": args.sequence_weight,
        "sequence_teacher_weight": args.sequence_teacher_weight,
        "sequence_every": args.sequence_every,
        "seed": args.seed,
        "best_loss": best_loss,
        "last_loss": losses[-1],
        "mean_last_100_loss": float(np.mean(losses[-100:])),
    }
    (args.output / "distillation-training.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
