#!/usr/bin/env python3
"""Distill teacher-ranked wake decisions into a one-logit causal student.

This is deliberately separate from the compact CTC trainer.  The qualified
teacher supervises the ordering of complete causal windows, while hard labels
anchor the decision boundary.  Deployment uses a two-frame rolling mean,
which makes a single anomalous frame insufficient to wake the device.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microwakeword.kizz_feature_archive import open_feature_archive
from microwakeword.kizz_phoneme_teacher import choose_validation_threshold
from microwakeword.ordered_state_model import SqueezeFrequency, model as build_student
from microwakeword.phoneme_student import compact_phone_contract
from tools.distill_kizz_phoneme_student import (
    INPUT_SHAPE,
    OUTPUT_FRAMES,
    device_validation_features,
    load_causal_decision_cache,
    require_teacher_gates,
    sha256_file,
    student_flags_for_architecture,
)

ROLLING_FRAMES = 2
AUXILIARY_MARGIN_WEIGHT = 0.5


def deployment_frame_scores(logits):
    """Combine wake and phonetic rejection evidence from one student."""
    values = tf.convert_to_tensor(logits)
    if values.shape.rank != 3:
        raise ValueError("student logits must be [batch,time,channels]")
    if values.shape[-1] == 1:
        return tf.squeeze(values, axis=-1)
    if values.shape[-1] != 4:
        raise ValueError("deployed decision student must emit one or four channels")
    wake = values[:, :, 0]
    auxiliary = values[:, :, 1:]
    phonetic_margin = auxiliary[:, :, 0] - tf.reduce_max(auxiliary[:, :, 1:], axis=-1)
    return wake + AUXILIARY_MARGIN_WEIGHT * phonetic_margin


def rolling_mean_scores(logits, frames: int = ROLLING_FRAMES):
    """Return one robust clip score from causal per-frame logits."""
    values = tf.convert_to_tensor(logits)
    if values.shape.rank == 3:
        values = deployment_frame_scores(values)
    if values.shape.rank != 2 or frames < 1:
        raise ValueError("logits must be [batch,time] and frames must be positive")
    if values.shape[1] is not None and frames > int(values.shape[1]):
        raise ValueError("rolling window exceeds output frames")
    windows = tf.signal.frame(values, frames, 1, axis=1)
    return tf.reduce_max(tf.reduce_mean(windows, axis=-1), axis=1)


def ranknet_loss(student_scores, teacher_scores, teacher_mask, min_delta=0.05):
    """Transfer teacher ordering without requiring matching score calibration."""
    student = tf.convert_to_tensor(student_scores)
    teacher = tf.cast(teacher_scores, student.dtype)
    mask = tf.cast(teacher_mask, tf.bool)
    teacher_delta = teacher[:, None] - teacher[None, :]
    student_delta = student[:, None] - student[None, :]
    comparable = mask[:, None] & mask[None, :]
    comparable &= tf.abs(teacher_delta) >= tf.cast(min_delta, student.dtype)
    # Count each unordered pair once.  Weight meaningful teacher gaps more.
    comparable &= tf.linalg.band_part(tf.ones_like(comparable), 0, -1)
    comparable &= ~tf.eye(tf.shape(student)[0], dtype=tf.bool)
    weights = tf.minimum(tf.abs(teacher_delta), 1.0)
    losses = tf.nn.softplus(-tf.sign(teacher_delta) * student_delta) * weights
    selected = tf.boolean_mask(losses, comparable)
    return tf.math.divide_no_nan(
        tf.reduce_sum(selected), tf.cast(tf.size(selected), student.dtype)
    )


def ranked_decision_loss(
    logits,
    labels,
    teacher_frame_scores,
    teacher_mask,
    *,
    hard_weight: float,
    teacher_rank_weight: float,
    negative_frame_weight: float,
    margin_weight: float,
    margin: float,
):
    """Optimize the exact rolling deployment score and teacher ordering."""
    frame_logits = deployment_frame_scores(logits)
    clip_scores = rolling_mean_scores(frame_logits)
    labels = tf.cast(labels, clip_scores.dtype)
    hard = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=clip_scores)
    )
    teacher_clip_scores = tf.reduce_max(teacher_frame_scores, axis=1)
    ranked = ranknet_loss(clip_scores, teacher_clip_scores, teacher_mask)
    negatives = labels < 0.5
    negative_frames = tf.boolean_mask(frame_logits, negatives)
    frame_suppression = tf.reduce_mean(tf.nn.softplus(negative_frames))
    positives = tf.boolean_mask(clip_scores, labels > 0.5)
    negative_clips = tf.boolean_mask(clip_scores, negatives)
    tail_margin = tf.reduce_mean(
        tf.nn.softplus(negative_clips[:, None] - positives[None, :] + margin)
    )
    total = (
        hard_weight * hard
        + teacher_rank_weight * ranked
        + negative_frame_weight * frame_suppression
        + margin_weight * tail_margin
    )
    return total, (hard, ranked, frame_suppression, tail_margin)


def score_features(model, features: np.ndarray, batch_size: int) -> np.ndarray:
    scores = []
    for start in range(0, len(features), batch_size):
        logits = model(
            np.asarray(features[start : start + batch_size], np.float32), training=False
        )
        scores.extend(rolling_mean_scores(logits).numpy().tolist())
    return np.asarray(scores, dtype=np.float64)


def _sha(path: Path) -> str:
    return sha256_file(path)


class RankedBatcher:
    """Balanced deterministic sampling with bounded hard-negative pressure."""

    def __init__(
        self,
        features,
        rows,
        causal_scores,
        overlay_positives,
        expanded_negatives,
        critical_collision_indexes,
        student_hard_negative_indexes,
        student_hard_expanded_indexes,
        noise_sources,
        *,
        batch_size,
        seed,
    ):
        self.features = features
        self.rows = rows
        self.causal_scores = causal_scores
        self.overlay_positive = overlay_positives
        self.expanded = expanded_negatives
        self.critical_collision = critical_collision_indexes
        self.student_hard_negative = student_hard_negative_indexes
        self.student_hard_expanded = student_hard_expanded_indexes
        self.critical_collision_set = set(
            int(index) for index in critical_collision_indexes
        )
        self.noise = [open_feature_archive(path) for path in noise_sources]
        self.batch_size = batch_size
        self.seed = seed
        train = [i for i, row in enumerate(rows) if row["split"] == "train"]
        self.clean_positive = np.asarray(
            [
                i
                for i in train
                if rows[i]["label"] == 1
                and rows[i].get("source_group") != "device_channel_positive"
            ]
        )
        self.device_positive = np.asarray(
            [
                i
                for i in train
                if rows[i]["label"] == 1
                and rows[i].get("source_group") == "device_channel_positive"
            ]
        )
        self.negative = np.asarray([i for i in train if rows[i]["label"] == 0])
        if (
            not len(self.clean_positive)
            or not len(self.device_positive)
            or not len(self.overlay_positive)
            or not len(self.negative)
        ):
            raise ValueError("training needs clean, device, overlay, and negative rows")
        self.expanded_order = np.random.default_rng(seed + 1).permutation(
            len(self.expanded)
        )

    @staticmethod
    def _window(values, rng):
        values = np.asarray(values, np.float32)
        if len(values) >= INPUT_SHAPE[0]:
            start = int(rng.integers(0, len(values) - INPUT_SHAPE[0] + 1))
            return values[start : start + INPUT_SHAPE[0]]
        result = np.zeros(INPUT_SHAPE, np.float32)
        result[: len(values)] = values
        return result

    def batch(self, step):
        rng = np.random.default_rng(self.seed + step)
        half = self.batch_size // 2
        x = np.zeros((self.batch_size,) + INPUT_SHAPE, np.float32)
        labels = np.zeros(self.batch_size, np.float32)
        teacher = np.zeros((self.batch_size, OUTPUT_FRAMES), np.float32)
        teacher_mask = np.zeros(self.batch_size, np.float32)
        auxiliary_labels = np.full(self.batch_size, 2, np.int32)
        for slot in range(half):
            labels[slot] = 1
            auxiliary_labels[slot] = 0
            variant = (step * half + slot) % 3
            if variant == 2:
                x[slot] = self.overlay_positive[
                    int(rng.integers(0, len(self.overlay_positive)))
                ]
            else:
                pool = self.device_positive if variant == 1 else self.clean_positive
                index = int(rng.choice(pool))
                x[slot] = self.features[index]
                teacher[slot] = self.causal_scores[index]
                teacher_mask[slot] = 1
        for offset in range(half):
            slot = half + offset
            mode = (step * half + offset) % 8
            if mode in (0, 1, 2):
                pool = (
                    self.critical_collision
                    if mode == 0
                    else self.student_hard_negative if mode == 1 else self.negative
                )
                index = int(rng.choice(pool))
                x[slot] = self.features[index]
                if index in self.critical_collision_set:
                    auxiliary_labels[slot] = 1
                teacher[slot] = self.causal_scores[index]
                teacher_mask[slot] = 1
            elif mode == 3:
                index = int(rng.choice(self.student_hard_expanded))
                x[slot] = self.expanded[index]
            elif mode == 4:
                index = int(
                    self.expanded_order[
                        (step * half + offset) % len(self.expanded_order)
                    ]
                )
                x[slot] = self.expanded[index]
            else:
                source = self.noise[(mode - 5) % len(self.noise)]
                x[slot] = self._window(source[int(rng.integers(0, len(source)))], rng)
        order = rng.permutation(self.batch_size)
        return (
            x[order],
            labels[order],
            teacher[order],
            teacher_mask[order],
            auxiliary_labels[order],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--teacher-causal-window-cache", type=Path, required=True)
    parser.add_argument("--overlay-positive-features", type=Path, required=True)
    parser.add_argument("--overlay-provenance", type=Path, required=True)
    parser.add_argument("--expanded-public-negatives", type=Path, required=True)
    parser.add_argument("--teacher-qualification", type=Path, required=True)
    parser.add_argument("--continuous-qualification", type=Path, required=True)
    parser.add_argument("--device-validation-quality-report", type=Path, required=True)
    parser.add_argument("--noise-source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-weights", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=24108)
    parser.add_argument("--hard-weight", type=float, default=0.5)
    parser.add_argument("--teacher-rank-weight", type=float, default=1.0)
    parser.add_argument("--negative-frame-weight", type=float, default=0.15)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    args = parser.parse_args()
    if args.batch_size < 8 or args.batch_size % 2 or args.steps < 1:
        parser.error("batch size must be even and >=8; steps must be positive")
    require_teacher_gates(args.teacher_qualification, args.continuous_qualification)
    corpus_path = args.corpus / "corpus.json"
    corpus = json.loads(corpus_path.read_text())
    rows = corpus["examples"]
    features = np.load(args.corpus / "features.npy", mmap_mode="r")
    _, cache = load_causal_decision_cache(
        args.teacher_causal_window_cache,
        representation="qualified_teacher_causal_student_endpoint_decisions",
        corpus_json=corpus_path,
        contract=compact_phone_contract(),
        expected_examples=len(rows),
    )
    causal_scores = np.asarray(cache["decision_score"], np.float32)
    if causal_scores.shape != (len(rows), OUTPUT_FRAMES):
        raise ValueError("teacher causal cache differs from corpus")
    expanded_meta = json.loads(
        (args.expanded_public_negatives / "metadata.json").read_text()
    )
    expanded = np.load(args.expanded_public_negatives / "features.npy", mmap_mode="r")
    if len(expanded) != int(expanded_meta["count"]):
        raise ValueError("expanded-negative metadata differs")
    overlays = np.load(args.overlay_positive_features, mmap_mode="r")
    if overlays.ndim != 3 or tuple(overlays.shape[1:]) != INPUT_SHAPE:
        raise ValueError("overlay-positive feature geometry differs")
    overlay_provenance = json.loads(args.overlay_provenance.read_text())
    train_overlay_rows = [
        row for row in overlay_provenance["examples"] if row["split"] == "train"
    ]
    if len(train_overlay_rows) != len(overlays):
        raise ValueError("overlay-positive feature order differs from provenance")
    overlay_indexes = [
        i for i, row in enumerate(train_overlay_rows) if row.get("variant") != "clean"
    ]
    overlays = overlays[overlay_indexes]
    device_features, device_rows = device_validation_features(
        args.device_validation_quality_report
    )
    validation_positive_idx = [
        i
        for i, row in enumerate(rows)
        if row["split"] == "validation" and row["label"] == 1
    ]
    validation_negative_idx = [
        i
        for i, row in enumerate(rows)
        if row["split"] == "validation" and row["label"] == 0
    ]
    validation_positive = np.asarray(features[validation_positive_idx], np.float32)
    validation_negative = np.asarray(features[validation_negative_idx], np.float32)
    validation_seconds = sum(
        float(rows[i]["duration_seconds"]) for i in validation_negative_idx
    )
    tf.keras.utils.set_random_seed(args.seed)
    flags = student_flags_for_architecture("dilated_temporal_memory", 1)
    flags.allow_scalar_output = True
    model = build_student(flags, INPUT_SHAPE, None)
    if args.init_weights:
        model.load_weights(args.init_weights)
    train_negative_indexes = np.asarray(
        [
            i
            for i, row in enumerate(rows)
            if row["split"] == "train" and row["label"] == 0
        ],
        dtype=np.int64,
    )
    source_manifest = json.loads(
        Path(corpus["manifests"]["source"]["path"]).read_text()
    )
    source_by_id = {row["source_id"]: row for row in source_manifest["examples"]}
    critical_collision_indexes = np.asarray(
        [
            i
            for i in train_negative_indexes
            if source_by_id.get(rows[int(i)].get("parent_source_id"), {}).get(
                "render_text"
            )
            in {"Kizz patrol", "His control"}
        ],
        dtype=np.int64,
    )
    if not len(critical_collision_indexes):
        raise ValueError("critical collision training pool is empty")
    if args.init_weights:
        corpus_negative_scores = score_features(
            model, features[train_negative_indexes], args.batch_size
        )
        corpus_order = np.argsort(corpus_negative_scores)[::-1]
        hard_negative_indexes = train_negative_indexes[
            corpus_order[: max(1, math.ceil(len(corpus_order) / 4))]
        ]
        expanded_scores = score_features(model, expanded, args.batch_size)
        expanded_order = np.argsort(expanded_scores)[::-1]
        hard_expanded_indexes = expanded_order[
            : max(1, math.ceil(len(expanded_order) * 0.05))
        ]
    else:
        teacher_clip = np.max(causal_scores, axis=1)
        corpus_order = np.argsort(teacher_clip[train_negative_indexes])[::-1]
        hard_negative_indexes = train_negative_indexes[
            corpus_order[: max(1, math.ceil(len(corpus_order) / 4))]
        ]
        hard_expanded_indexes = np.arange(len(expanded), dtype=np.int64)
    auxiliary_logits = tf.keras.layers.Conv2D(
        3, 1, padding="same", name="phonetic_rejection_logits"
    )(model.get_layer("encoder_hidden").output)
    auxiliary_logits = SqueezeFrequency(name="phonetic_rejection_sequence")(
        auxiliary_logits
    )
    combined_logits = tf.keras.layers.Concatenate(
        axis=-1, name="wake_and_phonetic_logits"
    )([model.output, auxiliary_logits])
    model = tf.keras.Model(model.input, combined_logits, name="ranked_decision_student")
    batcher = RankedBatcher(
        features,
        rows,
        causal_scores,
        overlays,
        expanded,
        critical_collision_indexes,
        hard_negative_indexes,
        hard_expanded_indexes,
        args.noise_source,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    @tf.function
    def train_batch(x, labels, teacher, teacher_mask, auxiliary_labels):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss, parts = ranked_decision_loss(
                logits,
                labels,
                teacher,
                teacher_mask,
                hard_weight=args.hard_weight,
                teacher_rank_weight=args.teacher_rank_weight,
                negative_frame_weight=args.negative_frame_weight,
                margin_weight=args.margin_weight,
                margin=args.margin,
            )
            auxiliary_clip_logits = tf.reduce_mean(logits[:, :, 1:], axis=1)
            auxiliary_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    auxiliary_labels, auxiliary_clip_logits, from_logits=True
                )
            )
            loss = loss + args.auxiliary_weight * auxiliary_loss
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, (*parts, auxiliary_loss)

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    ledger = []
    best_key = (-1, -1.0, -1, -1.0)
    best_step = None
    for step in range(args.steps):
        loss, parts = train_batch(
            *(tf.convert_to_tensor(v) for v in batcher.batch(step))
        )
        if step == 0 or (step + 1) % args.eval_every == 0:
            positive_scores = score_features(
                model, validation_positive, args.batch_size
            )
            negative_scores = score_features(
                model, validation_negative, args.batch_size
            )
            point = choose_validation_threshold(
                positive_scores,
                negative_scores,
                negative_exposure_seconds=validation_seconds,
                min_recall=0.90,
                max_faph=0.10,
            )
            negative_ceiling = float(np.max(negative_scores))
            zero_fp_recall = float(np.mean(positive_scores > negative_ceiling))
            device_scores = score_features(model, device_features, args.batch_size)
            threshold = point.get("threshold")
            device_accepted = (
                int(np.sum(device_scores > float(threshold)))
                if threshold is not None
                else 0
            )
            device_zero = int(np.sum(device_scores > negative_ceiling))
            qualified = bool(point.get("qualified")) and device_accepted >= 10
            item = {
                "step": step + 1,
                "loss": float(loss),
                "parts": [float(v) for v in parts],
                "operating_point": point,
                "zero_false_accept_recall": zero_fp_recall,
                "device_validation": {
                    "accepted_at_clean_operating_point": device_accepted,
                    "zero_false_accept_accepted": device_zero,
                    "required": 10,
                    "total": len(device_rows),
                },
            }
            path = checkpoints / f"step-{step + 1:04d}.weights.h5"
            model.save_weights(path)
            item["checkpoint"] = {"path": str(path.resolve()), "sha256": _sha(path)}
            ledger.append(item)
            key = (
                int(qualified),
                zero_fp_recall,
                device_zero,
                -int(point.get("false_accepts_at_recall_floor", 10**9)),
            )
            if key > best_key:
                best_key, best_step = key, step + 1
                model.save_weights(args.output / "best.weights.h5")
            print(json.dumps(item), flush=True)
    model.save_weights(args.output / "last.weights.h5")
    report = {
        "schema_version": 1,
        "recipe": "kizz_control_teacher_ranked_decision_v1",
        "deployment_score": {
            "type": "max_rolling_mean",
            "frames": ROLLING_FRAMES,
        },
        "parameter_count": model.count_params(),
        "best_step": best_step,
        "best_key": list(best_key),
        "ledger": ledger,
        "loss_contract": {
            "hard_weight": args.hard_weight,
            "teacher_rank_weight": args.teacher_rank_weight,
            "negative_frame_weight": args.negative_frame_weight,
            "margin_weight": args.margin_weight,
            "margin": args.margin,
            "auxiliary_weight": args.auxiliary_weight,
            "auxiliary_task": "deployed_wake_vs_critical_collision_vs_other",
            "auxiliary_margin_weight": AUXILIARY_MARGIN_WEIGHT,
        },
        "student_hard_mining": {
            "critical_collision_count": len(critical_collision_indexes),
            "critical_collision_texts": ["His control", "Kizz patrol"],
            "corpus_negative_count": len(hard_negative_indexes),
            "expanded_public_count": len(hard_expanded_indexes),
            "expanded_public_fraction": 0.05 if args.init_weights else 1.0,
        },
        "bindings": {
            "corpus": {"path": str(corpus_path.resolve()), "sha256": _sha(corpus_path)},
            "teacher_causal_window_cache": {
                "path": str(args.teacher_causal_window_cache.resolve()),
                "json_sha256": _sha(
                    args.teacher_causal_window_cache.with_suffix(".json")
                ),
                "npz_sha256": _sha(
                    args.teacher_causal_window_cache.with_suffix(".npz")
                ),
            },
            "teacher_qualification": {
                "path": str(args.teacher_qualification.resolve()),
                "sha256": _sha(args.teacher_qualification),
            },
            "continuous_qualification": {
                "path": str(args.continuous_qualification.resolve()),
                "sha256": _sha(args.continuous_qualification),
            },
            "device_validation": {
                "path": str(args.device_validation_quality_report.resolve()),
                "sha256": _sha(args.device_validation_quality_report),
            },
            "expanded_public_negatives": {
                "path": str(args.expanded_public_negatives.resolve()),
                "metadata_sha256": _sha(
                    args.expanded_public_negatives / "metadata.json"
                ),
                "features_sha256": _sha(
                    args.expanded_public_negatives / "features.npy"
                ),
            },
            "overlay_positive_features": {
                "path": str(args.overlay_positive_features.resolve()),
                "sha256": _sha(args.overlay_positive_features),
                "provenance_path": str(args.overlay_provenance.resolve()),
                "provenance_sha256": _sha(args.overlay_provenance),
                "selected_overlay_count": len(overlays),
            },
            "initialization": (
                {
                    "path": str(args.init_weights.resolve()),
                    "sha256": _sha(args.init_weights),
                }
                if args.init_weights
                else None
            ),
        },
    }
    (args.output / "distillation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
