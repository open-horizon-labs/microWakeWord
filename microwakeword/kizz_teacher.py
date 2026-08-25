# coding=utf-8
# Copyright 2026 Open Horizon Labs.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Offline full-context Kizz teacher and deterministic training batches.

The teacher is intentionally not firmware-compatible. It sees the complete
two-second feature window in both directions and emits soft Kizz state logits.
The only artifact that may cross into the student/firmware path is a hashed
teacher-logit cache or a distilled causal student.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tensorflow as tf
from mmap_ninja.ragged import RaggedMmap

from microwakeword.ordered_state import KIZZ_TOPOLOGY, ordered_state_sequence_score


INPUT_FRAMES = 260
FEATURE_BINS = 40
OUTPUT_FRAMES = 66
FRAME_STRIDE = 3


@tf.keras.utils.register_keras_serializable(package="microwakeword")
class TeacherDownsample(tf.keras.layers.Layer):
    """Pool a fixed input into a configurable 30-ms teacher timeline."""

    def __init__(self, output_frames: int = OUTPUT_FRAMES, **kwargs):
        super().__init__(**kwargs)
        if output_frames < 1:
            raise ValueError("output_frames must be positive")
        self.output_frames = int(output_frames)

    def call(self, inputs):
        batch = tf.shape(inputs)[0]
        required = self.output_frames * FRAME_STRIDE
        padding = tf.maximum(0, required - tf.shape(inputs)[1])
        padded = tf.pad(inputs, [[0, 0], [0, padding], [0, 0]])
        trimmed = padded[:, :required, :]
        reshaped = tf.reshape(
            trimmed,
            [batch, self.output_frames, FRAME_STRIDE, tf.shape(inputs)[-1]],
        )
        return tf.reduce_mean(reshaped, axis=2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_frames, input_shape[2])

    def get_config(self):
        return {**super().get_config(), "output_frames": self.output_frames}


@tf.keras.utils.register_keras_serializable(package="microwakeword")
class TeacherSequenceScore(tf.keras.layers.Layer):
    """Expose the same ordered-state score used by the deployed decoder."""

    def call(self, inputs):
        return ordered_state_sequence_score(inputs)


def build_teacher(
    *,
    input_frames: int = INPUT_FRAMES,
    feature_bins: int = FEATURE_BINS,
    hidden_size: int = 128,
    recurrent_layers: int = 7,
    output_frames: int = OUTPUT_FRAMES,
) -> tf.keras.Model:
    """Build a full-context, high-capacity non-causal teacher."""
    if input_frames != INPUT_FRAMES or feature_bins != FEATURE_BINS:
        raise ValueError("teacher input contract is [260, 40]")
    if hidden_size < 16 or recurrent_layers < 1 or recurrent_layers > 8:
        raise ValueError("teacher capacity is too small")
    if output_frames < 1:
        raise ValueError("teacher output_frames must be positive")

    inputs = tf.keras.Input(shape=(input_frames, feature_bins), name="features")
    net = tf.keras.layers.LayerNormalization(name="feature_norm")(inputs)
    net = tf.keras.layers.Conv1D(
        hidden_size,
        7,
        padding="same",
        activation="swish",
        name="local_encoder",
    )(net)
    # Symmetric dilations make this a full-window teacher while remaining
    # practical on a CPU-only training host. Seven blocks cover the complete
    # 260-frame window; the argument is retained as a capacity knob for tests
    # and ablations.
    dilations = (1, 2, 4, 8, 16, 32, 64, 128)[:recurrent_layers]
    for index, dilation in enumerate(dilations):
        residual = net
        net = tf.keras.layers.Conv1D(
            hidden_size,
            3,
            padding="same",
            dilation_rate=dilation,
            use_bias=False,
            name=f"context_conv_{index}",
        )(net)
        net = tf.keras.layers.LayerNormalization(name=f"context_norm_{index}")(net)
        net = tf.keras.layers.Activation("swish")(net)
        if residual.shape[-1] == net.shape[-1]:
            net = tf.keras.layers.Add(name=f"context_residual_{index}")([net, residual])
    net = tf.keras.layers.Dense(hidden_size, activation="swish", name="state_projection")(net)
    net = TeacherDownsample(
        output_frames=output_frames, name="teacher_30ms_frames"
    )(net)
    logits = tf.keras.layers.Dense(
        KIZZ_TOPOLOGY.state_count,
        name="teacher_state_logits",
    )(net)
    return tf.keras.Model(inputs, logits, name="kizz_offline_teacher")


@dataclass(frozen=True)
class NegativeSource:
    """A training-only RaggedMmap source with an immutable identity."""

    source_id: str
    path: Path


class TeacherBatchSequence(tf.keras.utils.Sequence):
    """Deterministic balanced batches from aligned positives and negatives."""

    def __init__(
        self,
        positive_features: Path,
        positive_targets: Path,
        negative_sources: Sequence[NegativeSource],
        *,
        batch_size: int = 32,
        seed: int = 24103,
        steps_per_epoch: int = 250,
        negative_state: int = 1,
        negative_source_weights: Sequence[float] | None = None,
    ) -> None:
        self.positive_features = np.load(positive_features, mmap_mode="r")
        self.positive_targets = np.load(positive_targets, mmap_mode="r")
        if self.positive_features.shape != (
            len(self.positive_targets),
            INPUT_FRAMES,
            FEATURE_BINS,
        ) or self.positive_targets.ndim != 2:
            raise ValueError(
                "positive arrays must have shapes [N, 260, 40] and [N, output_frames]"
            )
        self.output_frames = int(self.positive_targets.shape[1])
        if not negative_sources:
            raise ValueError("at least one negative source is required")
        self.negative_sources = tuple(negative_sources)
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError("batch_size must be even and at least two")
        if self.steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be positive")
        if negative_state not in (0, 1):
            raise ValueError("negative_state must be background (0) or silence (1)")
        self.negative_state = int(negative_state)
        self._negative_sets = []
        for source in self.negative_sources:
            if source.path.is_file() and source.path.suffix == ".npy":
                values = np.load(source.path, mmap_mode="r")
                if values.ndim != 3 or tuple(values.shape[1:]) != (
                    INPUT_FRAMES,
                    FEATURE_BINS,
                ):
                    raise ValueError(
                        f"fixed negative source {source.path} must have shape [N, 260, 40]"
                    )
                if len(values) == 0:
                    raise ValueError("negative sources must not be empty")
                self._negative_sets.append(values)
            else:
                mmap = RaggedMmap(source.path)
                if len(mmap) == 0:
                    raise ValueError("negative sources must not be empty")
                self._negative_sets.append(mmap)
        if negative_source_weights is None:
            self._negative_probabilities = np.full(
                len(self._negative_sets), 1.0 / len(self._negative_sets)
            )
        else:
            probabilities = np.asarray(negative_source_weights, dtype=np.float64)
            if probabilities.shape != (len(self._negative_sets),):
                raise ValueError("negative_source_weights must match negative sources")
            if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
                raise ValueError(
                    "negative_source_weights must be finite and non-negative"
                )
            total = float(np.sum(probabilities))
            if total <= 0:
                raise ValueError("negative_source_weights must have positive mass")
            self._negative_probabilities = probabilities / total

    def __len__(self):
        return self.steps_per_epoch

    def _rng(self, batch_index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + int(batch_index))

    @staticmethod
    def _negative_window(features: np.ndarray, start: int) -> np.ndarray:
        window = np.asarray(features[start : start + INPUT_FRAMES], dtype=np.float32)
        if window.shape[0] == INPUT_FRAMES:
            return window
        result = np.zeros((INPUT_FRAMES, FEATURE_BINS), dtype=np.float32)
        result[: window.shape[0]] = window
        return result

    def __getitem__(self, batch_index):
        rng = self._rng(batch_index)
        half = self.batch_size // 2
        positive_indices = rng.integers(0, len(self.positive_features), size=half)
        x_positive = np.asarray(self.positive_features[positive_indices], dtype=np.float32)
        y_positive = np.asarray(self.positive_targets[positive_indices], dtype=np.int32)

        x_negative = np.empty((half, INPUT_FRAMES, FEATURE_BINS), dtype=np.float32)
        y_negative = np.full(
            (half, self.output_frames), self.negative_state, dtype=np.int32
        )
        for row in range(half):
            source_index = int(
                rng.choice(len(self._negative_sets), p=self._negative_probabilities)
            )
            source_set = self._negative_sets[source_index]
            item_index = int(rng.integers(0, len(source_set)))
            item = np.asarray(source_set[item_index], dtype=np.float32)
            if item.shape == (INPUT_FRAMES, FEATURE_BINS):
                x_negative[row] = item
            else:
                start = int(rng.integers(0, max(1, len(item))))
                x_negative[row] = self._negative_window(item, start)

        x = np.concatenate([x_positive, x_negative], axis=0)
        y = np.concatenate([y_positive, y_negative], axis=0)
        labels = np.concatenate(
            [np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)]
        )
        order = rng.permutation(self.batch_size)
        return x[order], {"states": y[order], "label": labels[order]}


def teacher_loss(
    state_logits: tf.Tensor,
    targets: tf.Tensor,
    labels: tf.Tensor,
    *,
    frame_weight: float = 1.0,
    sequence_weight: float = 0.0,
) -> tf.Tensor:
    """Combine positive state fit, rejection-set fit, and sequence score.

    Negative frames are not forced into one arbitrary rejection class:
    background and silence are jointly valid rejection evidence.
    """
    positive_frame_loss = tf.keras.losses.sparse_categorical_crossentropy(
        targets, state_logits, from_logits=True
    )
    log_probs = tf.nn.log_softmax(state_logits, axis=-1)
    rejection_frame_loss = -tf.reduce_logsumexp(log_probs[:, :, :2], axis=-1)
    positive = tf.cast(tf.reshape(labels, [-1, 1]), positive_frame_loss.dtype)
    frame_loss = tf.where(positive > 0.5, positive_frame_loss, rejection_frame_loss)
    frame_loss = tf.reduce_mean(frame_loss)
    total = float(frame_weight) * frame_loss
    if sequence_weight:
        sequence_logits = ordered_state_sequence_score(state_logits)
        sequence_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels, logits=sequence_logits
            )
        )
        total += float(sequence_weight) * sequence_loss
    return total


__all__ = [
    "FEATURE_BINS",
    "FRAME_STRIDE",
    "INPUT_FRAMES",
    "NegativeSource",
    "OUTPUT_FRAMES",
    "TeacherBatchSequence",
    "build_teacher",
    "teacher_loss",
]
