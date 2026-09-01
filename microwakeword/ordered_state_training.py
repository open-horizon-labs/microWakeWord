"""Aligned frame-state supervision for ordered-state endpoint training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from microwakeword.ordered_state import KIZZ_TOPOLOGY
from microwakeword.ordered_state_model import acoustic_model
from microwakeword.provenance import sha256_file


def validate_expected_file_hashes(directory: Path, expected: dict | None) -> None:
    if expected is None:
        return
    if not isinstance(expected, dict) or not expected:
        raise ValueError("expected_files_sha256 must be a non-empty mapping")
    for filename, expected_hash in sorted(expected.items()):
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"missing hash-bound frame supervision file: {filename}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"frame supervision hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )


class OrderedStateFrameSupervisor:
    """Apply an auxiliary aligned-state update from fixed NumPy arrays."""

    def __init__(self, training_wrapper, optimizer, config):
        directory = Path(config["directory"])
        validate_expected_file_hashes(directory, config.get("expected_files_sha256"))
        self.features = np.load(directory / "features.npy", mmap_mode="r")
        self.targets = np.load(directory / "targets.npy", mmap_mode="r")
        weights_path = directory / "weights.npy"
        self.weights = (
            np.load(weights_path, mmap_mode="r")
            if weights_path.is_file()
            else np.ones(len(self.features), dtype=np.float32)
        )
        self.acoustic_model = acoustic_model(training_wrapper)
        self.optimizer = optimizer
        self.frame_weight = float(config.get("frame_weight", 0.0))
        self.batch_size = int(config.get("batch_size", training_wrapper.input_shape[0]))
        self.rng = np.random.default_rng(int(config.get("seed", 231)))
        self._validate()

    def _validate(self):
        if self.frame_weight <= 0:
            raise ValueError("aligned frame supervision requires positive frame_weight")
        if self.batch_size < 1:
            raise ValueError("aligned frame supervision batch_size must be positive")
        if self.features.ndim != 3:
            raise ValueError("aligned features must have shape [example, time, bin]")
        if tuple(self.features.shape[1:]) != tuple(self.acoustic_model.input_shape[1:]):
            raise ValueError("aligned feature shape does not match the training model")
        if self.targets.ndim != 2 or len(self.targets) != len(self.features):
            raise ValueError("aligned targets must have shape [example, output_time]")
        if self.targets.shape[1] != self.acoustic_model.output_shape[1]:
            raise ValueError(
                "aligned target time dimension does not match model output"
            )
        if self.weights.shape != (len(self.features),):
            raise ValueError("aligned weights must contain one value per example")
        if not len(self.features):
            raise ValueError("aligned frame supervision is empty")
        if np.any(~np.isfinite(self.weights)) or np.any(self.weights < 0):
            raise ValueError("aligned weights must be finite and non-negative")
        if not np.any(self.weights > 0):
            raise ValueError("aligned frame supervision needs a positive weight")
        minimum = int(np.min(self.targets))
        maximum = int(np.max(self.targets))
        if minimum < -1 or maximum >= KIZZ_TOPOLOGY.state_count:
            raise ValueError("aligned targets contain an invalid state index")
        if not np.any(self.targets >= 0):
            raise ValueError("aligned targets contain no supervised frames")

    def train_on_batch(self) -> float:
        indexes = self.rng.choice(
            len(self.features),
            size=self.batch_size,
            replace=len(self.features) < self.batch_size,
        )
        features = np.asarray(self.features[indexes], dtype=np.float32)
        targets = np.asarray(self.targets[indexes], dtype=np.int32)
        example_weights = np.asarray(self.weights[indexes], dtype=np.float32)
        if np.any(~np.isfinite(features)):
            raise ValueError("aligned features contain non-finite values")
        valid = targets >= 0
        safe_targets = np.where(valid, targets, 0)
        with tf.GradientTape() as tape:
            logits = self.acoustic_model(features, training=True)
            losses = tf.keras.losses.sparse_categorical_crossentropy(
                safe_targets, logits, from_logits=True
            )
            frame_weights = tf.cast(valid, losses.dtype) * tf.convert_to_tensor(
                example_weights[:, None], dtype=losses.dtype
            )
            denominator = tf.reduce_sum(frame_weights)
            loss = self.frame_weight * tf.math.divide_no_nan(
                tf.reduce_sum(losses * frame_weights), denominator
            )
        gradients = tape.gradient(loss, self.acoustic_model.trainable_variables)
        updates = [
            (gradient, variable)
            for gradient, variable in zip(
                gradients, self.acoustic_model.trainable_variables
            )
            if gradient is not None
        ]
        if not updates:
            raise ValueError("aligned frame supervision produced no gradients")
        self.optimizer.apply_gradients(updates)
        return float(loss.numpy())
