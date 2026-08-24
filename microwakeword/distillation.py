# coding=utf-8
"""Teacher-logit distillation helpers for the compact Kizz student."""

from __future__ import annotations

import tensorflow as tf

from microwakeword.ordered_state import ordered_state_sequence_score


def teacher_kl_loss(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    temperature: float = 2.0,
) -> tf.Tensor:
    """Return temperature-scaled framewise KL divergence."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    temperature = tf.cast(temperature, student_logits.dtype)
    teacher_probs = tf.nn.softmax(teacher_logits / temperature, axis=-1)
    teacher_log_probs = tf.nn.log_softmax(teacher_logits / temperature, axis=-1)
    student_log_probs = tf.nn.log_softmax(student_logits / temperature, axis=-1)
    kl = tf.reduce_sum(
        teacher_probs * (teacher_log_probs - student_log_probs), axis=-1
    )
    return tf.reduce_mean(kl) * temperature * temperature


def distillation_loss(
    student_logits: tf.Tensor,
    teacher_logits: tf.Tensor,
    hard_targets: tf.Tensor,
    *,
    temperature: float = 2.0,
    hard_weight: float = 0.5,
    teacher_weight: float = 0.5,
    sequence_logits: tf.Tensor | None = None,
    sequence_labels: tf.Tensor | None = None,
    sequence_weight: float = 0.0,
) -> tf.Tensor:
    """Combine hard frame states with teacher soft targets."""
    hard = tf.keras.losses.sparse_categorical_crossentropy(
        hard_targets, student_logits, from_logits=True
    )
    hard = tf.reduce_mean(hard)
    soft = teacher_kl_loss(student_logits, teacher_logits, temperature)
    total = float(hard_weight) * hard + float(teacher_weight) * soft
    if sequence_weight:
        if sequence_labels is None:
            raise ValueError("sequence_labels are required when sequence_weight is set")
        if sequence_logits is None:
            sequence_logits = ordered_state_sequence_score(student_logits)
        endpoint = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=sequence_labels, logits=sequence_logits
            )
        )
        total += float(sequence_weight) * endpoint
    return total


__all__ = ["distillation_loss", "teacher_kl_loss"]
