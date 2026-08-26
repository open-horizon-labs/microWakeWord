# coding=utf-8
# Copyright 2026 Open Horizon Labs.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Causal temporal encoder for an ordered-state wake-word decoder.

Unlike the binary MixedNet head, this model emits one vector of unnormalised
state logits at every inference step. Phrase-level temporal integration belongs
to :mod:`microwakeword.ordered_state`, not to another neural network.
"""

import ast
import json
from pathlib import Path

import tensorflow as tf

from microwakeword.layers import stream, strided_drop
from microwakeword.mixednet import MixConv
from microwakeword.ordered_state import KIZZ_TOPOLOGY, ordered_state_sequence_score

DEFAULT_STATE_COUNT = KIZZ_TOPOLOGY.state_count


def parse(text):
    """Parse comma/list architecture arguments using Python literal syntax."""
    if not text:
        return []
    result = ast.literal_eval(text)
    return result if isinstance(result, tuple) else [result]


def model_parameters(parser_nn):
    """Register the bounded ordered-state causal-convolution architecture."""
    parser_nn.add_argument("--pointwise_filters", default="96,96,96,96")
    parser_nn.add_argument("--residual_connection", default="0,0,0,0")
    parser_nn.add_argument("--repeat_in_block", default="1,1,1,1")
    parser_nn.add_argument(
        "--mixconv_kernel_sizes",
        default="[3], [5], [7], [9]",
        help="Local-context depthwise kernels for each causal block",
    )
    parser_nn.add_argument("--first_conv_filters", type=int, default=48)
    parser_nn.add_argument("--first_conv_kernel_size", type=int, default=5)
    parser_nn.add_argument("--stride", type=int, default=3)
    parser_nn.add_argument("--num_states", type=int, default=DEFAULT_STATE_COUNT)


def spectrogram_slices_dropped(flags):
    """Return input slices consumed by valid temporal convolutions."""
    if getattr(flags, "causal_memory", False):
        return int(getattr(flags, "warmup_output_drop", 0)) * int(flags.stride) + (
            int(flags.stride) - 1
        )
    dropped = 0
    if flags.first_conv_filters > 0:
        dropped += flags.first_conv_kernel_size - 1
    for repeat, kernel_sizes in zip(
        parse(flags.repeat_in_block), parse(flags.mixconv_kernel_sizes)
    ):
        dropped += repeat * (max(kernel_sizes) - 1) * flags.stride
    return dropped


def training_spectrogram_length(flags, output_frames):
    """Return the shortest input that emits ``output_frames`` state vectors.

    The first temporal convolution consumes feature frames with ``flags.stride``.
    The legacy scalar MixedNet length formula only adds the valid-convolution
    context and therefore under-counts the input whenever stride is greater than
    one.  Ordered-state training needs the declared output timeline itself: 66
    frames for a two-second sequence at the fixed 30 ms cadence.
    """
    output_frames = int(output_frames)
    if output_frames < 1:
        raise ValueError("ordered-state training requires at least one output frame")
    if getattr(flags, "causal_memory", False):
        emitted = output_frames + int(getattr(flags, "warmup_output_drop", 0))
        return (emitted - 1) * int(flags.stride) + int(flags.first_conv_kernel_size)
    receptive_field_frames = spectrogram_slices_dropped(flags) + 1
    return (output_frames - 1) * int(flags.stride) + receptive_field_frames


@tf.keras.utils.register_keras_serializable(package="microwakeword")
class SqueezeFrequency(tf.keras.layers.Layer):
    """Remove the singleton spatial-frequency axis without fixing time length."""

    def call(self, inputs):
        return tf.squeeze(inputs, axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + input_shape[3:]


@tf.keras.utils.register_keras_serializable(package="microwakeword")
class OrderedStateSequenceHead(tf.keras.layers.Layer):
    """Turn state logits into the deployed Viterbi completion logit."""

    def __init__(
        self,
        state_evidence_floor=None,
        self_loop_probability=0.6,
        next_state_probability=0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.state_evidence_floor = state_evidence_floor
        self.self_loop_probability = float(self_loop_probability)
        self.next_state_probability = float(next_state_probability)

    def call(self, inputs):
        score = ordered_state_sequence_score(
            inputs,
            state_evidence_floor=self.state_evidence_floor,
            self_loop_probability=self.self_loop_probability,
            next_state_probability=self.next_state_probability,
        )
        return tf.expand_dims(score, axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "state_evidence_floor": self.state_evidence_floor,
            "self_loop_probability": self.self_loop_probability,
            "next_state_probability": self.next_state_probability,
        }


def model(flags, shape, batch_size):
    """Build a streaming-convertible per-timestep state-logit model.

    The default maximum receptive field is 670 ms (including the 30 ms
    frontend window), substantially narrower than v19's phrase-level context.
    The external ordered-state decoder owns phrase duration and ordering.
    """
    pointwise_filters = parse(flags.pointwise_filters)
    repeat_in_block = parse(flags.repeat_in_block)
    mixconv_kernel_sizes = parse(flags.mixconv_kernel_sizes)
    residual_connections = parse(flags.residual_connection)
    causal_memory = bool(getattr(flags, "causal_memory", False))
    temporal_dilations = parse(getattr(flags, "temporal_dilations", ""))
    lengths = {
        len(pointwise_filters),
        len(repeat_in_block),
        len(mixconv_kernel_sizes),
        len(residual_connections),
    }
    if len(lengths) != 1:
        raise ValueError("all ordered-state block parameter lists must match")
    if causal_memory and len(temporal_dilations) != len(pointwise_filters):
        raise ValueError("causal-memory dilations must match temporal blocks")
    if flags.num_states < 3:
        raise ValueError("ordered-state model needs background, silence, and a path")

    input_audio = tf.keras.layers.Input(shape=shape, batch_size=batch_size)
    net = tf.keras.ops.expand_dims(input_audio, axis=2)

    if flags.first_conv_filters > 0:
        net = stream.Stream(
            cell=tf.keras.layers.Conv2D(
                flags.first_conv_filters,
                (flags.first_conv_kernel_size, 1),
                strides=(flags.stride, 1),
                padding="valid",
                use_bias=False,
            ),
            use_one_step=False,
            pad_time_dim="causal" if causal_memory else None,
            pad_freq_dim="valid",
        )(net)
        net = tf.keras.layers.Activation("relu")(net)

    blocks = list(
        zip(
            pointwise_filters,
            repeat_in_block,
            mixconv_kernel_sizes,
            residual_connections,
        )
    )
    for block_index, (filters, repeat, kernel_sizes, residual_enabled) in enumerate(
        blocks
    ):
        if residual_enabled:
            residual = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            residual = tf.keras.layers.BatchNormalization()(residual)

        for repeat_index in range(repeat):
            if causal_memory:
                net = stream.Stream(
                    cell=tf.keras.layers.DepthwiseConv2D(
                        kernel_size=(3, 1),
                        dilation_rate=(int(temporal_dilations[block_index]), 1),
                        padding="valid",
                        use_bias=False,
                    ),
                    use_one_step=True,
                    pad_time_dim="causal",
                    pad_freq_dim="valid",
                )(net)
            elif max(kernel_sizes) > 1:
                net = MixConv(kernel_size=kernel_sizes)(net)
            net = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            net = tf.keras.layers.BatchNormalization()(net)
            if residual_enabled:
                residual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(
                    residual
                )
                net = net + residual
            activation_name = (
                "encoder_hidden"
                if not causal_memory
                and block_index == len(blocks) - 1
                and repeat_index == repeat - 1
                else None
            )
            net = tf.keras.layers.Activation("relu", name=activation_name)(net)

    if causal_memory:
        warmup = int(getattr(flags, "warmup_output_drop", 0))
        if warmup < 0 or net.shape[1] is None or warmup >= int(net.shape[1]):
            raise ValueError("invalid causal-memory warm-up output drop")
        net = strided_drop.StridedDrop(warmup, name="causal_memory_warmup_drop")(net)
        net = tf.keras.layers.Activation("linear", name="encoder_hidden")(net)

    logits = tf.keras.layers.Conv2D(
        filters=flags.num_states,
        kernel_size=1,
        padding="same",
        use_bias=True,
        name="state_logits",
    )(net)
    logits = SqueezeFrequency(name="state_logits_sequence")(logits)
    return tf.keras.Model(input_audio, logits, name="ordered_state_mixednet")


def training_model(acoustic_model, loss_config=None):
    """Wrap state logits with the differentiable deployed sequence score."""
    loss_config = loss_config or {}
    probability = OrderedStateSequenceHead(
        state_evidence_floor=loss_config.get("state_evidence_floor"),
        self_loop_probability=float(loss_config.get("self_loop_probability", 0.6)),
        next_state_probability=float(loss_config.get("next_state_probability", 0.4)),
        name="ordered_state_sequence_head",
    )(acoustic_model.output)
    return tf.keras.Model(
        acoustic_model.input,
        probability,
        name="ordered_state_training_model",
    )


def acoustic_model(training_wrapper):
    """Extract per-frame state logits from a loaded training wrapper."""
    return tf.keras.Model(
        training_wrapper.input,
        training_wrapper.get_layer("state_logits_sequence").output,
        name="ordered_state_mixednet",
    )


def decoder_contract(loss_config, frame_step_seconds):
    """Return the score settings that training and evaluation must share."""
    loss_config = loss_config or {}
    frame_step_seconds = float(frame_step_seconds)
    if frame_step_seconds <= 0:
        raise ValueError("frame_step_seconds must be positive")
    return {
        "schema_version": 1,
        "state_count": KIZZ_TOPOLOGY.state_count,
        "frame_step_seconds": frame_step_seconds,
        "decoder_args": {
            "from_logits": True,
            "state_evidence_floor": loss_config.get("state_evidence_floor"),
            "self_loop_probability": float(
                loss_config.get("self_loop_probability", 0.6)
            ),
            "next_state_probability": float(
                loss_config.get("next_state_probability", 0.4)
            ),
        },
    }


def write_decoder_contract(path, contract):
    """Write a stable decoder contract beside a trained artifact."""
    Path(path).write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")


def receptive_field_ms(flags, feature_step_ms=10, frontend_window_ms=30):
    """Report the maximum acoustic receptive field of one emitted state vector."""
    if getattr(flags, "causal_memory", False):
        dilations = parse(getattr(flags, "temporal_dilations", ""))
        first_context = max(0, int(flags.first_conv_kernel_size) - 1)
        temporal_context = (
            2 * sum(int(value) for value in dilations) * int(flags.stride)
        )
        return frontend_window_ms + (first_context + temporal_context) * feature_step_ms
    slices = 1
    if flags.first_conv_filters > 0:
        slices += flags.first_conv_kernel_size - 1
    for repeat, kernel_sizes in zip(
        parse(flags.repeat_in_block), parse(flags.mixconv_kernel_sizes)
    ):
        slices += repeat * (max(kernel_sizes) - 1) * flags.stride
    return frontend_window_ms + (slices - 1) * feature_step_ms
