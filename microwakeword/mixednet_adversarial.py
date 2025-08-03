# Copyright 2025 Kevin Ahrendt.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MixedNet model with adversarial training for TTS robustness."""

import ast

import tensorflow as tf

from microwakeword.layers import stream, strided_drop, gradient_reversal
from microwakeword.mixednet import (
    parse, _split_channels, _get_shape_value,
    ChannelSplit, MixConv, SpatialAttention
)


def model_parameters(parser_nn):
    """MixedNetAdversarial model parameters."""

    # Include all base MixedNet parameters
    parser_nn.add_argument(
        "--pointwise_filters",
        type=str,
        default="48, 48, 48, 48",
        help="Number of filters in every MixConv block's pointwise convolution",
    )
    parser_nn.add_argument(
        "--residual_connection",
        type=str,
        default="0,0,0,0,0",
        help="Use a residual connection in each MixConv block",
    )
    parser_nn.add_argument(
        "--repeat_in_block",
        type=str,
        default="1,1,1,1",
        help="Number of repeating conv blocks inside of residual block",
    )
    parser_nn.add_argument(
        "--mixconv_kernel_sizes",
        type=str,
        default="[5], [9], [13], [21]",
        help="Kernel size lists for DepthwiseConv1D in time dim for every MixConv block",
    )
    parser_nn.add_argument(
        "--max_pool",
        type=int,
        default=0,
        help="apply max pool instead of average pool before final convolution and sigmoid activation",
    )
    parser_nn.add_argument(
        "--first_conv_filters",
        type=int,
        default=32,
        help="Number of filters on initial convolution layer. Set to 0 to disable.",
    )
    parser_nn.add_argument(
        "--first_conv_kernel_size",
        type=int,
        default="3",
        help="Temporal kernel size for the initial convolution layer.",
    )
    parser_nn.add_argument(
        "--spatial_attention",
        type=int,
        default=0,
        help="Add a spatial attention layer before the final pooling layer",
    )
    parser_nn.add_argument(
        "--pooled",
        type=int,
        default=0,
        help="Pool the temporal dimension before the final fully connected layer. Uses average pooling or max pooling depending on the max_pool argument",
    )
    parser_nn.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Striding in the time dimension of the initial convolution layer",
    )

    # Adversarial training specific parameters
    parser_nn.add_argument(
        "--adversarial_beta",
        type=float,
        default=0.5,
        help="Weight of adversarial loss in the total loss",
    )
    parser_nn.add_argument(
        "--adversarial_lambda",
        type=float,
        default=0.3,
        help="Gradient reversal scaling factor for adversarial training",
    )
    parser_nn.add_argument(
        "--adversarial_hidden_units",
        type=str,
        default="128,64",
        help="Hidden units for adversarial classifier layers",
    )
    parser_nn.add_argument(
        "--adversarial_dropout",
        type=float,
        default=0.5,
        help="Dropout rate for adversarial classifier",
    )


def spectrogram_slices_dropped(flags):
    """Computes the number of spectrogram slices dropped due to valid padding.

    Args:
        flags: data/model parameters

    Returns:
        int: number of spectrogram slices dropped
    """
    spectrogram_slices_dropped = 0

    if flags.first_conv_filters > 0:
        spectrogram_slices_dropped += flags.first_conv_kernel_size - 1

    for repeat, ksize in zip(
        parse(flags.repeat_in_block),
        parse(flags.mixconv_kernel_sizes),
    ):
        spectrogram_slices_dropped += (repeat * (max(ksize) - 1)) * flags.stride

    return spectrogram_slices_dropped


def model(flags, shape, batch_size):
    """MixedNetAdversarial model.

    This model extends MixedNet with an adversarial classifier branch
    that predicts whether the input is real speech or TTS-generated.
    The gradient reversal layer ensures the main model learns features
    that are invariant to the speech source.

    Args:
      flags: data/model parameters
      shape: shape of the input vector
      batch_size: batch size for training

    Returns:
      Keras model with two outputs: wake word prediction and TTS prediction
    """

    pointwise_filters = parse(flags.pointwise_filters)
    repeat_in_block = parse(flags.repeat_in_block)
    mixconv_kernel_sizes = parse(flags.mixconv_kernel_sizes)
    residual_connections = parse(flags.residual_connection)
    adversarial_hidden_units = parse(flags.adversarial_hidden_units)

    for list in (
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if len(pointwise_filters) != len(list):
            raise ValueError("all input lists have to be the same length")

    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # make it [batch, time, 1, feature]
    net = tf.keras.ops.expand_dims(net, axis=2)

    # Streaming Conv2D with 'valid' padding
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
            pad_time_dim=None,
            pad_freq_dim="valid",
        )(net)

        net = tf.keras.layers.Activation("relu")(net)

    # encoder - exactly the same as original MixedNet
    for filters, repeat, ksize, res in zip(
        pointwise_filters,
        repeat_in_block,
        mixconv_kernel_sizes,
        residual_connections,
    ):
        if res:
            residual = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            residual = tf.keras.layers.BatchNormalization()(residual)

        for _ in range(repeat):
            if max(ksize) > 1:
                net = MixConv(kernel_size=ksize)(net)
            net = tf.keras.layers.Conv2D(
                filters=filters, kernel_size=1, use_bias=False, padding="same"
            )(net)
            net = tf.keras.layers.BatchNormalization()(net)

            if res:
                residual = strided_drop.StridedDrop(residual.shape[1] - net.shape[1])(
                    residual
                )
                net = net + residual

            net = tf.keras.layers.Activation("relu")(net)

    # Save features before final pooling for adversarial classifier
    features = net

    # Continue with wake word classification branch
    if net.shape[1] > 1:
        if flags.spatial_attention:
            net = SpatialAttention(
                kernel_size=4,
                ring_buffer_size=net.shape[1] - 1,
            )(net)
        else:
            net = stream.Stream(
                cell=tf.keras.layers.Identity(),
                ring_buffer_size_in_time_dim=net.shape[1] - 1,
                use_one_step=False,
            )(net)

        if flags.pooled:
            if flags.max_pool:
                net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)
            else:
                net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)

    net = tf.keras.layers.Flatten()(net)
    wake_word_output = tf.keras.layers.Dense(1, activation="sigmoid", name="wake_word")(net)

    # Adversarial classifier branch
    # Apply gradient reversal to features
    adversarial_features = gradient_reversal.GradientReversal(lambda_=flags.adversarial_lambda)(features)

    # Global average pooling to aggregate temporal information
    adversarial_features = tf.keras.layers.GlobalAveragePooling2D()(adversarial_features)

    # Adversarial classifier
    for units in adversarial_hidden_units:
        adversarial_features = tf.keras.layers.Dense(units, activation="relu")(adversarial_features)
        adversarial_features = tf.keras.layers.Dropout(flags.adversarial_dropout)(adversarial_features)

    # Binary classification: 0 = real speech, 1 = TTS
    tts_output = tf.keras.layers.Dense(1, activation="sigmoid", name="tts_classifier")(adversarial_features)

    return tf.keras.Model(input_audio, [wake_word_output, tts_output])