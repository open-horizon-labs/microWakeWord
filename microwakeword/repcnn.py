# Copyright 2024 Kevin Ahrendt.
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

"""RepCNN model with re-parameterizable convolutional blocks."""

import ast

import tensorflow as tf

from microwakeword.layers import repconv_block, stream


def parse(text):
    """Parse model parameters.

    Args:
        text: string with layer parameters: '128,128' or "'relu','relu'".

    Returns:
        list of parsed parameters
    """
    if not text:
        return []
    res = ast.literal_eval(text)
    if isinstance(res, tuple):
        return res
    else:
        return [res]


def model_parameters(parser_nn):
    """RepCNN model parameters."""

    parser_nn.add_argument(
        "--num_blocks",
        type=int,
        default=2,
        help="Number of RepConv blocks in the model",
    )
    parser_nn.add_argument(
        "--block_filters",
        type=str,
        default="48, 64",
        help="Number of filters in each RepConv block",
    )
    parser_nn.add_argument(
        "--kernel_sizes",
        type=str,
        default="7, 5, 3",
        help="Kernel sizes for parallel branches in RepConv blocks (paper uses 7,5,3 + 1x1)",
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
        default=5,
        help="Temporal kernel size for the initial convolution layer.",
    )
    parser_nn.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Stride for the initial convolution layer.",
    )
    parser_nn.add_argument(
        "--use_batch_norm",
        type=int,
        default=1,
        help="Use batch normalization in RepConv blocks (0 or 1)",
    )
    parser_nn.add_argument(
        "--max_pool",
        type=int,
        default=0,
        help="Use max pooling instead of average pooling before final layer",
    )
    parser_nn.add_argument(
        "--pooled",
        type=int,
        default=0,
        help="Pool the temporal dimension before the final fully connected layer",
    )


def model(flags, shape, batch_size):
    """RepCNN model.

    Based on the paper: https://arxiv.org/html/2406.02652

    Args:
        flags: data/model parameters
        shape: shape of the input vector
        batch_size: batch size

    Returns:
        Keras model
    """

    # Parse parameters
    block_filters = parse(flags.block_filters)
    kernel_sizes = parse(flags.kernel_sizes)

    # Validate parameters
    if flags.num_blocks != len(block_filters):
        raise ValueError(
            f"Number of blocks ({flags.num_blocks}) must match "
            f"length of block_filters ({len(block_filters)})"
        )

    # Input layer
    input_audio = tf.keras.layers.Input(
        shape=shape,
        batch_size=batch_size,
    )
    net = input_audio

    # Make it [batch, time, 1, feature]
    net = tf.keras.ops.expand_dims(net, axis=2)

    # Initial convolution with streaming support
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

        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

    # RepConv blocks
    for i in range(flags.num_blocks):
        filters = block_filters[i]

        # RepConvBlock already uses Conv2D with 'same' padding internally
        # No need for Stream wrapper as the block handles its own convolutions
        net = repconv_block.RepConvBlock(
            filters=filters,
            kernel_sizes=kernel_sizes,
            use_batch_norm=bool(flags.use_batch_norm),
            activation="relu",
            name=f"repconv_block_{i}",
        )(net)

    # Global pooling
    if flags.pooled:
        # Pool over both time and frequency dimensions
        if flags.max_pool:
            net = tf.keras.layers.GlobalMaxPooling2D()(net)
        else:
            net = tf.keras.layers.GlobalAveragePooling2D()(net)
    else:
        # Flatten without pooling - keeps temporal information
        net = tf.keras.layers.Flatten()(net)

    # Final dense layer for binary classification
    net = tf.keras.layers.Dense(1)(net)
    net = tf.keras.layers.Activation("sigmoid")(net)

    model = tf.keras.Model(inputs=input_audio, outputs=net)

    return model


def reparameterize_model(model):
    """Apply re-parameterization to all RepConvBlock layers in the model.

    This should be called after training to convert multi-branch blocks
    into single efficient convolutions for inference.

    Args:
        model: Trained Keras model containing RepConvBlock layers

    Returns:
        Model with re-parameterized blocks
    """

    for layer in model.layers:
        if isinstance(layer, repconv_block.RepConvBlock):
            layer.reparameterize()
        # Handle Stream-wrapped layers
        elif hasattr(layer, "cell") and isinstance(
            layer.cell, repconv_block.RepConvBlock
        ):
            layer.cell.reparameterize()

    return model
