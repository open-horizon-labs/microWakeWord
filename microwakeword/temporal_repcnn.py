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

"""Model based on 1D depthwise MixedConvs and 1x1 convolutions in time + residual."""

import ast

import tensorflow as tf

from microwakeword.layers import stream, strided_drop


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
    """MixedNet model parameters."""

    parser_nn.add_argument(
        "--branches",
        type=str,
        default="[2,2], [2,2]",
        help="",
    )
    parser_nn.add_argument(
        "--depth_multipliers",
        type=str,
        default="[2,1], [1,1]",
        help="",
    )
    parser_nn.add_argument(
        "--kernel_sizes",
        type=str,
        default="[7,9],[11,13]",
        help="",
    )
    parser_nn.add_argument(
        "--pointwise_filters",
        type=str,
        default="32, 32",
        help="Number of filters in each block's pointwise convolution",
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


    for block_kernel_sizes in parse(flags.kernel_sizes):
        for kernel_size in block_kernel_sizes:
            spectrogram_slices_dropped += (kernel_size-1)*flags.stride
        
    return spectrogram_slices_dropped

class TemporalRepConvBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        branches,
        depth_multiplier,
        kernel_size,
        use_batch_norm=True,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.branches = branches
        self.depth_multplier = depth_multiplier
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.activation = tf.keras.activations.get(activation)
        self.reparameterized = False
    
    def build(self, input_shape):
        super().build(input_shape)
        
        # Store input shape for re-parameterization
        self.input_shape_stored = input_shape
        
        # Input channels
        self.in_channels = input_shape[-1]
        
        # Create parallel Conv2D branches for each kernel size
        self.conv_branches = []
        self.bn_branches = []
        
        for _ in range(self.branches):
            conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.kernel_size, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multplier,
                use_bias=False,
            )
            self.conv_branches.append(conv)

            if self.use_batch_norm:
                bn = tf.keras.layers.BatchNormalization()
                self.bn_branches.append(bn)
        
        self.conv_1x1 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multplier,
                use_bias=False,
            )
        
        if self.use_batch_norm:
            self.bn_1x1 = tf.keras.layers.BatchNormalization()

        # For re-parameterized mode, create a single convolution
        # Pre-create it to avoid state issues during re-parameterization
        self.reparam_conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.kernel_size, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multplier,
                use_bias=False,
            )
    
    def call(self, inputs, training=None):
        """Forward pass through the block.

        Args:
            inputs: Input tensor [batch, time, 1, features]
            training: Boolean indicating training mode

        Returns:
            Output tensor after convolution and activation
        """        
        net = inputs
        
        # If re-parameterized, use single convolution
        if self.reparameterized and self.reparam_conv is not None:
            x = self.reparam_conv(net)
            if self.activation is not None:
                x = self.activation(x)
            return x
        
        # Training mode: use multiple branches
        outputs = []

        # Process convolution branches
        for i, conv in enumerate(self.conv_branches):
            x = conv(net)
            if self.use_batch_norm and i < len(self.bn_branches):
                x = self.bn_branches[i](x, training=training)
            outputs.append(x)

        # Process 1x1 convolution branch
        dropped_net = strided_drop.StridedDrop(self.kernel_size-1)(net)
        x_1x1 = self.conv_1x1(dropped_net)
        if self.use_batch_norm:
            x_1x1 = self.bn_1x1(x_1x1, training=training)
        outputs.append(x_1x1)

        # Sum all branches
        x = tf.add_n(outputs)

        # Apply activation
        if self.activation is not None:
            x = self.activation(x)

        return x

def model(flags, shape, batch_size):
    """


    Returns:
      Keras model for training
    """

    branches = parse(flags.branches)
    depth_multipliers = parse(flags.depth_multipliers)
    kernel_sizes = parse(flags.kernel_sizes)
    pointwise_filters = parse(flags.pointwise_filters)


    for list in (
        depth_multipliers,
        kernel_sizes,
        pointwise_filters,
    ):
        if len(branches) != len(list):
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

        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)


    for block_branches, block_depth_multipliers, block_kernel_sizes, block_pointwise_filter in zip(branches, depth_multipliers, kernel_sizes, pointwise_filters):
        for branch, depth_multiplier, kernel_size in zip(block_branches, block_depth_multipliers, block_kernel_sizes):
            net = stream.Stream(
                cell=tf.keras.layers.Identity(),
                ring_buffer_size_in_time_dim=kernel_size-1,
                use_one_step=False,
            )(net)

            net = TemporalRepConvBlock(branch, depth_multiplier, kernel_size)(net)
            
        net = tf.keras.layers.Conv2D(
                filters=block_pointwise_filter, kernel_size=1, use_bias=False, padding="same"
            )(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

    if net.shape[1] > 1:
        net = stream.Stream(
            cell=tf.keras.layers.Identity(),
            ring_buffer_size_in_time_dim=net.shape[1] - 1,
            use_one_step=False,
        )(net)

        if flags.pooled:
            # We want to use either Global Max Pooling or Global Average Pooling, but the esp-nn operator optimizations only benefit regular pooling operations

            if flags.max_pool:
                net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)
            else:
                net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)
        else:
            net = tf.keras.layers.Flatten()(net)

    # Final dense layer for binary classification
    net = tf.keras.layers.Dense(1)(net)
    net = tf.keras.layers.Activation("sigmoid")(net)

    model = tf.keras.Model(inputs=input_audio, outputs=net)

    return model
