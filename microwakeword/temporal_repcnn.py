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

"""Model based on RepCNN wake word model adjusted to support streaming."""

import ast

import tensorflow as tf

from microwakeword.layers import stream, strided_drop

import numpy as np

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
    """Temporal RepCNN model parameters."""

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
        "--smaller_kernel_sizes",
        type=str,
        default="",
        help="Smaller kernel sizes for additional branch in each block (comma-separated). Leave empty to disable.",
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
    parser_nn.add_argument(
        "--adaptive_pooling",
        type=int,
        default=0,
        help="Use adaptive pooling that combines both average and max pooling with learnable weights (0=disabled, 1=enabled)",
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
        reparameterized=False,  # Add as parameter for proper serialization
        smaller_kernel_size=None,  # Optional smaller kernel branch
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.branches = branches
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.use_batch_norm = use_batch_norm
        self.activation = tf.keras.activations.get(activation)
        self.reparameterized = reparameterized
        self.smaller_kernel_size = smaller_kernel_size

        # Create reparam_conv in __init__ to avoid state tracking issues
        # Always use bias=True for reparam_conv since after reparameterization
        # we always have bias (either from original bias or folded batch norm)
        self.reparam_conv = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(self.kernel_size, 1),
            strides=1,
            padding="valid",
            depth_multiplier=self.depth_multiplier,
            use_bias=True,
        )

    def build(self, input_shape):
        super().build(input_shape)

        # Store input shape for re-parameterization
        self.input_shape_stored = input_shape

        # Input channels
        self.in_channels = input_shape[-1]

        # If already reparameterized, only build the reparam_conv
        if self.reparameterized:
            # Initialize empty lists to avoid attribute errors
            self.conv_branches = []
            self.bn_branches = []
            self.conv_1x1 = None
            self.bn_1x1 = None
            self.conv_smaller = None
            self.bn_smaller = None
            return

        # Create parallel Conv2D branches for each kernel size
        self.conv_branches = []
        self.bn_branches = []

        for _ in range(self.branches):
            conv = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.kernel_size, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multiplier,
                use_bias=not self.use_batch_norm,  # Use bias when NOT using batch norm
            )
            self.conv_branches.append(conv)

            if self.use_batch_norm:
                bn = tf.keras.layers.BatchNormalization()
                self.bn_branches.append(bn)

        # Create smaller kernel branch if specified
        if self.smaller_kernel_size is not None:
            self.conv_smaller = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(self.smaller_kernel_size, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multiplier,
                use_bias=not self.use_batch_norm,
            )

            if self.use_batch_norm:
                self.bn_smaller = tf.keras.layers.BatchNormalization()
        else:
            self.conv_smaller = None
            self.bn_smaller = None

        self.conv_1x1 = tf.keras.layers.DepthwiseConv2D(
                kernel_size=(1, 1),
                strides=1,
                padding="valid",
                depth_multiplier=self.depth_multiplier,
                use_bias=not self.use_batch_norm,  # Use bias when NOT using batch norm
            )

        if self.use_batch_norm:
            self.bn_1x1 = tf.keras.layers.BatchNormalization()

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
        if self.reparameterized:
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

        # Process smaller kernel branch if it exists
        if self.conv_smaller is not None:
            dropped_net_smaller = strided_drop.StridedDrop(self.kernel_size - self.smaller_kernel_size)(net)
            x_smaller = self.conv_smaller(dropped_net_smaller)
            if self.use_batch_norm:
                x_smaller = self.bn_smaller(x_smaller, training=training)
            outputs.append(x_smaller)

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

    def reparameterize(self):
        """Merge multiple branches into a single depthwise convolution.

        This should be called after training to convert the multi-branch
        architecture into a single efficient convolution for inference.
        """

        if self.reparameterized:
            return

        # Initialize merged kernel with zeros
        # For depthwise conv: [kernel_height, kernel_width, in_channels, depth_multiplier]
        merged_kernel = np.zeros((self.kernel_size, 1, self.in_channels, self.depth_multiplier))

        # Since depthwise conv doesn't have a bias by default in our branches,
        # we'll accumulate the bias from batch norm folding
        # Output channels = in_channels * depth_multiplier
        output_channels = self.in_channels * self.depth_multiplier
        merged_bias = np.zeros(output_channels)

        # Merge each convolution branch (all have the same kernel size)
        for i, conv in enumerate(self.conv_branches):
            # Get conv weights - includes bias if not using batch norm
            weights = conv.get_weights()
            conv_weights = weights[0]  # [kernel_size, 1, in_channels, depth_multiplier]

            # If we have bias (when not using batch norm), add it
            if len(weights) > 1:
                merged_bias += weights[1]

            # Apply batch norm if present
            if self.use_batch_norm and i < len(self.bn_branches):
                bn = self.bn_branches[i]
                gamma, beta, moving_mean, moving_var = bn.get_weights()

                # Fold BN parameters into convolution
                # For depthwise conv, BN operates on output_channels = in_channels * depth_multiplier
                std = np.sqrt(moving_var + bn.epsilon)
                scale = gamma / std

                # Reshape scale and bias for proper broadcasting
                # conv_weights shape: [kernel_size, 1, in_channels, depth_multiplier]
                # We need to scale each of the output channels independently
                scale_reshaped = scale.reshape(self.in_channels, self.depth_multiplier)

                # Scale kernel weights - broadcast across kernel_size dimension
                for c in range(self.in_channels):
                    for d in range(self.depth_multiplier):
                        conv_weights[:, :, c, d] *= scale_reshaped[c, d]

                # Add bias contribution
                merged_bias += beta - moving_mean * scale

            # Add to merged kernel
            merged_kernel += conv_weights

        # Merge 1x1 convolution branch
        weights_1x1 = self.conv_1x1.get_weights()
        conv_1x1_weights = weights_1x1[0]  # [1, 1, in_channels, depth_multiplier]

        # If we have bias (when not using batch norm), add it
        if len(weights_1x1) > 1:
            merged_bias += weights_1x1[1]

        # Apply batch norm to 1x1 branch if present
        if self.use_batch_norm:
            bn = self.bn_1x1
            gamma, beta, moving_mean, moving_var = bn.get_weights()
            std = np.sqrt(moving_var + bn.epsilon)
            scale = gamma / std

            # Scale 1x1 kernel weights
            scale_reshaped = scale.reshape(self.in_channels, self.depth_multiplier)
            for c in range(self.in_channels):
                for d in range(self.depth_multiplier):
                    conv_1x1_weights[:, :, c, d] *= scale_reshaped[c, d]

            # Add bias contribution
            merged_bias += beta - moving_mean * scale

        # Merge smaller kernel branch if it exists
        if self.conv_smaller is not None:
            weights_smaller = self.conv_smaller.get_weights()
            conv_smaller_weights = weights_smaller[0]  # [smaller_kernel_size, 1, in_channels, depth_multiplier]

            # If we have bias (when not using batch norm), add it
            if len(weights_smaller) > 1:
                merged_bias += weights_smaller[1]

            # Apply batch norm to smaller branch if present
            if self.use_batch_norm:
                bn = self.bn_smaller
                gamma, beta, moving_mean, moving_var = bn.get_weights()
                std = np.sqrt(moving_var + bn.epsilon)
                scale = gamma / std

                # Scale smaller kernel weights
                scale_reshaped = scale.reshape(self.in_channels, self.depth_multiplier)
                for c in range(self.in_channels):
                    for d in range(self.depth_multiplier):
                        conv_smaller_weights[:, :, c, d] *= scale_reshaped[c, d]

                # Add bias contribution
                merged_bias += beta - moving_mean * scale

            # Add smaller kernel weights at the appropriate offset
            # Offset aligns with StridedDrop behavior
            offset = self.kernel_size - self.smaller_kernel_size
            merged_kernel[offset : offset + self.smaller_kernel_size, :, :, :] += conv_smaller_weights

        # Add 1x1 weights to the LAST position of merged kernel
        # This aligns with how StridedDrop shifts the 1x1 input
        merged_kernel[self.kernel_size - 1 : self.kernel_size, :, :, :] += conv_1x1_weights

        # Build the reparam_conv layer if not already built
        if not self.reparam_conv.built:
            self.reparam_conv.build(self.input_shape_stored)
            # After building, the layer has random weights. We need to ensure
            # we completely replace them with our merged weights.

        # Set weights with both kernel and bias
        self.reparam_conv.set_weights([merged_kernel, merged_bias])

        self.reparameterized = True

    def get_weights(self):
        """Override get_weights to return only reparam_conv weights when reparameterized."""
        if self.reparameterized:
            return self.reparam_conv.get_weights()
        return super().get_weights()

    def set_weights(self, weights):
        """Override set_weights to handle reparameterized state."""
        if self.reparameterized:
            self.reparam_conv.set_weights(weights)
        else:
            super().set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "branches": self.branches,
                "depth_multiplier": self.depth_multiplier,
                "kernel_size": self.kernel_size,
                "use_batch_norm": self.use_batch_norm,
                "activation": tf.keras.activations.serialize(self.activation),
                "reparameterized": self.reparameterized,
                "smaller_kernel_size": self.smaller_kernel_size,
            }
        )
        return config

def model(flags, shape, batch_size):
    """
    Based on the paper:
    RepCNN: Micro-sized, Mighty Models for Wakeword Detection
    Arnav Kundu, Prateeth Naya∗, Priyanka Padmanabhan, Devang Naik
    https://arxiv.org/pdf/2406.02652

    Returns:
      Keras model for training
    """

    branches = parse(flags.branches)
    depth_multipliers = parse(flags.depth_multipliers)
    kernel_sizes = parse(flags.kernel_sizes)
    pointwise_filters = parse(flags.pointwise_filters)
    smaller_kernel_sizes = parse(flags.smaller_kernel_sizes) if flags.smaller_kernel_sizes else []


    for list in (
        depth_multipliers,
        kernel_sizes,
        pointwise_filters,
    ):
        if len(branches) != len(list):
            raise ValueError("all input lists have to be the same length")

    # Validate smaller_kernel_sizes if provided
    if smaller_kernel_sizes and len(smaller_kernel_sizes) != len(branches):
        raise ValueError("smaller_kernel_sizes must have the same length as other block parameters")

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


    for block_idx, (block_branches, block_depth_multipliers, block_kernel_sizes, block_pointwise_filter) in enumerate(zip(branches, depth_multipliers, kernel_sizes, pointwise_filters)):
        # Get smaller kernel size for this block if provided
        block_smaller_kernel = smaller_kernel_sizes[block_idx] if smaller_kernel_sizes else None

        for branch, depth_multiplier, kernel_size in zip(block_branches, block_depth_multipliers, block_kernel_sizes):
            net = stream.Stream(
                cell=tf.keras.layers.Identity(),
                ring_buffer_size_in_time_dim=kernel_size-1,
                use_one_step=False,
            )(net)

            net = TemporalRepConvBlock(branch, depth_multiplier, kernel_size, smaller_kernel_size=block_smaller_kernel)(net)

        net = tf.keras.layers.Conv2D(
                filters=block_pointwise_filter, kernel_size=1, use_bias=False, padding="same"
            )(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Activation("relu")(net)

    if net.shape[1] > 1:
        # Use ReLU instead of Identity as the streaming cell.
        # This is critical for TFLite quantization: when wrapping Identity, the TFLite
        # converter cannot infer quantization parameters for the internal state buffer,
        # causing it to remain float32. This creates expensive dequantize→concat→quantize
        # cycles before the pooling layers.
        #
        # ReLU provides quantization metadata, allowing the state buffer to be int8.
        # Since values have already passed through ReLU activation, this is functionally
        # a no-op (all values ≥0), but enables proper quantization.
        #
        # For TFLite Micro deployment, this is optimal: a single ReLU operation
        # (cheap compare+select) is faster than dequantize+quantize operations
        # (expensive multiply+divide+round).
        net = stream.Stream(
            cell=tf.keras.layers.ReLU(max_value=None),
            ring_buffer_size_in_time_dim=net.shape[1] - 1,
            use_one_step=False,
        )(net)

        if flags.pooled:
            if flags.adaptive_pooling:
                # Adaptive pooling: combine average and max pooling with learnable weights
                # Apply both pooling operations
                avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)
                max_pool = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)

                # Reshape to flatten spatial dimensions
                avg_pool = tf.keras.layers.Reshape((-1,))(avg_pool)
                max_pool = tf.keras.layers.Reshape((-1,))(max_pool)

                # Concatenate both pooling outputs
                net = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

                # Note: The final Dense layer will learn how to weight these features
                # This is more flexible than explicit weighting as it allows the model
                # to learn complex interactions between avg and max pooled features
            else:
                # Standard pooling based on max_pool flag
                if flags.max_pool:
                    net = tf.keras.layers.MaxPooling2D(pool_size=(net.shape[1], 1))(net)
                else:
                    net = tf.keras.layers.AveragePooling2D(pool_size=(net.shape[1], 1))(net)
                net = tf.keras.layers.Reshape((-1,))(net)
        else:
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
        if isinstance(layer, TemporalRepConvBlock):
            layer.reparameterize()
        # Handle Stream-wrapped layers
        elif hasattr(layer, "cell") and isinstance(
            layer.cell, TemporalRepConvBlock
        ):
            layer.cell.reparameterize()

    return model
