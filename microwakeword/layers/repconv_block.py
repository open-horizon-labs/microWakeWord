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

"""Re-parameterizable Convolutional Block for RepCNN model."""

import numpy as np
import tensorflow as tf


class RepConvBlock(tf.keras.layers.Layer):
    """Re-parameterizable Convolutional Block.

    During training, uses multiple parallel branches with different kernel sizes
    for better gradient flow. During inference, branches are merged into a single
    efficient convolution through re-parameterization.

    Based on: https://arxiv.org/html/2406.02652

    Args:
        filters: Number of output filters
        kernel_sizes: List of kernel sizes for parallel branches (e.g., [7, 9, 11, 13])
        use_batch_norm: Whether to use batch normalization
        activation: Activation function to use (default: 'relu')
    """

    def __init__(
        self,
        filters,
        kernel_sizes=None,
        use_batch_norm=True,
        activation="relu",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]  # Paper uses these sizes + 1x1

        self.filters = filters
        self.kernel_sizes = kernel_sizes
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

        for kernel_size in self.kernel_sizes:
            # Use regular Conv2D with same output filters for all branches
            conv = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=(kernel_size, 1),
                strides=(1, 1),
                padding="same",  # Use same padding for simplicity
                use_bias=False,
                name=f"conv_k{kernel_size}",
            )
            self.conv_branches.append(conv)

            if self.use_batch_norm:
                bn = tf.keras.layers.BatchNormalization(name=f"bn_k{kernel_size}")
                self.bn_branches.append(bn)

        # 1x1 convolution branch (pointwise)
        self.conv_1x1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            name="conv_1x1",
        )

        if self.use_batch_norm:
            self.bn_1x1 = tf.keras.layers.BatchNormalization(name="bn_1x1")

        # For re-parameterized mode, create a single convolution
        # Pre-create it to avoid state issues during re-parameterization
        self.reparam_conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(max(self.kernel_sizes), 1),
            strides=(1, 1),
            padding="same",
            use_bias=True,
            name="reparam_conv",
        )

    def call(self, inputs, training=None):
        """Forward pass through the block.

        Args:
            inputs: Input tensor [batch, time, 1, features]
            training: Boolean indicating training mode

        Returns:
            Output tensor after convolution and activation
        """

        # If re-parameterized, use single convolution
        if self.reparameterized and self.reparam_conv is not None:
            x = self.reparam_conv(inputs)
            if self.activation is not None:
                x = self.activation(x)
            return x

        # Training mode: use multiple branches
        outputs = []

        # Process convolution branches
        for i, conv in enumerate(self.conv_branches):
            x = conv(inputs)
            if self.use_batch_norm and i < len(self.bn_branches):
                x = self.bn_branches[i](x, training=training)
            outputs.append(x)

        # Process 1x1 convolution branch
        x_1x1 = self.conv_1x1(inputs)
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
        """Merge multiple branches into a single convolution.

        This should be called after training to convert the multi-branch
        architecture into a single efficient convolution for inference.
        """

        if self.reparameterized:
            return

        # Get the maximum kernel size
        max_kernel = max(self.kernel_sizes)

        # Initialize merged kernel with zeros
        # Shape: [kernel_height, kernel_width, in_channels, out_channels]
        merged_kernel = np.zeros((max_kernel, 1, self.in_channels, self.filters))
        merged_bias = np.zeros(self.filters)

        # Merge each convolution branch
        for i, (conv, kernel_size) in enumerate(zip(self.conv_branches, self.kernel_sizes)):
            # Get conv weights [kernel_height, kernel_width, in_channels, out_channels]
            conv_weights = conv.get_weights()[0]
            
            # Pad kernel to max size (center the smaller kernels)
            pad_total = max_kernel - kernel_size
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            
            padded_kernel = np.pad(
                conv_weights,
                ((pad_top, pad_bottom), (0, 0), (0, 0), (0, 0)),
                mode="constant",
            )

            # Apply batch norm if present
            if self.use_batch_norm and i < len(self.bn_branches):
                bn = self.bn_branches[i]
                gamma, beta, moving_mean, moving_var = bn.get_weights()

                # Fold BN parameters into convolution
                # For each output channel: w_new = gamma/std * w, b_new = beta - gamma/std * mean
                std = np.sqrt(moving_var + bn.epsilon)
                scale = gamma / std
                
                # Scale kernel weights for each output channel
                for j in range(self.filters):
                    padded_kernel[:, :, :, j] *= scale[j]
                    
                # Add bias contribution
                merged_bias += beta - moving_mean * scale
            
            # Add to merged kernel
            merged_kernel += padded_kernel

        # Merge 1x1 convolution branch
        conv_1x1_weights = self.conv_1x1.get_weights()[0]  # [1, 1, in_channels, filters]

        # Apply batch norm to 1x1 branch if present
        if self.use_batch_norm:
            bn = self.bn_1x1
            gamma, beta, moving_mean, moving_var = bn.get_weights()
            std = np.sqrt(moving_var + bn.epsilon)
            scale = gamma / std

            # Scale 1x1 kernel weights
            for j in range(self.filters):
                conv_1x1_weights[:, :, :, j] *= scale[j]
            
            # Add bias contribution
            merged_bias += beta - moving_mean * scale

        # Add 1x1 weights to center of merged kernel
        center = max_kernel // 2
        merged_kernel[center : center + 1, :, :, :] += conv_1x1_weights

        # Build the reparam_conv layer if not already built
        if not self.reparam_conv.built:
            self.reparam_conv.build(self.input_shape_stored)
        
        # Set weights to the pre-created re-parameterized convolution
        self.reparam_conv.set_weights([merged_kernel, merged_bias])

        self.reparameterized = True

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_sizes": self.kernel_sizes,
                "use_batch_norm": self.use_batch_norm,
                "activation": tf.keras.activations.serialize(self.activation),
                "reparameterized": self.reparameterized,
            }
        )
        return config