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

"""Gradient Reversal Layer for adversarial training."""

import tensorflow as tf


@tf.custom_gradient
def gradient_reversal(x, lambda_):
    """Gradient reversal operation with scaling factor lambda."""
    def grad(dy):
        return -lambda_ * dy, None
    return x, grad


class GradientReversal(tf.keras.layers.Layer):
    """Gradient Reversal Layer that reverses gradients during backpropagation.
    
    During forward pass, this layer acts as an identity function.
    During backward pass, it reverses the gradient and scales by lambda.
    
    This is used for adversarial training to make features domain-invariant.
    """
    
    def __init__(self, lambda_=1.0, **kwargs):
        """Initialize the gradient reversal layer.
        
        Args:
            lambda_: Gradient reversal scaling factor. Default is 1.0.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.lambda_ = lambda_
        
    def call(self, x):
        """Forward pass - acts as identity, gradient reversal happens in backward pass."""
        return gradient_reversal(x, self.lambda_)
    
    def get_config(self):
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({'lambda_': self.lambda_})
        return config