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

"""Utilities for adversarial models."""

import tensorflow as tf
from microwakeword.layers import modes, stream


def extract_wake_word_model(adversarial_model):
    """Extract just the wake word classification branch from adversarial model.
    
    This creates a new model that only outputs wake word predictions,
    excluding the adversarial TTS classifier branch. This is used for
    deployment where we only need wake word detection.
    
    Args:
        adversarial_model: The full adversarial model with two outputs
        
    Returns:
        A new model with only the wake word output
    """
    # Get the wake word output (first output)
    wake_word_output = adversarial_model.outputs[0]
    
    # Create new model with same input but only wake word output
    wake_word_model = tf.keras.Model(
        inputs=adversarial_model.inputs,
        outputs=wake_word_output,
        name="wake_word_only"
    )
    
    return wake_word_model


def convert_adversarial_model_to_streaming(adversarial_model, config, mode):
    """Convert adversarial model to streaming mode for deployment.
    
    This function first extracts just the wake word branch, then converts
    it to streaming mode.
    
    Args:
        adversarial_model: The full adversarial model
        config: Configuration dictionary
        mode: Streaming mode (e.g., STREAM_INTERNAL_STATE_INFERENCE)
        
    Returns:
        Streaming model with only wake word output
    """
    # First extract wake word only model
    wake_word_model = extract_wake_word_model(adversarial_model)
    
    # Convert to streaming using the standard streaming conversion
    # This requires the model conversion utilities from the main codebase
    input_data_shape = modes.get_input_data_shape(config, mode)
    
    # Set streaming mode
    modes.set_mode(wake_word_model, mode)
    
    # Create new input for streaming
    input_audio = tf.keras.layers.Input(
        shape=input_data_shape,
        batch_size=1,
        name="input_audio"
    )
    
    # Run model in streaming mode
    output = wake_word_model(input_audio)
    
    # Create streaming model
    streaming_model = tf.keras.Model(inputs=input_audio, outputs=output)
    
    return streaming_model