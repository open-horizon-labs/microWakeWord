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

"""MixedNet encoder with LSTM decoder using CTC loss for wake word detection."""

import ast
import tensorflow as tf
from microwakeword.layers import stream, strided_drop
from microwakeword import mixednet


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
    """MixedNet+LSTM+CTC model parameters."""
    
    # Inherit all MixedNet encoder parameters
    mixednet.model_parameters(parser_nn)
    
    # Add LSTM decoder specific parameters
    parser_nn.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Dimension of encoder output embeddings fed to LSTM decoder",
    )
    parser_nn.add_argument(
        "--lstm_units",
        type=str,
        default="64",
        help="Number of LSTM units per layer (comma-separated for multiple layers)",
    )
    parser_nn.add_argument(
        "--wake_word_phrase",
        type=str,
        default="hey jarvis",
        help="Wake word phrase to detect (space-separated words)",
    )
    parser_nn.add_argument(
        "--ctc_merge_repeated",
        type=int,
        default=1,
        help="Whether to merge repeated labels in CTC decoding",
    )


def build_vocab(wake_word_phrase):
    """Build vocabulary from wake word phrase.
    
    Args:
        wake_word_phrase: String with space-separated wake words
        
    Returns:
        vocab: Dictionary mapping words to indices
        vocab_size: Size of vocabulary including blank token
    """
    words = wake_word_phrase.lower().split()
    # Index 0 is reserved for CTC blank token
    vocab = {word: idx + 1 for idx, word in enumerate(words)}
    vocab['<blank>'] = 0
    vocab_size = len(words) + 1  # +1 for blank token
    
    return vocab, vocab_size


def create_encoder(flags, shape, batch_size):
    """Create MixedNet encoder that outputs embeddings instead of binary classification.
    
    Args:
        flags: Model parameters
        shape: Input shape
        batch_size: Batch size
        
    Returns:
        encoder_model: Keras model that outputs embeddings
        encoder_output_steps: Number of time steps in encoder output
    """
    # Get the base MixedNet model
    base_model = mixednet.model(flags, shape, batch_size)
    
    # Remove the final Dense(1) + sigmoid layer
    # The model currently ends with Flatten -> Dense(1, sigmoid)
    # We want to intercept at the Flatten output
    
    # Find the layer before the final Dense layer
    penultimate_layer = None
    for layer in base_model.layers[:-1]:
        if isinstance(layer, tf.keras.layers.Flatten):
            penultimate_layer = layer
            break
    
    if penultimate_layer is None:
        raise ValueError("Could not find Flatten layer in MixedNet model")
    
    # Create new model that outputs embeddings
    encoder_input = base_model.input
    
    # Get the output from the layer before flattening
    pre_flatten_output = penultimate_layer.input
    
    # Instead of flattening completely, we want to preserve time dimension
    # Current shape should be (batch, time, 1, features)
    # We want (batch, time, embedding_dim)
    
    # Apply Dense layer to each time step to get embeddings
    embedding_layer = tf.keras.layers.Dense(flags.embedding_dim, name="encoder_embeddings")
    
    # Reshape to (batch, time, features) if needed
    if len(pre_flatten_output.shape) == 4:
        reshaped = tf.keras.layers.Reshape(
            (-1, pre_flatten_output.shape[-1])
        )(pre_flatten_output)
    else:
        reshaped = pre_flatten_output
    
    # Apply embedding layer
    encoder_output = tf.keras.layers.TimeDistributed(embedding_layer)(reshaped)
    
    encoder_model = tf.keras.Model(inputs=encoder_input, outputs=encoder_output)
    
    # Calculate output time steps
    # This depends on the network architecture and padding
    encoder_output_steps = encoder_output.shape[1]
    if encoder_output_steps is None:
        encoder_output_steps = -1  # Will be calculated dynamically during training
    
    return encoder_model, encoder_output_steps


def create_decoder(flags, encoder_output, vocab_size):
    """Create LSTM decoder with CTC output.
    
    Args:
        flags: Model parameters
        encoder_output: Output tensor from encoder
        vocab_size: Size of vocabulary (including blank token)
        
    Returns:
        decoder_output: Logits for CTC loss (batch, time, vocab_size)
    """
    lstm_units = parse(flags.lstm_units)
    
    net = encoder_output
    
    # Add LSTM layers
    for i, units in enumerate(lstm_units):
        # For streaming, we need to use stateful LSTM or convert later
        # For now, use regular LSTM for training
        net = tf.keras.layers.LSTM(
            units=int(units),
            return_sequences=True,
            name=f"decoder_lstm_{i}"
        )(net)
    
    # Output layer for CTC
    decoder_output = tf.keras.layers.Dense(
        vocab_size,
        activation=None,  # CTC loss expects logits
        name="ctc_output"
    )(net)
    
    return decoder_output


def model(flags, shape, batch_size):
    """MixedNet+LSTM+CTC model for wake word detection.
    
    Args:
        flags: Model parameters
        shape: Input shape
        batch_size: Batch size
        
    Returns:
        Keras model for training with CTC loss
    """
    # Build vocabulary
    vocab, vocab_size = build_vocab(flags.wake_word_phrase)
    
    # Create encoder
    encoder_model, encoder_steps = create_encoder(flags, shape, batch_size)
    
    # Get encoder output
    input_audio = encoder_model.input
    encoder_output = encoder_model.output
    
    # Create decoder
    decoder_output = create_decoder(flags, encoder_output, vocab_size)
    
    # Create full model
    model = tf.keras.Model(inputs=input_audio, outputs=decoder_output)
    
    # Store vocabulary info as model attributes for later use
    model.vocab = vocab
    model.vocab_size = vocab_size
    model.encoder_steps = encoder_steps
    
    return model