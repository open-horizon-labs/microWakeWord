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

"""Streaming inference support for CTC-based wake word models."""

import tensorflow as tf
import numpy as np
from microwakeword import ctc_utils


class StreamingCTCInference:
    """Streaming inference for MixedNet+LSTM+CTC models.
    
    This class handles:
    1. Buffering encoder outputs
    2. Maintaining LSTM state across streaming calls
    3. CTC decoding on accumulated outputs
    """
    
    def __init__(self, model, window_size=20, step_size=1):
        """Initialize streaming inference.
        
        Args:
            model: Trained CTC model
            window_size: Number of encoder outputs to accumulate for CTC decoding
            step_size: Number of new outputs before running CTC decode
        """
        self.model = model
        self.vocab = model.vocab
        self.window_size = window_size
        self.step_size = step_size
        
        # Extract encoder and decoder parts
        self._setup_streaming_model()
        
        # Buffers
        self.encoder_buffer = []
        self.lstm_states = None
        self.steps_since_decode = 0
        
    def _setup_streaming_model(self):
        """Setup encoder and decoder for streaming."""
        # The model architecture is:
        # Input -> Encoder (MixedNet) -> LSTM Decoder -> CTC Output
        
        # We need to split this into parts for streaming
        # For now, we'll use the full model and handle buffering externally
        # In a production system, you'd want to properly convert the LSTM to stateful
        pass
    
    def reset(self):
        """Reset streaming state."""
        self.encoder_buffer = []
        self.lstm_states = None
        self.steps_since_decode = 0
    
    def process_frame(self, audio_frame):
        """Process a single audio frame in streaming mode.
        
        Args:
            audio_frame: Audio input for one time step
            
        Returns:
            wake_word_detected: Boolean indicating if wake word was detected
            confidence: Confidence score (0-1)
        """
        # Run through the model
        # In streaming mode, we'd process one frame at a time through the encoder
        # For now, we'll accumulate frames and process in windows
        
        # This is a simplified implementation
        # A full implementation would require converting the model to streaming mode
        
        # Add frame to buffer
        self.encoder_buffer.append(audio_frame)
        
        # Check if we should run decoding
        self.steps_since_decode += 1
        
        if self.steps_since_decode >= self.step_size and len(self.encoder_buffer) >= self.window_size:
            # Take the last window_size frames
            window = np.array(self.encoder_buffer[-self.window_size:])
            
            # Run through model
            # Shape: (1, window_size, features)
            window_batch = np.expand_dims(window, axis=0)
            
            # Get predictions
            logits = self.model(window_batch, training=False)
            
            # Decode
            sequence_length = tf.constant([logits.shape[1]])
            _, wake_word_detected = ctc_utils.ctc_decode(
                logits, sequence_length, self.vocab
            )
            
            self.steps_since_decode = 0
            
            # Calculate confidence from logits
            # Simple approach: use max probability from CTC output
            probs = tf.nn.softmax(logits, axis=-1)
            confidence = tf.reduce_max(probs).numpy()
            
            return bool(wake_word_detected[0]), float(confidence)
        
        return False, 0.0
    
    def process_stream(self, audio_stream, frame_size=160):
        """Process an audio stream.
        
        Args:
            audio_stream: Iterator yielding audio chunks
            frame_size: Size of each frame (160 = 10ms at 16kHz)
            
        Yields:
            (timestamp, wake_word_detected, confidence)
        """
        timestamp = 0
        
        for audio_chunk in audio_stream:
            # Process chunk frame by frame
            for i in range(0, len(audio_chunk), frame_size):
                frame = audio_chunk[i:i + frame_size]
                if len(frame) == frame_size:
                    detected, confidence = self.process_frame(frame)
                    yield timestamp, detected, confidence
                    timestamp += frame_size / 16000.0  # Convert to seconds


def convert_to_streaming_model(model, flags):
    """Convert a trained CTC model to streaming format.
    
    This is a placeholder for the full streaming conversion.
    A complete implementation would:
    1. Convert MixedNet encoder to use streaming layers
    2. Convert LSTM to stateful LSTM
    3. Handle CTC decoding in streaming fashion
    
    Args:
        model: Trained non-streaming model
        flags: Model configuration flags
        
    Returns:
        Streaming-compatible model
    """
    # For now, return the original model
    # Full implementation would require modifying the model architecture
    return model