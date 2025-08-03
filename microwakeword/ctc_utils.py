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

"""CTC loss and decoding utilities for wake word detection."""

import tensorflow as tf
import numpy as np


def create_ctc_labels(batch_labels, vocab, max_label_length=None):
    """Convert binary labels to CTC sequence labels.
    
    Args:
        batch_labels: Binary labels (batch_size,) where 1 = wake word, 0 = not wake word
        vocab: Vocabulary dictionary mapping words to indices
        max_label_length: Maximum label length (for padding)
        
    Returns:
        labels: Sparse tensor labels for CTC loss
        label_lengths: Length of each label sequence
    """
    batch_size = tf.shape(batch_labels)[0]
    
    # Get wake word sequence (excluding blank token)
    wake_words = [word for word in vocab.keys() if word != '<blank>']
    wake_word_indices = [vocab[word] for word in wake_words]
    
    # Create label sequences
    label_sequences = []
    label_lengths = []
    
    for i in range(batch_size):
        if batch_labels[i] == 1:
            # Positive sample: use wake word sequence
            label_sequences.append(wake_word_indices)
            label_lengths.append(len(wake_word_indices))
        else:
            # Negative sample: empty sequence
            label_sequences.append([])
            label_lengths.append(0)
    
    # Pad sequences if needed
    if max_label_length is None:
        max_label_length = max(len(seq) for seq in label_sequences) if label_sequences else 0
    
    # Convert to dense tensor with padding
    padded_labels = []
    for seq in label_sequences:
        padded_seq = seq + [0] * (max_label_length - len(seq))
        padded_labels.append(padded_seq)
    
    labels = tf.constant(padded_labels, dtype=tf.int32)
    label_lengths = tf.constant(label_lengths, dtype=tf.int32)
    
    # Convert to sparse tensor for CTC loss
    indices = []
    values = []
    for i, seq in enumerate(label_sequences):
        for j, val in enumerate(seq):
            indices.append([i, j])
            values.append(val)
    
    if indices:
        sparse_labels = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=[batch_size, max_label_length]
        )
    else:
        # All negative samples - create empty sparse tensor
        sparse_labels = tf.SparseTensor(
            indices=tf.zeros([0, 2], dtype=tf.int64),
            values=tf.zeros([0], dtype=tf.int32),
            dense_shape=[batch_size, max_label_length]
        )
    
    return sparse_labels, label_lengths


class CTCLoss(tf.keras.losses.Loss):
    """CTC loss wrapper for Keras."""
    
    def __init__(self, blank_index=0, logits_time_major=False, name="ctc_loss"):
        super().__init__(name=name)
        self.blank_index = blank_index
        self.logits_time_major = logits_time_major
    
    def call(self, y_true, y_pred):
        """Compute CTC loss.
        
        Args:
            y_true: Tuple of (labels, label_lengths, logit_lengths)
            y_pred: Logits from model (batch, time, vocab_size)
            
        Returns:
            CTC loss value
        """
        # Unpack y_true
        labels = y_true[0]
        label_lengths = y_true[1]
        logit_lengths = y_true[2]
        
        # Compute CTC loss
        loss = tf.nn.ctc_loss(
            labels=labels,
            logits=y_pred,
            label_length=label_lengths,
            logit_length=logit_lengths,
            logits_time_major=self.logits_time_major,
            blank_index=self.blank_index
        )
        
        return tf.reduce_mean(loss)


def ctc_decode(logits, sequence_length, vocab, merge_repeated=True):
    """Decode CTC output to wake word predictions.
    
    Args:
        logits: Model output logits (batch, time, vocab_size)
        sequence_length: Valid length of each sequence
        vocab: Vocabulary dictionary
        merge_repeated: Whether to merge repeated tokens
        
    Returns:
        decoded_sequences: List of decoded word sequences
        wake_word_detected: Binary array indicating wake word detection
    """
    # Perform CTC beam search decoding
    decoded, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf.transpose(logits, [1, 0, 2]),  # Make time major
        sequence_length=sequence_length,
        merge_repeated=merge_repeated
    )
    
    # Convert sparse tensor to dense
    decoded_dense = tf.sparse.to_dense(decoded[0], default_value=-1)
    
    # Convert indices to words
    reverse_vocab = {v: k for k, v in vocab.items()}
    wake_words = [word for word in vocab.keys() if word != '<blank>']
    expected_sequence = [vocab[word] for word in wake_words]
    
    decoded_sequences = []
    wake_word_detected = []
    
    for sequence in decoded_dense.numpy():
        # Remove padding (-1 values)
        valid_indices = sequence[sequence >= 0]
        
        # Convert to words
        words = [reverse_vocab.get(idx, '') for idx in valid_indices if idx != 0]  # Skip blank
        decoded_sequences.append(words)
        
        # Check if it matches wake word sequence
        detected = (list(valid_indices[valid_indices != 0]) == expected_sequence)
        wake_word_detected.append(detected)
    
    return decoded_sequences, np.array(wake_word_detected, dtype=np.float32)


class CTCMetrics:
    """Metrics for CTC-based wake word detection."""
    
    def __init__(self, vocab):
        self.vocab = vocab
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.true_positives = 0
        self.false_positives = 0
        self.true_negatives = 0
        self.false_negatives = 0
    
    def update(self, y_true_binary, logits, sequence_lengths):
        """Update metrics with batch predictions.
        
        Args:
            y_true_binary: Binary ground truth (1 = wake word, 0 = not)
            logits: Model output logits
            sequence_lengths: Valid sequence lengths
        """
        _, predictions = ctc_decode(logits, sequence_lengths, self.vocab)
        
        for true_label, pred_label in zip(y_true_binary, predictions):
            if true_label == 1 and pred_label == 1:
                self.true_positives += 1
            elif true_label == 0 and pred_label == 1:
                self.false_positives += 1
            elif true_label == 0 and pred_label == 0:
                self.true_negatives += 1
            else:  # true_label == 1 and pred_label == 0
                self.false_negatives += 1
    
    def get_metrics(self):
        """Calculate and return metrics."""
        total = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
        if total == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        accuracy = (self.true_positives + self.true_negatives) / total
        
        precision = self.true_positives / (self.true_positives + self.false_positives) \
            if (self.true_positives + self.false_positives) > 0 else 0.0
        
        recall = self.true_positives / (self.true_positives + self.false_negatives) \
            if (self.true_positives + self.false_negatives) > 0 else 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall) \
            if (precision + recall) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }