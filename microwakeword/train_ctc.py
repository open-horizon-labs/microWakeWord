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

"""Training utilities for CTC-based wake word models."""

import os
import tensorflow as tf
from absl import logging
import numpy as np

from microwakeword import train, ctc_utils


def compile_ctc_model(model, config, vocab):
    """Compile model with CTC loss and appropriate metrics.
    
    Args:
        model: Keras model with CTC output
        config: Training configuration
        vocab: Vocabulary dictionary
    """
    # Create custom loss that handles CTC
    ctc_loss = ctc_utils.CTCLoss()
    
    # Use Adam optimizer
    learning_rate = config.get("learning_rates", [0.001])[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile with CTC loss
    model.compile(
        optimizer=optimizer,
        loss=ctc_loss,
        run_eagerly=False  # Set to True for debugging
    )
    
    return model


def prepare_ctc_batch(spectrograms, labels, weights, vocab, model):
    """Prepare batch data for CTC training.
    
    Args:
        spectrograms: Input spectrograms (batch, time, features)
        labels: Binary labels (batch,)
        weights: Sample weights (batch,)
        vocab: Vocabulary dictionary
        model: Model to get encoder output steps
        
    Returns:
        Tuple of (spectrograms, ctc_labels) ready for training
    """
    batch_size = tf.shape(spectrograms)[0]
    
    # Get number of time steps in encoder output
    # This is needed for logit_length in CTC loss
    encoder_steps = model.encoder_steps
    if encoder_steps is None or encoder_steps == -1:
        # Calculate dynamically
        dummy_output = model(spectrograms[:1], training=False)
        encoder_steps = dummy_output.shape[1]
    
    # Create CTC labels
    sparse_labels, label_lengths = ctc_utils.create_ctc_labels(labels, vocab)
    
    # Create logit lengths (all sequences have same length after encoder)
    logit_lengths = tf.fill([batch_size], encoder_steps)
    
    # Pack labels for CTC loss
    ctc_labels = (sparse_labels, label_lengths, logit_lengths)
    
    return spectrograms, ctc_labels, weights


def train_ctc_model(model, config, data_processor, flags):
    """Train model with CTC loss.
    
    Args:
        model: Keras model with CTC output
        config: Training configuration
        data_processor: Data handler for loading batches
        flags: Model flags/parameters
    """
    # Get vocabulary from model
    vocab = model.vocab
    
    # Compile model
    model = compile_ctc_model(model, config, vocab)
    
    # Setup checkpointing
    checkpoint_path = os.path.join(config["train_dir"], "best_weights")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # Setup metrics tracking
    ctc_metrics = ctc_utils.CTCMetrics(vocab)
    
    # Training parameters
    training_steps = config["training_steps"][0]
    batch_size = config["batch_size"]
    eval_step_interval = config.get("eval_step_interval", 500)
    
    # Custom training loop for better control
    best_val_loss = float('inf')
    
    for step in range(training_steps):
        # Get training batch
        train_fingerprints, train_ground_truth, train_weights = data_processor.get_data(
            "training",
            batch_size=batch_size,
            features_length=config["spectrogram_length"],
            truncation_strategy="random",
        )
        
        # Prepare CTC batch
        x_train, y_train_ctc, sample_weights = prepare_ctc_batch(
            train_fingerprints, train_ground_truth, train_weights, vocab, model
        )
        
        # Training step
        with tf.GradientTape() as tape:
            predictions = model(x_train, training=True)
            loss = model.loss(y_train_ctc, predictions)
            
            # Apply sample weights
            weighted_loss = loss * sample_weights
            total_loss = tf.reduce_mean(weighted_loss)
        
        # Update weights
        gradients = tape.gradient(total_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Periodic evaluation
        if step % eval_step_interval == 0:
            # Get validation batch
            val_fingerprints, val_ground_truth, val_weights = data_processor.get_data(
                "validation",
                batch_size=batch_size,
                features_length=config["spectrogram_length"],
                truncation_strategy="truncate_start",
            )
            
            # Prepare CTC batch
            x_val, y_val_ctc, _ = prepare_ctc_batch(
                val_fingerprints, val_ground_truth, val_weights, vocab, model
            )
            
            # Validation step
            val_predictions = model(x_val, training=False)
            val_loss = model.loss(y_val_ctc, val_predictions)
            val_loss = tf.reduce_mean(val_loss)
            
            # Update metrics
            logit_lengths = y_val_ctc[2]
            ctc_metrics.reset()
            ctc_metrics.update(val_ground_truth, val_predictions, logit_lengths)
            metrics = ctc_metrics.get_metrics()
            
            logging.info(
                f"Step {step}/{training_steps} - "
                f"Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Accuracy: {metrics['accuracy']:.4f}, "
                f"Precision: {metrics['precision']:.4f}, "
                f"Recall: {metrics['recall']:.4f}"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(checkpoint_path)
                logging.info(f"Saved best model with val_loss: {val_loss:.4f}")
    
    # Save final weights
    final_path = os.path.join(config["train_dir"], "last_weights")
    model.save_weights(final_path)
    
    return model


def validate_ctc_nonstreaming(config, data_processor, model, test_set):
    """Validate CTC model in non-streaming mode.
    
    Args:
        config: Training configuration
        data_processor: Data handler
        model: Trained model
        test_set: Test set name
        
    Returns:
        Dictionary of metrics
    """
    vocab = model.vocab
    ctc_metrics = ctc_utils.CTCMetrics(vocab)
    
    # Get test data
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    
    # Process in batches
    batch_size = 32
    num_samples = len(testing_ground_truth)
    
    all_predictions = []
    
    for i in range(0, num_samples, batch_size):
        batch_end = min(i + batch_size, num_samples)
        batch_fingerprints = testing_fingerprints[i:batch_end]
        batch_ground_truth = testing_ground_truth[i:batch_end]
        
        # Get predictions
        logits = model(batch_fingerprints, training=False)
        
        # Get sequence lengths
        encoder_steps = logits.shape[1]
        sequence_lengths = tf.fill([batch_end - i], encoder_steps)
        
        # Decode predictions
        _, wake_word_detected = ctc_utils.ctc_decode(
            logits, sequence_lengths, vocab
        )
        
        all_predictions.extend(wake_word_detected)
        
        # Update metrics
        ctc_metrics.update(batch_ground_truth, logits, sequence_lengths)
    
    # Get final metrics
    metrics = ctc_metrics.get_metrics()
    
    # Add additional metrics for compatibility
    metrics["loss"] = 0.0  # CTC loss not computed during validation
    metrics["auc"] = 0.0  # Would need to compute from predictions
    
    # Compute ambient metrics if available
    if data_processor.get_mode_size(test_set + "_ambient") > 0:
        ambient_fingerprints, ambient_ground_truth, _ = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        
        # Process ambient data
        ambient_false_positives = 0
        for i in range(0, len(ambient_ground_truth), batch_size):
            batch_end = min(i + batch_size, len(ambient_ground_truth))
            batch_fingerprints = ambient_fingerprints[i:batch_end]
            
            logits = model(batch_fingerprints, training=False)
            sequence_lengths = tf.fill([batch_end - i], logits.shape[1])
            
            _, wake_word_detected = ctc_utils.ctc_decode(
                logits, sequence_lengths, vocab
            )
            
            ambient_false_positives += np.sum(wake_word_detected)
        
        duration_hours = data_processor.get_mode_duration(test_set + "_ambient") / 3600.0
        metrics["ambient_false_positives"] = ambient_false_positives
        metrics["ambient_false_positives_per_hour"] = ambient_false_positives / duration_hours
    else:
        metrics["ambient_false_positives"] = 0
        metrics["ambient_false_positives_per_hour"] = 0
    
    # Add placeholder metrics for compatibility
    metrics["recall_at_no_faph"] = metrics["recall"]  # Simplified
    metrics["cutoff_for_no_faph"] = 0.5
    metrics["average_viable_recall"] = metrics["recall"]
    
    return metrics