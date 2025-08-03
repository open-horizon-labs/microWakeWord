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

"""Simplified adversarial training without Keras fit() API."""

import os
import numpy as np
import tensorflow as tf
from absl import logging

from microwakeword.train_adversarial import validate_adversarial_nonstreaming


def train_adversarial_simple(
    model,
    epochs,
    batch_size,
    flags,
    config,
    data_processor,
    checkpoint_path,
    best_checkpoint_path,
    tensorboard_path,
    optimizer,
    losses,
    metrics,
    restore_checkpoint,
    class_weights,
):
    """Simple adversarial training that bypasses Keras fit() entirely."""
    
    # Initialize optimizer if needed
    if optimizer is None:
        learning_rate = config.get("learning_rates", [0.001])[0]
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Loss functions
    wake_loss_fn = losses["wake_word"]
    tts_loss_fn = losses["tts_classifier"]
    
    # Restore checkpoint if requested
    if restore_checkpoint and os.path.exists(checkpoint_path):
        logging.info(f"Restoring from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    
    # Track best validation metric
    best_val_recall = 0.0
    
    # Calculate steps per epoch
    training_samples = data_processor.get_mode_size("training")
    steps_per_epoch = training_samples // batch_size
    
    logging.info(f"Starting training: {epochs} epochs, {steps_per_epoch} steps per epoch")
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"\nEpoch {epoch + 1}/{epochs}")
        
        epoch_wake_loss = 0.0
        epoch_tts_loss = 0.0
        epoch_total_loss = 0.0
        
        for step in range(steps_per_epoch):
            # Get batch of training data
            (
                train_fingerprints,
                train_ground_truth,
                train_tts_labels,
                train_weights
            ) = data_processor.get_data(
                "training",
                batch_size=batch_size,
                features_length=config["spectrogram_length"],
                truncation_strategy="truncate_start",
                augmentation_policy={
                    "freq_mix_prob": config.get("freq_mix_augmentation_prob", [0])[0],
                    "time_mask_max_size": config.get("time_mask_max_size", [0])[0],
                    "time_mask_count": config.get("time_mask_count", [0])[0],
                    "freq_mask_max_size": config.get("freq_mask_max_size", [0])[0],
                    "freq_mask_count": config.get("freq_mask_count", [0])[0],
                }
            )
            
            # Apply class weights to sample weights
            wake_word_sample_weights = train_weights.copy()
            for i in range(len(train_ground_truth)):
                if train_ground_truth[i] == 1:
                    wake_word_sample_weights[i] *= class_weights["wake_word"][1]
                else:
                    wake_word_sample_weights[i] *= class_weights["wake_word"][0]
            
            # Reshape labels
            train_ground_truth = train_ground_truth.reshape(-1, 1)
            train_tts_labels = train_tts_labels.reshape(-1, 1)
            
            # Training step
            with tf.GradientTape() as tape:
                # Forward pass
                outputs = model(train_fingerprints, training=True)
                wake_pred = outputs[0]
                tts_pred = outputs[1]
                
                # Calculate losses
                wake_loss = wake_loss_fn(train_ground_truth, wake_pred)
                tts_loss = tts_loss_fn(train_tts_labels, tts_pred)
                
                # Apply sample weights to wake word loss
                wake_loss = tf.reduce_mean(wake_loss * wake_word_sample_weights.reshape(-1, 1))
                tts_loss = tf.reduce_mean(tts_loss)
                
                # Total loss
                total_loss = wake_loss + tts_loss
            
            # Backward pass
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Track losses
            epoch_wake_loss += float(wake_loss)
            epoch_tts_loss += float(tts_loss)
            epoch_total_loss += float(total_loss)
            
            # Log progress
            if step % 100 == 0 and step > 0:
                avg_wake_loss = epoch_wake_loss / (step + 1)
                avg_tts_loss = epoch_tts_loss / (step + 1)
                avg_total_loss = epoch_total_loss / (step + 1)
                logging.info(
                    f"Step {step}/{steps_per_epoch} - "
                    f"loss: {avg_total_loss:.4f} - "
                    f"wake_loss: {avg_wake_loss:.4f} - "
                    f"tts_loss: {avg_tts_loss:.4f}"
                )
        
        # End of epoch summary
        avg_wake_loss = epoch_wake_loss / steps_per_epoch
        avg_tts_loss = epoch_tts_loss / steps_per_epoch
        avg_total_loss = epoch_total_loss / steps_per_epoch
        logging.info(
            f"Epoch {epoch + 1} completed - "
            f"avg_loss: {avg_total_loss:.4f} - "
            f"avg_wake_loss: {avg_wake_loss:.4f} - "
            f"avg_tts_loss: {avg_tts_loss:.4f}"
        )
        
        # Save checkpoint
        model.save_weights(checkpoint_path)
        
        # Validate every eval_step_interval epochs
        if (epoch + 1) % config.get("eval_step_interval", 500) == 0:
            logging.info("Running validation...")
            
            # Need to compile model for validation metrics
            if not hasattr(model, 'compiled'):
                model.compile(
                    optimizer=optimizer,
                    loss=losses,
                    metrics={
                        "wake_word": [
                            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                            tf.keras.metrics.Recall(name="recall"),
                            tf.keras.metrics.Precision(name="precision"),
                            tf.keras.metrics.AUC(name="auc"),
                            tf.keras.metrics.TruePositives(name="tp"),
                            tf.keras.metrics.FalsePositives(name="fp"),
                            tf.keras.metrics.FalseNegatives(name="fn")
                        ],
                        "tts_classifier": [
                            tf.keras.metrics.BinaryAccuracy(name="accuracy")
                        ]
                    }
                )
            
            val_metrics = validate_adversarial_nonstreaming(
                config, data_processor, model, "validation"
            )
            
            logging.info(f"Validation metrics at epoch {epoch + 1}:")
            logging.info(f"  Wake word accuracy: {val_metrics['accuracy']:.4f}")
            logging.info(f"  Wake word recall: {val_metrics['recall']:.4f}")
            logging.info(f"  Wake word precision: {val_metrics['precision']:.4f}")
            logging.info(f"  TTS classifier accuracy: {val_metrics['tts_accuracy']:.4f}")
            logging.info(f"  Ambient FA/hour: {val_metrics['ambient_false_positives_per_hour']:.4f}")
            
            # Save best model
            if val_metrics['recall'] > best_val_recall:
                best_val_recall = val_metrics['recall']
                model.save_weights(best_checkpoint_path)
                logging.info(f"Saved new best model with recall: {best_val_recall:.4f}")
    
    return model