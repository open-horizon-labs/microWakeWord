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

"""Alternative training approach for Keras 3.x compatibility."""

import os
import numpy as np
import tensorflow as tf
from absl import logging

from microwakeword.train_adversarial import validate_adversarial_nonstreaming


def create_training_step_function(model, optimizer, loss_fns, loss_weights=None):
    """Create a custom training step function for adversarial training.
    
    This bypasses Keras's built-in training loop to avoid sample weight issues.
    """
    if loss_weights is None:
        loss_weights = {"wake_word": 1.0, "tts_classifier": 1.0}
    
    @tf.function
    def train_step(x, y_wake, y_tts, sample_weights_wake):
        with tf.GradientTape() as tape:
            # Forward pass
            outputs = model(x, training=True)
            y_wake_pred = outputs[0]
            y_tts_pred = outputs[1]
            
            # Calculate losses
            wake_loss = loss_fns["wake_word"](y_wake, y_wake_pred)
            tts_loss = loss_fns["tts_classifier"](y_tts, y_tts_pred)
            
            # Apply sample weights to wake word loss
            wake_loss = wake_loss * sample_weights_wake
            wake_loss = tf.reduce_mean(wake_loss)
            tts_loss = tf.reduce_mean(tts_loss)
            
            # Combined loss
            total_loss = (loss_weights["wake_word"] * wake_loss + 
                         loss_weights["tts_classifier"] * tts_loss)
        
        # Backward pass
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Update metrics
        model.compiled_metrics.update_state(
            {"wake_word": y_wake, "tts_classifier": y_tts},
            {"wake_word": y_wake_pred, "tts_classifier": y_tts_pred}
        )
        
        return {
            "loss": total_loss,
            "wake_loss": wake_loss,
            "tts_loss": tts_loss
        }
    
    return train_step


def train_adversarial_keras3(
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
    """Alternative training function for Keras 3.x compatibility.
    
    This function uses a custom training loop to avoid sample weight issues.
    """
    # Compile model without sample weight
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
    )
    
    # Create custom training step
    train_step = create_training_step_function(model, optimizer, losses)
    
    # Restore checkpoint if requested
    if restore_checkpoint and os.path.exists(checkpoint_path):
        logging.info(f"Restoring from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    
    # Track best validation metric
    best_val_recall = 0.0
    
    # Calculate steps per epoch
    training_samples = data_processor.get_mode_size("training")
    steps_per_epoch = training_samples // batch_size
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Reset metrics
        model.reset_metrics()
        
        # Training for one epoch
        epoch_losses = []
        
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
            
            # Convert to tensors
            x_batch = tf.constant(train_fingerprints, dtype=tf.float32)
            y_wake_batch = tf.constant(train_ground_truth.reshape(-1, 1), dtype=tf.float32)
            y_tts_batch = tf.constant(train_tts_labels.reshape(-1, 1), dtype=tf.float32)
            sample_weights_batch = tf.constant(wake_word_sample_weights, dtype=tf.float32)
            
            # Execute training step
            step_losses = train_step(x_batch, y_wake_batch, y_tts_batch, sample_weights_batch)
            epoch_losses.append(step_losses["loss"].numpy())
            
            # Log progress
            if step % 100 == 0:
                metrics_values = {m.name: m.result().numpy() for m in model.metrics}
                logging.info(f"Step {step}/{steps_per_epoch} - loss: {step_losses['loss']:.4f} - {metrics_values}")
        
        # Save checkpoint
        model.save_weights(checkpoint_path)
        
        # Validate every eval_step_interval epochs
        if (epoch + 1) % config.get("eval_step_interval", 500) == 0:
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