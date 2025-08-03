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

"""Training functions for adversarial TTS-robust models."""

import contextlib
import os

from absl import logging
import numpy as np
import tensorflow as tf

from microwakeword.train import swap_attribute


def validate_adversarial_nonstreaming(config, data_processor, model, test_set):
    """Validate the adversarial model in non-streaming mode.
    
    Args:
        config: Training configuration dictionary
        data_processor: AdversarialDataProcessor instance
        model: Trained model with two outputs [wake_word, tts_classifier]
        test_set: Which test set to use ("validation" or "testing")
        
    Returns:
        Dictionary of metrics including wake word and TTS classification performance
    """
    testing_fingerprints, testing_ground_truth, testing_tts_labels, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)
    testing_tts_labels = testing_tts_labels.reshape(-1, 1)
    
    model.reset_metrics()
    
    # Evaluate model on both tasks
    result = model.evaluate(
        testing_fingerprints,
        {"wake_word": testing_ground_truth, "tts_classifier": testing_tts_labels},
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )
    
    metrics = {}
    # Wake word metrics
    metrics["accuracy"] = result["wake_word_accuracy"]
    metrics["recall"] = result["wake_word_recall"]
    metrics["precision"] = result["wake_word_precision"]
    metrics["auc"] = result["wake_word_auc"]
    metrics["loss"] = result["loss"]
    metrics["wake_word_loss"] = result["wake_word_loss"]
    
    # TTS classifier metrics
    metrics["tts_accuracy"] = result["tts_classifier_accuracy"]
    metrics["tts_loss"] = result["tts_classifier_loss"]
    
    # Initialize ambient metrics
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0
    
    test_set_fp = result["wake_word_fp"]
    
    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            ambient_testing_tts_labels,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)
        ambient_testing_tts_labels = ambient_testing_tts_labels.reshape(-1, 1)
        
        # Evaluate without updating metrics
        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(
                ambient_testing_fingerprints,
                {"wake_word": ambient_testing_ground_truth, "tts_classifier": ambient_testing_tts_labels},
                batch_size=1024,
                return_dict=True,
                verbose=0,
            )
        
        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )
        
        # Calculate ambient metrics for wake word performance
        all_true_positives = ambient_predictions["wake_word_tp"]
        ambient_false_positives = ambient_predictions["wake_word_fp"] - test_set_fp
        all_false_negatives = ambient_predictions["wake_word_fn"]
        
        metrics["auc"] = ambient_predictions["wake_word_auc"]
        metrics["loss"] = ambient_predictions["loss"]
        metrics["tts_accuracy"] = ambient_predictions["tts_classifier_accuracy"]
        
        recall_at_cutoffs = all_true_positives / (
            all_true_positives + all_false_negatives
        )
        faph_at_cutoffs = ambient_false_positives / duration_of_ambient_set
        
        target_faph_cutoff_probability = 1.0
        for index, cutoff in enumerate(np.linspace(0.0, 1.0, 101)):
            if faph_at_cutoffs[index] == 0:
                target_faph_cutoff_probability = cutoff
                recall_at_no_faph = recall_at_cutoffs[index]
                break
                
        if faph_at_cutoffs[0] > 2:
            # Use linear interpolation to estimate recall at 2 faph
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1
                
            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]
            
            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            recall_at_2faph = recall_at_cutoffs[0]
            
        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]
        
        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])
                
        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )
        
        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives
        metrics["ambient_false_positives_per_hour"] = ambient_false_positives / duration_of_ambient_set
        metrics["average_viable_recall"] = average_viable_recall
        
    return metrics


def train_adversarial_model(
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
    weighted_metrics,
    restore_checkpoint,
    class_weights,
):
    """Train the adversarial model with dual objectives.
    
    Args:
        model: The adversarial model with two outputs
        epochs: Number of epochs to train
        batch_size: Batch size for training
        flags: Command line flags
        config: Training configuration
        data_processor: AdversarialDataProcessor instance
        checkpoint_path: Path to save checkpoints
        best_checkpoint_path: Path to save best model
        tensorboard_path: Path for tensorboard logs
        optimizer: Optimizer instance
        losses: Dictionary of loss functions for each output
        metrics: Dictionary of metrics for each output
        weighted_metrics: Dictionary of weighted metrics
        restore_checkpoint: Whether to restore from checkpoint
        class_weights: Dictionary of class weights for each output
        
    Returns:
        Trained model
    """
    # Compile model with dual outputs
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        weighted_metrics=weighted_metrics,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    
    # Best model callback based on wake word performance
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path,
        save_weights_only=True,
        save_best_only=True,
        monitor='val_wake_word_recall',
        mode='max',
        verbose=1,
    )
    callbacks.append(best_model_callback)
    
    # Tensorboard callback
    if tensorboard_path:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=tensorboard_path,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard_callback)
    
    # Restore checkpoint if requested
    if restore_checkpoint and os.path.exists(checkpoint_path + '.index'):
        logging.info(f"Restoring from checkpoint: {checkpoint_path}")
        model.load_weights(checkpoint_path)
    
    # Training loop
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Get training data
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
        
        train_ground_truth = train_ground_truth.reshape(-1, 1)
        train_tts_labels = train_tts_labels.reshape(-1, 1)
        
        # Train for one epoch
        history = model.fit(
            train_fingerprints,
            {"wake_word": train_ground_truth, "tts_classifier": train_tts_labels},
            batch_size=batch_size,
            epochs=1,
            sample_weight={"wake_word": train_weights, "tts_classifier": np.ones_like(train_weights)},
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )
        
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
            
    return model