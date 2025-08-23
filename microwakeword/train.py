# Copyright 2023 The Google Research Authors.
# Modifications copyright 2024 Kevin Ahrendt.
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

import contextlib
import os

from absl import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.util import tf_decorator


@contextlib.contextmanager
def swap_attribute(obj, attr, temp_value):
    """Temporarily swap an attribute of an object."""
    original_value = getattr(obj, attr)
    setattr(obj, attr, temp_value)

    try:
        yield
    finally:
        setattr(obj, attr, original_value)


def validate_nonstreaming(config, data_processor, model, test_set):
    # Handle both regular and adversarial data processors
    is_adversarial = hasattr(data_processor, '__class__') and 'Adversarial' in data_processor.__class__.__name__

    data_result = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )

    if is_adversarial:
        # Adversarial data processor returns 4 values
        testing_fingerprints, testing_ground_truth, testing_tts_labels, _ = data_result
        testing_ground_truth = testing_ground_truth.reshape(-1, 1)
        testing_tts_labels = testing_tts_labels.reshape(-1, 1)

        # For adversarial models, we need to pass both outputs
        y_test = {
            "wake_word": testing_ground_truth,
            "tts_classifier": testing_tts_labels
        }
    else:
        # Regular data processor returns 3 values
        testing_fingerprints, testing_ground_truth, _ = data_result
        testing_ground_truth = testing_ground_truth.reshape(-1, 1)
        y_test = testing_ground_truth

    model.reset_metrics()

    result = model.evaluate(
        testing_fingerprints,
        y_test,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    metrics = {}

    if is_adversarial:
        # For adversarial models, metrics are prefixed with output names
        metrics["accuracy"] = result.get("wake_word_accuracy", result.get("accuracy", 0))
        metrics["recall"] = result.get("wake_word_recall", result.get("recall", 0))
        metrics["precision"] = result.get("wake_word_precision", result.get("precision", 0))
        metrics["auc"] = result.get("wake_word_auc", result.get("auc", 0))
        metrics["loss"] = result.get("wake_word_loss", result.get("loss", 0))

        # Extract FP for wake word
        if "wake_word_fp" in result:
            test_set_fp = result["wake_word_fp"].numpy()
        elif "fp" in result:
            test_set_fp = result["fp"].numpy()
        else:
            test_set_fp = 0
    else:
        # Regular models have unprefixed metrics
        metrics["accuracy"] = result["accuracy"]
        metrics["recall"] = result["recall"]
        metrics["precision"] = result["precision"]
        metrics["auc"] = result["auc"]
        metrics["loss"] = result["loss"]
        test_set_fp = result["fp"].numpy()

    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    if data_processor.get_mode_size("validation_ambient") > 0:
        ambient_data_result = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )

        if is_adversarial:
            # Adversarial data processor returns 4 values
            ambient_testing_fingerprints, ambient_testing_ground_truth, ambient_tts_labels, _ = ambient_data_result
            ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)
            ambient_tts_labels = ambient_tts_labels.reshape(-1, 1)

            # For adversarial models, we need to pass both outputs
            y_ambient = {
                "wake_word": ambient_testing_ground_truth,
                "tts_classifier": ambient_tts_labels
            }
        else:
            # Regular data processor returns 3 values
            ambient_testing_fingerprints, ambient_testing_ground_truth, _ = ambient_data_result
            ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)
            y_ambient = ambient_testing_ground_truth

        # XXX: tf no longer provides a way to evaluate a model without updating metrics
        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(
                ambient_testing_fingerprints,
                y_ambient,
                batch_size=1024,
                return_dict=True,
                verbose=0,
            )

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
        if is_adversarial:
            # Extract wake word metrics for adversarial models
            all_true_positives = ambient_predictions.get("wake_word_tp", ambient_predictions.get("tp", [0])).numpy()
            ambient_fp = ambient_predictions.get("wake_word_fp", ambient_predictions.get("fp", [0])).numpy()
            ambient_false_positives = ambient_fp - test_set_fp
            all_false_negatives = ambient_predictions.get("wake_word_fn", ambient_predictions.get("fn", [0])).numpy()

            metrics["auc"] = ambient_predictions.get("wake_word_auc", ambient_predictions.get("auc", 0))
            metrics["loss"] = ambient_predictions.get("wake_word_loss", ambient_predictions.get("loss", 0))
        else:
            all_true_positives = ambient_predictions["tp"].numpy()
            ambient_false_positives = ambient_predictions["fp"].numpy() - test_set_fp
            all_false_negatives = ambient_predictions["fn"].numpy()

            metrics["auc"] = ambient_predictions["auc"]
            metrics["loss"] = ambient_predictions["loss"]

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

            # Increase index until we find a faph less than 2
            index_of_first_viable = 1
            while faph_at_cutoffs[index_of_first_viable] > 2:
                index_of_first_viable += 1

            x0 = faph_at_cutoffs[index_of_first_viable - 1]
            y0 = recall_at_cutoffs[index_of_first_viable - 1]
            x1 = faph_at_cutoffs[index_of_first_viable]
            y1 = recall_at_cutoffs[index_of_first_viable]

            recall_at_2faph = (y0 * (x1 - 2.0) + y1 * (2.0 - x0)) / (x1 - x0)
        else:
            # Lowest faph is already under 2, assume the recall is constant before this
            index_of_first_viable = 0
            recall_at_2faph = recall_at_cutoffs[0]

        x_coordinates = [2.0]
        y_coordinates = [recall_at_2faph]

        for index in range(index_of_first_viable, len(recall_at_cutoffs)):
            if faph_at_cutoffs[index] != x_coordinates[-1]:
                # Only add a point if it is a new faph
                # This ensures if a faph rate is repeated, we use the highest recall
                x_coordinates.append(faph_at_cutoffs[index])
                y_coordinates.append(recall_at_cutoffs[index])

        # Use trapezoid rule to estimate the area under the curve, then divide by 2.0 to get the average recall
        average_viable_recall = (
            np.trapz(np.flip(y_coordinates), np.flip(x_coordinates)) / 2.0
        )

        metrics["recall_at_no_faph"] = recall_at_no_faph
        metrics["cutoff_for_no_faph"] = target_faph_cutoff_probability
        metrics["ambient_false_positives"] = ambient_false_positives[50]
        metrics["ambient_false_positives_per_hour"] = faph_at_cutoffs[50]
        metrics["average_viable_recall"] = average_viable_recall

    return metrics


def train(model, config, data_processor):
    # Detect if we're using adversarial training based on data processor type
    is_adversarial = hasattr(data_processor, '__class__') and 'Adversarial' in data_processor.__class__.__name__

    # Get adversarial beta from config flags if available
    adversarial_beta = 0.5  # Default to balanced training
    if is_adversarial and 'flags' in config:
        # config['flags'] is a dict, not an object
        adversarial_beta = config['flags'].get('adversarial_beta', 0.5)

    # Get hard negative mining settings from config (YAML or command-line)
    use_hard_negative_mining = config.get("use_hard_negative_mining", 0)
    if use_hard_negative_mining == 0 and 'flags' in config:
        use_hard_negative_mining = config['flags'].get('use_hard_negative_mining', 0)

    hard_negative_k = config.get("hard_negative_k", 50)
    if 'flags' in config and config['flags'].get('hard_negative_k') is not None:
        hard_negative_k = config['flags'].get('hard_negative_k')

    hard_negative_start_step = config.get("hard_negative_start_step", 0)
    if 'flags' in config and config['flags'].get('hard_negative_start_step') is not None:
        hard_negative_start_step = config['flags'].get('hard_negative_start_step')

    # Assign default training settings if not set in the configuration yaml
    if not (training_steps_list := config.get("training_steps")):
        training_steps_list = [20000]
    if not (learning_rates_list := config.get("learning_rates")):
        learning_rates_list = [0.001]
    if not (mix_up_prob_list := config.get("mix_up_augmentation_prob")):
        mix_up_prob_list = [0.0]
    if not (freq_mix_prob_list := config.get("freq_mix_augmentation_prob")):
        freq_mix_prob_list = [0.0]
    if not (time_mask_max_size_list := config.get("time_mask_max_size")):
        time_mask_max_size_list = [5]
    if not (time_mask_count_list := config.get("time_mask_count")):
        time_mask_count_list = [2]
    if not (freq_mask_max_size_list := config.get("freq_mask_max_size")):
        freq_mask_max_size_list = [5]
    if not (freq_mask_count_list := config.get("freq_mask_count")):
        freq_mask_count_list = [2]
    if not (positive_class_weight_list := config.get("positive_class_weight")):
        positive_class_weight_list = [1.0]
    if not (negative_class_weight_list := config.get("negative_class_weight")):
        negative_class_weight_list = [1.0]

    # Ensure all training setting lists are as long as the training step iterations
    def pad_list_with_last_entry(list_to_pad, desired_length):
        while len(list_to_pad) < desired_length:
            last_entry = list_to_pad[-1]
            list_to_pad.append(last_entry)

    training_step_iterations = len(training_steps_list)
    pad_list_with_last_entry(learning_rates_list, training_step_iterations)
    pad_list_with_last_entry(mix_up_prob_list, training_step_iterations)
    pad_list_with_last_entry(freq_mix_prob_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(time_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_max_size_list, training_step_iterations)
    pad_list_with_last_entry(freq_mask_count_list, training_step_iterations)
    pad_list_with_last_entry(positive_class_weight_list, training_step_iterations)
    pad_list_with_last_entry(negative_class_weight_list, training_step_iterations)

    optimizer = tf.keras.optimizers.Adam()
    cutoffs = np.linspace(0.0, 1.0, 101).tolist()

    # Get focal loss settings from config (YAML or command-line)
    use_focal_loss = config.get("use_focal_loss", 0)
    if use_focal_loss == 0 and 'flags' in config:
        use_focal_loss = config['flags'].get('use_focal_loss', 0)

    focal_alpha = config.get("focal_alpha", 0.25)
    if 'flags' in config and config['flags'].get('focal_alpha') is not None:
        focal_alpha = config['flags'].get('focal_alpha')

    focal_gamma = config.get("focal_gamma", 2.0)
    if 'flags' in config and config['flags'].get('focal_gamma') is not None:
        focal_gamma = config['flags'].get('focal_gamma')

    # Create base loss function based on focal loss setting
    if use_focal_loss:
        logging.info(f"Using focal loss with alpha={focal_alpha}, gamma={focal_gamma}")
        base_loss = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=focal_alpha,
            gamma=focal_gamma,
            from_logits=False
        )
    else:
        base_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    if is_adversarial:
        # For adversarial models, use focal loss only for wake word
        loss = {
            "wake_word": base_loss,
            "tts_classifier": tf.keras.losses.BinaryCrossentropy(from_logits=False)
        }

        # Loss weights with adversarial lambda
        loss_weights = {
            "wake_word": 1.0 - adversarial_beta,
            "tts_classifier": adversarial_beta
        }

        metrics = {
            "wake_word": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
                tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
                tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
                tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.BinaryCrossentropy(name="loss"),
            ],
            "tts_classifier": [
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.BinaryCrossentropy(name="loss"),
            ]
        }

        model.compile(optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics)
    else:
        # Regular single-output model
        loss = base_loss

        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.TruePositives(name="tp", thresholds=cutoffs),
            tf.keras.metrics.FalsePositives(name="fp", thresholds=cutoffs),
            tf.keras.metrics.TrueNegatives(name="tn", thresholds=cutoffs),
            tf.keras.metrics.FalseNegatives(name="fn", thresholds=cutoffs),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.BinaryCrossentropy(name="loss"),
        ]

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # We un-decorate the `tf.function`, it's very slow to manually run training batches
    model.make_train_function()
    _, model.train_function = tf_decorator.unwrap(model.train_function)

    # Configure checkpointer and restore if available
    checkpoint_directory = os.path.join(config["train_dir"], "restore/")
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    # Configure TensorBoard summaries
    train_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "train")
    )
    validation_writer = tf.summary.create_file_writer(
        os.path.join(config["summaries_dir"], "validation")
    )

    training_steps_max = np.sum(training_steps_list)

    best_minimization_quantity = 10000
    best_maximization_quantity = 0.0
    best_no_faph_cutoff = 1.0

    # Accumulated statistics for the current evaluation interval
    accumulated_wake_accuracy = 0.0
    accumulated_wake_recall = 0.0
    accumulated_wake_precision = 0.0
    accumulated_wake_loss = 0.0
    accumulated_tts_accuracy = 0.0
    accumulated_tts_loss = 0.0
    accumulated_total_loss = 0.0

    # Hard negative mining statistics
    accumulated_num_positives = 0
    accumulated_num_negatives = 0
    accumulated_num_selected_negatives = 0

    for training_step in range(1, training_steps_max + 1):
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate = learning_rates_list[i]
                mix_up_prob = mix_up_prob_list[i]
                freq_mix_prob = freq_mix_prob_list[i]
                time_mask_max_size = time_mask_max_size_list[i]
                time_mask_count = time_mask_count_list[i]
                freq_mask_max_size = freq_mask_max_size_list[i]
                freq_mask_count = freq_mask_count_list[i]
                positive_class_weight = positive_class_weight_list[i]
                negative_class_weight = negative_class_weight_list[i]
                break

        model.optimizer.learning_rate.assign(learning_rate)

        augmentation_policy = {
            "mix_up_prob": mix_up_prob,
            "freq_mix_prob": freq_mix_prob,
            "time_mask_max_size": time_mask_max_size,
            "time_mask_count": time_mask_count,
            "freq_mask_max_size": freq_mask_max_size,
            "freq_mask_count": freq_mask_count,
        }

        # Get data - handle both regular and adversarial data processors
        data_result = data_processor.get_data(
            "training",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="default",
            augmentation_policy=augmentation_policy,
        )

        if is_adversarial:
            # Adversarial data processor returns 4 values
            train_fingerprints, train_ground_truth, train_tts_labels, train_sample_weights = data_result
            train_ground_truth = train_ground_truth.reshape(-1, 1)
            train_tts_labels = train_tts_labels.reshape(-1, 1)

            # # Apply class weights to wake word samples only
            # class_weights = {0: negative_class_weight, 1: positive_class_weight}
            # wake_word_weights = train_sample_weights * np.vectorize(class_weights.get)(
            #     train_ground_truth
            # )

            # Format outputs as dict for multi-output model
            y_train = {
                "wake_word": train_ground_truth,
                "tts_classifier": train_tts_labels
            }

            # For Keras 3.x compatibility, we need to use a different approach
            # Instead of dict sample weights, we'll apply the weights in the loss
            # by using a custom training step
            keras_version = tuple(map(int, tf.keras.__version__.split('.')[:2]))
            if keras_version >= (3, 0):
                # Manual gradient computation for Keras 3.x
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = model(train_fingerprints, training=True)
                    wake_pred = predictions[0]
                    tts_pred = predictions[1]

                    # Calculate losses
                    if use_focal_loss:
                        wake_loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                            alpha=focal_alpha,
                            gamma=focal_gamma,
                            from_logits=False,
                            reduction='none'
                        )
                    else:
                        wake_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
                    tts_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

                    wake_losses = wake_loss_fn(train_ground_truth, wake_pred)
                    tts_losses = tts_loss_fn(train_tts_labels, tts_pred)

                    # Apply hard negative mining if enabled
                    if use_hard_negative_mining and training_step >= hard_negative_start_step:
                        # Apply weights and squeeze for processing
                        weighted_wake_losses = tf.squeeze(wake_losses) #* wake_word_weights.flatten()

                        # Separate positive and negative indices
                        positive_mask = tf.cast(train_ground_truth.flatten() == 1, tf.float32)
                        negative_mask = tf.cast(train_ground_truth.flatten() == 0, tf.float32)

                        # Get indices of negatives
                        negative_indices = tf.where(negative_mask > 0)
                        negative_loss_values = tf.gather(weighted_wake_losses, negative_indices)

                        # Determine how many negatives to keep
                        num_negatives = tf.shape(negative_indices)[0]
                        k = tf.minimum(hard_negative_k, num_negatives)

                        # Get top K negative indices
                        if num_negatives > 0:
                            _, top_k_indices = tf.nn.top_k(negative_loss_values[:, 0], k=k)
                            selected_negative_indices = tf.gather(negative_indices, top_k_indices)

                            # Create selection mask
                            selection_mask = positive_mask  # Start with all positives

                            # Add selected negatives to mask
                            updates = tf.ones(tf.shape(selected_negative_indices)[0])
                            selection_mask = tf.tensor_scatter_nd_update(
                                selection_mask,
                                selected_negative_indices,
                                updates
                            )
                        else:
                            # No negatives in batch, use all positives
                            selection_mask = positive_mask

                        # Apply selection mask to wake losses
                        masked_wake_losses = weighted_wake_losses * selection_mask
                        num_selected = tf.reduce_sum(selection_mask)
                        weighted_wake_loss = tf.reduce_sum(masked_wake_losses) / tf.maximum(num_selected, 1.0)

                        # Track hard negative mining statistics
                        num_pos = int(tf.reduce_sum(positive_mask))
                        num_neg = int(tf.reduce_sum(negative_mask))
                        num_selected_neg = int(tf.reduce_sum(selection_mask * negative_mask))

                        # # Log occasionally
                        # if training_step % 100 == 0:
                        #     logging.info(
                        #         "Hard negative mining (adversarial): %d/%d negatives selected, %d positives (step %d)",
                        #         num_selected_neg, num_neg, num_pos, training_step
                        #     )
                    else:
                        # Standard weighted loss without hard negative mining
                        weighted_wake_loss = tf.reduce_mean(wake_losses) # * wake_word_weights.reshape(-1, 1))

                    tts_loss = tf.reduce_mean(tts_losses)

                    # Total loss with adversarial lambda
                    total_loss = (1.0 - adversarial_beta)*weighted_wake_loss + adversarial_beta * tts_loss

                # Compute gradients and update
                gradients = tape.gradient(total_loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Update metrics manually
                # We'll track the key metrics ourselves
                wake_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    tf.round(wake_pred), train_ground_truth), tf.float32))
                wake_tp = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 1, tf.round(wake_pred) == 1), tf.float32))
                wake_fp = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 0, tf.round(wake_pred) == 1), tf.float32))
                wake_fn = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 1, tf.round(wake_pred) == 0), tf.float32))
                wake_recall = wake_tp / (wake_tp + wake_fn + 1e-7)
                wake_precision = wake_tp / (wake_tp + wake_fp + 1e-7)
                tts_accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    tf.round(tts_pred), train_tts_labels), tf.float32))

                # Build result list to match expected format
                result = [
                    float(total_loss),  # 0: total loss
                    float(wake_accuracy),  # 1: wake accuracy
                    float(wake_recall),  # 2: wake recall
                    float(wake_precision),  # 3: wake precision
                    0.0, 0.0, 0.0, 0.0,  # 4-7: placeholders for TP/FP/TN/FN
                    0.5,  # 8: AUC placeholder
                    0.0,  # 9: placeholder
                    float(weighted_wake_loss),  # 10: wake loss
                    float(tts_accuracy),  # 11: tts accuracy
                    float(tts_loss),  # 12: tts loss
                ]
            else:
                # Keras 2.x can handle dict sample weights
                sample_weights = {
                    "wake_word": wake_word_weights,
                    "tts_classifier": np.ones_like(train_sample_weights)
                }

                result = model.train_on_batch(
                    train_fingerprints,
                    y_train,
                    sample_weight=sample_weights,
                )
        else:
            # Regular data processor returns 3 values
            train_fingerprints, train_ground_truth, train_sample_weights = data_result
            train_ground_truth = train_ground_truth.reshape(-1, 1)

            class_weights = {0: negative_class_weight, 1: positive_class_weight}
            combined_weights = train_sample_weights * np.vectorize(class_weights.get)(
                train_ground_truth
            )

            # Check if we should use hard negative mining
            if use_hard_negative_mining and training_step >= hard_negative_start_step:
                # Use custom gradient computation for hard negative mining
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = model(train_fingerprints, training=True)

                    # Calculate unreduced losses
                    if use_focal_loss:
                        loss_fn = tf.keras.losses.BinaryFocalCrossentropy(
                            alpha=focal_alpha,
                            gamma=focal_gamma,
                            from_logits=False,
                            reduction='none'
                        )
                    else:
                        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')

                    losses_unreduced = loss_fn(train_ground_truth, predictions)
                    losses_unreduced = tf.squeeze(losses_unreduced)  # Remove extra dimension

                    # Apply sample weights
                    weighted_losses = losses_unreduced * combined_weights.flatten()

                    # Separate positive and negative indices
                    positive_mask = tf.cast(train_ground_truth.flatten() == 1, tf.float32)
                    negative_mask = tf.cast(train_ground_truth.flatten() == 0, tf.float32)

                    # Get indices of positives and negatives
                    positive_losses = weighted_losses * positive_mask
                    negative_losses = weighted_losses * negative_mask

                    # Sort negative losses and get top K
                    negative_indices = tf.where(negative_mask > 0)
                    negative_loss_values = tf.gather(weighted_losses, negative_indices)

                    # Determine how many negatives to keep
                    num_negatives = tf.shape(negative_indices)[0]
                    k = tf.minimum(hard_negative_k, num_negatives)

                    # Get top K negative indices
                    if num_negatives > 0:
                        _, top_k_indices = tf.nn.top_k(negative_loss_values[:, 0], k=k)
                        selected_negative_indices = tf.gather(negative_indices, top_k_indices)

                        # Create selection mask
                        selection_mask = positive_mask  # Start with all positives

                        # Add selected negatives to mask
                        updates = tf.ones(tf.shape(selected_negative_indices)[0])
                        selection_mask = tf.tensor_scatter_nd_update(
                            selection_mask,
                            selected_negative_indices,
                            updates
                        )
                    else:
                        # No negatives in batch, use all positives
                        selection_mask = positive_mask

                    # Apply selection mask to losses
                    masked_losses = weighted_losses * selection_mask

                    # Calculate mean loss over selected samples
                    num_selected = tf.reduce_sum(selection_mask)
                    total_loss = tf.reduce_sum(masked_losses) / tf.maximum(num_selected, 1.0)

                # Compute gradients and update
                gradients = tape.gradient(total_loss, model.trainable_variables)
                model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                # Update metrics manually
                accuracy = tf.reduce_mean(tf.cast(tf.equal(
                    tf.round(predictions), train_ground_truth), tf.float32))
                tp = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 1, tf.round(predictions) == 1), tf.float32))
                fp = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 0, tf.round(predictions) == 1), tf.float32))
                fn = tf.reduce_sum(tf.cast(
                    tf.logical_and(train_ground_truth == 1, tf.round(predictions) == 0), tf.float32))
                recall = tp / (tp + fn + 1e-7)
                precision = tp / (tp + fp + 1e-7)

                # Build result list to match expected format
                result = [
                    float(total_loss),  # 0: total loss
                    float(accuracy),  # 1: accuracy
                    float(recall),  # 2: recall
                    float(precision),  # 3: precision
                    0.0, 0.0, 0.0, 0.0,  # 4-7: placeholders for TP/FP/TN/FN
                    0.5,  # 8: AUC placeholder
                    float(total_loss),  # 9: loss
                ]

                # Track hard negative mining statistics
                num_pos = int(tf.reduce_sum(positive_mask))
                num_neg = int(tf.reduce_sum(negative_mask))
                num_selected_neg = int(tf.reduce_sum(selection_mask * negative_mask))

                # Log occasionally
                if training_step % 100 == 0:
                    logging.info(
                        "Hard negative mining: %d/%d negatives selected, %d positives (step %d)",
                        num_selected_neg, num_neg, num_pos, training_step
                    )
            else:
                # Standard training without hard negative mining
                result = model.train_on_batch(
                    train_fingerprints,
                    train_ground_truth,
                    sample_weight=combined_weights,
                )

        # Initialize hard negative mining stats for this batch
        num_pos = 0
        num_neg = 0
        num_selected_neg = 0

        # Extract metrics based on model type
        if is_adversarial:
            # For adversarial models, result is a list with metrics for both outputs
            # The order depends on the metrics defined, but we can extract by position
            # Total loss is first, then wake_word metrics, then tts_classifier metrics
            total_loss = result[0]
            # Wake word metrics start at index 1
            wake_accuracy = result[1]
            wake_recall = result[2]
            wake_precision = result[3]
            wake_loss = result[10]  # Wake word loss metric
            # TTS classifier accuracy is after wake word metrics
            tts_accuracy = result[11]
            tts_loss = result[12]
        else:
            # Regular model metrics
            wake_accuracy = result[1]
            wake_recall = result[2]
            wake_precision = result[3]
            wake_loss = result[9]
            total_loss = wake_loss
            tts_accuracy = None
            tts_loss = None

        # Accumulate statistics across mini-batches
        mini_batch_num = (training_step - 1) % config["eval_step_interval"] + 1

        # Reset accumulation at the start of each evaluation interval
        if mini_batch_num == 1:
            accumulated_wake_accuracy = 0.0
            accumulated_wake_recall = 0.0
            accumulated_wake_precision = 0.0
            accumulated_wake_loss = 0.0
            accumulated_tts_accuracy = 0.0
            accumulated_tts_loss = 0.0
            accumulated_total_loss = 0.0
            accumulated_num_positives = 0
            accumulated_num_negatives = 0
            accumulated_num_selected_negatives = 0

        # Add current batch metrics to accumulation
        accumulated_wake_accuracy += wake_accuracy
        accumulated_wake_recall += wake_recall
        accumulated_wake_precision += wake_precision
        accumulated_wake_loss += wake_loss
        accumulated_total_loss += total_loss

        # Accumulate hard negative mining statistics
        accumulated_num_positives += num_pos
        accumulated_num_negatives += num_neg
        accumulated_num_selected_negatives += num_selected_neg

        if is_adversarial:
            accumulated_tts_accuracy += tts_accuracy
            accumulated_tts_loss += tts_loss

        # Calculate running averages
        avg_wake_accuracy = accumulated_wake_accuracy / mini_batch_num
        avg_wake_recall = accumulated_wake_recall / mini_batch_num
        avg_wake_precision = accumulated_wake_precision / mini_batch_num
        avg_wake_loss = accumulated_wake_loss / mini_batch_num
        avg_total_loss = accumulated_total_loss / mini_batch_num

        if is_adversarial:
            avg_tts_accuracy = accumulated_tts_accuracy / mini_batch_num
            avg_tts_loss = accumulated_tts_loss / mini_batch_num

        # Print the running statistics in the current validation epoch
        if is_adversarial:
            print(
                "Validation Batch #{:d}: Acc={:.3f}; Rec={:.3f}; Prec={:.3f}; Loss={:.4f} (wake={:.4f}, tts={:.4f}); TTS_Acc={:.3f}; Mini-Batch #{:d}/{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    avg_wake_accuracy,
                    avg_wake_recall,
                    avg_wake_precision,
                    avg_total_loss,
                    avg_wake_loss,
                    avg_tts_loss,
                    avg_tts_accuracy,
                    mini_batch_num,
                    config["eval_step_interval"],
                ),
                end="\r",
            )
        else:
            print(
                "Validation Batch #{:d}: Acc={:.3f}; Rec={:.3f}; Prec={:.3f}; Loss={:.4f}; Mini-Batch #{:d}/{:d}".format(
                    (training_step // config["eval_step_interval"] + 1),
                    avg_wake_accuracy,
                    avg_wake_recall,
                    avg_wake_precision,
                    avg_wake_loss,
                    mini_batch_num,
                    config["eval_step_interval"],
                ),
                end="\r",
            )

        is_last_step = training_step == training_steps_max
        if (training_step % config["eval_step_interval"]) == 0 or is_last_step:
            if is_adversarial:
                logging.info(
                    "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, loss %f (wake %f, tts %f), tts_acc %.2f%%",
                    *(
                        training_step,
                        learning_rate,
                        avg_wake_accuracy * 100,
                        avg_wake_recall * 100,
                        avg_wake_precision * 100,
                        avg_total_loss,
                        avg_wake_loss,
                        avg_tts_loss,
                        avg_tts_accuracy * 100,
                    ),
                )
            else:
                logging.info(
                    "Step #%d: rate %f, accuracy %.2f%%, recall %.2f%%, precision %.2f%%, cross entropy %f",
                    *(
                        training_step,
                        learning_rate,
                        avg_wake_accuracy * 100,
                        avg_wake_recall * 100,
                        avg_wake_precision * 100,
                        avg_wake_loss,
                    ),
                )

            with train_writer.as_default():
                if is_adversarial:
                    tf.summary.scalar("loss/total", avg_total_loss, step=training_step)
                    tf.summary.scalar("loss/wake_word", avg_wake_loss, step=training_step)
                    tf.summary.scalar("loss/tts_classifier", avg_tts_loss, step=training_step)
                    tf.summary.scalar("accuracy/wake_word", avg_wake_accuracy, step=training_step)
                    tf.summary.scalar("accuracy/tts_classifier", avg_tts_accuracy, step=training_step)
                    tf.summary.scalar("recall", avg_wake_recall, step=training_step)
                    tf.summary.scalar("precision", avg_wake_precision, step=training_step)
                    tf.summary.scalar("auc", result[8], step=training_step)
                else:
                    tf.summary.scalar("loss", avg_wake_loss, step=training_step)
                    tf.summary.scalar("accuracy", avg_wake_accuracy, step=training_step)
                    tf.summary.scalar("recall", avg_wake_recall, step=training_step)
                    tf.summary.scalar("precision", avg_wake_precision, step=training_step)
                    tf.summary.scalar("auc", result[8], step=training_step)

                # Add hard negative mining summaries if enabled
                if use_hard_negative_mining and training_step >= hard_negative_start_step:
                    avg_positives = accumulated_num_positives / mini_batch_num
                    avg_negatives = accumulated_num_negatives / mini_batch_num
                    avg_selected_negatives = accumulated_num_selected_negatives / mini_batch_num
                    selection_ratio = avg_selected_negatives / max(avg_negatives, 1.0)

                    tf.summary.scalar("hard_negative_mining/num_positives", avg_positives, step=training_step)
                    tf.summary.scalar("hard_negative_mining/num_negatives", avg_negatives, step=training_step)
                    tf.summary.scalar("hard_negative_mining/num_selected_negatives", avg_selected_negatives, step=training_step)
                    tf.summary.scalar("hard_negative_mining/selection_ratio", selection_ratio, step=training_step)

                train_writer.flush()

            model.save_weights(
                os.path.join(config["train_dir"], "last_weights.weights.h5")
            )

            nonstreaming_metrics = validate_nonstreaming(
                config, data_processor, model, "validation"
            )
            model.reset_metrics()  # reset metrics for next validation epoch of training
            logging.info(
                "Step %d (nonstreaming): Validation: recall at no faph = %.3f with cutoff %.2f, accuracy = %.2f%%, recall = %.2f%%, precision = %.2f%%, ambient false positives = %d, estimated false positives per hour = %.5f, loss = %.5f, auc = %.5f, average viable recall = %.9f",
                *(
                    training_step,
                    nonstreaming_metrics["recall_at_no_faph"] * 100,
                    nonstreaming_metrics["cutoff_for_no_faph"],
                    nonstreaming_metrics["accuracy"] * 100,
                    nonstreaming_metrics["recall"] * 100,
                    nonstreaming_metrics["precision"] * 100,
                    nonstreaming_metrics["ambient_false_positives"],
                    nonstreaming_metrics["ambient_false_positives_per_hour"],
                    nonstreaming_metrics["loss"],
                    nonstreaming_metrics["auc"],
                    nonstreaming_metrics["average_viable_recall"],
                ),
            )

            with validation_writer.as_default():
                tf.summary.scalar(
                    "loss", nonstreaming_metrics["loss"], step=training_step
                )
                tf.summary.scalar(
                    "accuracy", nonstreaming_metrics["accuracy"], step=training_step
                )
                tf.summary.scalar(
                    "recall", nonstreaming_metrics["recall"], step=training_step
                )
                tf.summary.scalar(
                    "precision", nonstreaming_metrics["precision"], step=training_step
                )
                tf.summary.scalar(
                    "recall_at_no_faph",
                    nonstreaming_metrics["recall_at_no_faph"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "auc",
                    nonstreaming_metrics["auc"],
                    step=training_step,
                )
                tf.summary.scalar(
                    "average_viable_recall",
                    nonstreaming_metrics["average_viable_recall"],
                    step=training_step,
                )
                validation_writer.flush()

            os.makedirs(os.path.join(config["train_dir"], "train"), exist_ok=True)

            model.save_weights(
                os.path.join(
                    config["train_dir"],
                    "train",
                    f"{int(best_minimization_quantity * 10000)}_weights_{training_step}.weights.h5",
                )
            )

            current_minimization_quantity = 0.0
            if config["minimization_metric"] is not None:
                current_minimization_quantity = nonstreaming_metrics[
                    config["minimization_metric"]
                ]
            current_maximization_quantity = nonstreaming_metrics[
                config["maximization_metric"]
            ]
            current_no_faph_cutoff = nonstreaming_metrics["cutoff_for_no_faph"]

            # Save model weights if this is a new best model
            if (
                (
                    (
                        current_minimization_quantity <= config["target_minimization"]
                    )  # achieved target false positive rate
                    and (
                        (
                            current_maximization_quantity > best_maximization_quantity
                        )  # either accuracy improved
                        or (
                            best_minimization_quantity > config["target_minimization"]
                        )  # or this is the first time we met the target
                    )
                )
                or (
                    (
                        current_minimization_quantity > config["target_minimization"]
                    )  # we haven't achieved our target
                    and (
                        current_minimization_quantity < best_minimization_quantity
                    )  # but we have decreased since the previous best
                )
                or (
                    (
                        current_minimization_quantity == best_minimization_quantity
                    )  # we tied a previous best
                    and (
                        current_maximization_quantity > best_maximization_quantity
                    )  # and we increased our accuracy
                )
            ):
                best_minimization_quantity = current_minimization_quantity
                best_maximization_quantity = current_maximization_quantity
                best_no_faph_cutoff = current_no_faph_cutoff

                # overwrite the best model weights
                model.save_weights(
                    os.path.join(config["train_dir"], "best_weights.weights.h5")
                )
                checkpoint.save(file_prefix=checkpoint_prefix)

            logging.info(
                "So far the best minimization quantity is %.3f with best maximization quantity of %.5f%%; no faph cutoff is %.2f",
                best_minimization_quantity,
                (best_maximization_quantity * 100),
                best_no_faph_cutoff,
            )

    # Save checkpoint after training
    checkpoint.save(file_prefix=checkpoint_prefix)
    model.save_weights(os.path.join(config["train_dir"], "last_weights.weights.h5"))
