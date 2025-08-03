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
    testing_fingerprints, testing_ground_truth, _ = data_processor.get_data(
        test_set,
        batch_size=config["batch_size"],
        features_length=config["spectrogram_length"],
        truncation_strategy="truncate_start",
    )
    testing_ground_truth = testing_ground_truth.reshape(-1, 1)

    model.reset_metrics()

    result = model.evaluate(
        testing_fingerprints,
        testing_ground_truth,
        batch_size=1024,
        return_dict=True,
        verbose=0,
    )

    metrics = {}
    metrics["accuracy"] = result["accuracy"]
    metrics["recall"] = result["recall"]
    metrics["precision"] = result["precision"]

    metrics["auc"] = result["auc"]
    metrics["loss"] = result["loss"]
    metrics["recall_at_no_faph"] = 0
    metrics["cutoff_for_no_faph"] = 0
    metrics["ambient_false_positives"] = 0
    metrics["ambient_false_positives_per_hour"] = 0
    metrics["average_viable_recall"] = 0

    test_set_fp = result["fp"].numpy()

    if data_processor.get_mode_size("validation_ambient") > 0:
        (
            ambient_testing_fingerprints,
            ambient_testing_ground_truth,
            _,
        ) = data_processor.get_data(
            test_set + "_ambient",
            batch_size=config["batch_size"],
            features_length=config["spectrogram_length"],
            truncation_strategy="split",
        )
        ambient_testing_ground_truth = ambient_testing_ground_truth.reshape(-1, 1)

        # XXX: tf no longer provides a way to evaluate a model without updating metrics
        with swap_attribute(model, "reset_metrics", lambda: None):
            ambient_predictions = model.evaluate(
                ambient_testing_fingerprints,
                ambient_testing_ground_truth,
                batch_size=1024,
                return_dict=True,
                verbose=0,
            )

        duration_of_ambient_set = (
            data_processor.get_mode_duration("validation_ambient") / 3600.0
        )

        # Other than the false positive rate, all other metrics are accumulated across
        # both test sets
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
    
    # Get adversarial lambda from config flags if available
    adversarial_lambda = 1.0
    if is_adversarial and 'flags' in config:
        # config['flags'] is a dict, not an object
        adversarial_lambda = config['flags'].get('adversarial_lambda', 1.0)
    
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

    if is_adversarial:
        # For adversarial models, use dict of losses and metrics
        loss = {
            "wake_word": tf.keras.losses.BinaryCrossentropy(from_logits=False),
            "tts_classifier": tf.keras.losses.BinaryCrossentropy(from_logits=False)
        }
        
        # Loss weights with adversarial lambda
        loss_weights = {
            "wake_word": 1.0,
            "tts_classifier": adversarial_lambda
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
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        
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
            
            # Apply class weights to wake word samples only
            class_weights = {0: negative_class_weight, 1: positive_class_weight}
            wake_word_weights = train_sample_weights * np.vectorize(class_weights.get)(
                train_ground_truth
            )
            
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
                    wake_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
                    tts_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
                    
                    wake_losses = wake_loss_fn(train_ground_truth, wake_pred)
                    tts_losses = tts_loss_fn(train_tts_labels, tts_pred)
                    
                    # Apply sample weights to wake word loss
                    weighted_wake_loss = tf.reduce_mean(wake_losses * wake_word_weights.reshape(-1, 1))
                    tts_loss = tf.reduce_mean(tts_losses)
                    
                    # Total loss with adversarial lambda
                    total_loss = weighted_wake_loss + adversarial_lambda * tts_loss
                
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
            
            result = model.train_on_batch(
                train_fingerprints,
                train_ground_truth,
                sample_weight=combined_weights,
            )

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
        
        # Add current batch metrics to accumulation
        accumulated_wake_accuracy += wake_accuracy
        accumulated_wake_recall += wake_recall
        accumulated_wake_precision += wake_precision
        accumulated_wake_loss += wake_loss
        accumulated_total_loss += total_loss
        
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
