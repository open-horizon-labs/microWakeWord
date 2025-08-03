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

"""Extended data loading functionality for adversarial TTS training."""

import numpy as np

from microwakeword.data import (
    MmapFeatureGenerator, ClipsHandlerWrapperGenerator, 
    FeatureHandler, spec_augment
)
from microwakeword.audio.clips import Clips
from microwakeword.audio.augmentation import Augmentation
from microwakeword.audio.spectrograms import SpectrogramGeneration


class AdversarialMmapFeatureGenerator(MmapFeatureGenerator):
    """Extended MmapFeatureGenerator that includes TTS labels.
    
    This class extends the base FeatureSetProvider to track whether
    samples are from TTS or real speech for adversarial training.
    """
    
    def __init__(
        self,
        path: str,
        label: bool,
        is_tts: bool,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
        stride: int,
        step: float,
        fixed_right_cutoffs: list[int] = [0],
    ):
        """Initialize the adversarial feature provider.
        
        Args:
            path: Input directory to the Ragged MMaps
            label: The class each spectrogram represents (wake word or not)
            is_tts: Whether this dataset contains TTS-generated speech
            sampling_weight: The sampling weight for how frequently to sample
            penalty_weight: The penalizing weight for incorrect predictions
            truncation_strategy: How to truncate if spectrogram is too long
            stride: The stride in the model's first layer
            step: The window step duration (in seconds)
            fixed_right_cutoffs: List of spectrogram slices to cutoff on the right
        """
        super().__init__(
            path, label, sampling_weight, penalty_weight,
            truncation_strategy, stride, step, fixed_right_cutoffs
        )
        self.is_tts = float(is_tts)


class AdversarialClipsHandlerWrapperGenerator(ClipsHandlerWrapperGenerator):
    """Extended ClipsHandlerWrapperGenerator that includes TTS labels."""
    
    def __init__(
        self,
        spectrogram_generation,
        label: bool,
        is_tts: bool,
        sampling_weight: float,
        penalty_weight: float,
        truncation_strategy: str,
    ):
        """Initialize the adversarial clips handler.
        
        Args:
            spectrogram_generation: Object that handles generating spectrograms
            label: The class each spectrogram represents (wake word or not)
            is_tts: Whether this dataset contains TTS-generated speech
            sampling_weight: The sampling weight for how frequently to sample
            penalty_weight: The penalizing weight for incorrect predictions
            truncation_strategy: How to truncate if spectrogram is too long
        """
        super().__init__(
            spectrogram_generation, label, sampling_weight,
            penalty_weight, truncation_strategy
        )
        self.is_tts = float(is_tts)


class AdversarialFeatureHandler(FeatureHandler):
    """Extended FeatureHandler for adversarial TTS training.
    
    This processor returns an additional label for each sample indicating
    whether it's from TTS (1.0) or real speech (0.0).
    """
    
    def __init__(self, config: dict):
        """Initialize the adversarial feature handler.
        
        Args:
            config: Training configuration dictionary
        """
        # Initialize empty list before calling parent
        self.feature_providers = []
        
        # Process features with adversarial providers
        for feature_set in config["features"]:
            if feature_set.get("type", "mmap") == "mmap":
                # Determine if this is TTS data
                is_tts = feature_set.get("is_tts", "generated" in feature_set["features_dir"])
                
                self.feature_providers.append(
                    AdversarialMmapFeatureGenerator(
                        path=feature_set["features_dir"],
                        label=feature_set["truth"],
                        is_tts=is_tts,
                        sampling_weight=feature_set["sampling_weight"],
                        penalty_weight=feature_set["penalty_weight"],
                        truncation_strategy=feature_set["truncation_strategy"],
                        stride=config["stride"],
                        step=config["window_step_ms"] / 1000.0,
                        fixed_right_cutoffs=feature_set.get("fixed_right_cutoffs", [0]),
                    )
                )
            elif feature_set.get("type") == "clips":
                # Handle clips type with adversarial wrapper
                clips_handler = Clips(**feature_set["clips_settings"])
                augmentation_applier = Augmentation(**feature_set["augmentation_settings"])
                spectrogram_generator = SpectrogramGeneration(
                    clips_handler,
                    augmentation_applier,
                    **feature_set["spectrogram_generation_settings"],
                )
                is_tts = feature_set.get("is_tts", False)
                
                self.feature_providers.append(
                    AdversarialClipsHandlerWrapperGenerator(
                        spectrogram_generator,
                        feature_set["truth"],
                        is_tts,
                        feature_set["sampling_weight"],
                        feature_set["penalty_weight"],
                        feature_set["truncation_strategy"],
                    )
                )
    
    def get_data(
        self,
        mode: str,
        batch_size: int,
        features_length: int,
        truncation_strategy: str = "default",
        augmentation_policy: dict = {
            "freq_mix_prob": 0.0,
            "time_mask_max_size": 0,
            "time_mask_count": 0,
            "freq_mask_max_size": 0,
            "freq_mask_count": 0,
        },
    ):
        """Gets spectrograms with wake word and TTS labels.
        
        Args:
            mode: which training set to use
            batch_size: number of spectrograms in the sample for training mode
            features_length: the length of the spectrograms
            truncation_strategy: how to truncate spectrograms longer than features_length
            augmentation_policy: dictionary that specifies augmentation settings
            
        Returns:
            data: spectrograms in a NumPy array
            labels: wake word labels (0 or 1)
            tts_labels: TTS labels (0 for real, 1 for TTS)
            weights: penalizing weight for incorrect predictions
        """
        
        if mode == "training":
            sample_count = batch_size
        elif mode in ("validation", "testing"):
            sample_count = self.get_mode_size(mode)
            
        data = []
        labels = []
        tts_labels = []
        weights = []
        
        if mode == "training":
            random_feature_providers = np.random.choice(
                [
                    provider
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ],
                size=sample_count,
                replace=True,
                p=[
                    provider.sampling_weight
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ] / np.sum([
                    provider.sampling_weight
                    for provider in self.feature_providers
                    if provider.get_mode_size("training")
                ])
            )
            
            for provider in random_feature_providers:
                spectrogram = provider.get_random_spectrogram(
                    "training", features_length, truncation_strategy
                )
                spectrogram = spec_augment(
                    spectrogram,
                    augmentation_policy["time_mask_max_size"],
                    augmentation_policy["time_mask_count"],
                    augmentation_policy["freq_mask_max_size"],
                    augmentation_policy["freq_mask_count"],
                )
                
                data.append(spectrogram)
                labels.append(float(provider.label))
                tts_labels.append(float(provider.is_tts))
                weights.append(float(provider.penalty_weight))
        else:
            for provider in self.feature_providers:
                generator = provider.get_feature_generator(
                    mode, features_length, truncation_strategy
                )
                
                for spectrogram in generator:
                    data.append(spectrogram)
                    labels.append(provider.label)
                    tts_labels.append(provider.is_tts)
                    weights.append(provider.penalty_weight)
        
        if truncation_strategy != "none":
            # Spectrograms are all the same length, convert to numpy array
            data = np.array(data)
        labels = np.array(labels)
        tts_labels = np.array(tts_labels)
        weights = np.array(weights)
        
        if truncation_strategy == "none":
            # Spectrograms may be of different length
            return data, labels, tts_labels, weights
            
        indices = np.arange(labels.shape[0])
        
        # Randomize the order of the data, weights, and labels
        np.random.shuffle(indices)
        
        # FreqMix augmentation if enabled
        if mode == "training" and augmentation_policy["freq_mix_prob"] > 0:
            num_samples_to_augment = int(
                augmentation_policy["freq_mix_prob"] * sample_count
            )
            
            # Select samples for augmentation
            augment_indices = indices[:num_samples_to_augment]
            
            # For each sample to augment, find a pair to mix with
            for idx in augment_indices:
                # Find another sample with same wake word label
                same_label_indices = indices[labels[indices] == labels[idx]]
                if len(same_label_indices) > 1:
                    # Remove current index from candidates
                    candidates = same_label_indices[same_label_indices != idx]
                    pair_idx = np.random.choice(candidates)
                    
                    # FreqMix: swap frequency bins between samples
                    freq_bins = data.shape[2]
                    swap_size = np.random.randint(1, freq_bins // 2)
                    swap_start = np.random.randint(0, freq_bins - swap_size)
                    
                    # Swap frequency bins
                    temp = data[idx, :, swap_start:swap_start+swap_size].copy()
                    data[idx, :, swap_start:swap_start+swap_size] = data[pair_idx, :, swap_start:swap_start+swap_size]
                    data[pair_idx, :, swap_start:swap_start+swap_size] = temp
        
        return data[indices], labels[indices], tts_labels[indices], weights[indices]