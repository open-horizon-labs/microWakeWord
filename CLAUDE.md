# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

microWakeWord is a TensorFlow-based wake word detection training framework that produces models suitable for TensorFlow Lite for Microcontrollers. The project focuses on creating custom wake word models with low false accept/reject rates for low-power devices.

## Key Commands

### Installation
```bash
# Install microWakeWord (requires Python 3.10+)
pip install -e .

# macOS users need to install a forked version of pymicro-features first:
pip install 'git+https://github.com/puddly/pymicro-features@puddly/minimum-cpp-version'
```

### Training a Model
```bash
# Generate wake word samples using Piper
python3 piper-sample-generator/generate_samples.py "wake_word" \
    --max-samples 1000 \
    --batch-size 100 \
    --output-dir generated_samples

# Train the standard MixedNet model using the configuration
python -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block  "1, 1, 1, 1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3

# Train the MixedNet+CTC model (experimental)
python -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 0 \
    --use_weights "best_weights" \
    mixednet_ctc \
    --wake_word_phrase "hey jarvis" \
    --embedding_dim 64 \
    --lstm_units "64,32" \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block  "1, 1, 1, 1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3

# Train the adversarial TTS-robust model
python -m microwakeword.model_train_eval \
    --training_config='training_adversarial.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet_adversarial \
    --adversarial_beta 0.5 \
    --adversarial_lambda 1.0 \
    --adversarial_hidden_units "128,64" \
    --adversarial_dropout 0.5 \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block  "1, 1, 1, 1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3

# Train with focal loss for better handling of class imbalance
python -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    --use_focal_loss 1 \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    mixednet \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block  "1, 1, 1, 1" \
    --mixconv_kernel_sizes '[5], [7,11], [9,15], [23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3
```

### Focal Loss Configuration

Binary focal loss can be used as an alternative to standard binary cross entropy to better handle class imbalance in wake word detection:

**When to use focal loss:**
- When you have significantly more negative samples than positive wake word samples
- When the model is overfitting to the majority class (too many false negatives)
- When standard class weighting isn't sufficient to address imbalance

**Configuration options:**
- `--use_focal_loss 1`: Enable focal loss (default: 0)
- `--focal_alpha`: Class balancing factor (default: 0.25 for rare positive class)
- `--focal_gamma`: Focusing parameter (default: 2.0, higher values focus more on hard examples)

**YAML configuration:**
```yaml
# In your training_parameters.yaml
use_focal_loss: 1  # 0=standard BCE, 1=focal loss
focal_alpha: 0.25  # Class balance factor
focal_gamma: 2.0   # Focusing parameter
```

Note: Focal loss works with all model types (MixedNet, Inception, adversarial). For adversarial models, focal loss is only applied to the wake word classification branch, while the TTS classifier maintains standard binary cross entropy.

### Hard Negative Mining Configuration

Hard negative mining improves training efficiency by focusing on the most challenging negative samples. We support three strategies: fixed-K (from RepCNN paper), percentile-based selection, and curriculum learning.

**When to use hard negative mining:**
- When you have many easy negative samples that dominate training
- To improve model's ability to distinguish difficult negative cases
- When training time is limited and you want to focus on informative samples
- Can be combined with focal loss for even better handling of class imbalance

**Strategy 1: Fixed-K (Original RepCNN approach)**
Selects the top K hardest negative samples per batch:
```yaml
hard_negative_mining:
  enabled: 1
  strategy: fixed_k
  k: 50              # Number of hardest negatives to select
  start_step: 1000   # Warm-up period before starting
```

**Strategy 2: Percentile-based selection**
Selects negatives within a percentile range of loss values:
```yaml
hard_negative_mining:
  enabled: 1
  strategy: percentile
  percentile_lower: 0    # Select hardest 20% of negatives
  percentile_upper: 20   # (0 = hardest, 100 = easiest)
  max_k: 100            # Safety limit on number selected
  start_step: 1000      # Warm-up period
```

**Strategy 3: Curriculum learning (Recommended)**
Gradually focuses on harder examples during training:
```yaml
hard_negative_mining:
  enabled: 1
  strategy: curriculum
  max_k: 100        # Safety limit
  start_step: 0     # Can start immediately
  
hard_negative_curriculum:
  stages:
    # No mining initially (warm-up)
    - step: 0
      lower_pct: 100
      upper_pct: 100
    
    # Start with moderately hard examples
    - step: 1000
      lower_pct: 30   # Avoid outliers/noise
      upper_pct: 80
    
    # Gradually focus on harder examples
    - step: 5000
      lower_pct: 20
      upper_pct: 60
    
    # Focus on hardest examples
    - step: 15000
      lower_pct: 0
      upper_pct: 20
```

**How curriculum learning improves training:**
- **Early training (30-80th percentile)**: Learns core patterns without overfitting to noise
- **Mid training (20-60th percentile)**: Refines decision boundaries
- **Late training (0-20th percentile)**: Polishes performance on edge cases

**Command-line example (backwards compatibility):**
```bash
python -m microwakeword.model_train_eval \
    --training_config='training_parameters_hnm_example.yaml' \
    --train 1 \
    mixednet
```

**Monitoring:**
- TensorBoard tracks mining statistics under `hard_negative_mining/` namespace
- Curriculum stages show `percentile_lower` and `percentile_upper` over time
- Logs show selection ratio and sample counts
- Compatible with all model types (MixedNet, Inception, adversarial, CTC)

Note: See `training_parameters_hnm_example.yaml` for a complete configuration example. Hard negative mining can be combined with focal loss and class weighting for optimal performance.

### Testing
```bash
# Run unit tests
pytest tests/unit -v

# Run integration tests (fast only)
pytest tests/integration -v -m "not slow"

# Run all tests
pytest

# Run with coverage
pytest tests/unit -v --cov=microwakeword --cov-report=term
```

The project uses pytest for testing with unit and integration test suites. Slow integration tests are marked and can be excluded.

### Code Quality Tools
```bash
# Run linting
ruff check .

# Run formatting
ruff format .

# Install pre-commit hooks (recommended)
./scripts/install-git-hooks.sh

# Run pre-commit checks manually
pre-commit run --all-files
```

The project uses:
- **Ruff** for linting and code formatting
- **Pre-commit hooks** for automated code quality checks
- **Git hooks** to prevent direct commits to main branch

## Architecture

### Core Components

1. **Audio Processing** (`microwakeword/audio/`)
   - `audio_utils.py`: Audio file handling and manipulation
   - `augmentation.py`: Audio augmentation pipeline (noise, reverb, EQ, etc.)
   - `clips.py`: Managing audio clips and dataset splits
   - `spectrograms.py`: Spectrogram generation for model input
   - `cGenerateFeatures.so`: Native library for feature extraction

2. **Model Architecture** 
   - `mixednet.py`: MixedNet model using MixConv depthwise convolutions
   - `mixednet_ctc.py`: MixedNet encoder + LSTM decoder with CTC loss (experimental)
   - `mixednet_adversarial.py`: MixedNet with adversarial training for TTS robustness
   - `inception.py`: Inception-based model architecture
   - `layers/`: Custom TensorFlow layers for streaming inference
     - `stream.py`: Streaming layer implementations
     - `modes.py`: Training/inference mode management
     - `delay.py`, `strided_drop.py`: Streaming-specific layers
     - `gradient_reversal.py`: Gradient reversal layer for adversarial training

3. **Training Pipeline**
   - `train.py`: Unified training loop supporting both regular and adversarial models
   - `train_ctc.py`: CTC-specific training loop for MixedNet+CTC models
   - `model_train_eval.py`: Main entry point for training and evaluation
   - `data.py`: Data loading and preprocessing with Ragged Mmap support
   - `data_adversarial.py`: Extended data pipeline with TTS labels for adversarial training
   - `ctc_utils.py`: CTC loss, decoding, and metrics utilities
   - `utils_adversarial.py`: Utilities for extracting wake word model from adversarial model

4. **Inference**
   - `inference.py`: Model class for loading and running inference
   - `streaming_ctc.py`: Streaming inference support for CTC models
   - Converts non-streaming models to streaming for real-time detection
   - Supports quantization for embedded deployment

### Data Pipeline

1. **Sample Generation**: Uses Piper TTS to generate wake word samples
2. **Augmentation**: Applies various augmentations (background noise, reverb, pitch shift, etc.)
3. **Feature Extraction**: Converts audio to 40 Mel-like features every 10ms
4. **Storage**: Uses Ragged Mmap format for efficient disk-based training

### Model Training Process

1. Trains in non-streaming mode on full spectrograms
2. Uses SpecAugment for additional robustness
3. Two-stage optimization: minimize false accepts, then maximize accuracy
4. Converts to streaming model after training
5. Quantizes for embedded deployment

### MixedNet+CTC Architecture (Experimental)

The MixedNet+CTC model combines:
- **Encoder**: MixedNet CNN that outputs embeddings instead of binary classification
- **Decoder**: LSTM layers that process embeddings sequentially
- **CTC Loss**: Enables training without precise temporal alignment of wake words

Benefits:
- No need for careful alignment of wake word samples in training data
- Better handling of variable-length wake words
- Word-level understanding of the wake phrase

Parameters:
- `--wake_word_phrase`: Space-separated wake words (e.g., "hey jarvis")
- `--embedding_dim`: Size of encoder output embeddings (default: 64)
- `--lstm_units`: Comma-separated LSTM layer sizes (e.g., "64,32")

### MixedNet+Adversarial Architecture

The MixedNet+Adversarial model improves robustness to TTS-generated speech using domain-adversarial training:
- **Main Branch**: Standard wake word classification
- **Adversarial Branch**: Predicts if input is TTS or real speech
- **Gradient Reversal Layer**: Forces model to learn TTS-invariant features by reversing gradients during backpropagation

Benefits:
- 8-12% accuracy improvement on real speech (based on paper: https://arxiv.org/html/2408.10463v1)
- Reduced overfitting to TTS-specific artifacts
- No inference overhead (adversarial branch removed during deployment)

Parameters:
- `--adversarial_beta`: Loss weight balance (default: 0.5)
  - Controls balance between wake word loss (1-β) and TTS classifier loss (β)
  - Total loss = `(1-β) * wake_word_loss + β * tts_classifier_loss`
- `--adversarial_lambda`: Gradient reversal strength (default: 1.0)
  - Scales the reversed gradients in the gradient reversal layer
  - Higher values create stronger domain-invariant features
- `--adversarial_hidden_units`: Hidden layers for adversarial classifier (e.g., "128,64")
- `--adversarial_dropout`: Dropout rate in adversarial classifier (default: 0.5)

Training requires labeling data as TTS or real in the configuration file:
```yaml
features:
- features_dir: /path/to/tts_samples
  is_tts: true  # Mark as TTS-generated
- features_dir: /path/to/real_samples
  is_tts: false  # Mark as real speech
```

Training Notes:
- The unified `train.py` automatically detects adversarial models and handles multi-output training
- Validation metrics track both wake word performance and TTS classification accuracy
- Training displays accumulated statistics across mini-batches for smoother progress monitoring
- Keras 3.x compatibility is handled with manual gradient computation when needed

## Important Considerations

- Models are trained with a 16kHz sample rate
- Window duration: 30ms, stride: 10ms
- Default clip duration: 1500ms maximum
- Python 3.10+ is required (supports 3.10, 3.11, 3.12)
- Training requires significant experimentation with hyperparameters
- The `basic_training_notebook.ipynb` provides a starting point but won't produce production-ready models without tuning

## Development Workflow

1. **Branch Protection**: Direct commits to main are blocked. Use feature branches and pull requests.
2. **Pre-commit Hooks**: Automatically run linting and formatting checks before commits.
3. **CI/CD Pipeline**: 
   - GitHub Actions run tests on Python 3.10, 3.11, and 3.12
   - Linting with Ruff is enforced on all PRs
   - Slow integration tests run only on main branch pushes
4. **Testing**: Write unit tests for new features and integration tests for end-to-end workflows