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
```

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
   - `inception.py`: Inception-based model architecture
   - `layers/`: Custom TensorFlow layers for streaming inference
     - `stream.py`: Streaming layer implementations
     - `modes.py`: Training/inference mode management
     - `delay.py`, `strided_drop.py`: Streaming-specific layers

3. **Training Pipeline**
   - `train.py`: Core training loop and validation logic
   - `train_ctc.py`: CTC-specific training loop for MixedNet+CTC models
   - `model_train_eval.py`: Main entry point for training and evaluation
   - `data.py`: Data loading and preprocessing with Ragged Mmap support
   - `ctc_utils.py`: CTC loss, decoding, and metrics utilities

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