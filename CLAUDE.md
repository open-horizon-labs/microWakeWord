# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

microWakeWord is a TensorFlow-based wake word detection training framework that produces models suitable for TensorFlow Lite for Microcontrollers. The project focuses on creating custom wake word models with low false accept/reject rates for low-power devices.

## Key Commands

### Installation
```bash
# Install microWakeWord (requires Python 3.10)
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

# Train the model using the configuration
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
```

### Testing
The project uses the `microwakeword.test` module for accuracy evaluation. Tests are typically run during the training process.

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
   - `inception.py`: Inception-based model architecture
   - `layers/`: Custom TensorFlow layers for streaming inference
     - `stream.py`: Streaming layer implementations
     - `modes.py`: Training/inference mode management
     - `delay.py`, `strided_drop.py`: Streaming-specific layers

3. **Training Pipeline**
   - `train.py`: Core training loop and validation logic
   - `model_train_eval.py`: Main entry point for training and evaluation
   - `data.py`: Data loading and preprocessing with Ragged Mmap support

4. **Inference**
   - `inference.py`: Model class for loading and running inference
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

## Important Considerations

- Models are trained with a 16kHz sample rate
- Window duration: 30ms, stride: 10ms
- Default clip duration: 1500ms maximum
- Python 3.10 is required (not 3.11+)
- Training requires significant experimentation with hyperparameters
- The `basic_training_notebook.ipynb` provides a starting point but won't produce production-ready models without tuning