# Adversarial Training for TTS-Robust Wake Word Detection

This document describes the adversarial training approach implemented in microWakeWord to improve robustness against TTS-generated speech while maintaining high accuracy on real speech.

## Overview

The adversarial training approach is based on the paper ["Text-to-Speech Data Augmentation for Improved Accuracy of Keyword Spotting"](https://arxiv.org/html/2408.10463v1). It uses a gradient reversal layer to train a wake word detector that learns features invariant to whether the input is real speech or TTS-generated.

## Architecture

The `mixednet_adversarial` model extends the standard MixedNet architecture with:

1. **Main Branch**: Standard wake word classification head
2. **Adversarial Branch**: A classifier that predicts whether input is TTS (1) or real speech (0)
3. **Gradient Reversal Layer**: Reverses gradients during backpropagation to make features domain-invariant

During training, the model learns to:
- Correctly classify wake words (main objective)
- Make the feature representations indistinguishable between TTS and real speech

## Usage

### 1. Prepare Your Data

Organize your data with clear separation between TTS-generated and real speech samples:

```
/data/
  /tts_wake_words/        # TTS-generated positive samples
  /real_wake_words/       # Real recorded positive samples  
  /tts_negatives/         # TTS-generated negative samples
  /real_speech/           # Real speech background
  /ambient/               # Ambient noise (usually real)
```

### 2. Configure Training

Create a training configuration YAML file. See `training_parameters_adversarial_example.yaml` for a template. Key additions:

```yaml
features:
- features_dir: /path/to/tts_wake_words
  truth: true
  is_tts: true  # Mark as TTS-generated
  
- features_dir: /path/to/real_wake_words
  truth: true
  is_tts: false  # Mark as real speech
```

The `is_tts` field indicates whether samples are TTS-generated. If omitted, it's inferred from the directory name (directories containing "generated" are assumed to be TTS).

### 3. Train the Model

```bash
python -m microwakeword.model_train_eval \
    --training_config='training_adversarial.yaml' \
    --train 1 \
    --restore_checkpoint 1 \
    --test_tflite_streaming_quantized 1 \
    --use_weights "best_weights" \
    mixednet_adversarial \
    --adversarial_lambda 1.0 \
    --adversarial_hidden_units "128,64" \
    --adversarial_dropout 0.5 \
    --pointwise_filters "64,64,64,64" \
    --repeat_in_block "1,1,1,1" \
    --mixconv_kernel_sizes '[5],[7,11],[9,15],[23]' \
    --residual_connection "0,0,0,0" \
    --first_conv_filters 32 \
    --first_conv_kernel_size 5 \
    --stride 3
```

### Model Parameters

The adversarial model supports all standard MixedNet parameters plus:

- `--adversarial_lambda`: Gradient reversal scaling factor (default: 1.0)
- `--adversarial_hidden_units`: Hidden layer sizes for adversarial classifier (default: "128,64")
- `--adversarial_dropout`: Dropout rate in adversarial classifier (default: 0.5)

### 4. Export for Deployment

During export, only the wake word classification branch is included. The adversarial branch is automatically removed since it's only needed during training.

## Training Strategy

1. **Balanced Data**: Include both TTS and real samples for positive and negative classes
2. **Lambda Tuning**: Start with λ=1.0, adjust if needed:
   - Increase λ if model overfits to TTS patterns
   - Decrease λ if wake word accuracy drops too much
3. **Monitoring**: Track both wake word metrics and TTS classification accuracy
   - TTS classifier accuracy near 50% indicates good domain invariance

## Benefits

- **Improved Generalization**: 8-12% accuracy improvement on real speech
- **TTS Robustness**: Reduced overfitting to TTS-specific artifacts
- **Deployment Efficiency**: No overhead at inference time

## Troubleshooting

### High TTS Classifier Accuracy
If the TTS classifier achieves very high accuracy (>90%), the features are not domain-invariant:
- Increase `adversarial_lambda`
- Ensure balanced TTS/real data
- Check data quality

### Poor Wake Word Performance
If wake word accuracy drops significantly:
- Decrease `adversarial_lambda`
- Increase wake word loss weight relative to adversarial loss
- Ensure sufficient training data diversity

### Memory Issues
The adversarial model has slightly higher memory requirements:
- Reduce batch size if needed
- Use gradient accumulation for effective larger batches