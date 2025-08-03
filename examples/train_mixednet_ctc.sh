#!/bin/bash
# Example script for training MixedNet+LSTM+CTC model

# Train the model with CTC loss
python -m microwakeword.model_train_eval \
    --training_config='training_parameters.yaml' \
    --train 1 \
    --restore_checkpoint 0 \
    --test_tflite_streaming_quantized 0 \
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