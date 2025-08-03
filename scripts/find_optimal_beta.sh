#!/bin/bash
# Script to search for optimal beta value

# Coarse search first
for beta in 0.1 0.3 0.5 0.7 0.9; do
    echo "Training with beta=$beta"
    python -m microwakeword.model_train_eval \
        --training_config='training_adversarial.yaml' \
        --train 1 \
        --test_tf_nonstreaming 1 \
        --use_weights "best_weights" \
        mixednet_adversarial \
        --adversarial_beta $beta \
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
    
    # Move results to a beta-specific folder
    mv trained_models/model trained_models/model_beta_${beta}
done

# After coarse search, do fine search around best value
# For example, if 0.5 was best, try:
# for beta in 0.4 0.45 0.5 0.55 0.6; do ...