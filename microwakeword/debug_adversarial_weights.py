"""Debug script for adversarial training sample weight issues."""

import tensorflow as tf
import numpy as np
from microwakeword import mixednet_adversarial

def test_sample_weights():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {tf.keras.__version__}")
    
    # Create minimal flags object
    class Flags:
        adversarial_lambda = 1.0
        adversarial_hidden_units = "128,64"
        adversarial_dropout = 0.5
        pointwise_filters = "32,32"
        repeat_in_block = "1,1"
        mixconv_kernel_sizes = "[5],[7]"
        residual_connection = "0,0"
        first_conv_filters = 16
        first_conv_kernel_size = 3
        spatial_attention = 0
        pooled = 1
        max_pool = 0
        stride = 1
    
    flags = Flags()
    
    # Create a small model
    model = mixednet_adversarial.model(flags, shape=(100, 40), batch_size=None)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss={
            "wake_word": tf.keras.losses.BinaryCrossentropy(),
            "tts_classifier": tf.keras.losses.BinaryCrossentropy()
        },
        metrics={
            "wake_word": ["accuracy"],
            "tts_classifier": ["accuracy"]
        }
    )
    
    # Create dummy data
    batch_size = 10
    dummy_x = np.random.randn(batch_size, 100, 40)
    dummy_y_wake = np.random.randint(0, 2, (batch_size, 1))
    dummy_y_tts = np.random.randint(0, 2, (batch_size, 1))
    
    # Create sample weights
    sample_weights_wake = np.random.rand(batch_size)
    sample_weights_tts = np.ones(batch_size)
    
    print("\nTest 1: Dictionary sample weights")
    try:
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=5,
            epochs=1,
            sample_weight={"wake_word": sample_weights_wake, "tts_classifier": sample_weights_tts},
            verbose=1
        )
        print("✓ Success with dictionary sample weights")
    except Exception as e:
        print(f"✗ Failed: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("\nTest 2: List sample weights")
    try:
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=5,
            epochs=1,
            sample_weight=[sample_weights_wake, sample_weights_tts],
            verbose=1
        )
        print("✓ Success with list sample weights")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTest 3: Single sample weight (applied to all outputs)")
    try:
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=5,
            epochs=1,
            sample_weight=sample_weights_wake,
            verbose=1
        )
        print("✓ Success with single sample weight")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\nTest 4: No sample weight")
    try:
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=5,
            epochs=1,
            verbose=1
        )
        print("✓ Success without sample weight")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Test with class weights simulation
    print("\nTest 5: Simulated class weights in sample weights")
    try:
        # Simulate class weight application
        class_weight_multiplier = np.where(dummy_y_wake == 1, 1.0, 40.0).flatten()
        weighted_samples = sample_weights_wake * class_weight_multiplier
        
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=5,
            epochs=1,
            sample_weight={"wake_word": weighted_samples, "tts_classifier": sample_weights_tts},
            verbose=1
        )
        print("✓ Success with class-weighted sample weights")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sample_weights()