"""Debug script for adversarial training issues."""

import tensorflow as tf
import numpy as np
from microwakeword import mixednet_adversarial

# Create a simple test model to debug the structure
def test_model_structure():
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
    model = mixednet_adversarial.model(flags, shape=(100, 40), batch_size=1)
    
    print("\nModel summary:")
    model.summary()
    
    print("\nModel outputs:")
    for i, output in enumerate(model.outputs):
        print(f"  Output {i}: {output.name} - shape {output.shape}")
    
    # Compile the model
    print("\nCompiling model...")
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
    
    # Test with dummy data
    print("\nTesting model.fit with dummy data...")
    dummy_x = np.random.randn(10, 100, 40)
    dummy_y_wake = np.random.randint(0, 2, (10, 1))
    dummy_y_tts = np.random.randint(0, 2, (10, 1))
    
    try:
        # Try different ways of passing the data
        print("\nTrying with dictionary outputs...")
        history = model.fit(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            batch_size=2,
            epochs=1,
            verbose=1
        )
        print("Success with dictionary outputs!")
    except Exception as e:
        print(f"Failed with dictionary outputs: {e}")
        print(f"Error type: {type(e)}")
        
        # Try with list outputs
        print("\nTrying with list outputs...")
        try:
            history = model.fit(
                dummy_x,
                [dummy_y_wake, dummy_y_tts],
                batch_size=2,
                epochs=1,
                verbose=1
            )
            print("Success with list outputs!")
        except Exception as e2:
            print(f"Failed with list outputs: {e2}")
    
    # Test evaluate
    print("\nTesting model.evaluate...")
    try:
        result = model.evaluate(
            dummy_x,
            {"wake_word": dummy_y_wake, "tts_classifier": dummy_y_tts},
            verbose=0,
            return_dict=True
        )
        print("Evaluation result keys:", list(result.keys()))
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    test_model_structure()