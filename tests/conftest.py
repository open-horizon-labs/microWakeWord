"""Shared test fixtures and configuration for microWakeWord tests."""

import numpy as np
import pytest
import tensorflow as tf
import yaml


@pytest.fixture
def sample_audio():
    """Generate sample audio data for testing."""
    # 1 second of audio at 16kHz
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)

    # Generate a simple sine wave
    frequency = 440  # A4 note
    t = np.linspace(0, duration, samples)
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

    return audio, sample_rate


@pytest.fixture
def short_audio_chunks():
    """Generate short audio chunks for streaming tests."""
    # 10ms chunks at 16kHz = 160 samples
    chunk_size = 160
    num_chunks = 100

    chunks = []
    for i in range(num_chunks):
        # Create varying frequencies to simulate speech
        frequency = 200 + i * 5
        t = np.linspace(0, 0.01, chunk_size)
        chunk = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        chunks.append(chunk)

    return chunks


@pytest.fixture
def temp_audio_file(tmp_path, sample_audio):
    """Create a temporary audio file for testing."""
    from scipy.io import wavfile

    audio_data, sample_rate = sample_audio
    file_path = tmp_path / "test_audio.wav"

    # Convert to int16 for wav file
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(file_path, sample_rate, audio_int16)

    return file_path


@pytest.fixture
def minimal_model_config():
    """Minimal configuration for model creation."""
    return {
        "input_shape": (150, 40),  # 1.5 seconds at 10ms steps, 40 features
        "num_classes": 2,
        "pointwise_filters": [32, 32],  # Smaller for faster tests
        "first_conv_filters": 16,
        "first_conv_kernel_size": 3,
        "stride": 1,
    }


@pytest.fixture
def training_config(tmp_path):
    """Create a minimal training configuration for testing."""
    config = {
        "window_step_ms": 10,
        "train_dir": str(tmp_path / "models"),
        "features": [
            {
                "features_dir": str(tmp_path / "features"),
                "sampling_weight": 1.0,
                "penalty_weight": 1.0,
                "truth": True,
                "truncation_strategy": "truncate_start",
                "type": "mmap",
            }
        ],
        "training_steps": [10],  # Very small for testing
        "positive_class_weight": [1],
        "negative_class_weight": [1],
        "learning_rates": [0.001],
        "batch_size": 2,
        "eval_step_interval": 5,
        "clip_duration_ms": 1500,
        "time_mask_max_size": [0],
        "time_mask_count": [0],
        "freq_mask_max_size": [0],
        "freq_mask_count": [0],
        "maximization_metric": "accuracy",
        "target_minimization": None,
        "minimization_metric": None,
    }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path, config


@pytest.fixture(scope="session")
def tensorflow_setup():
    """Configure TensorFlow for testing."""
    # Disable GPU for consistent testing
    tf.config.set_visible_devices([], "GPU")

    # Set memory growth to avoid OOM in tests
    physical_devices = tf.config.list_physical_devices("CPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)


# Markers for test categorization
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
