# microWakeWord Testing Plan

## Overview

This document outlines a comprehensive testing strategy for the microWakeWord project, covering both basic unit tests and complex integration tests including model training.

## Testing Framework

**Recommended Framework**: pytest
- Modern, flexible testing framework
- Excellent fixture support for setup/teardown
- Good integration with TensorFlow testing utilities
- Parametrized testing for multiple test cases
- Easy mocking capabilities

**Additional Testing Dependencies**:
```bash
pip install pytest pytest-cov pytest-mock pytest-tensorflow
```

## Testing Structure

```
microWakeWord/
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Shared fixtures and test configuration
│   ├── unit/                          # Unit tests for individual components
│   │   ├── __init__.py
│   │   ├── audio/
│   │   │   ├── test_audio_utils.py
│   │   │   ├── test_augmentation.py
│   │   │   ├── test_clips.py
│   │   │   └── test_spectrograms.py
│   │   ├── layers/
│   │   │   ├── test_delay.py
│   │   │   ├── test_stream.py
│   │   │   └── test_modes.py
│   │   ├── test_data.py
│   │   ├── test_utils.py
│   │   └── test_metrics.py
│   ├── integration/                   # Integration tests
│   │   ├── __init__.py
│   │   ├── test_model_creation.py
│   │   ├── test_inference.py
│   │   └── test_training_pipeline.py
│   └── fixtures/                      # Test data and resources
│       ├── audio_samples/
│       ├── configs/
│       └── models/
```

## Basic Functionality Tests

### 1. Audio Processing Tests (`tests/unit/audio/`)

#### test_audio_utils.py
- Test audio file loading/saving
- Test audio format conversions
- Test resampling functionality
- Test silence removal
- Edge cases: empty files, corrupted files, various formats

#### test_augmentation.py
- Test individual augmentation functions
- Test augmentation pipeline
- Test parameter validation
- Test deterministic behavior with fixed seeds
- Mock external dependencies (audio files)

#### test_clips.py
- Test clip loading and splitting
- Test train/val/test split functionality
- Test clip duration handling
- Test file pattern matching

#### test_spectrograms.py
- Test spectrogram generation
- Test feature extraction
- Test sliding window functionality
- Test with various audio lengths

### 2. Layer Tests (`tests/unit/layers/`)

#### test_stream.py
- Test streaming layer conversions
- Test state management
- Test different modes (training vs inference)
- Test dimension handling

#### test_delay.py
- Test delay buffer functionality
- Test state updates
- Test edge cases (empty buffers, overflow)

### 3. Data Pipeline Tests (`tests/unit/test_data.py`)
- Test data loading from RaggedMmap
- Test batch generation
- Test data augmentation integration
- Test shuffling and sampling

### 4. Utility Tests (`tests/unit/test_utils.py`)
- Test metric computation functions
- Test file I/O utilities
- Test configuration parsing

## Complex Functionality Tests

### 1. Model Creation Tests (`tests/integration/test_model_creation.py`)
```python
import pytest
import tensorflow as tf
from microwakeword.mixednet import MixedNet
from microwakeword.inception import inception
from microwakeword.layers.modes import Modes

class TestModelCreation:
    @pytest.fixture
    def model_config(self):
        return {
            'input_shape': (150, 40),
            'num_classes': 2,
            'pointwise_filters': [64, 64, 64, 64],
            'first_conv_filters': 32,
            'first_conv_kernel_size': 5,
            'stride': 3
        }
    
    def test_mixednet_creation(self, model_config):
        """Test MixedNet model can be created with valid config."""
        model = MixedNet(**model_config)
        assert isinstance(model, tf.keras.Model)
        assert model.input_shape == (None, 150, 40)
        assert model.output_shape == (None, 2)
    
    def test_inception_creation(self):
        """Test Inception model creation."""
        model = inception(
            input_shape=(150, 40),
            num_classes=2,
            filters=[32, 64, 128],
            dropout=0.1
        )
        assert isinstance(model, tf.keras.Model)
    
    def test_model_modes(self, model_config):
        """Test model can switch between training and inference modes."""
        model = MixedNet(**model_config)
        
        # Test training mode
        model.set_mode(Modes.TRAINING)
        output = model(tf.random.normal((1, 150, 40)))
        assert output.shape == (1, 2)
        
        # Test non-streaming inference
        model.set_mode(Modes.NON_STREAM_INFERENCE)
        output = model(tf.random.normal((1, 150, 40)))
        assert output.shape == (1, 2)
```

### 2. Training Pipeline Tests (`tests/integration/test_training_pipeline.py`)
```python
import pytest
import tempfile
import yaml
from pathlib import Path
from microwakeword.train import train_model
from microwakeword.model_train_eval import main

class TestTrainingPipeline:
    @pytest.fixture
    def minimal_training_config(self, tmp_path):
        """Create a minimal training configuration for testing."""
        config = {
            'window_step_ms': 10,
            'train_dir': str(tmp_path / 'models'),
            'features': [{
                'features_dir': str(tmp_path / 'features'),
                'sampling_weight': 1.0,
                'penalty_weight': 1.0,
                'truth': True,
                'truncation_strategy': 'truncate_start',
                'type': 'mmap'
            }],
            'training_steps': [100],  # Small number for testing
            'positive_class_weight': [1],
            'negative_class_weight': [1],
            'learning_rates': [0.001],
            'batch_size': 8,
            'eval_step_interval': 50,
            'clip_duration_ms': 1500,
            'maximization_metric': 'accuracy'
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def mock_training_data(self, tmp_path):
        """Create mock training data for testing."""
        # This would create minimal RaggedMmap files
        # with synthetic data for testing
        pass
    
    def test_training_initialization(self, minimal_training_config, mock_training_data):
        """Test that training can be initialized with config."""
        # Test configuration loading
        # Test model creation
        # Test data pipeline initialization
        pass
    
    def test_short_training_run(self, minimal_training_config, mock_training_data):
        """Test a very short training run to ensure pipeline works."""
        # Run training for just a few steps
        # Verify checkpoints are created
        # Verify metrics are logged
        pass
    
    @pytest.mark.slow
    def test_full_training_cycle(self, minimal_training_config, mock_training_data):
        """Test a complete (but small) training cycle."""
        # Run training to completion
        # Test model conversion to streaming
        # Test quantization
        # Verify final model files
        pass
```

### 3. Inference Tests (`tests/integration/test_inference.py`)
```python
import pytest
import numpy as np
from microwakeword.inference import Model

class TestInference:
    @pytest.fixture
    def test_model_path(self, tmp_path):
        """Create or reference a test model."""
        # Either create a minimal model or use a pre-trained test model
        pass
    
    def test_model_loading(self, test_model_path):
        """Test model can be loaded for inference."""
        model = Model(test_model_path)
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_single_prediction(self, test_model_path):
        """Test single audio prediction."""
        model = Model(test_model_path)
        
        # Create synthetic audio (1 second at 16kHz)
        audio = np.random.randn(16000).astype(np.float32)
        
        prediction = model.predict(audio)
        assert isinstance(prediction, float)
        assert 0 <= prediction <= 1
    
    def test_streaming_prediction(self, test_model_path):
        """Test streaming inference with audio chunks."""
        model = Model(test_model_path)
        
        # Simulate streaming audio in chunks
        chunk_size = 160  # 10ms at 16kHz
        for _ in range(100):
            chunk = np.random.randn(chunk_size).astype(np.float32)
            prediction = model.predict_stream(chunk)
            assert 0 <= prediction <= 1
```

## Test Data Management

### Fixtures Directory Structure
```
tests/fixtures/
├── audio_samples/
│   ├── wake_word_sample.wav      # Real wake word sample
│   ├── negative_sample.wav       # Non-wake word sample
│   ├── silence.wav              # Silent audio
│   └── noise.wav                # Background noise
├── configs/
│   ├── minimal_training.yaml    # Minimal valid config
│   ├── invalid_config.yaml      # For error testing
│   └── test_augmentation.yaml   # Augmentation test config
└── models/
    └── tiny_test_model.tflite   # Pre-trained minimal model
```

## Mocking Strategy

For unit tests, mock:
- File I/O operations
- External audio files
- TensorFlow model training (for testing pipeline logic)
- Time-consuming operations

Example mock usage:
```python
from unittest.mock import Mock, patch
import pytest

@patch('microwakeword.audio.audio_utils.load_audio')
def test_audio_processing(mock_load):
    mock_load.return_value = (np.zeros(16000), 16000)
    # Test audio processing without actual file I/O
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e .
        pip install pytest pytest-cov pytest-mock
    
    - name: Run unit tests
      run: pytest tests/unit -v --cov=microwakeword
    
    - name: Run integration tests (fast)
      run: pytest tests/integration -v -m "not slow"
```

## Performance Testing

### Memory Usage Tests
```python
import tracemalloc
import pytest

def test_memory_usage_during_inference():
    tracemalloc.start()
    
    # Run inference
    model = Model("path/to/model")
    for _ in range(1000):
        model.predict(np.random.randn(160))
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory usage is reasonable
    assert peak < 100 * 1024 * 1024  # Less than 100MB
```

### Speed Tests
```python
import time

def test_inference_speed():
    model = Model("path/to/model")
    audio_chunk = np.random.randn(160)
    
    start = time.time()
    for _ in range(1000):
        model.predict(audio_chunk)
    duration = time.time() - start
    
    # Should process 1000 chunks (10 seconds) faster than real-time
    assert duration < 10.0
```

## Test Execution Commands

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=microwakeword --cov-report=html

# Run only unit tests
pytest tests/unit

# Run only fast tests
pytest -m "not slow"

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/audio/test_audio_utils.py

# Run tests in parallel
pytest -n auto
```

## Next Steps

1. Set up the test directory structure
2. Install testing dependencies
3. Create conftest.py with shared fixtures
4. Start with unit tests for core functionality
5. Gradually add integration tests
6. Set up CI/CD pipeline
7. Add performance benchmarks