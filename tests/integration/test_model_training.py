"""Integration tests for model training pipeline."""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile
import yaml
from unittest.mock import patch, Mock

from microwakeword.train import train_model
from microwakeword.mixednet import MixedNet
from microwakeword.layers.modes import Modes
from microwakeword.data import Data
from mmap_ninja.ragged import RaggedMmap


@pytest.mark.integration
class TestModelTrainingPipeline:
    """Test the complete model training pipeline."""
    
    @pytest.fixture
    def mock_training_data(self, tmp_path):
        """Create mock training data using RaggedMmap."""
        # Create directories
        features_dir = tmp_path / "features"
        train_dir = features_dir / "training" / "test_mmap"
        val_dir = features_dir / "validation" / "test_mmap"
        
        train_dir.mkdir(parents=True)
        val_dir.mkdir(parents=True)
        
        # Create mock spectrograms (150 time steps, 40 features)
        num_samples = 20
        spectrogram_shape = (150, 40)
        
        # Training data
        train_data = []
        for i in range(num_samples):
            # Create a random spectrogram
            spectrogram = np.random.randn(*spectrogram_shape).astype(np.float32)
            train_data.append(spectrogram)
        
        # Validation data
        val_data = []
        for i in range(num_samples // 2):
            spectrogram = np.random.randn(*spectrogram_shape).astype(np.float32)
            val_data.append(spectrogram)
        
        # Save using RaggedMmap
        RaggedMmap.from_lists(
            out_dir=str(train_dir),
            lists=[train_data],
            batch_size=10
        )
        
        RaggedMmap.from_lists(
            out_dir=str(val_dir),
            lists=[val_data],
            batch_size=5
        )
        
        return features_dir
    
    @pytest.fixture
    def training_config_with_data(self, tmp_path, mock_training_data):
        """Create a training configuration with mock data."""
        config = {
            'window_step_ms': 10,
            'train_dir': str(tmp_path / 'models'),
            'features': [{
                'features_dir': str(mock_training_data),
                'sampling_weight': 1.0,
                'penalty_weight': 1.0,
                'truth': True,
                'truncation_strategy': 'truncate_start',
                'type': 'mmap'
            }],
            'training_steps': [20],  # Very few steps for testing
            'positive_class_weight': [1],
            'negative_class_weight': [1],
            'learning_rates': [0.001],
            'batch_size': 4,
            'eval_step_interval': 10,
            'clip_duration_ms': 1500,
            'time_mask_max_size': [0],
            'time_mask_count': [0],
            'freq_mask_max_size': [0],
            'freq_mask_count': [0],
            'maximization_metric': 'accuracy',
            'target_minimization': None,
            'minimization_metric': None
        }
        
        config_path = tmp_path / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path, config
    
    def test_model_initialization(self, minimal_model_config):
        """Test that model can be initialized properly."""
        model = MixedNet(**minimal_model_config)
        
        assert isinstance(model, tf.keras.Model)
        assert len(model.layers) > 0
        
        # Test that model can process input
        test_input = tf.random.normal((1, 150, 40))
        output = model(test_input, training=True)
        
        assert output.shape == (1, 2)
        assert not tf.reduce_any(tf.math.is_nan(output))
    
    def test_model_compilation(self, minimal_model_config):
        """Test model compilation with optimizer and loss."""
        model = MixedNet(**minimal_model_config)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Check that model is compiled
        assert model.optimizer is not None
        assert model.loss is not None
    
    @pytest.mark.slow
    def test_short_training_run(self, training_config_with_data, minimal_model_config):
        """Test a very short training run."""
        config_path, config = training_config_with_data
        
        # Create model
        model = MixedNet(**minimal_model_config)
        
        # Create data loader
        data = Data(config)
        
        # Get a batch to test
        batch = next(data.train_data_generator())
        assert batch[0].shape[0] == config['batch_size']
        assert batch[0].shape[1:] == (150, 40)
        assert batch[1].shape == (config['batch_size'],)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        
        # Train for one epoch
        history = model.fit(
            data.train_data_generator(),
            steps_per_epoch=5,
            epochs=1,
            validation_data=data.validation_data_generator(),
            validation_steps=2
        )
        
        assert 'loss' in history.history
        assert 'accuracy' in history.history
        assert len(history.history['loss']) == 1
    
    def test_model_checkpointing(self, tmp_path, minimal_model_config):
        """Test that model checkpoints can be saved and loaded."""
        model = MixedNet(**minimal_model_config)
        
        # Save model
        checkpoint_path = tmp_path / "checkpoint"
        model.save_weights(str(checkpoint_path))
        
        assert checkpoint_path.with_suffix('.index').exists()
        
        # Create new model and load weights
        new_model = MixedNet(**minimal_model_config)
        new_model.load_weights(str(checkpoint_path))
        
        # Compare predictions
        test_input = tf.random.normal((1, 150, 40))
        tf.random.set_seed(42)
        
        # Need to run once to build the model
        _ = model(test_input)
        _ = new_model(test_input)
        
        # Now compare with same input
        output1 = model(test_input, training=False)
        output2 = new_model(test_input, training=False)
        
        np.testing.assert_array_almost_equal(
            output1.numpy(), output2.numpy(), decimal=5
        )
    
    @pytest.mark.slow
    def test_model_mode_switching(self, minimal_model_config):
        """Test switching between training and streaming inference modes."""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        
        from microwakeword import utils
        
        # Create and compile model
        model = MixedNet(**minimal_model_config)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )
        
        # Get flags for the model
        flags = {
            'desired_samples': 16000,
            'window_size_samples': 480,
            'window_stride_samples': 160,
            'sample_rate': 16000,
            'window_size_ms': 30.0,
            'window_stride_ms': 10.0,
            'clip_duration_ms': 1500,
            'use_spec_augment': False,
            'feature_type': 'mfcc',
            'training': False
        }
        
        # Convert to streaming model
        with patch('microwakeword.utils.modes.get_input_data_shape', return_value=(1, 40)):
            streaming_model = utils.convert_to_streaming_inference(
                model,
                flags,
                Modes.STREAM_INTERNAL_STATE_INFERENCE
            )
        
        assert streaming_model is not None
        
        # Test streaming inference with chunks
        chunk_size = 160  # 10ms at 16kHz
        num_chunks = 10
        
        for i in range(num_chunks):
            # Create audio chunk
            audio_chunk = np.random.randn(chunk_size).astype(np.float32)
            
            # In real usage, this would go through feature extraction
            # For testing, we'll create mock features
            mock_features = np.random.randn(1, 1, 40).astype(np.float32)
            
            # Run inference
            output = streaming_model(mock_features)
            
            # Should output probability
            assert output.shape == (1, 1, 2)  # batch, time, classes