"""Unit tests for audio utility functions."""

import pytest
import numpy as np
from unittest.mock import patch, Mock
import tempfile
from pathlib import Path

from microwakeword.audio.audio_utils import (
    load_audio, save_clip, resample_audio, remove_silence
)


class TestLoadAudio:
    """Test the load_audio function."""
    
    @patch('microwakeword.audio.audio_utils.metadata.load')
    def test_load_wav_file(self, mock_metadata):
        """Test loading a WAV file."""
        # Mock the metadata module
        mock_audio_file = Mock()
        mock_audio_file.streaminfo.sample_rate = 16000
        mock_audio_file.streaminfo.channels = 1
        mock_audio_file.audio_data = [np.array([0.1, 0.2, 0.3], dtype=np.float32)]
        
        mock_metadata.return_value = mock_audio_file
        
        audio, sr = load_audio("test.wav")
        
        assert sr == 16000
        assert len(audio) == 3
        assert np.array_equal(audio, np.array([0.1, 0.2, 0.3]))
    
    @patch('microwakeword.audio.audio_utils.metadata.load')
    def test_load_stereo_to_mono(self, mock_metadata):
        """Test that stereo audio is converted to mono."""
        # Mock stereo audio
        mock_audio_file = Mock()
        mock_audio_file.streaminfo.sample_rate = 16000
        mock_audio_file.streaminfo.channels = 2
        
        # Stereo data (2 channels)
        left_channel = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        right_channel = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        stereo_data = np.stack([left_channel, right_channel], axis=0)
        
        mock_audio_file.audio_data = [stereo_data]
        mock_metadata.return_value = mock_audio_file
        
        audio, sr = load_audio("test.wav")
        
        assert sr == 16000
        assert audio.ndim == 1  # Should be mono
        # Check that it's the average of the two channels
        expected = (left_channel + right_channel) / 2
        np.testing.assert_array_almost_equal(audio, expected)
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(Exception):
            load_audio("nonexistent_file.wav")


class TestSaveClip:
    """Test the save_clip function."""
    
    def test_save_audio_clip(self, tmp_path):
        """Test saving an audio clip."""
        # Create test audio
        audio = np.array([0.1, 0.2, 0.3, -0.1, -0.2], dtype=np.float32)
        file_path = tmp_path / "test_output.wav"
        
        save_clip((audio, 16000), str(file_path))
        
        assert file_path.exists()
        
        # Try to load it back to verify
        import scipy.io.wavfile as wavfile
        sr, loaded_audio = wavfile.read(file_path)
        
        assert sr == 16000
        # Convert back to float32 range [-1, 1]
        loaded_float = loaded_audio.astype(np.float32) / 32767.0
        np.testing.assert_array_almost_equal(audio, loaded_float, decimal=4)
    
    def test_save_clipped_audio(self, tmp_path):
        """Test that audio values outside [-1, 1] are clipped."""
        # Create audio with values outside valid range
        audio = np.array([2.0, -2.0, 0.5, -0.5], dtype=np.float32)
        file_path = tmp_path / "test_clipped.wav"
        
        save_clip((audio, 16000), str(file_path))
        
        import scipy.io.wavfile as wavfile
        sr, loaded_audio = wavfile.read(file_path)
        
        # Check that values were clipped
        loaded_float = loaded_audio.astype(np.float32) / 32767.0
        assert np.max(loaded_float) <= 1.0
        assert np.min(loaded_float) >= -1.0


class TestResampleAudio:
    """Test the resample_audio function."""
    
    def test_no_resample_needed(self):
        """Test when no resampling is needed."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        resampled = resample_audio(audio, 16000, 16000)
        
        np.testing.assert_array_equal(audio, resampled)
    
    def test_downsample(self):
        """Test downsampling from 48kHz to 16kHz."""
        # Create a simple signal at 48kHz
        duration = 0.1  # 100ms
        original_sr = 48000
        target_sr = 16000
        
        t = np.linspace(0, duration, int(duration * original_sr))
        # 1kHz sine wave
        audio = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        
        resampled = resample_audio(audio, original_sr, target_sr)
        
        # Check that length is approximately correct
        expected_length = int(len(audio) * target_sr / original_sr)
        assert abs(len(resampled) - expected_length) <= 1
        
        # Check that the signal still contains the frequency
        # (This is a basic check - proper testing would use FFT)
        assert np.max(resampled) > 0.9
        assert np.min(resampled) < -0.9
    
    def test_upsample(self):
        """Test upsampling from 8kHz to 16kHz."""
        audio = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        resampled = resample_audio(audio, 8000, 16000)
        
        # Should have approximately double the samples
        assert len(resampled) == len(audio) * 2


class TestRemoveSilence:
    """Test the remove_silence function."""
    
    def test_remove_leading_silence(self):
        """Test removing silence from the beginning."""
        # Create audio with silence at the start
        silence = np.zeros(1000)
        signal = np.random.randn(1000) * 0.5
        audio = np.concatenate([silence, signal])
        
        trimmed, _ = remove_silence((audio, 16000))
        
        # Should have removed most of the silence
        assert len(trimmed) < len(audio)
        assert len(trimmed) >= len(signal)
    
    def test_remove_trailing_silence(self):
        """Test removing silence from the end."""
        signal = np.random.randn(1000) * 0.5
        silence = np.zeros(1000)
        audio = np.concatenate([signal, silence])
        
        trimmed, _ = remove_silence((audio, 16000))
        
        assert len(trimmed) < len(audio)
        assert len(trimmed) >= len(signal)
    
    def test_no_silence_to_remove(self):
        """Test when there's no silence to remove."""
        # Create audio that's all signal
        audio = np.random.randn(1000) * 0.5
        
        trimmed, _ = remove_silence((audio, 16000))
        
        # Should be mostly unchanged (maybe small trim at edges)
        assert len(trimmed) >= len(audio) * 0.9
    
    def test_all_silence(self):
        """Test when audio is all silence."""
        audio = np.zeros(1000) + 0.0001  # Very quiet but not zero
        
        trimmed, _ = remove_silence((audio, 16000))
        
        # Should keep some minimum length
        assert len(trimmed) > 0
    
    @patch('webrtcvad.Vad')
    def test_vad_integration(self, mock_vad_class):
        """Test that VAD is properly used."""
        # Mock the VAD
        mock_vad = Mock()
        mock_vad.is_speech.return_value = True
        mock_vad_class.return_value = mock_vad
        
        audio = np.random.randn(16000)  # 1 second
        trimmed, _ = remove_silence((audio, 16000))
        
        # Check that VAD was created and used
        mock_vad_class.assert_called_once()
        assert mock_vad.is_speech.called