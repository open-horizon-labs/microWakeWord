"""Unit tests for metric computation functions."""

import pytest
import numpy as np
from microwakeword.test import (
    compute_metrics, 
    metrics_to_string,
    compute_false_accepts_per_hour,
    generate_roc_curve
)


class TestComputeMetrics:
    """Test the compute_metrics function."""
    
    def test_perfect_classification(self):
        """Test metrics for perfect classification."""
        metrics = compute_metrics(
            true_positives=100,
            true_negatives=100,
            false_positives=0,
            false_negatives=0
        )
        
        assert metrics['accuracy'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['false_positive_rate'] == 0.0
        assert metrics['false_negative_rate'] == 0.0
        assert metrics['count'] == 200
    
    def test_no_true_positives(self):
        """Test metrics when there are no true positives."""
        metrics = compute_metrics(
            true_positives=0,
            true_negatives=100,
            false_positives=10,
            false_negatives=10
        )
        
        assert metrics['accuracy'] == pytest.approx(100/120)
        assert metrics['recall'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['false_positive_rate'] == pytest.approx(10/110)
        assert metrics['false_negative_rate'] == 1.0
    
    def test_balanced_errors(self):
        """Test metrics with balanced errors."""
        metrics = compute_metrics(
            true_positives=80,
            true_negatives=70,
            false_positives=30,
            false_negatives=20
        )
        
        assert metrics['accuracy'] == pytest.approx(150/200)
        assert metrics['recall'] == pytest.approx(80/100)
        assert metrics['precision'] == pytest.approx(80/110)
        assert metrics['false_positive_rate'] == pytest.approx(30/100)
        assert metrics['false_negative_rate'] == pytest.approx(20/100)
    
    def test_empty_data(self):
        """Test metrics with no data."""
        metrics = compute_metrics(0, 0, 0, 0)
        
        assert np.isnan(metrics['accuracy'])
        assert np.isnan(metrics['recall'])
        assert np.isnan(metrics['precision'])
        assert np.isnan(metrics['false_positive_rate'])
        assert np.isnan(metrics['false_negative_rate'])
        assert metrics['count'] == 0


class TestMetricsToString:
    """Test the metrics_to_string function."""
    
    def test_format_output(self):
        """Test that metrics are formatted correctly."""
        metrics = {
            'accuracy': 0.95,
            'recall': 0.90,
            'precision': 0.85,
            'false_positive_rate': 0.05,
            'false_negative_rate': 0.10,
            'count': 1000
        }
        
        result = metrics_to_string(metrics)
        
        assert 'accuracy = 95.00%' in result
        assert 'recall = 90.00%' in result
        assert 'precision = 85.00%' in result
        assert 'fpr = 5.00%' in result
        assert 'fnr = 10.00%' in result
        assert '(N=1000)' in result


class TestComputeFalseAcceptsPerHour:
    """Test the compute_false_accepts_per_hour function."""
    
    def test_no_false_accepts(self):
        """Test when there are no false accepts."""
        # Create probabilities that are all below threshold
        probabilities = [np.ones(100) * 0.1 for _ in range(10)]
        cutoffs = np.array([0.5, 0.7, 0.9])
        
        faph = compute_false_accepts_per_hour(probabilities, cutoffs)
        
        assert np.all(faph == 0)
    
    def test_single_false_accept(self):
        """Test with a single false accept."""
        # Create probabilities with one spike
        probs = np.ones(100) * 0.1
        probs[50] = 0.8
        probabilities = [probs]
        cutoffs = np.array([0.5])
        
        faph = compute_false_accepts_per_hour(
            probabilities, 
            cutoffs,
            ignore_slices_after_accept=10,
            step_s=0.01  # 10ms steps
        )
        
        # 1 false accept in 1 second = 3600 per hour
        assert faph[0] == pytest.approx(3600.0)
    
    def test_cooldown_period(self):
        """Test that cooldown period prevents multiple detections."""
        # Create probabilities with multiple spikes close together
        probs = np.ones(100) * 0.1
        probs[50:55] = 0.8  # 5 consecutive high values
        probabilities = [probs]
        cutoffs = np.array([0.5])
        
        faph = compute_false_accepts_per_hour(
            probabilities,
            cutoffs,
            ignore_slices_after_accept=10,
            step_s=0.01
        )
        
        # Should only count as 1 false accept due to cooldown
        assert faph[0] == pytest.approx(3600.0)


class TestGenerateROCCurve:
    """Test the generate_roc_curve function."""
    
    def test_basic_roc_curve(self):
        """Test basic ROC curve generation."""
        false_accepts_per_hour = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        false_rejections = np.array([0.01, 0.05, 0.10, 0.20, 0.40, 0.80])
        cutoffs = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95])
        
        x, y, c = generate_roc_curve(
            false_accepts_per_hour,
            false_rejections,
            cutoffs,
            max_faph=2.0
        )
        
        # Check that coordinates are in descending order
        assert all(x[i] >= x[i+1] for i in range(len(x)-1))
        
        # Check that first point respects max_faph
        assert x[0] <= 2.0
        
        # Check that we have corresponding cutoffs
        assert len(x) == len(y) == len(c)
    
    def test_interpolation_at_max_faph(self):
        """Test that interpolation works when first point exceeds max_faph."""
        false_accepts_per_hour = np.array([5.0, 3.0, 1.0])
        false_rejections = np.array([0.1, 0.2, 0.4])
        cutoffs = np.array([0.3, 0.5, 0.7])
        
        x, y, c = generate_roc_curve(
            false_accepts_per_hour,
            false_rejections,
            cutoffs,
            max_faph=2.0
        )
        
        # First point should be at max_faph
        assert x[0] == 2.0
        
        # Y value should be interpolated
        assert 0.2 < y[0] < 0.4  # Should be between the two points