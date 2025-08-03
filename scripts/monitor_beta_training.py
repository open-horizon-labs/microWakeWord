#!/usr/bin/env python3
"""Monitor training metrics to understand beta's effect."""

import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def analyze_tensorboard_logs(log_dir, beta_value):
    """Analyze TensorBoard logs to understand beta's effect."""
    
    # Key metrics to monitor
    metrics_to_track = {
        'wake_loss_ratio': [],  # Ratio of wake loss to total loss
        'tts_accuracy': [],     # How well adversarial classifier performs
        'wake_accuracy': [],    # Main task performance
        'gradient_conflict': [] # Estimate of gradient conflict
    }
    
    # Read TensorBoard logs
    for event_file in Path(log_dir).glob('**/events.out.tfevents.*'):
        for event in tf.compat.v1.train.summary_iterator(str(event_file)):
            for value in event.summary.value:
                if 'loss/wake_word' in value.tag:
                    wake_loss = value.simple_value
                elif 'loss/tts_classifier' in value.tag:
                    tts_loss = value.simple_value
                elif 'loss/total' in value.tag:
                    total_loss = value.simple_value
                    # Calculate contribution ratio
                    if total_loss > 0:
                        metrics_to_track['wake_loss_ratio'].append(
                            wake_loss / total_loss
                        )
                elif 'accuracy/tts_classifier' in value.tag:
                    metrics_to_track['tts_accuracy'].append(value.simple_value)
                elif 'accuracy/wake_word' in value.tag:
                    metrics_to_track['wake_accuracy'].append(value.simple_value)
    
    return metrics_to_track

def diagnose_beta_setting(metrics, beta):
    """Diagnose if beta is well-tuned based on training metrics."""
    
    print(f"\nDiagnosing beta={beta}:")
    print("-" * 40)
    
    # Check TTS classifier accuracy
    if metrics['tts_accuracy']:
        avg_tts_acc = np.mean(metrics['tts_accuracy'][-100:])  # Last 100 steps
        
        if avg_tts_acc > 0.9:
            print("⚠️  TTS classifier too accurate (>90%)")
            print("   → Beta might be too LOW")
            print("   → Adversarial training not effective")
            print("   → Try increasing beta to 0.6-0.8")
        elif avg_tts_acc < 0.6:
            print("✓  TTS classifier confused (60% acc)")
            print("   → Good sign! Features are domain-invariant")
        else:
            print(f"   TTS classifier accuracy: {avg_tts_acc:.2%}")
    
    # Check wake word performance stability
    if metrics['wake_accuracy']:
        wake_acc = metrics['wake_accuracy']
        if len(wake_acc) > 200:
            early = np.mean(wake_acc[:100])
            late = np.mean(wake_acc[-100:])
            
            if late < early * 0.9:
                print("⚠️  Wake word accuracy degrading")
                print("   → Beta might be too HIGH")
                print("   → Adversarial loss dominating")
                print("   → Try decreasing beta to 0.3-0.4")
            elif late > early * 1.1:
                print("✓  Wake word accuracy improving")
            else:
                print("✓  Wake word accuracy stable")
    
    # Check loss balance
    if metrics['wake_loss_ratio']:
        avg_ratio = np.mean(metrics['wake_loss_ratio'][-100:])
        expected_ratio = (1 - beta)  # Expected contribution
        
        if abs(avg_ratio - expected_ratio) > 0.1:
            print(f"ℹ️  Actual wake loss contribution: {avg_ratio:.2%}")
            print(f"   Expected based on beta: {expected_ratio:.2%}")
            print("   → Losses may have different scales")

def find_optimal_beta_early_stopping():
    """Strategy: Start with high beta, reduce if wake word performance drops."""
    
    print("\nAdaptive Beta Strategy:")
    print("-" * 40)
    print("""
    1. Start with beta=0.7 (strong adversarial signal)
    2. Monitor for 500 steps:
       - If wake accuracy drops >5%: reduce beta to 0.5
       - If TTS accuracy stays >85%: increase beta to 0.8
       - If both are stable: keep current beta
    3. After 1000 steps, fix beta for rest of training
    
    This helps find the "edge" where adversarial training helps
    without destroying the main task.
    """)

def theoretical_guidance():
    """Provide theoretical guidance for beta selection."""
    
    print("\nTheoretical Beta Selection Guide:")
    print("=" * 50)
    print("""
    Beta controls the trade-off between:
    - (1-β): Wake word detection (main task)
    - β: Domain confusion (adversarial task)
    
    GUIDELINES BY SCENARIO:
    
    1. Mostly TTS Training Data (>80% TTS):
       → Use HIGH beta (0.6-0.8)
       → Need strong adversarial signal to prevent overfitting
    
    2. Balanced TTS/Real Data (40-60% each):
       → Use MEDIUM beta (0.4-0.6)
       → Default 0.5 is often optimal
    
    3. Mostly Real Data (<20% TTS):
       → Use LOW beta (0.2-0.4)
       → Don't need much domain adaptation
    
    4. High False Accept Requirements:
       → Start with LOWER beta (0.3-0.5)
       → Prioritize main task performance
    
    5. Deploying to Very Different TTS:
       → Use HIGHER beta (0.6-0.8)
       → Maximize domain invariance
    
    SIGNS OF GOOD BETA:
    ✓ TTS classifier accuracy: 50-70% (confused)
    ✓ Wake word metrics stable or improving
    ✓ Validation on real data better than TTS-only baseline
    ✓ Gradient norms balanced between two losses
    
    SIGNS OF BAD BETA:
    ✗ TTS classifier >90% accurate (beta too low)
    ✗ Wake word accuracy dropping (beta too high)
    ✗ Training unstable/oscillating (beta at extremes)
    ✗ No improvement over baseline (beta irrelevant)
    """)

if __name__ == "__main__":
    theoretical_guidance()
    find_optimal_beta_early_stopping()
    
    # Example: analyze existing logs
    # log_dir = "trained_models/model/logs"
    # beta = 0.5
    # metrics = analyze_tensorboard_logs(log_dir, beta)
    # diagnose_beta_setting(metrics, beta)