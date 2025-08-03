#!/usr/bin/env python3
"""Analyze results from different beta values to find optimal setting."""

import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_metrics(model_dir):
    """Load metrics from a trained model directory."""
    metrics = {}
    
    # Try to load testing metrics
    metrics_file = Path(model_dir) / "testing_set_metrics.txt"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    try:
                        metrics[key.strip()] = float(value.strip())
                    except:
                        metrics[key.strip()] = value.strip()
    
    # Load training config to get beta value
    config_file = Path(model_dir) / "training_config.yaml"
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            if 'flags' in config and 'adversarial_beta' in config['flags']:
                metrics['beta'] = config['flags']['adversarial_beta']
    
    return metrics

def plot_beta_analysis(results_dir="trained_models"):
    """Plot metrics vs beta values."""
    
    # Collect all results
    all_metrics = []
    for model_dir in Path(results_dir).glob("model_beta_*"):
        if model_dir.is_dir():
            metrics = load_metrics(model_dir)
            if metrics and 'beta' in metrics:
                all_metrics.append(metrics)
    
    if not all_metrics:
        print("No results found!")
        return
    
    # Sort by beta
    all_metrics.sort(key=lambda x: x.get('beta', 0))
    
    # Extract data for plotting
    betas = [m['beta'] for m in all_metrics]
    
    # Create subplots for different metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Performance vs Beta Value', fontsize=16)
    
    # Plot different metrics
    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('recall', 'Recall', axes[0, 1]),
        ('precision', 'Precision', axes[0, 2]),
        ('ambient_false_positives_per_hour', 'False Accepts/Hour', axes[1, 0]),
        ('auc', 'AUC', axes[1, 1]),
        ('average_viable_recall', 'Avg Viable Recall', axes[1, 2]),
    ]
    
    for metric_key, metric_name, ax in metrics_to_plot:
        values = [m.get(metric_key, 0) for m in all_metrics]
        ax.plot(betas, values, 'bo-')
        ax.set_xlabel('Beta')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Beta')
        ax.grid(True, alpha=0.3)
        
        # Highlight best value
        if metric_key == 'ambient_false_positives_per_hour':
            best_idx = np.argmin(values)  # Lower is better
        else:
            best_idx = np.argmax(values)  # Higher is better
        
        ax.plot(betas[best_idx], values[best_idx], 'r*', markersize=15)
        ax.annotate(f'β={betas[best_idx]:.2f}', 
                   xy=(betas[best_idx], values[best_idx]),
                   xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('beta_analysis.png', dpi=150)
    plt.show()
    
    # Print summary
    print("\nBeta Analysis Summary:")
    print("-" * 50)
    for m in all_metrics:
        print(f"Beta {m['beta']:.2f}:")
        print(f"  Accuracy: {m.get('accuracy', 0):.4f}")
        print(f"  Recall: {m.get('recall', 0):.4f}")
        print(f"  Precision: {m.get('precision', 0):.4f}")
        print(f"  FA/Hour: {m.get('ambient_false_positives_per_hour', 0):.4f}")
        print()
    
    # Find optimal beta based on a composite score
    scores = []
    for m in all_metrics:
        # Composite score: maximize recall while minimizing false accepts
        # You can adjust these weights based on your priorities
        score = (
            m.get('recall', 0) * 1.0 +  # Weight recall heavily
            m.get('precision', 0) * 0.5 +  # Some weight to precision
            (1.0 - min(m.get('ambient_false_positives_per_hour', 1), 1)) * 0.8  # Penalize false accepts
        )
        scores.append(score)
    
    best_idx = np.argmax(scores)
    print(f"\nRecommended beta: {all_metrics[best_idx]['beta']:.2f}")
    print(f"  (Based on composite score prioritizing recall and low FA/hour)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze beta hyperparameter results')
    parser.add_argument('--results-dir', default='trained_models',
                       help='Directory containing model_beta_* subdirectories')
    args = parser.parse_args()
    
    plot_beta_analysis(args.results_dir)