#!/usr/bin/env python3
"""Suggest initial beta value based on dataset composition."""

import yaml
import argparse
from pathlib import Path

def analyze_dataset_composition(config_file):
    """Analyze the ratio of TTS vs real data in training config."""
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    tts_samples = 0
    real_samples = 0
    
    for feature in config.get('features', []):
        # Get sample count (approximate from directory size or config)
        sample_count = feature.get('sample_count', 1000)  # Default estimate
        
        if feature.get('is_tts', False):
            tts_samples += sample_count
        else:
            real_samples += sample_count
    
    total_samples = tts_samples + real_samples
    tts_ratio = tts_samples / total_samples if total_samples > 0 else 0
    
    print(f"\nDataset Composition Analysis:")
    print(f"  TTS samples: {tts_samples} ({tts_ratio:.1%})")
    print(f"  Real samples: {real_samples} ({(1-tts_ratio):.1%})")
    
    # Suggest beta based on composition
    if tts_ratio > 0.8:
        suggested_beta = 0.7
        print(f"\n→ Suggested beta: {suggested_beta}")
        print("  Reason: Heavy TTS dataset (>80%) needs strong adversarial signal")
        print("  Also try: 0.6, 0.8")
    elif tts_ratio > 0.6:
        suggested_beta = 0.6
        print(f"\n→ Suggested beta: {suggested_beta}")
        print("  Reason: Majority TTS dataset (60-80%)")
        print("  Also try: 0.5, 0.7")
    elif tts_ratio > 0.4:
        suggested_beta = 0.5
        print(f"\n→ Suggested beta: {suggested_beta}")
        print("  Reason: Balanced dataset (40-60% TTS)")
        print("  Also try: 0.4, 0.6")
    elif tts_ratio > 0.2:
        suggested_beta = 0.4
        print(f"\n→ Suggested beta: {suggested_beta}")
        print("  Reason: Minority TTS dataset (20-40%)")
        print("  Also try: 0.3, 0.5")
    else:
        suggested_beta = 0.3
        print(f"\n→ Suggested beta: {suggested_beta}")
        print("  Reason: Minimal TTS dataset (<20%)")
        print("  Also try: 0.2, 0.4")
    
    print("\nFine-tuning tips:")
    print("  - Monitor TTS classifier accuracy (should be 50-70%)")
    print("  - If TTS acc >85%, increase beta")
    print("  - If wake word performance drops, decrease beta")
    print("  - Best beta often ±0.1 from initial suggestion")
    
    return suggested_beta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Suggest beta value based on dataset')
    parser.add_argument('config_file', help='Path to training config YAML')
    args = parser.parse_args()
    
    analyze_dataset_composition(args.config_file)