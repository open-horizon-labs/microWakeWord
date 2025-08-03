#!/usr/bin/env python3
"""Visualize the interaction between beta and lambda parameters."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def effective_adversarial_strength(beta, lambda_val):
    """Calculate effective adversarial influence."""
    return beta * lambda_val

def predict_tts_accuracy(beta, lambda_val):
    """Predict TTS classifier accuracy based on parameters."""
    # Empirical model based on typical behavior
    strength = effective_adversarial_strength(beta, lambda_val)
    
    # Sigmoid-like function: high strength = low accuracy
    base_acc = 0.9  # Starting accuracy without adversarial
    min_acc = 0.5   # Minimum (random) accuracy
    
    # Strength affects accuracy
    tts_acc = min_acc + (base_acc - min_acc) * np.exp(-2 * strength)
    
    # Add some noise/complexity
    tts_acc += 0.05 * np.sin(beta * 10) * np.cos(lambda_val * 10)
    
    return np.clip(tts_acc, min_acc, base_acc)

def predict_wake_performance(beta, lambda_val, base_performance=0.85):
    """Predict wake word performance impact."""
    strength = effective_adversarial_strength(beta, lambda_val)
    
    # Too much adversarial training hurts performance
    if strength > 0.8:
        penalty = (strength - 0.8) * 0.3
    else:
        penalty = 0
    
    # Some adversarial training helps (up to a point)
    if strength < 0.6:
        bonus = strength * 0.05
    else:
        bonus = 0.03
    
    performance = base_performance - penalty + bonus
    return np.clip(performance, 0.5, 0.95)

def plot_interaction_surface():
    """Create 3D surface plots showing beta-lambda interaction."""
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create parameter grids
    beta_range = np.linspace(0.1, 0.9, 50)
    lambda_range = np.linspace(0.2, 2.0, 50)
    beta_grid, lambda_grid = np.meshgrid(beta_range, lambda_range)
    
    # Calculate metrics for each combination
    tts_acc_grid = np.zeros_like(beta_grid)
    wake_perf_grid = np.zeros_like(beta_grid)
    strength_grid = np.zeros_like(beta_grid)
    
    for i in range(len(beta_range)):
        for j in range(len(lambda_range)):
            b = beta_grid[j, i]
            l = lambda_grid[j, i]
            tts_acc_grid[j, i] = predict_tts_accuracy(b, l)
            wake_perf_grid[j, i] = predict_wake_performance(b, l)
            strength_grid[j, i] = effective_adversarial_strength(b, l)
    
    # Plot 1: Effective Strength
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(beta_grid, lambda_grid, strength_grid,
                             cmap=cm.viridis, alpha=0.8)
    ax1.set_xlabel('Beta')
    ax1.set_ylabel('Lambda')
    ax1.set_zlabel('Effective Strength')
    ax1.set_title('Effective Adversarial Strength\n(β × λ)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot 2: TTS Accuracy
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(beta_grid, lambda_grid, tts_acc_grid,
                             cmap=cm.RdYlGn_r, alpha=0.8)
    ax2.set_xlabel('Beta')
    ax2.set_ylabel('Lambda')
    ax2.set_zlabel('TTS Accuracy')
    ax2.set_title('Predicted TTS Classifier Accuracy\n(Lower is better)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # Plot 3: Wake Performance
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(beta_grid, lambda_grid, wake_perf_grid,
                             cmap=cm.RdYlGn, alpha=0.8)
    ax3.set_xlabel('Beta')
    ax3.set_ylabel('Lambda')
    ax3.set_zlabel('Wake Performance')
    ax3.set_title('Predicted Wake Word Performance\n(Higher is better)')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # Plot 4: Contour plot of TTS accuracy
    ax4 = fig.add_subplot(2, 3, 4)
    contour1 = ax4.contour(beta_grid, lambda_grid, tts_acc_grid, 
                           levels=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
    ax4.clabel(contour1, inline=True, fontsize=8)
    ax4.set_xlabel('Beta')
    ax4.set_ylabel('Lambda')
    ax4.set_title('TTS Accuracy Contours\n(Target: 0.55-0.65)')
    
    # Highlight good region
    good_region = (tts_acc_grid >= 0.55) & (tts_acc_grid <= 0.65)
    ax4.contourf(beta_grid, lambda_grid, good_region.astype(float), 
                 levels=[0.5, 1.5], colors=['green'], alpha=0.2)
    
    # Plot 5: Contour plot of wake performance
    ax5 = fig.add_subplot(2, 3, 5)
    contour2 = ax5.contour(beta_grid, lambda_grid, wake_perf_grid,
                           levels=[0.7, 0.75, 0.8, 0.82, 0.84, 0.86, 0.88])
    ax5.clabel(contour2, inline=True, fontsize=8)
    ax5.set_xlabel('Beta')
    ax5.set_ylabel('Lambda')
    ax5.set_title('Wake Performance Contours\n(Higher is better)')
    
    # Plot 6: Optimal region combining both metrics
    ax6 = fig.add_subplot(2, 3, 6)
    
    # Define optimality score
    optimality = np.zeros_like(beta_grid)
    for i in range(len(beta_range)):
        for j in range(len(lambda_range)):
            tts = tts_acc_grid[j, i]
            wake = wake_perf_grid[j, i]
            
            # Good TTS confusion (0.55-0.65)
            if 0.55 <= tts <= 0.65:
                tts_score = 1.0
            elif 0.5 <= tts <= 0.7:
                tts_score = 0.5
            else:
                tts_score = 0.0
            
            # Good wake performance (>0.82)
            wake_score = max(0, (wake - 0.75) / 0.15)
            
            optimality[j, i] = tts_score * wake_score
    
    im = ax6.imshow(optimality, extent=[0.1, 0.9, 0.2, 2.0], 
                    origin='lower', aspect='auto', cmap=cm.RdYlGn)
    ax6.set_xlabel('Beta')
    ax6.set_ylabel('Lambda')
    ax6.set_title('Optimal Parameter Region\n(Green = Best)')
    fig.colorbar(im, ax=ax6)
    
    # Add common configurations
    configs = [
        (0.5, 1.0, 'Default'),
        (0.7, 0.8, 'High TTS'),
        (0.3, 1.5, 'Low TTS'),
        (0.6, 1.2, 'Fast'),
        (0.4, 0.8, 'Conservative')
    ]
    
    for beta, lam, label in configs:
        ax6.plot(beta, lam, 'ko', markersize=8)
        ax6.annotate(label, (beta, lam), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    
    plt.suptitle('Beta-Lambda Interaction Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig('beta_lambda_interaction.png', dpi=150)
    plt.show()

def plot_training_dynamics():
    """Show how the interaction changes during training."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Simulate training progression
    steps = np.linspace(0, 10000, 100)
    
    # Different beta-lambda combinations
    configs = [
        (0.5, 1.0, 'Balanced (β=0.5, λ=1.0)'),
        (0.7, 0.8, 'High Beta (β=0.7, λ=0.8)'),
        (0.3, 1.5, 'High Lambda (β=0.3, λ=1.5)'),
        (0.7, 1.5, 'Both High (β=0.7, λ=1.5)'),
    ]
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for idx, (beta, lambda_val, label) in enumerate(configs):
        color = colors[idx]
        
        # Simulate metrics over time
        tts_accs = []
        wake_accs = []
        
        for step in steps:
            # Early training: TTS learns quickly
            progress = step / 10000
            
            # TTS accuracy trajectory
            if progress < 0.2:
                # Quick initial learning
                tts_base = 0.9 - progress * 2 * beta * lambda_val
            else:
                # Convergence to confused state
                target_tts = predict_tts_accuracy(beta, lambda_val)
                tts_base = target_tts + (0.9 - target_tts) * np.exp(-5 * (progress - 0.2))
            
            tts_acc = tts_base + np.random.normal(0, 0.02)
            tts_accs.append(tts_acc)
            
            # Wake accuracy trajectory
            if progress < 0.1:
                # Initial learning
                wake_acc = 0.5 + progress * 3
            else:
                # Affected by adversarial training
                wake_target = predict_wake_performance(beta, lambda_val)
                wake_acc = wake_target + np.random.normal(0, 0.01)
                
                # Add some oscillation if parameters too high
                if beta * lambda_val > 0.8:
                    wake_acc += 0.05 * np.sin(progress * 20)
            
            wake_accs.append(wake_acc)
        
        # Plot TTS accuracy
        axes[0, 0].plot(steps, tts_accs, color=color, label=label, alpha=0.7)
        
        # Plot wake accuracy
        axes[0, 1].plot(steps, wake_accs, color=color, label=label, alpha=0.7)
        
        # Plot effective gradient magnitude (simulated)
        grad_magnitude = [beta * lambda_val * (0.9 - tts) for tts in tts_accs]
        axes[1, 0].plot(steps, grad_magnitude, color=color, label=label, alpha=0.7)
        
        # Plot stability (inverse of variance)
        window = 20
        stability = []
        for i in range(len(wake_accs)):
            if i < window:
                stability.append(1.0)
            else:
                variance = np.var(wake_accs[i-window:i])
                stability.append(1.0 / (1.0 + variance * 100))
        axes[1, 1].plot(steps, stability, color=color, label=label, alpha=0.7)
    
    # Configure plots
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('TTS Classifier Accuracy')
    axes[0, 0].set_title('TTS Confusion Over Time')
    axes[0, 0].axhline(y=0.6, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Training Steps')
    axes[0, 1].set_ylabel('Wake Word Accuracy')
    axes[0, 1].set_title('Wake Performance Over Time')
    axes[0, 1].axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Gradient Magnitude')
    axes[1, 0].set_title('Effective Adversarial Gradient')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Training Steps')
    axes[1, 1].set_ylabel('Training Stability')
    axes[1, 1].set_title('Stability Score (1 = stable)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Dynamics with Different β-λ Combinations', fontsize=14)
    plt.tight_layout()
    plt.savefig('beta_lambda_dynamics.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("Generating beta-lambda interaction visualizations...")
    
    print("\n1. Creating 3D surface plots...")
    plot_interaction_surface()
    
    print("\n2. Creating training dynamics plots...")
    plot_training_dynamics()
    
    print("\nVisualizations saved as:")
    print("  - beta_lambda_interaction.png")
    print("  - beta_lambda_dynamics.png")
    
    print("\nKey insights:")
    print("  • Effective strength = β × λ")
    print("  • Target TTS accuracy: 55-65%")
    print("  • Optimal region typically: β∈[0.4,0.6], λ∈[0.8,1.2]")
    print("  • High β×λ (>0.8) risks wake performance degradation")
    print("  • Low β×λ (<0.3) provides minimal adaptation benefit")