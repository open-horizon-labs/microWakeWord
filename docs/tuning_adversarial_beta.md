# Tuning the Adversarial Beta Parameter

## Overview

Beta (β) controls the balance between wake word detection and domain adversarial training in the loss function:
- **Wake word loss weight**: (1 - β)
- **TTS classifier loss weight**: β
- **Total loss** = (1-β) × wake_word_loss + β × tts_classifier_loss

Finding the optimal beta is crucial for effective adversarial training.

## Quick Reference: Signs of Good vs Bad Beta

### ✅ Signs of GOOD Beta Value

1. **TTS Classifier Confusion (50-70% accuracy)**
   - The adversarial classifier struggles to distinguish TTS from real speech
   - This means features are becoming domain-invariant
   - Ideal range: 55-65% accuracy (barely better than random)

2. **Stable or Improving Wake Word Metrics**
   - Wake word accuracy maintains or improves during training
   - Recall and precision don't degrade significantly
   - Loss curves smooth, not oscillating

3. **Validation Performance Gap**
   - Better performance on real speech validation than baseline
   - Reduced gap between TTS and real speech performance
   - Consistent improvements across different test sets

4. **Balanced Loss Contributions**
   - Both losses contribute meaningfully to gradients
   - Neither loss dominates completely
   - Gradient norms from both branches are comparable

### ❌ Signs of BAD Beta Value

#### Beta Too LOW (< optimal)

1. **TTS Classifier Too Accurate (>85%)**
   - Adversarial branch easily distinguishes TTS from real
   - Features retain TTS-specific artifacts
   - Domain adaptation is ineffective

2. **No Improvement Over Baseline**
   - Model performs similarly to non-adversarial version
   - Overfitting to TTS characteristics persists
   - Real speech performance doesn't improve

3. **Training Logs Show:**
   ```
   TTS_Acc=0.92  ← Too high! Beta too low
   Wake_Acc=0.88  ← Good but won't generalize
   ```

#### Beta Too HIGH (> optimal)

1. **Wake Word Performance Degradation**
   - Main task accuracy drops significantly (>10%)
   - Recall plummets while precision might stay high
   - Model "forgets" how to detect wake words

2. **Training Instability**
   - Loss curves oscillate wildly
   - Metrics jump erratically between epochs
   - Gradient explosions or vanishing

3. **Training Logs Show:**
   ```
   TTS_Acc=0.51  ← Very confused (maybe too much)
   Wake_Acc=0.65  ← Degraded significantly!
   ```

## Monitoring During Training

### Key Metrics to Watch

```python
# In your training logs, monitor these patterns:

# GOOD pattern (beta well-tuned):
Epoch 100: Wake_Acc=0.82, TTS_Acc=0.68, Wake_Loss=0.35, TTS_Loss=0.62
Epoch 200: Wake_Acc=0.84, TTS_Acc=0.64, Wake_Loss=0.32, TTS_Loss=0.65
Epoch 300: Wake_Acc=0.85, TTS_Acc=0.61, Wake_Loss=0.30, TTS_Loss=0.67
# → Wake metrics improving, TTS getting more confused

# BAD pattern (beta too low):
Epoch 100: Wake_Acc=0.82, TTS_Acc=0.88, Wake_Loss=0.35, TTS_Loss=0.28
Epoch 200: Wake_Acc=0.84, TTS_Acc=0.91, Wake_Loss=0.32, TTS_Loss=0.22
# → TTS classifier learning too well

# BAD pattern (beta too high):
Epoch 100: Wake_Acc=0.82, TTS_Acc=0.55, Wake_Loss=0.35, TTS_Loss=0.69
Epoch 200: Wake_Acc=0.76, TTS_Acc=0.52, Wake_Loss=0.48, TTS_Loss=0.69
# → Wake performance degrading
```

### TensorBoard Indicators

Look for these patterns in TensorBoard:

1. **Loss Ratio Plot**
   - Plot `wake_loss / total_loss` over time
   - Should roughly equal (1 - β)
   - Large deviations suggest scale mismatch

2. **Performance Correlation**
   - Plot wake_accuracy vs tts_accuracy
   - Good beta: negative correlation (one up, other down)
   - Bad beta: both high (too low) or wake drops alone (too high)

3. **Gradient Norms**
   - Monitor gradient magnitudes from both branches
   - Should be same order of magnitude
   - 10x+ difference suggests beta needs adjustment

## Diagnostic Checklist

Run through this checklist after training for 1000-2000 steps:

### 1. TTS Classifier Check
```python
if tts_accuracy > 0.85:
    print("⚠️ Beta likely TOO LOW")
    print("→ Try increasing by 0.1-0.2")
elif tts_accuracy < 0.55:
    print("⚠️ Might be okay, but check wake metrics")
    if wake_accuracy_dropping:
        print("→ Beta likely TOO HIGH")
else:
    print("✓ TTS confusion in good range")
```

### 2. Wake Word Performance Check
```python
early_wake_acc = metrics[:500].mean()
recent_wake_acc = metrics[-500:].mean()

if recent_wake_acc < early_wake_acc * 0.9:
    print("⚠️ Beta likely TOO HIGH")
    print("→ Adversarial loss dominating")
elif recent_wake_acc > early_wake_acc * 1.05:
    print("✓ Wake performance improving!")
```

### 3. Validation Set Comparison
```python
# Compare performance on different data types
tts_validation_acc = validate(model, tts_val_set)
real_validation_acc = validate(model, real_val_set)

gap = abs(tts_validation_acc - real_validation_acc)
if gap < 0.05:
    print("✓ Good generalization across domains")
elif gap > 0.15:
    print("⚠️ Large performance gap")
    if tts_validation_acc > real_validation_acc:
        print("→ Still overfitting to TTS, consider higher beta")
```

## Beta Selection Strategy

### Initial Selection Based on Dataset

| TTS Data % | Suggested β | Range to Test | Rationale |
|------------|-------------|---------------|-----------|
| >80%       | 0.7         | 0.6-0.8       | Heavy TTS needs strong adversarial signal |
| 60-80%     | 0.6         | 0.5-0.7       | Significant TTS presence |
| 40-60%     | 0.5         | 0.4-0.6       | Balanced dataset (default) |
| 20-40%     | 0.4         | 0.3-0.5       | Some TTS augmentation |
| <20%       | 0.3         | 0.2-0.4       | Minimal TTS data |

### Fine-Tuning Process

1. **Coarse Search** (Train for 2000 steps each)
   ```bash
   for beta in 0.3 0.5 0.7; do
       # Quick training run
       # Evaluate metrics
   done
   ```

2. **Analyze Results**
   - Plot wake_accuracy vs beta
   - Plot tts_accuracy vs beta
   - Find beta where tts_acc ≈ 60-65%

3. **Fine Search** (±0.1 around best)
   ```bash
   # If 0.5 was best, try:
   for beta in 0.4 0.45 0.5 0.55 0.6; do
       # Full training
   done
   ```

## Advanced Diagnostics

### Gradient Conflict Analysis

High gradient conflict indicates the two objectives are fighting:

```python
def analyze_gradient_conflict(model, batch):
    with tf.GradientTape(persistent=True) as tape:
        outputs = model(batch)
        wake_loss = compute_wake_loss(outputs[0])
        tts_loss = compute_tts_loss(outputs[1])
    
    wake_grads = tape.gradient(wake_loss, model.shared_layers)
    tts_grads = tape.gradient(tts_loss, model.shared_layers)
    
    # Compute cosine similarity
    conflict = cosine_similarity(wake_grads, tts_grads)
    
    if conflict < -0.5:
        print("High gradient conflict - this is GOOD for adversarial training!")
    elif conflict > 0.5:
        print("Gradients aligned - adversarial training not working")
```

### Loss Landscape Visualization

Monitor how the losses interact:

```python
# During training, track:
wake_loss_trajectory = []
tts_loss_trajectory = []

# After training, plot:
plt.scatter(wake_loss_trajectory, tts_loss_trajectory, c=range(len(wake_loss_trajectory)))
plt.xlabel('Wake Loss')
plt.ylabel('TTS Loss')
plt.colorbar(label='Training Step')

# Good beta: Creates a Pareto frontier
# Bad beta: One loss dominates (vertical or horizontal line)
```

## Common Scenarios and Solutions

### Scenario 1: "TTS classifier won't get confused"
- **Symptom**: TTS accuracy stays >85% even with high beta
- **Causes**: 
  - TTS and real data too different
  - Model capacity too high for adversarial branch
- **Solutions**:
  - Increase beta to 0.8-0.9
  - Reduce adversarial_hidden_units (e.g., "64,32")
  - Increase adversarial_dropout (e.g., 0.7)

### Scenario 2: "Wake word performance crashes"
- **Symptom**: Wake accuracy drops >15% immediately
- **Causes**:
  - Beta too high for dataset
  - Learning rate too high
- **Solutions**:
  - Reduce beta to 0.2-0.3
  - Lower learning rate by 2-5x
  - Use gradual beta scheduling (start low, increase)

### Scenario 3: "No improvement over baseline"
- **Symptom**: Adversarial model = regular model performance
- **Causes**:
  - Beta too low
  - Not enough TTS diversity
  - Features already domain-invariant
- **Solutions**:
  - Increase beta to 0.6-0.7
  - Add more diverse TTS voices
  - Check if problem actually exists (test on deployment TTS)

## Beta Scheduling (Advanced)

Instead of fixed beta, you can schedule it:

```python
def get_beta(epoch, total_epochs):
    # Start low, increase gradually
    if epoch < total_epochs * 0.1:
        return 0.1  # Warm-up: focus on main task
    elif epoch < total_epochs * 0.5:
        # Linear increase
        progress = (epoch - total_epochs * 0.1) / (total_epochs * 0.4)
        return 0.1 + progress * 0.4  # Ramp up to 0.5
    else:
        return 0.5  # Fixed for fine-tuning
```

## Summary: Beta Tuning Decision Tree

```
Start Training
    ↓
After 1000 steps, check TTS accuracy
    ↓
TTS Acc > 85%? → Increase beta by 0.1-0.2
    ↓
TTS Acc < 55%? → Check wake metrics
    ↓           ↘
    ↓            Wake dropping? → Decrease beta by 0.1-0.2
    ↓
TTS Acc 55-85%? → Check validation
    ↓
Real speech improved? → ✓ Good beta!
    ↓
No improvement? → Adjust by ±0.1 and retry
```

## Quick Commands for Evaluation

```bash
# Check current training metrics
grep "TTS_Acc" training.log | tail -20

# Compare validation results
python scripts/analyze_beta_results.py

# Get beta suggestion for your dataset
python scripts/suggest_beta.py training_adversarial.yaml

# Monitor gradient statistics (if logged)
tensorboard --logdir=trained_models/model/logs
```

Remember: The optimal beta depends on your specific dataset, deployment scenario, and performance requirements. Use these guidelines as a starting point, but always validate with your actual use case.