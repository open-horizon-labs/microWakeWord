# Beta Tuning Quick Reference Card

## 🎯 Target Metrics
- **TTS Classifier Accuracy**: 55-65% (confused = good!)
- **Wake Word Accuracy**: Stable or improving
- **Real Speech Performance**: Better than baseline

## 🚦 Quick Diagnosis

### Look at TTS Classifier Accuracy:
| TTS Acc | Diagnosis | Action |
|---------|-----------|--------|
| >85% | Beta TOO LOW | Increase by 0.1-0.2 |
| 70-85% | Borderline low | Maybe increase by 0.1 |
| 55-70% | ✅ GOOD RANGE | Check wake metrics |
| 45-55% | Borderline high | Check if wake dropping |
| <45% | Beta TOO HIGH | Decrease by 0.1-0.2 |

### Look at Wake Word Performance:
| Wake Trend | Diagnosis | Action |
|------------|-----------|--------|
| Improving | ✅ Good! | Keep current beta |
| Stable | ✅ Okay | Check validation data |
| Dropping <5% | Monitor | Might be normal |
| Dropping >10% | Beta too high | Decrease by 0.1-0.2 |

## 📊 Starting Points by Dataset

```
>80% TTS  → Start with β=0.7
60-80%    → Start with β=0.6
40-60%    → Start with β=0.5 (default)
20-40%    → Start with β=0.4
<20% TTS  → Start with β=0.3
```

## 🔍 Training Log Patterns

### ✅ GOOD (β=0.5 working well):
```
Step 1000: Wake_Acc=0.82, TTS_Acc=0.68, Wake_Loss=0.35
Step 2000: Wake_Acc=0.84, TTS_Acc=0.64, Wake_Loss=0.32
Step 3000: Wake_Acc=0.85, TTS_Acc=0.61, Wake_Loss=0.30
```

### ❌ BAD (β too low, try 0.6-0.7):
```
Step 1000: Wake_Acc=0.82, TTS_Acc=0.88, Wake_Loss=0.35
Step 2000: Wake_Acc=0.84, TTS_Acc=0.91, Wake_Loss=0.32
```

### ❌ BAD (β too high, try 0.3-0.4):
```
Step 1000: Wake_Acc=0.82, TTS_Acc=0.52, Wake_Loss=0.35
Step 2000: Wake_Acc=0.74, TTS_Acc=0.51, Wake_Loss=0.48
```

## 🛠️ Tuning Workflow

1. **Quick Test** (2000 steps, 3 values):
   ```bash
   for beta in 0.3 0.5 0.7; do train_quick; done
   ```

2. **Check Results**:
   ```bash
   grep "TTS_Acc" log.txt | tail -5
   ```

3. **Fine-tune** (±0.1 around best):
   ```bash
   # If 0.5 worked best:
   for beta in 0.4 0.5 0.6; do train_full; done
   ```

## ⚡ Emergency Fixes

| Problem | Quick Fix |
|---------|-----------|
| TTS Acc stays >90% | β→0.8, reduce adversarial_hidden_units |
| Wake Acc crashes | β→0.3, lower learning rate 5x |
| Training unstable | β→0.5, check gradient clipping |
| No improvement | β→0.6, add TTS diversity |

## 📈 Success Indicators

✅ **You have good beta when:**
- TTS accuracy: 55-65%
- Wake accuracy: stable/improving
- Validation gap (TTS vs Real) < 5%
- Training stable, losses balanced
- Real speech performance > baseline

## 🎬 Final Check

After full training, validate:
```python
if tts_acc > 0.7: print("Consider higher beta next time")
if wake_dropped > 0.1: print("Consider lower beta next time")
if real_speech_improved: print("Success! Beta was good")
```