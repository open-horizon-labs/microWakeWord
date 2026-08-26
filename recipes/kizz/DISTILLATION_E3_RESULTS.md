# Kizz Control E3 distillation results

## Aim and hard gates

E3 tested whether the qualified Kizz Control phoneme teacher could transfer its
clean and device-channel separation into an ESP32-sized causal student. A
candidate could advance to INT8 only if one checkpoint satisfied all of these
conditions at one validation-selected threshold:

- clean validation recall at least 90% with at most 0.1 false accepts/hour;
- at least 10 of 11 held-out, individually qualified device replays accepted;
- then, and only then, locked positives at least 23/26, target positives at
  least 20/24, and 0/62 reviewed false wakes.

No E3 candidate passed the first two gates. Nothing from E3 was quantized or
flashed.

## Data-contract correction

The original student checkpoint selector saw only clean validation positives.
The corpus now binds a separate device-validation quality report and guarantees
that these recordings are:

- individually qualified and hash-bound;
- excluded from training and the locked evaluation set;
- represented across the active TTS providers; and
- used only for checkpoint selection.

The frozen teacher accepts 10/11 of this filtered device set at its qualified
clean threshold. That is the transfer ceiling for the current held-out set and
the source of the 10/11 student gate.

## Results

| Experiment | Change from E2 | Clean result | Device result | Decision |
| --- | --- | --- | --- | --- |
| E2 rescored | Original dilated CTC student, every checkpoint rescored on device validation | Some checkpoints pass the clean gate | Best 6/11 at the clean operating point | Baseline only |
| Combined KD continuation | E2 initialization plus posterior, occupation, and global representation KD | Regressed | At most 5/11 | Reject |
| Paired clean/device consistency | Exact parent binding and bounded probability consistency loss | Unstable/regressed | 0/11 in the joint run | Reject |
| Low-weight posterior KD | Temperature-aware posterior KD from scratch | Regressed | 1/11 before stop | Reject |
| Wide causal student | 163,364 parameters instead of 94,836 | Regressed | Best 6/11 | Capacity alone falsified |
| Temporal hidden-state KD | Frame-aligned 96-D PCA projection of the qualified teacher hidden sequence | No clean-qualified checkpoint; best clean zero-FP recall 70.7% | At most 5/11 at zero FP | Reject |
| Exact-view posterior KD | Teacher rescored the exact 300 generated noisy views | Frequently hundreds of false accepts; stopped at step 1,200 | At most 2/11 | Reject |
| Exact-view sequence KD | Transfer final teacher decision on each exact noisy view | No clean-qualified checkpoint; stopped at step 2,200 | Best 4/11 | Reject |
| Teacher-agreement gate | Suppress sequence KD when teacher binary decision disagrees with the label | Worse than E2; stopped at step 2,100 | Best 2/11 | Reject |
| Direct binary decision prototype | One causal wake score rather than 20 CTC logits | Best 5 false accepts at the 90% recall floor; no qualifying threshold | Up to 11/11 at the clean zero-FP boundary | Reject; useful capacity signal |
| Direct decision plus expanded negatives | 22,984 public negatives, 3:1 negative ratio, lower learning rate | Best observed 12 false accepts at the 90% recall floor | At most 2/11 | Reject |

## Exact-view teacher audit

The exact materialization reproduces the original overlay frontend and target
arrays byte-for-byte. The teacher accepts 264/300 generated overlay positives
at its frozen threshold:

| Source | Accepted / total |
| --- | ---: |
| AssemblyAI | 122/136 |
| Deepgram | 26/44 |
| ElevenLabs | 55/56 |
| Kokoro | 61/64 |

The 36 teacher-rejected true positives explain why blindly adding teacher
decision loss conflicts with hard labels. They do not, however, explain the
whole transfer failure: masking those disagreements also underperformed E2.

## What E3 established

1. The prior 72.6% aggregate INT8 result was not enough evidence for device
   deployment. Explicit device-channel checkpoint selection changes the answer.
2. The teacher is strong enough on held-out device replays (10/11); the current
   student transfer path is not.
3. More parameters, generic posterior KD, hidden-state KD, and simple paired-view
   consistency do not recover the teacher's ranking.
4. A binary causal student can retain all device examples, so acoustic capacity
   is not obviously absent, but its clean negative ranking remains inadequate.
5. The current validation exposure is still too short to estimate a production
   0.1 FAPH rate precisely. The zero-observed-FP gate is intentionally strict;
   it is a promotion gate, not a confidence interval claim.

## Review

**Aim:** produce a genuinely qualified ESP32 wake model, not merely a better
training loss.

**Status:** Salvage.

- Necessary: device-channel checkpoint selection and exact provenance are
  necessary and reusable.
- Aligned: the experiments directly test teacher-to-student transfer.
- Sufficient: no; no candidate reached the promotion gates.
- Mechanism: the tested transfer mechanisms are explicit and falsifiable.
- Risks retired: capacity-only, global/frame representation KD, exact-view
  posterior KD, exact-view decision KD, paired consistency, and a direct binary
  head were tested and rejected in their current forms.
- Drift: execution expanded from one distillation correction into an E3
  tournament. The route is to preserve the contracts/results and reframe the
  student objective before another run.

Human verification is not requested because no hardware claim or artifact is
being promoted.

## Dissent

The strongest argument against stopping before firmware is that the direct
binary prototype reached 11/11 device acceptance at a zero-FP threshold. That
does not justify flashing it: the same checkpoints failed clean validation,
sometimes badly, and threshold selection had no qualifying operating point.

The strongest argument against more distillation work is that 11 held-out
device clips are too few and may overstate the teacher. That is true, but it
does not invalidate the immediate conclusion: every student failed even this
small gate. Enlarging device validation is necessary before claiming success,
not a reason to weaken the current gate.

**Recommendation:** RECONSIDER the student objective and evaluation design
before another training tournament. Keep E2 as the best CTC baseline; do not
quantize or flash E3.

