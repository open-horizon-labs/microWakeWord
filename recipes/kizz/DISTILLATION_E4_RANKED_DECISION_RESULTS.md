# Kizz Control E4 teacher-ranked decision distillation

## Aim and unchanged promotion gates

E4 tested whether a compact causal decision student could preserve the
qualified teacher's device-channel recall while learning its ordering over
ordinary speech and explicit phonetic collisions.  Promotion still requires
one validation-selected operating point with:

- at least 90% clean validation recall at no more than 0.1 FAPH; and
- at least 10/11 held-out qualified device replays accepted at that point.

No E4 checkpoint passed both gates.  No E4 artifact is eligible for INT8
conversion, packaging, or firmware flash.

## Mechanism

The E4 student is a 92,993-parameter dilated causal encoder initialized from
the strongest E3 binary checkpoint.  Unlike the unstable one-frame prototype,
its deployed decision uses a rolling mean over adjacent outputs.  The loss
combines:

- balanced hard wake/non-wake labels;
- RankNet transfer of the qualified teacher's causal-window ordering;
- bounded current-student hard-negative mining with a random reserve;
- equal clean/device/overlay positive scheduling; and
- explicit phonetic collision curricula.

The final ablation adds three phonetic logits to the same encoder (wake,
critical collision, other) and subtracts collision evidence from wake evidence.
This is one 93k-parameter model, not a second neural verifier.

## Results

| Run | Material change | Best clean result at 90% recall | Device result | Decision |
| --- | --- | ---: | ---: | --- |
| E4 v1 | Fresh ranked decision student, proportional positive sampling | 15 false accepts | 0/11 above zero-FP boundary | Reject; device-channel schedule regressed |
| E4 v2 | Equal clean/device/overlay positives; initialize from E3 direct decision | 3 false accepts | 10/11 at useful checkpoints | Continue |
| E4 v3 | Current-student corpus/public hard mining | 3 false accepts | 10/11 | Reject; undirected mining did not move collisions |
| E4 v4 | Oversample `Kizz patrol` and `His control` from training voices | 3 false accepts | 10/11 | Reject; more hard labels alone were insufficient |
| E4 v5 | Intended training-only phonetic auxiliary task | 2 false accepts | 10/11 | Invalid mechanism claim: auxiliary collision labels were wired to the wrong branch |
| E4 v6 | Correct the auxiliary-label branch and add an adversarial test | 3 false accepts | 10/11 | Reject; learned representation did not alter the deployed scalar decision |
| E4 v7 | Deploy phonetic evidence in the same four-logit student | 3 false accepts at fixed weight | 10/11 | Reject |
| E4 v7 score sweep | Sweep rolling windows 1/2/3 and collision weights 0–5 without retraining | **2 false accepts**, 29/41 (70.73%) zero-FP recall | **10/11** | Best E4 result, still unqualified |

The best deterministic E4 v7 composition is checkpoint step 2000, a
three-frame rolling mean, and collision weight 0.375.  At that composition the
90%-recall threshold still accepts two clean negatives.  The zero-FP boundary
accepts 29/41 clean positives and 10/11 device positives.

## Error localization

At E4 v2 step 1200, all three clean false accepts at the 90%-recall floor were
held-out phonetic collisions:

- ElevenLabs `Kizz patrol`;
- ElevenLabs `His control`; and
- Deepgram `Kizz patrol`.

The phonetic curriculum later moved the Deepgram example below the floor, but
the two held-out ElevenLabs collisions remained.  Public speech was not the
limiting clean-validation error in E4.

## Conclusion

Teacher-ranked scalar decisions improve the direct binary baseline, and the
device-channel schedule is necessary to preserve target-channel behavior.
However, neither hard mining nor a tiny phonetic head recovers the teacher's
full collision separation.  The E4 stop trigger is therefore active: do not
continue weight sweeps or flash this model.  The next recipe needs sequence
structure that directly distinguishes the canonical phone path from collision
paths while retaining the successful E4 device-channel schedule and selection
gate.

## E5 follow-up: transfer the E4 encoder into CTC

E5 tested the immediate hybrid suggested by the E4 result: initialize the
20-logit canonical/collision CTC student from the device-capable E4 binary
encoder, while replacing only the incompatible output head.

Two bounded schedules failed:

1. **All layers trainable.** The random CTC head began learning and briefly
   reached 7/11 device replays at step 400. By steps 600–1100, device acceptance
   collapsed to 0–1/11 while clean false accepts fell to 3–5. This is direct
   catastrophic-forgetting evidence.
2. **Encoder frozen; output head only.** Clean false accepts briefly reached 2
   at step 400, but device acceptance remained 0/11 (maximum 2/11 by step 800).
   The binary encoder's representation is therefore not linearly decodable as
   the required compact phone paths by a replacement head.

E5 was stopped at step 1100 for the all-layer run and completed its planned
800-step frozen-head bound. Neither schedule approached the 10/11 device gate,
so no unfreeze continuation, INT8 conversion, or firmware work was justified.
