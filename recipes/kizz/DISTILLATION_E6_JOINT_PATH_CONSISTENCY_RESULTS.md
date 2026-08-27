# Kizz Control E6 joint path-consistency preflight

## Decision

Rejected at prequalification. Neither bounded run approached the required
clean/device trajectory, so no checkpoint is eligible for INT8 conversion,
packaging, or firmware testing.

The unchanged gates remain:

- clean validation recall >= 90% at <= 0.1 FAPH;
- >= 10/11 held-out qualified device replays at that clean-selected threshold;
- only after those pass, the locked aligned, target-channel, and false-wake
  gates.

## Recipe

E6 added `--paired-path-consistency-weight` to the existing 20-logit causal
CTC student. For aligned device/clean training pairs, it compares the
differentiable deployed canonical fit and canonical-vs-worst-collision margin
at the same causal endpoint. The final version uses symmetric gradients for
both views. Device examples still receive hard ordered-state targets, CTC
path loss, and explicit collision losses from the same step; the model starts
from random weights rather than an E4/E5 checkpoint.

This differs from the older posterior consistency experiment because the
agreement target is the deployment path score, not the framewise posterior.
It also differs from E5 because the agreement and sequence objectives are
present from the first update rather than introduced after binary convergence.

## Results

| Run | Steps | Learning rate | Path consistency | Best clean result at 90% recall | Device result | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E6 stop-gradient preflight | 600 | 0.0005 | 0.5 | 114 false accepts at step 600; 0.0% zero-FP recall | 0/11 at clean point; 1/11 at zero-FP boundary | Reject; implementation invalidated because clean-pair masks were not enabled by the new flag |
| E6 symmetric preflight | 300 | 0.0005 | 0.5 | 218 false accepts at step 300; 0.0% zero-FP recall | 0/11 at clean point; 0/11 at zero-FP boundary | Reject; trajectory gate violated |

The first run is retained only as a code-path diagnostic. Its path loss was
zero until the parent-materialization guard was corrected. The second run is
the valid experiment. It had 870 clean false accepts at step 200 and 218 at
step 300, while device acceptance remained 0/11 throughout evaluated
checkpoints.

## Interpretation and next pivot

The failure is not evidence that path-level invariance is impossible, but it
does falsify this direct symmetric score-matching formulation at the tested
weight and learning rate. The clean branch does not establish a stable
canonical path quickly enough for paired score matching to bootstrap device
invariance; the shared model remains far from a useful operating point.

Do not extend E6, sweep its weight, or initialize firmware from it. The next
recipe needs an explicit supervised anchor for both members of each aligned
pair (or a teacher-derived paired target), with a bounded warm-up/anti-
forgetting schedule. That should be designed and adversarially checked before
another training launch.

## Verification

- Focused distillation/unit suite: 48 tests passed after the final change.
- Python bytecode compilation and `git diff --check`: passed.
- Full historical suite remains 626 passed, 3 skipped from the prior
  checkpoint; it was not rerun in this environment because the TensorFlow
  environment does not provide the repository's `pytest` command.
- No candidate was quantized, packaged, flashed, or physically tested.

## Follow-up research: E7-E12

The active branch continued with four bounded tests after E6:

| Run | Mechanism | Best observed clean result at 90% recall | Device result | Decision |
| --- | --- | ---: | ---: | --- |
| E7 | Symmetric path agreement plus supervised clean-parent path anchor, random start | 192 false accepts at step 300 | 0/11 at clean point and zero-FP boundary | Reject |
| E8 | Same pair anchor added to the best ordinary CTC student, low-rate fine-tune | 7 false accepts at step 100 | 0/11 at clean point; 0/11 at zero-FP boundary | Reject |
| E9 | 20 CTC logits plus one jointly trained binary logit, random start | 7 false accepts at step 600 | 7/11 at zero-FP boundary | Reject |
| E10 | E9 plus teacher-ranked binary ordering and negative-frame suppression | 9 false accepts at step 600 | 3/11 at zero-FP boundary | Reject |
| E12 | 20 fresh CTC channels plus the complete E4 four-channel decision head, initialized from E4 | 3 false accepts and 10/11 at zero-FP boundary at step 1; 41 false accepts and 7/11 by step 100 | 5/11 by step 300 | Reject; CTC updates induce rapid decision forgetting |

E11 is not counted as valid model evidence: its initializer copied only E4's
scalar wake channel and omitted the three phonetic-rejection channels. The
implementation was corrected before E12, and an exact-score structural check
matched the E4 four-channel decision output within 0.001 on fresh inputs.

An offline complementarity check combined the qualified E4 decision model and
the best ordinary CTC student without changing either model. The best simple
normalized linear combination still had 2 clean false accepts at the 90%
recall floor and 10/11 device acceptance. A two-threshold AND rule reached at
best 37/41 clean positives with 1 false accept and only 4/11 device positives;
no combination passed both hard gates. This is diagnostic evidence, not a
deployment proposal.

## Fact-check and salvage conclusion

The recorded claims above were checked against the run `distillation.json`
artifacts, checkpoint ledgers, source code, and the fixed corpus/cache hashes.
The claims about E6, E7, E9, E10, and E12 are verified. E8's step-100 result
is verified from its ledger. E11 is explicitly excluded because its transfer
mechanism was invalid. The historical E2-E5 claims remain bound to their
existing reports and were not reinterpreted from loss or AUC.

The salvage result is now narrower and stronger:

1. Within the current 94-95k-parameter causal student family and the existing
   clean/device/overlay schedule, CTC and device-channel decision learning
   have repeatedly demonstrated incompatible update directions.
2. Pair consistency does not bootstrap the missing representation, whether
   trained from random initialization, added to E2, or added to a fresh joint
   head.
3. Retaining the E4 representation preserves device behavior only until the
   CTC objective updates the shared encoder; retaining the full E4 decision
   head does not prevent this forgetting.
4. Separate-model complementarity is real but insufficient on the current
   validation set: it cannot reach zero clean false accepts while retaining
   the device gate.

This supports a scoped feasibility conclusion: no qualifying single compact
student has been found, and the current representation-sharing approach is
not feasible under the tested capacity, data, and objective contracts. It
does not prove that Kizz Control is impossible. A credible next phase would
need a materially different resource decision—more held-out collision/device
data, a larger or explicitly separated dual-branch model, or a reviewed
cascade/ensemble—and must be designed before more weight sweeps.
