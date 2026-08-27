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
