# Kizz experiment findings

This file records decisions that should constrain the next experiment. It is
organized by failure mechanism rather than run order.

## Current baseline: v19 checkpoint 100

V19 at cutoff `0.70` and sliding window `1` remains the physical baseline. In
its selection replay, checkpoint 100 accepted 8/8 expanded positives and 3/8
confusables. A repeat accepted 4/4 positives and 3/4 confusables. Checkpoint 200
accepted only 5/8 positives; checkpoint 400 accepted 1/4 diverse positives.
Checkpoint 100 therefore preserved the best replay recall of that run.

The broader device-corpus replay still exposes weak separation. At `0.70`, v19
accepted 67/93 positives and 34/68 hard negatives, with 0/14 ambient captures.
It also accepted only two of four fresh current-room positive replays. A later
45-second room guard produced one false wake at `.749`. V19 is useful because
its live recall felt better than the later candidates, not because these results
meet the qualification bar.

## Findings

### Synthetic-only search did not transfer

Early combined and fee-family models used random clip holdouts that could reuse
speaker identities across train and test. They were useful search diagnostics,
not speaker-generalization evidence.

| Experiment | Offline result | Decision |
| --- | --- | --- |
| Combined v6 across nine spellings | At cutoff `.39`, intended acceptance ranged from 75.2% to 95.4%, while several unseen spellings reached only 31.3%–44.3% and confusables reached 21.1%–25.0%. | Rejected: weak unseen-pronunciation recall and high collision acceptance. |
| Combined v6 at `.78` | Zero observed ambient false accepts, but intended acceptance fell to 44.0%–68.0%. | Rejected: thresholding removed too much recall. |
| Fee-family v1 | Intended acceptance reached 94.5%–98.6%, but bare `Kizz`, `kiss`, and `Hiffy Kiss` reached 34.0%, 40.4%, and 66.9%. | Rejected: the model learned the suffix and nearby sounds. |
| Fee-family v2 with stronger negative pressure | Bare-word collisions fell, but conjunction negatives still reached 31.6%–38.4%. | Rejected: more negative weight did not create a clean full-phrase boundary. |

The framework now splits synthetic voices by identity before synthesis and
requires independent adult and child voice IDs for validation and test.

### Training-corpus success was misleading

An early quality-masked candidate accepted 17/17 device positives and 0/8 hard
negatives at `.70`. Sixteen positives were training inputs, and the seventeenth
used the same human speaker under an invalid second identity. Physical speaker
replay then accepted only 1/8 positives. The split was repaired, and training
captures no longer count as qualification evidence.

The human-referenced quality mask remains useful as a source screen. Tightening
the accepted positive voiced span from 322–1,540 ms to 483–1,100 ms rejected
6,229 of 67,150 generated clips while preserving hard negatives. The mistake
was treating a cleaner training domain as evidence of physical transfer.

### A threshold cannot repair overlapping scores

Several natural positives scored near zero while `Hey Kizz` and ordinary room
speech scored above `.70`. Lowering the cutoff could not recover zero-scoring
positives and increased false wakes; raising it removed already marginal
positives. Cutoff search remains necessary after training, but it is not a
substitute for class separation.

### Better aggregate replay can hide current-speaker failure

V27 and v28 pushed down hard-negative accepts on the accumulated device corpus.
V28 reached 88/93 positive accepts, 3/68 hard-negative accepts, and 0/14 ambient
accepts at `.70`, yet accepted only one of four fresh current-room positives.
It was rejected before flashing. The fresh-speaker diagnostic is now a required
gate alongside aggregate reporting.

### Fine-tuning choices produced opposite failures

V29 continued from v19 with batch normalization unfrozen. Positive recall
collapsed to 22/93. V30 froze batch normalization and recovered all four fresh
diagnostic positives, but accepted 40/68 hard negatives and 1/14 ambient
captures. Architecture and batch-normalization state must be pinned in the
training configuration; neither setting is presumed safe.

### Balanced pressure did not qualify v31

V31 continued the v30 path with equal positive and negative effective pressure.
At cutoff `.68`, it accepted 80/93 positives, 34/68 hard negatives, 0/14 ambient
captures, and 4/4 fresh diagnostics. At `.70`, it accepted 78/93 positives,
31/68 hard negatives, 0/14 ambient captures, and 3/4 fresh diagnostics.

On physical Kizz, v31 detected four held-out adult and child speaker replays,
but their peaks included `.690` and `.710`. It then false-woke repeatedly on
room audio at `.68`. Resetting all TensorFlow Lite resource variables and
streaming state after a detection did not stop the false wakes. Raising the
cutoff above the false peaks would also reject the marginal positives. V31 was
rejected as a separation failure, not tuned further.

### Model ensembles did not recover the missing domain

Combining the broad, device-specialist, and v28 scores did not recover the fresh
zero-scoring positives without accepting more confusables. An ensemble helps
only when its members make complementary errors; these models shared the same
missing-speaker and room-channel failures.

## Rules for the next candidate

1. Keep v19 checkpoint 100 and cutoff `.70` as the physical control.
2. Change one declared factor per comparison; audit shared sources and config.
3. Preserve clean sources and derive augmentation reproducibly.
4. Balance source groups and pronunciation variants. Do not let
   corpus size set their frequency.
5. Keep detector misses, false wakes, long household speech, music, and room
   noise as labeled device evidence.
6. Reject a candidate that fails the fresh-speaker diagnostic, even when its
   aggregate replay improves.
7. Flash only after speaker-independent synthetic and device reports pass, then
   measure live recall and false wakes on the quantized artifact.

The implementation and research references for these controls live in
[Techniques and references](../../documentation/techniques.md).
