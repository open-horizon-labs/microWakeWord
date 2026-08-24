# Kizz experiment findings

This file records decisions that should constrain the next experiment. It is
organized by failure mechanism rather than run order.

For the corpus sources, generated voices, physical fixtures, augmentation,
sampling ledger, training configuration, and salvage guardrails behind these
results, see the [Kizz training reference](TRAINING_REFERENCE.md).

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

## V32 controlled recipe reboot

The live household failures invalidate another small continuation of the v19
recipe. The deployed control is specifically checkpoint 100, not the final v19
export:

```text
checkpoint-candidates/v19-live-preserved-from-git.tflite
sha256 76250d0cef49f893df4724ea6cce0e87b8a8d0d63cf10fbe23c0e624298871ff
```

V19 trained `High Five Kizz` and eight `Kids` spellings as positives. That
history is preserved as evidence, but v32 does not inherit its label contract.
The separate `corpus.v32-canonical.yaml` recipe has one positive text:
`Hi-Fi Kizz`. Former `Kids` aliases and `High Five Kizz` are negatives. The
baseline `corpus.yaml` remains unchanged so this comparison is auditable.

V32 adds 48 connected household sentences rather than another short collision
list. Thirty-two sentence identities train; eight validate; eight test. Their
base Piper speakers are also disjoint, with 100 validation and 100 test speaker
IDs. Each sentence receives 96 deterministic voice/prosody renders, producing
4,608 connected-speech negatives. Ordinary sentences and phonetic collisions
place confusing fragments at the beginning, middle, and end.

The existing negative feature archive is the primary broad-speech source:
roughly 899 hours across speech, dinner-party, and non-speech data. Hard mining
uses the exact quantized deployed control, keeps source split and frame
coordinates, selects diverse local peaks, and retains a deterministic random
reserve. `dinner_party_eval` remains physically excluded from mining. A full
Python reference scan is sharded and resumable. Later scans pass v19 and the
new candidate together so the miner unions both models' local peaks before
applying quotas; v19 alone does not define what is hard.

The device captures from 2026-08-23 are deployment anchors, not training data.
Their copied snapshot contains 47 hash-verified observations: 18 reviewed
no-command false wakes and 29 speech-unconfirmed wakes. Its manifest explicitly
sets `training_eligible: false`. Quarantined device clips require a human review
and explicit promotion step; archive mining does not change that rule.

The flashed pre-roll artifact produced a separate exact-path proof capture:
48,000 pre-wake samples plus 96,000 post-wake samples at 16 kHz, with callback
probability `.706` at cutoff `.70`. Offline replay of the same nine-second WAV
with the deployed v19 quantized artifact peaks at `.9882` in the window ending
3.15 seconds into the capture. That proof manifest also sets
`training_eligible: false`.

Legacy aggregate device positives are not reused: they mix canonical prompts
with `hi_phi` and `hiffy` variants. An explicit promotion manifest instead
selects 20 canonical train captures with recorded phrase spans across five
speaker IDs. Every WAV and original capture record is hash-bound. Each promoted
positive is deterministically cropped or padded to a two-second window that is
guaranteed to contain the complete phrase. Span-less physical test captures are
not converted into per-window positive features; they remain raw full-stream
holdouts. The configuration boundary rejects `observations`, `false-wakes`, and
`evidence` paths.

The first v32 feature build exposed an inherited quality-policy bug: the
positive rule rejecting any source that would be cropped also rejected all
4,608 connected-sentence negatives. V32 now applies that truncation rule only
to positives. Long negative recordings are intentionally scored and trained as
streams because every model window inside ordinary speech remains negative.
The corrected mask accepts all 4,608 connected sentences while still rejecting
overlong positive synthesis. The bounded feature build contains 168,114
windows and records a one-example-per-speaker/phrase cap to preserve diversity
without materializing repeated near-duplicates.

Two same-architecture product candidates test whether inherited shortcuts can
be safely unlearned rather than pretending initialization is the only changed
factor:

1. checkpoint-100 initialization, with its exact weights and a bounded low-rate
   adaptation schedule;
2. scratch initialization, with the full recipe schedule appropriate for a new
   model.

The schedules intentionally differ, so this is not a causal initialization
ablation. It compares a bounded low-rate adaptation recipe against a full
scratch recipe; any winning mechanism needs a later matched-schedule ablation.

Both candidates are exported and evaluated quantized. Cutoffs are selected on
validation only, comparing sliding windows 1, 3, and 5 from one inference trace
per clip. The selector refuses to emit an operating point above the firmware's
`.99` maximum or one with zero validation recall. The untouched
`dinner_party_eval` set is the public-source holdout; sentence-disjoint test
negatives, prior real-speaker positives, and the 2026-08-23 deployment anchors
remain closed until the candidate and cutoff are frozen. V19 is not
replaced unless a fresh exact-artifact physical run shows both household false
wake improvement and no unacceptable wake-recall regression.

### Negative-source coverage decision

The current archive is unusually relevant to this failure before counting its
size. [VOiCES](https://voices18.github.io/) contributes far-field English read speech from hundreds of source
speakers replayed in real reverberant rooms with television, music, babble, and
HVAC distractors. [CHiME-6](https://chimechallenge.github.io/chime6/overview.html) contributes natural multi-speaker dinner-party speech
from twenty real homes. FMA, FSD50K, and WHAM! contribute music, sound events,
and noise. The connected Piper corpus realizes all 500 train, 100 validation,
and 100 test base-speaker IDs with no overlap.

This covers the observed family/TV/room mechanism better than adding another
large close-talk corpus blindly. It does not prove accent, age, or spontaneous
style completeness. [Common Voice](https://commonvoice.mozilla.org/en/datasets) is the next source for a measured
speaker/accent gap; GigaSpeech XS is the next source for a measured
podcast/YouTube style gap. [GigaSpeech](https://github.com/SpeechColab/GigaSpeech) access is gated by an external agreement
and contact-information form, so it is not silently downloaded or accepted.
Either source is added only with speaker/source-disjoint splits and an untouched
evaluation partition, after v32 reports which gap remains.

The v19 algorithm-v2 scan scored 647,283 feature items across nine eligible
archives and excluded `dinner_party_eval`. The merged corpus contains 1,152
hard windows: exactly 32 per source in each score band `.5-.7`, `.7-.8`,
`.8-.9`, and `.9-1.01`, plus 204 seeded random-reserve windows. Its manifest is
bound to the deployed v19 SHA, every source feature hash, every RaggedMmap
layout hash, and miner config hash `54f161ba3c1f47ba580cae970a7ce8063837ae54feec584e4be9b26cae098fa0`.

### V32 result: better rejection, unacceptable recall

The controlled reboot did not qualify a replacement model. The v19-initialized
candidate could not emit a deployable validation point. Scratch initialization
did emit one at cutoff `.80392`, window `1`, accepting 193/1,715 validation
positives and 4/5,076 hard negatives. That 11.3% recall was too low to deploy,
but it was sufficient to run the single planned candidate-remine pass.

The dual-model remine rescored all 647,283 eligible archive items against both
the exact v19 control and the scratch artifact, excluded `dinner_party_eval`,
and retained 1,152 diverse hard windows plus 204 deterministic random-reserve
windows. Its config hash is
`613cab32ce215d428627622887bb0af014cc469dec5e9269ecd9027e3063e1e2`.
The bounded 1,500-step, frozen-batch-normalization refinement produced quantized
artifact SHA-256
`5a0e24a1287113db2c1e16f00770ea2bb0e527708b2e03ebfc51bfa7590f81dc`.

| Frozen evaluation | v19 control `.70/1` | v32 remine `.64706/1` |
| --- | ---: | ---: |
| Synthetic test canonical positives | 791/1,528 | 204/1,528 |
| Synthetic test hard negatives | 1,094/5,076 | 11/5,076 |
| Raw device test positives | 15/17 | 6/17 |
| Raw device test hard negatives | 8/12 | 4/12 |

The v32 validation point accepted 225/1,715 positives and 4/5,076 hard
negatives. All 18 reviewed false-wake anchors and all 29 speech-unconfirmed
anchors scored zero, as did the nine-second pre-roll proof capture that v19
scored `.9882`. The untouched internal testing ROC brackets the selected cutoff
between `.87` (100% false-reject rate, zero observed false accepts per hour) and
`.46` (94.34% false-reject rate, `.187` estimated false accepts per hour).
The raw device test is useful playback evidence but is not qualification-eligible:
that split has no registered human speakers and no ambient-negative captures.

This is a real separation improvement and a failed product model. The recipe
can teach this architecture to reject the newly represented speech domain, but
not while preserving acceptable speaker/channel recall. V19 therefore remains
the capture-only physical control. No v32 artifact is flashed. The next model
work must change a declared mechanism—capacity, representation, sequential
verification, or wake phrase—not silently continue this run or reopen the test
sets for tuning.
