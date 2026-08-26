# Kizz Control distillation tournament v1

Status: **completed offline; no candidate qualified for quantization, continuous
evaluation, packaging, or firmware flash**.

This tournament tests three transfer mechanisms selected after the v11 student
failed to preserve the qualified teacher's decision quality, their combination,
and a bounded-capacity architecture alternative. Every candidate uses the same
pronunciation-clean Kizz Control corpus, compact 20-token phone contract,
validation split, and locked evidence described in
[`KIZZ_CONTROL_PHONEME_TEACHER_STUDENT_V1.md`](KIZZ_CONTROL_PHONEME_TEACHER_STUDENT_V1.md).

## Promotion gate

A float checkpoint may proceed only when all four conditions pass:

- validation reaches at least 90% recall with at most 0.1 observed false
  accepts/hour;
- at least 23/26 frozen aligned positives are accepted;
- at least 20/24 frozen StackChan target-channel positives are accepted; and
- 0/62 locked household false wakes are accepted.

Only a clip-qualified float candidate may be quantized and evaluated on the
100-hour continuous corpus. Only a qualified INT8 artifact may be packaged or
flashed. The tournament triggered the first stop condition, so no INT8 or
firmware artifact was produced.

## Implemented mechanisms

### A — temperature posterior distillation

The student minimizes KL divergence to temperature-softened teacher frame
posteriors. The implementation uses the standard `T²` scale so changing the
temperature does not silently change the gradient magnitude. The measured run
uses temperature 2.

### B — sequence-conditioned temporal distillation

`microwakeword.ctc_occupancy.ctc_state_occupation_log_probs` computes exact CTC
forward/backward state occupations conditioned on the canonical transcript.
The student matches those occupations instead of imitating unconstrained frame
posteriors. A bounded causal delay chooses one delay for each example, never a
different shift for each frame.

The alignment diagnostic found that B already removed nearly all systematic
lag: at B step 1000, 154/158 positive examples preferred delay 0 and 4/158
preferred delay 1. A separate +30 ms run therefore tests the residual lag
hypothesis rather than assuming it.

### C — intermediate representation transfer

The qualified teacher's final hidden sequence is time-averaged and projected
to 96 dimensions using train-only PCA. A training-only Dense adapter maps the
student hidden representation to that target. The adapter is discarded at
runtime. The cache binds teacher weights, qualification reports, source rows,
and PCA metadata by hash; validation and test rows never fit the projection.

### D — bounded temporal-residual student

D tests capacity without changing the input, output, decoder, or 670 ms
receptive-field contract. It has five residual MixConv blocks and 96,212
parameters, versus 47,828 for the control student. D2 adds teacher sequence
ranking and explicit ordinary/collision-negative margins. D3 concentrates that
supervision on the positive/negative tails. D4 is one final low-learning-rate
qualification-margin attempt; it regressed and was stopped.

## Deployment-equivalent scoring contract

The original training ledger selected checkpoints incorrectly in two ways:

1. it sent raw logits to a scorer whose input contract is log probabilities;
2. it maximized over every internal endpoint, while firmware scores suffixes
   ending at the latest frame.

`_student_scores` now applies log-softmax and uses the repository's accelerated
suffix forward-sum implementation with only the deployment endpoint. Future
training selects checkpoints by:

```text
(qualified,
 zero_false_accept_recall,
 -false_accepts_at_90_percent_recall,
 recall,
 separation)
```

Ties resolve to the earliest checkpoint. The exact-audio evaluator and corrected
cached-feature selector agree; float16 feature caching differs from the
deployment frontend by mean absolute error `0.0003148`, which is not the source
of the earlier discrepancy. Legacy tournament outputs were hash-verified and
rescored with `tools/rescore_kizz_distillation_checkpoints.py` before locked
evaluation.

## Results

The validation column is recall at a threshold accepting zero validation
negatives. `FP@90` is the number of validation negatives accepted at the first
threshold reaching 90% recall. Locked columns use the validation-selected
threshold or, when validation did not qualify, the validation zero-FP threshold
for diagnostic comparison only.

| Candidate | Transfer / architecture | Validation zero-FP recall | FP@90 | Aligned | Target channel | False wakes | Qualified |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| A | Temperature-2 posterior KD, control | 31.71% | 67 | 4/26 | 7/24 | 0/62 | No |
| B0 | CTC occupation KD, exact alignment, control | 41.46% | 6 | 3/26 | 6/24 | 0/62 | No |
| B1 | CTC occupation KD, +30 ms delay, control | 26.83% | 15 | 2/26 | 8/24 | 0/62 | No |
| C | Teacher representation KD, control | 46.34% | 11 | 5/26 | 7/24 | 0/62 | No |
| B+C | Occupation plus representation KD, control | 43.90% | 17 | 3/26 | 6/24 | 0/62 | No |
| D | Representation KD, temporal-residual | 46.34% | 13 | 11/26 | 9/24 | 0/62 | No |
| D2 | D plus sequence/negative boundaries | 73.17% | 2 | 13/26 | 13/24 | 0/62 | No |
| D3 | D2 plus tail separation | **78.05%** | **2** | **14/26** | **15/24** | **0/62** | **No** |
| E1 | Causal endpoint/listwise KD, temporal-residual | 78.05% | 2 | Not consumed | Not consumed | Not consumed | No |
| E2 | Causal endpoint/listwise KD, dilated causal memory | **82.93%** | **1** | **16/26** | **12/24** | **0/62** | **No** |

D4's best checkpoint was its initialization at 73.17% zero-FP recall and one
validation false accept at the 90% recall floor. Later checkpoints regressed,
so D4 did not displace D3 and was not allowed to consume the locked evaluation.

The immutable D3 weights are:

```text
/private/tmp/kizz-training/kizz-control-c1/tournament-d-stage3-tail-separation-v1/checkpoints/step-0001.weights.h5
SHA-256: 1abc9558f408d17f59a8c3ae21a7e9c5968a461d4fee1642906e4fb1d293a500
```

The complete local reports remain under
`/private/tmp/kizz-training/kizz-control-c1/tournament-*`. The private audio,
caches, and model weights are intentionally not committed.

## Causal-window distillation extension

E1 and E2 address a mismatch left by the original tournament: the teacher had
been reduced to one clip-level target even though firmware makes a causal
decision at every student endpoint. The extension caches the qualified
teacher's raw canonical fit, collision margin, eligibility, and decision score
at all 66 student endpoints. Training rotates among random, teacher-hard or
teacher/student-disagreement, and terminal endpoints. A standardized listwise
loss transfers ordering within each sampled batch.

The cache scorer is the Numba-compiled suffix forward-sum decoder. Tests compare
its accepted and raw fits, collision margins, eligibility, and decision scores
against the portable reference. A 3,473-example by 66-endpoint teacher cache
now builds in roughly 34 seconds on the M4 Pro, rather than spending minutes in
TensorFlow dispatch without completing.

E1 kept D3's 96,212-parameter temporal-residual architecture and fine-tuned it
for 1,000 steps. No trained checkpoint beat the initialization; the selector
retained step 1 at 78.05% zero-FP validation recall. Locked evidence was not
consumed for E1.

E2 uses five causal dilated depthwise blocks with dilations 1, 2, 4, 8, and 16,
a 1.93-second receptive field, 94,836 parameters, and exact float
streaming/non-streaming tail equivalence. Its selected step 2,800 reached 82.93%
zero-FP validation recall with one validation false accept at the 90% recall
floor. At the validation-selected zero-FP threshold it accepted 16/26 aligned
positives, 12/24 target-channel positives, and 0/62 false wakes. It therefore
failed three float gates and was not quantized or flashed.

The immutable E2 diagnostic weights are:

```text
/private/tmp/kizz-training/kizz-control-c1/distill-e2-dilated-listwise-v1/best.weights.h5
SHA-256: 6f2f0a8820373fc04c8175a46661405e63f10f539590d923f02def5e640aaf4a
```

A low-rate tail-separation continuation was stopped after 175 steps. It raised
validation false accepts at the 90% recall floor from 1 to 72 by step 100.
Clip-label ranking is invalid at randomly sampled causal prefixes: an early
prefix from a positive clip need not yet contain a wake. The trainer now rejects
causal-window transfer combined with clip-level ranking or tail-ranking losses.

## Decision and learning

D2/D3 show that the student can recover substantial teacher separation when
trained on the actual sequence decision boundary. A, B, C, and B+C do not show
that frame-level or representation transfer alone is sufficient. The +30 ms
ablation makes broad temporal misalignment less likely as the dominant failure.

E2 shows that longer causal memory and endpoint-specific teacher transfer can
improve the validation boundary and aligned recall, but target-channel recall
regressed. Quantization cannot repair an already-unqualified float model, and
flashing it would turn a controlled experiment into threshold tuning on
hardware. The next experiment must improve target-channel invariance using
training/validation evidence only; it must not fine-tune against locked evidence.
