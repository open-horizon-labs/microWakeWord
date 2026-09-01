# Kizz teacher → student training attempt: salvage report

Date: 2026-08-25

## Short version

V19 recognized real wake phrases but also woke on ordinary speech. We stopped
using its training data, built a new mixed-source corpus, and tested two larger
offline models as teachers for a small firmware model. C, the model that used
the same audio frontend as the device, was the only teacher worth keeping. D,
the raw-waveform WavLM model, failed the false-wake test. Distilling C produced
a very selective 8-bit student, but it missed too many real wakes. The next
run should improve the data and teacher-to-student transfer, not simply make a
model larger.

## Salvage report

**Reason:** This phase replaced v19's training lineage after repeated attempts
to trade its good recall for acceptable precision. The work is being salvaged
before the next restart so the data, teacher gate, alignment, and evaluation
lessons remain explicit.

**Original aim:** Build a new Kizz wake model that keeps useful recall while
substantially reducing ordinary-speech, household, music, and ambient false
wakes on the ESP32-S3.

**Reality:** v19 remained a useful recall control but had unacceptable live
precision. Retaining its old synthetic lineage made later results difficult to
trust, so the active training path was rebuilt from a clean, source-bound
manifest. Two offline teachers were then compared before distillation.

## Learnings

1. **The old corpus was not a safe foundation.** We excluded the
   old Piper lineage, inherited feature caches, duplicate room-scale exports,
   and unproven source identities. v19 remains a comparison control, not an
   input source for the new recipe.
2. **Fresh synthesis needs acoustic and provider diversity, not just more
   clips.** The new manifest contains 4,706 examples: 1,848
   positives and 2,858 negatives. Positive sources include AssemblyAI,
   Deepgram, ElevenLabs, Kokoro, device-rendered anchors, and deterministic
   speech/music/noise overlays. Negatives include LibriSpeech speech and MUSAN
   music, noise, and speech. Fifteen household false wakes were held out.
3. **The file list is part of the model.** C and D used the same versioned
   manifest. Feature caches carried its hash. We removed duplicate audio before
   setting source quotas. Held-out false wakes could not enter training,
   distillation, or threshold selection.
4. **C and D answered different questions.** C kept the device's 16 kHz audio
   frontend and described the wake as 23 ordered sound states. D used
   `microsoft/wavlm-base-plus` directly on the waveform, giving us a genuinely
   different Mac-only model to test.
5. **C was the only teacher worth distilling.** On the shared fixed-window
   comparison, C reached 90.08% recall with 0 false accepts per hour and 0/15
   held-out wake accepts. D reached 90% recall only at 72.68 FAPH (false
   accepts per hour) and accepted 13/15 held-out
   false wakes. The full sliding-window D evaluations also failed.
6. **We must qualify a teacher before distillation.** The aligned C teacher
   was accepted only under an explicit experimental gate: 88.10% recall,
   0/560 observed negative accepts, 0.00 FAPH (false accepts per hour), and
   0/15 held-out false wakes
   on 1,456 seconds of negative exposure. This was enough to test transfer,
   not enough to call the teacher production-qualified.
7. **The two models did not see the same moment in time.** The teacher produced
   87 outputs from 260 frontend frames. The small streaming model produced the
   later 66-frame slice that matches its runtime delay. Copying output number 1
   to output number 1 was wrong. The corrected cache used teacher frames
   `[21:87]`.
8. **The student learned rejection better than recognition.** The aligned
   stateful 8-bit student reached 72.62% recall at 0/560 observed false accepts
   and 0/15 held-out accepts. At the first 90% recall point it had 2/560
   accepts, or 4.95 FAPH on only 1,456 seconds. This is a promising control,
   not a release result.
9. **The deployed model's output matters as much as its input.** The student
   emitted 23 state scores, while the old firmware expected one probability.
   Matching input shapes did not make the artifact compatible. We needed a
   decoder in firmware and an 8-bit evaluator that preserved model memory.
10. **Training loss is not detector quality.** D initially picked checkpoints
    by training loss and left dropout active in a supposedly frozen WavLM
    backbone. The corrected D path fixed both defects, added padding masks,
    balanced negative pressure, temporal rejection loss, and validation-based
    checkpoint selection, but still failed. The recipe/data boundary remained
    the limiting issue.

## Frame shifts

- **“v19 needs more hard negatives.”** → **“The v19 lineage and materialized
  features must be treated as untrusted until source identity and provenance
  are proven.”**
- **“A larger teacher will solve precision.”** → **“A teacher is useful only if
  it passes the same detector-level gate as the student.”**
- **“Matching frames by index is enough for distillation.”** → **“Teacher and
  student timelines must be aligned in the deployed streaming geometry.”**
- **“A good fixed-window score is a qualified wake model.”** → **“Fixed-window
  screening, sliding-window false-activation exposure, held-out anchors,
  stateful INT8 scoring, and physical tests are separate gates.”**

## New guardrails

1. Start every new Kizz recipe from an explicit eligible manifest. Existing
   Piper audio, caches, checkpoints, and duplicate recordings are excluded
   unless a named ablation re-admits them.
2. Require one source-manifest hash through raw data, feature materialization,
   teacher qualification, distillation cache, student conversion, and reports.
3. Do not distill or flash an unqualified teacher. AUC, training loss, one
   encouraging replay, or known-anchor rejection cannot substitute for the
   qualification report.
4. Select teacher checkpoints on held-out detector metrics, never minimum
   training loss.
5. Preserve distinct negative classes: background/silence, ordinary connected
   speech, music/noise, close phrase collisions, and reviewed device false
   wakes. Do not collapse them into a misleading single target without an
   explicit decoder objective.
6. Evaluate the exact streaming state contract after quantization. Reset and
   carry-state modes must be named; threshold selection must happen before
   untouched test and held-out anchor evaluation.
7. Keep all live false-wake evidence quarantined until a human promotes it.
   Captures are evidence, not automatic corpus truth.
8. Stop at the first failed gate and write the diagnosis before starting
   another capacity, checkpoint, or distillation sweep.

## Missing context

- The clean-slate-v2 corpus still lacks enough natural human positives and
  long-form domestic/TV exposure for a production claim.
- The aligned teacher's 0.00 FAPH result used only 1,456 seconds of negatives;
  it is statistically fragile.
- The student misses were concentrated in noisy overlays and shorter positives,
  so the next pass needs to determine whether the loss is alignment, duration,
  channel robustness, or student capacity.
- Physical qualification must measure the exact flashed artifact, not infer it
  from desktop or fixed-window reports.

## Reusable fragments

- `recipes/kizz/data-balance-contract-clean-slate-v2.yaml`
- `tools/build_kizz_clean_slate_v1.py`
- `tools/build_kizz_teacher_features_v1.py`
- `tools/qualify_kizz_teacher_npy.py`
- `tools/cache_kizz_teacher_logits.py`
- `tools/distill_kizz_student.py`
- `tools/convert_distilled_student.py`
- `tools/score_ordered_state_streams.py`
- `recipes/kizz/CLEAN_SLATE_V2_C_D_RESULTS.md`
- `recipes/kizz/TEACHER_DISTILLATION_ALIGNED_V1.md`

## Fresh-start recommendation

Keep v19 only as the recall/precision comparison control. Keep C as the
surviving teacher architecture and D as a rejected representation experiment.
For the next pass, expand held-out household and long-form ambient exposure,
add natural human positive sessions, preserve the aligned teacher/student
geometry, and require a larger sliding-window qualification set before any
firmware decision. Do not restart by adding capacity or resurrecting old Piper
data.
