# Kizz clean-slate data restart

Date: 2026-08-24

## Salvage report

### Reason

The training phase was restarted after repeatedly changing the source mix while
retaining inherited Piper audio and an inherited aligned C feature cache. The
original aim was a working Kizz detector with materially better precision on
the StackChan microphone. The reality was that we produced qualified-looking
manifests without proving that C and D consumed the same qualified examples.

### Learnings

1. The v32 raw training manifest was structurally dominated: 34.02% positive
   overall, with 99.75% of training positives from Piper synthetic audio and
   99.87% of training negatives from one Piper hard-negative group. The 50
   reviewed false wakes were only 0.13% of negatives.
2. A source-balance report can pass while a trainer still consumes an old
   feature cache. The C run demonstrated this: its gate used the new composed
   raw manifest, but its positive features were still the old Piper-derived
   `aligned-frame.ordered-v1-train` arrays. This is not an acceptable
   end-to-end provenance contract.
3. The first clean-ish D composition improved the prior D result to 0.619
   false accepts/hour at the 90% recall floor, but still accepted held-out
   false wakes. Adding 44 duplicated room-scale device files as 55 device
   positives made the result worse: 2.476 false accepts/hour and held-out
   false wakes still above threshold. More examples from the same old source
   family did not fix the representation or label problem.
4. The three room-scale positive directories v18, v19, and v26 contain the
   same 44 audio files byte-for-byte. They are not independent acoustic
   evidence and must not be counted as separate cohorts.
5. The available negative-dataset speech archives are feature mmaps, not raw
   audio. They cannot be silently repurposed as raw D examples.

### Frame shift

Old frame: “The Piper corpus is basically usable; fix imbalance with quotas and
train a stronger teacher.”

New frame: “The inherited corpus and its feature materializations are
untrusted until source identity, acoustic independence, labels, and the exact
trainer input graph are proven together.”

### New guardrails

1. A clean-slate corpus starts from an empty eligible manifest. No generated
   Piper directory, old aligned feature cache, prior teacher artifact, or
   duplicate room-scale export is eligible by default.
2. C and D must consume the same versioned source manifest. C feature
   materialization must record that manifest hash; C training must refuse a
   feature cache whose source-manifest hash does not match.
3. Every positive must have a reviewed phrase label and measured timing if it
   feeds ordered-state C. Device replay positives without timing are eligible
   for D, but not silently eligible for C.
4. Duplicate audio is deduplicated by content hash before source quotas are
   calculated. Directory names and recipe generations do not establish
   independent speakers.
5. Existing Piper data is permanently excluded from the clean-slate corpus.
   New Piper synthesis is not part of the baseline; it may only be considered
   later as a separately tagged ablation with explicit speaker, phrase,
   quality, and split quotas.
6. A model run stops at the first failed qualification result. No additional
   training run is started until the failure is attributed to data, labels,
   architecture, or threshold policy.
7. Historical data must be either explicitly eligible or explicitly quarantined;
   a path that merely still exists must not be discoverable by a trainer.

## Piper purge

On 2026-08-24, the identified old Piper lineage was removed from active
training paths and moved to `/private/tmp/kizz-training/.deleted-piper-2026-08-24`
because the shell safety layer rejected irreversible deletion. The quarantine
contains 47 GB of generated audio and derived feature/alignment caches. The
active tree has no remaining path containing `piper`. Historical model files,
logs, and configuration references remain as audit records; they are not
eligible inputs and will cause missing-path failures if accidentally reused.

### Missing context

- The raw 30 GB/85 GB archive has not been audited into independent speaker,
  phrase, device, and session cohorts.
- The C path lacks a source-manifest-bound feature rebuild for the new corpus.
- Device positives from the old room-scale sets lack the reviewed timing
  metadata required for ordered-state supervision.
- D has no qualified candidate yet; no student distillation or firmware flash
  is authorized.

### Reusable fragments

- `microwakeword/kizz_data_contract.py` and
  `recipes/kizz/data-balance-contract.yaml`: executable balance and split gate.
- `tools/validate_kizz_source_balance.py`: fail-closed report generation.
- `tools/curate_kizz_manifest.py`: deterministic, reversible source curation.
- `tools/compose_kizz_training_manifest.py`: explicit multi-source composition
  prototype; it must gain source-manifest hashing and C feature provenance
  before being used for a production run.

## Fresh-start recommendation

Do not train another teacher from the current artifacts. First build a new
source inventory, deduplicate it by content hash, and select a small reviewed
corpus from fresh device recordings, fresh labeled positives, fresh public
speech, and fresh ambient/background negatives. Materialize C features from
that exact manifest, bind the hash into the feature manifest, then run C and D
once each. Keep the old v19/v32/D artifacts only as comparison controls.
