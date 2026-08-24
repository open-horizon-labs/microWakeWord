# Kizz data-balance guardrails

The v32 regression was a corpus contract failure, not a tuning mystery. The
raw manifest was 34.02% positive by file count, with 99.75% of positives from
one Piper synthetic group and 99.87% of negatives from one Piper hard-negative
group. The 50 reviewed device false-wake anchors were only 0.13% of the
negative class. A uniform file sampler therefore had no protection against
source domination.

`data-balance-contract.yaml` is now a hard pre-training gate for both offline
teacher paths:

- every train, validation, and test split contains both classes;
- every example has a real path, source group, split, speaker ID, and session ID;
- paths occur once and speakers/sessions do not cross splits;
- the training class ratio is 40–60% positive;
- positives contain Piper synthetic, labeled TTS, and device replay, with no
  group above 50% and at least 10% each from labeled TTS and device replay;
- negatives contain Piper hard negatives, public speech, background/no-speech,
  and device false wakes, with no group above 50% and minimum coverage for the
  three non-Piper groups.

Run the gate directly:

```sh
uv run --python 3.12 --with pyyaml python \
  tools/validate_kizz_source_balance.py \
  --manifest /path/to/manifest.json \
  --contract recipes/kizz/data-balance-contract.yaml \
  --output /path/to/balance-report.json
```

Both `tools/train_kizz_teacher.py` (C) and
`tools/train_kizz_pretrained_teacher.py` (D) require this contract and abort
before model construction if it is not qualified. The report and its hashes
are recorded in the teacher artifact for provenance.

## Reset procedure

The current v32 generated files remain historical controls and are not
qualified for another teacher. Build a new manifest from explicitly named
source groups, cap Piper data at the contract limits, add device-channel
positive replay and public/background negatives, and rerun the gate. Only a
green report may feed feature materialization, teacher training, or
distillation.

Raw Piper archives and quarantined device evidence are retained. A curated
manifest is the reversible first reset: it excludes harmful or redundant
files without destroying the evidence needed to reproduce or audit the
decision. Permanent deletion requires a separate exact-path inventory after a
qualified replacement corpus and held-out evaluation exist.
