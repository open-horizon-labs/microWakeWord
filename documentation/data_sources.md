# Data Sources for Training Wake Words

## Synthetic speech

[The Open Horizon Labs Piper sample generator fork](https://github.com/open-horizon-labs/piper-sample-generator)
generates synthetic wake-word samples in speaker-independent cohorts. The
LibriTTS-R model supplies many speaker embeddings but no reliable age labels;
these samples do not establish child coverage. Recipes may add identity-disjoint,
age-labeled voices through the supplemental TTS catalog. The Kizz example uses
[ElevenLabs Voice Design](https://elevenlabs.io/docs/eleven-creative/voices/voice-design)
and records each resolved voice ID, declared age cohort, model, seed, and
settings. We also use
[openWakeWord](https://github.com/dscripka/openWakeWord) to generate adversarial
phrases.

## Augmentation

Training may use licensed background audio from:

- [ESC-50: Dataset for Environmental Sound Classification](https://github.com/karolpiczak/ESC-50) - (Creative Commons Attribution-NonCommercial 3.0 Unported.)
- [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475) - (Various Creative Commons Licenses.)
- [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840) - (Creative Commons Attribution 4.0 International License.)
- [WHAM!: Extending Speech Separation to Noisy Environments](https://arxiv.org/abs/1907.01160) - (Creative Commons Attribution-NonCommercial 4.0 International License.)

We reverberate samples with room impulse responses from
[BIRD: Big Impulse Response Dataset](https://arxiv.org/abs/2010.09930).

## Ambient negatives

### Ambient Speech

- [Voices Obscured in Complex Environmental Settings (VOICES) corpus](https://arxiv.org/abs/1804.05053) - (Creative Commons Attribution 4.0 License.)
- [Common Voice: A Massively-Multilingual Speech Corpus](https://arxiv.org/abs/1912.06670) - (Creative Commons License.)

### Ambient Background

- [FSD50K: An Open Dataset of Human-Labeled Sound Events](https://arxiv.org/abs/2010.00475)
- [FMA: A Dataset For Music Analysis](https://arxiv.org/abs/1612.01840) - reverberated with room impulse responses
- [WHAM!: Extending Speech Separation to Noisy Environments](https://arxiv.org/abs/1907.01160)

## Validation and test

Generate positive and negative samples from speaker IDs excluded from training.
Keep the primary validation and test wake clips clean; report a separate,
deterministic acoustic-stress cohort when adding backgrounds or room responses.
Assign background files to either training augmentation or a separately reported
stress set by source identity; never let the same recording serve both. Keep
both out of clean validation. Estimate false accepts per
hour during training with VOiCES and the
[DiPCo Dinner Party Corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus)
(Community Data License Agreement – Permissive Version 1.0). Measure streaming
false accepts per hour with DiPCo after training.
