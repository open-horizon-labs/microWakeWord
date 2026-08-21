# Data Sources for Training Wake Words

## Synthetic speech

[Piper sample generator](https://github.com/rhasspy/piper-sample-generator)
generates synthetic wake-word samples. We also use
[openWakeWord](https://github.com/dscripka/openWakeWord) to generate adversarial
phrases.

## Augmentation

We augment generated samples with background audio from:

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

Generate separate positive and negative samples for validation and testing; use
the same augmentations as training. Split FSD50K, FMA, and WHAM! 90/10 between
training and testing; keep both out of validation. Estimate false accepts per
hour during training with VOiCES and the
[DiPCo Dinner Party Corpus](https://www.amazon.science/publications/dipco-dinner-party-corpus)
(Community Data License Agreement – Permissive Version 1.0). Measure streaming
false accepts per hour with DiPCo after training.
