# Kizz wake-word training reference

Kizz is the worked case study for training a custom microWakeWord model in this
fork. The wake phrase is useful precisely because it is difficult: **HiPhi** is
an invented spelling, speakers pronounce it several ways, **Kizz** is close to
`kids`, `kiss`, `quiz`, `his`, and `this`, and the complete phrase must work for
adults and children through a small embedded microphone in an occupied room.

This document preserves how the corpus and experiments were built. It includes
methods that fed v19, exploratory sources that did not, and controls added after
failed runs. The generic commands live in [Usage](../../documentation/USAGE.md);
the result ledger lives in [Experiments](EXPERIMENTS.md).

## Current status

V19 checkpoint 100, used at cutoff `0.70` with a one-frame sliding window, is
the best live baseline from this effort. It is not a qualified release. Its
quantized TensorFlow Lite artifact has SHA-256:

```text
76250d0cef49f893df4724ea6cce0e87b8a8d0d63cf10fbe23c0e624298871ff
```

The artifact survives because it worked better on physical Kizz than later
candidates. It still accepted too many device hard negatives and missed fresh
speakers. Preserve it as the control while improving the evidence and training
method.

## Salvage report

**Salvaged:** 2026-08-23  
**Reason:** the first recipe documentation compressed the selected v19 settings
and omitted most of the corpus-building and diagnostic work.  
**Original aim:** train a local **HiPhi Kizz** detector that responds naturally
for different household speakers without waking on nearby speech.

### What changed our understanding

1. A large synthetic corpus can report strong holdout metrics while failing on
   the target microphone. Voice identity, synthesis family, room channel, and
   capture session are separate domains.
2. Splitting by phrase or WAV is not independent evaluation. A source voice may
   appear in only one of train, validation, or test.
3. Piper supplies useful breadth but no reliable age labels. Designed adult and
   teen voices supplied explicit age cohorts and a second synthesis family.
4. Clean TTS and the same TTS played through Kizz are different evidence. The
   latter includes the laptop speaker, room, distance, microphone, gain, and
   frontend.
5. Detector misses must be captured. Otherwise the training corpus contains
   successful wakes and systematically loses the failures it needs most.
6. A cutoff can trade recall for false wakes only when positive and negative
   scores are separated. It cannot recover positives near zero while rejecting
   negatives above `.70`.
7. Offline aggregate scores hid failures for the current speaker. A fresh
   speaker/session challenge must run before flashing.
8. Runtime streaming state matters. Isolated clips and continuous
   carry-until-detection replay answer different questions; physical operation
   remains the acceptance test.
9. Training share and learning pressure are different. Error penalties and
   class weights can make a minority source dominate gradients even when its
   sample quota looks balanced.
10. More negative data is not automatically better. V29–v31 showed that extra
    separation pressure could destroy recall or move both classes together.

### Frame shifts

| Earlier frame | Replacement frame | Evidence |
| --- | --- | --- |
| Generate enough TTS and tune a cutoff | Build independent acoustic domains and demand class separation | Natural positives scored near zero while some room speech exceeded `.70`. |
| Split files or phrases | Split immutable speaker identities and recording sessions | One apparent held-out device positive came from the training speaker under a second ID. |
| Synthetic speech is a stand-in for device speech | Clean synthesis and device-channel replay are separate sources | A model that accepted 17/17 direct device-training clips accepted only 1/8 when those phrases were replayed into Kizz. |
| The largest corpus should carry the most weight | Declare per-batch source and pronunciation shares | Tens of thousands of Piper clips otherwise overwhelmed designed voices and scarce device evidence. |
| A failed voice turn means the wake model failed | Observe wake detection, state transitions, command capture, transport, STT, and command execution separately | Several turns woke correctly but later produced “speech recognition unavailable” because command audio did not reach the server. |
| Training requires the physical product online | Simulated enrollment is a first-class fixture; hardware is the final acoustic gate | The enrollment protocol can prove that misses are retained without Kizz attached. |
| The trainer is part of UHC | Enrollment has its own configured endpoint | Training and production voice have different lifecycle and deployment needs. |
| Kizz-specific training code | Device-profile evidence over a shared framework | Future microphone products should reuse the corpus contract and test one shared model first. |

### Missing context that should have been gathered earlier

- independent natural adult and child speakers assigned to immutable splits;
- multiple sessions and normal-distance recordings for each speaker;
- an inventory of every temporary generator, fixture, model, manifest, and
  report before summarizing the recipe;
- a physical-test record binding model hash, threshold, sliding window,
  microphone gain, firmware, and active server instance;
- separate telemetry for microphone input, wake probability, detector state,
  captured command bytes, transmitted bytes, STT result, and command result;
- the provenance and availability of the v18a weights used to initialize v19.

### Ownership and coordination breakdowns

- Temporary scripts and generated catalogs contained much of the experiment
  knowledge, while the repository docs described only the polished pipeline.
  Anything needed to repeat a successful experiment must move into a checked-in
  tool, manifest, or case-study record.
- Wake-word, post-wake capture, STT, and command execution were debugged in one
  live loop. A success or failure at one layer was too easily attributed to
  another. Each layer needs its own result and recovery telemetry.
- Model, cutoff, gain, and server process changed during physical tests without
  one immutable run manifest. Future comparisons must print and persist the
  active configuration before the first utterance.
- The first documentation pass declared the recipe covered after recording the
  winning settings. The reviewer boundary should include an inventory check:
  every source and fixture in the training workspace is either documented,
  promoted, or recorded as discarded.

### Reusable fragments

The speaker-split generators, catalog format, quality mask, background ledger,
device enrollment contract, phrase alignment, stratified sampler, ablation
audits, cutoff selector, and dual-state evaluator are the reusable result. The
[tool ledger](#tools-and-what-each-preserves) maps each technique to code.

### Guardrails carried forward

- Preserve clean WAVs; derive masks, aligned clips, acoustic variants, and
  features reproducibly.
- Bind every generated or derived corpus to its recipe, source model, voice
  identity, settings, split, and hashes.
- Assign a human or synthetic voice identity to one split before generation.
- Never treat Piper speaker IDs as adult or child labels.
- Cap large homogeneous sources by speaker and phrase instead of feeding every
  available clip into each run.
- Keep sampling groups single-class and report realized batch quotas plus
  weighted pressure.
- Treat validation as the only cutoff-selection source. Open test after the
  candidate and cutoff are frozen.
- Retain live misses, false wakes, and ambient audio as labeled evidence.
- Keep Kizz-specific labels and results in this recipe; keep enrollment,
  provenance, balancing, and evaluation machinery device-neutral.
- Compare one shared model across microphone profiles before creating
  device-specific models.
- Flash only the quantized artifact that passed the declared offline gates, then
  record physical results against its hash.

## Corpus design

### Positive class

The product intent is one full-phrase class, not a bare `Kizz` detector. The
recipe uses alternative spellings to induce plausible acoustic readings:

- `Hi-Fi Kizz`, `High-Fi Kizz`, `Hi Phi Kizz`;
- `Hee Fee Kizz`, `High Fee Kizz`;
- `Hee Fye Kizz`, `High Fye Kizz`;
- `Hippy Kizz`, `Hiffy Kizz`;
- kids-like endings observed in real speech, such as `Hi-Fi Kids` and
  `hiffy kids`.

V19 also treated `High Five Kizz` as positive. Later listening and phonetic
inspection found that the `/v/` in “five” is an extra consonant rather than a
normal Hi-Fi realization. The current [`corpus.yaml`](corpus.yaml) moves it to
hard negatives. That label change means the current recipe is a successor to
v19, not a byte-for-byte reconstruction.

### Negative classes

The negative set was designed in layers:

| Layer | Examples | Failure it challenges |
| --- | --- | --- |
| Suffix-only neighbors | `Kizz`, `kids`, `kiss`, `quiz`, `fizz`, `his`, `this` | Learning only the final syllable |
| Prefix-only speech | `Hi-Fi`, `Hi Phi` | Waking before the full phrase finishes |
| Valid prefix, wrong suffix | `Hi-Fi Kiss`, `Hippy Kiss`, `Hee Fye Kiss` | Treating any similar final consonant as Kizz |
| Wrong prefix, valid suffix | `Wi-Fi Kizz`, `Sci-Fi Kizz`, `Happy Kizz`, `Hey Kizz` | Learning only the suffix |
| Connected-speech collisions | “My five kids are waiting by the door” | False wakes spanning word boundaries in ordinary speech |
| Downstream commands without a wake | “Play Prince in the front family room” | Confusing likely command vocabulary with the wake phrase |
| General household speech | dinner conversation, TV, podcasts | False accepts outside hand-designed phonetic foils |
| Non-speech | room tone, appliances, outdoor noise, music | Acoustic false accepts |

The short phrase list lives in [`corpus.yaml`](corpus.yaml). Sentences mined from
observed false wakes and related sound neighborhoods live in
[`counterexamples.yaml`](counterexamples.yaml). Plausible readings withheld from
training live in [`probes.yaml`](probes.yaml).

## Every speech and audio source we tried

| Source | How it was made | Role | Did it feed v19? |
| --- | --- | --- | --- |
| Piper LibriTTS-R generator | Mixed pairs of speaker embeddings while varying pace and synthesis noise | Broad pronunciation and voice coverage | Yes, with per-speaker/phrase caps |
| ElevenLabs Voice Design + TTS | Designed eight persistent adult/teen voices, then rendered each phrase with seeded voice-setting variants | Independent synthesis family and labeled age cohorts | Yes |
| Kokoro | Local pilot using four stock voices | Quick second-family listening and collision probe | No; exploratory and not provenance-complete |
| macOS `say` | Six system voices at chosen speaking rates | Early physical enrollment and microphone fixture | No; pipeline diagnostic |
| Natural human speech | Directed and live wake attempts recorded by Kizz | Real pronunciation and microphone evidence | Yes, training split only |
| ElevenLabs played through Kizz | Clean generated WAVs played from the laptop at recorded volume, delay, and distance | Device/room-channel adaptation | Yes |
| Ambient Kizz captures | Quiet room, conversation, and later unattended room audio | Background training and false-wake evaluation | Yes, according to split and experiment |
| Upstream negative feature archives | Speech, dinner-party speech, and no-speech archives | Broad negative coverage and checkpoint selection | Yes |
| ESC-50 | Indoor/outdoor categories split by official fold | Training augmentation and independent stress audio | Yes for augmentation; fold 5 stayed stress-only |
| MIT room impulse responses | Convolution during feature generation | Simulated room acoustics | Yes |
| Deepgram Nova-3 timestamps | Word/utterance timing over captured WAVs | Proposed phrase spans for review | Metadata only; transcript text was not label truth |

The important distinction is between a **clean source**, a **derived acoustic
variant**, and a **device recording**. A clean ElevenLabs WAV and a recording of
that WAV through Kizz may share a synthetic speaker identity, but the second
contains the physical channel and must carry its own session and conditions.

## Piper generation

The Open Horizon Labs
[`piper-sample-generator`](https://github.com/open-horizon-labs/piper-sample-generator)
fork was pinned at commit `35d2f2d`. It adds speaker-range isolation, seeded
generation, and per-WAV provenance to the upstream generator.

### Speaker construction

The LibriTTS-R generator does not use one fixed voice per WAV. It selects two
base speaker embeddings and blends them with spherical interpolation. V19 used
the first 700 base speakers because later LibriTTS-R identities were more prone
to artifacts:

| Split | Half-open base-speaker range | Planned sample share |
| --- | ---: | ---: |
| Train | `0..560` | 80% |
| Validation | `560..630` | 10% |
| Test | `630..700` | 10% |

The ranges are disjoint before synthesis. A generated WAV records both base
speaker IDs, interpolation weight, phrase, and synthesis settings. These IDs
carry no trustworthy age label.

### Variation grid

The generator cycled through:

| Parameter | Values |
| --- | --- |
| Speaker interpolation | `0.2`, `0.4`, `0.6`, `0.8` |
| Length scale | `0.72`, `0.85`, `1.0`, `1.15`, `1.32` |
| Synthesis noise | `0.55`, `0.667`, `0.8`, `0.95`, `1.15` |
| Duration noise | `0.65`, `0.8`, `0.95` |
| Seed | Recipe seed `231`, advanced deterministically by phrase and split |

The v19 generation manifest contained 67,150 Piper WAVs: 38,300 positives and
28,850 hard negatives. Generation volume did not determine training volume.
Feature building capped Piper at one clip per participating speaker and phrase,
then the stratified sampler assigned the Piper groups their batch shares.

This is why keeping the large corpus is still useful: it supplies a broad pool
from which deterministic, diverse subsets can be drawn. Feeding the whole pool
on every run would let one synthetic family drown out ElevenLabs and device
evidence.

## ElevenLabs voice design and phrase rendering

ElevenLabs served two different jobs:

1. **Voice Design** created persistent voice identities from written acoustic
   descriptions.
2. **Text to Speech** rendered wake phrases and hard negatives with those fixed
   identities.

The distinction matters. Regenerating a preview does not create an independent
speaker split; the persistent provider voice ID defines identity.

### Designed voices

[`elevenlabs-voice-designs.yaml`](elevenlabs-voice-designs.yaml) declares eight
roles:

| Name | Split | Cohort | Intended voice |
| --- | --- | --- | --- |
| `kizz-adult-train-a` | Train | Adult | Warm conversational American woman, medium pitch and pace |
| `kizz-adult-train-b` | Train | Adult | Relaxed conversational American man, medium-low pitch |
| `kizz-child-train-a` | Train | Child | Conversational American teenage boy |
| `kizz-child-train-b` | Train | Child | Conversational American teenage girl |
| `kizz-adult-validation` | Validation | Adult | Conversational Canadian woman |
| `kizz-child-validation` | Validation | Child | Casual, slightly quick American teenage boy |
| `kizz-adult-test` | Test | Adult | Relaxed British man with a light accent |
| `kizz-child-test` | Test | Child | Casual, slightly quick American teenage girl |

The provider was asked for natural household speech, not announcer or character
voices. Voice Design used `eleven_ttv_v3`, guidance `5`, seeds `231` through
`238`, and a paragraph of ordinary conversational preview text. Enhancement was
disabled. Every returned preview was saved for listening; preview zero was
selected for each catalog entry and promoted to a persistent voice.

The historical resolved catalog used more specific prompts for its first two
voices than the later checked-in template. Adult train A was “a natural American
English woman in her thirties” speaking casually to a nearby household device,
warm but unperformed, with clear consonants and no narration style. Adult train
B was “a natural American English man in his forties” speaking to a device
across a room, relaxed and clear without sounding rehearsed or theatrical. The
remaining six resolved descriptions match the roles summarized above. Preserve
the resolved catalog with a run; the design template alone cannot recreate the
same provider identities.

Phrase rendering used `eleven_multilingual_v2`, English, mono signed-16 PCM at
16 kHz. Per-WAV seeds were derived from the recipe seed, phrase index, voice
index, and sample index. Three render settings rotated across samples:

| Variant | Stability | Similarity boost | Speed |
| --- | ---: | ---: | ---: |
| Relaxed/slow | `0.35` | `0.75` | `0.9` |
| Neutral | `0.50` | `0.80` | `1.0` |
| Stable/fast | `0.70` | `0.85` | `1.1` |

For v19, each training voice rendered 12 examples per positive phrase and three
per hard-negative phrase. Each validation and test voice rendered four examples
per positive phrase and one per hard negative. With the historical 18-positive,
28-negative phrase set, that produced 1,600 ElevenLabs WAVs:

| Split and cohort | Positives | Hard negatives |
| --- | ---: | ---: |
| Train adults | 432 | 168 |
| Train children/teens | 432 | 168 |
| Validation adults | 72 | 28 |
| Validation children/teens | 72 | 28 |
| Test adults | 72 | 28 |
| Test children/teens | 72 | 28 |

The generated voice catalog records the provider, TTS model, voice IDs, split,
age cohort, design description, and seed. The generation manifest records the
voice identity and settings for each WAV. Provider voice IDs and generated audio
are workspace artifacts rather than repository content.

### Designed hard negatives

The first designed-voice pass concentrated on positives. A later pass rendered
all 28 short confusable phrases in the same adult and teen voice families.
Without this, “provider” and “class” were correlated: ElevenLabs often meant
positive while Piper supplied most confusables. Designed negatives removed that
shortcut. V19 gave these their own sampling group rather than hiding them inside
one undifferentiated synthetic pool.

### Long counterexample sentences

Observed room false wakes were expanded into connected sentences: kids/five,
Wi-Fi, “if he is,” quiz/kiss, ordinary commands, and unrelated household speech.
These were rendered as a separate ElevenLabs corpus so a later experiment could
add them without changing the original phrase sources. That work informed
v20–v31; it did not create the v19 artifact.

## Other synthesis pilots

### Kokoro

A local Kokoro pilot rendered the wake phrase with `af_heart`, `am_fenrir`,
`bf_emma`, and `bm_fable`. The same voices rendered four collision probes:
`Hee Fee Kiss`, `Hey Kizz`, `Hi-Fi Kiss`, and `High Fee Kiss`.

It was useful for hearing how another model family realized the name, but the
pilot had no repository tool, split-bound identity manifest, or full phrase
matrix. It was therefore excluded from v19 and cannot count as independent
evaluation. The reusable lesson is to add a synthesis family through the same
catalog and provenance contract, not as loose WAVs.

### macOS system voices

An early physical fixture used `say` to render 16 positive and negative
utterances with six voices:

| Voice | Assigned split | Rate |
| --- | --- | ---: |
| Samantha | Train | 165 wpm |
| Daniel | Train | 155 wpm |
| Reed, English (US) | Train | 160 wpm |
| Flo, English (UK) | Train | 155 wpm |
| Karen | Validation | 170 wpm |
| Moira | Test | 160 wpm |

Each AIFF was played with `afplay` into Kizz and captured for three seconds.
This exercised the enrollment path and exposed microphone/transport problems.
It was not used as v19 training or qualification evidence: system voices were a
small, correlated fixture family, and the early corpus contract was less strict.

## Recording synthetic speech through Kizz

Clean synthesis does not model the target device. We therefore replayed
split-reserved ElevenLabs training voices from the laptop speaker into Kizz and
recorded the microphone output through the enrollment service.

### First playback fixture

- four ElevenLabs training identities: two adult and two teen;
- three positives: `Hi-Fi Kizz`, `Hi Phi Kizz`, `Hiffy Kizz`;
- three hard negatives: `Hi-Fi Kiss`, `high five kids`, `Wi-Fi Kizz`;
- laptop system volume 50%;
- approximately 15 cm speaker-to-device distance;
- 2.5-second Kizz capture window;
- the middle clean source WAV from each phrase/voice directory.

### Expanded voice-setting fixture

A second pass synthesized new slow and fast examples with the same four
training identities, played them at 50%, and captured 2.6 seconds. This added two
more examples for each positive phrase and one more for each selected negative.
It recorded the source WAV hash, voice identity, age cohort, render settings,
volume, distance, and session.

### V19 multi-condition fixture

The final v19 adaptation pass replayed one existing clean source per phrase and
voice under two physical conditions:

| Condition | Laptop volume | Playback delay inside capture | Purpose |
| --- | ---: | ---: | --- |
| `quiet-early` | 42% | 180 ms | Quieter speech near the start of the feature window |
| `loud-late` | 60% | 700 ms | Louder speech at a different window phase |

Both used a 2-second recording at roughly 15 cm. The clean source was retained;
the Kizz recording became a separate `synthetic_playback` capture. This pass was
the practical bridge between designed voices and the Kizz microphone channel.

Playback recordings are valuable training evidence, but not human validation.
The source speaker identity and split still apply, and laptop playback cannot
substitute for a person speaking at normal household distance.

## Human and ambient captures

The standalone enrollment service asks a named device to record a bounded
attempt whether or not the current detector fires. Each manifest entry includes
the intended truth, speaker, session, device profile, firmware, conditions,
provisional `detected` result, WAV hash, and immutable split.

Human collection included:

- individual natural wake attempts;
- back-to-back prompted sessions;
- explicit live misses after a wake phrase failed to trigger;
- short collision phrases such as `Kizz`, `kiss`, `fizz`, `Hey Kizz`, `Hi-Fi`,
  `his`, and `this`;
- ordinary conversation captured as a hard negative;
- attempts at different distances and microphone settings.

The first training corpus overrepresented one adult speaker. That evidence was
still useful for adapting to their live pronunciation, but it could not measure
speaker generalization. Later adult and child test replays exposed that gap.

The v19 provenance froze the device evidence at:

| Corpus role | Count |
| --- | ---: |
| Training positives | 77 |
| Training hard negatives | 56 |
| Training ambient negatives | 2 |
| Held-out positives | 14 |
| Held-out hard negatives | 10 |

Of the training phrase captures, 108 came from split-reserved ElevenLabs voices
played through Kizz and 25 came from one human speaker. That imbalance is one
reason v19 remains a control rather than a qualified model.

Ambient collection included quiet room tone and unattended household audio.
These captures serve two jobs: training-only background material and held-out
false-wake measurement. A clip cannot serve both jobs.

## Finding the phrase inside a capture

A two-second feature window can lose a wake phrase when the recording is longer
or the speech begins late. Three alignment methods were used:

1. **Human review:** record `start_ms` and `end_ms` for the intended phrase.
2. **Deepgram Nova-3 timestamps:** use utterance/word timing as a proposed span,
   then review it. Transcription was often wrong—one `Hi-Fi Kizz` became “I
   fight his”—while the time interval was still useful.
3. **Normalized cross-correlation:** for known speaker-playback fixtures, locate
   the voiced part of the clean source WAV inside the Kizz recording and record
   the correlation score.

Feature building keeps the measured phrase plus 250 ms of context. The source
recording and hash remain unchanged. Random truncation is the fallback for long
captures without reliable spans; start/end truncation is allowed only when the
capture protocol guarantees edge alignment.

## Source screening and acoustic augmentation

### Quality mask

Eight reviewed human wake spans initially established a 560–880 ms observed
range. Adding a 25% margin produced a 483–1,100 ms accepted positive span. The
v19 source mask also required:

- RMS above `-50 dBFS`;
- clipped-sample fraction no greater than `0.001`;
- source duration no greater than 1,700 ms, leaving placement room inside the
  two-second training window;
- non-silent audio.

The v19 combined manifest contained 68,750 clean WAVs. The mask accepted 58,175
and rejected 10,575. Positive timing limits did not apply to hard negatives.
The mask was a source-quality screen, not a claim that accepted TTS sounded
human or matched Kizz.

### Background corpus

[`prepare_background_corpus.py`](../../tools/prepare_background_corpus.py)
partitioned ESC-50 by environment and official fold:

| Environment | Training clips | Stress-only clips |
| --- | ---: | ---: |
| Indoor | 642 | 160 |
| Outdoor | 640 | 160 |

Folds 1–4 supplied augmentation; fold 5 remained stress evidence. Indoor
categories included appliance, human-noise, door, clock, and household sounds.
Outdoor categories included traffic, weather, animals, machinery, water, and
fire. Training-only Kizz room tone was added to the indoor pool with its device
profile and source hash.

### Feature-time transformations

V19's `normal_room` feature build applied transformations only to training:

| Transformation | Probability |
| --- | ---: |
| Gain, from `-35` to `0 dB` | `1.00` |
| Background mix | `0.80` |
| Room impulse response | `0.60` |
| Colored noise | `0.35` |
| Parametric EQ | `0.15` |
| Gain transition | `0.15` |
| Tanh distortion | `0.10` |
| Pitch shift | `0.10` |
| Band-stop filter | `0.10` |

Speech-to-background ratio ranged from 3 to 20 dB, and temporal jitter ranged
from 150 to 300 ms. Validation and test features stayed clean. Later tooling
added a `challenging` profile at -6 to 6 dB SNR so noise may exceed speech; that
profile was not part of v19 and should be tested as a declared ablation rather
than silently replacing the baseline.

## How v19 sampled the evidence

The corpus directories were highly unequal, so v19 used stratified batches.
The planned share controlled the number of examples seen; the weighted-pressure
share also included per-source error penalties.

| Group | Planned share | Realized sample share | Weighted-pressure share |
| --- | ---: | ---: | ---: |
| Piper positives | 15% | 15.625% | 11.441% |
| Piper hard negatives | 15% | 15.625% | 22.883% |
| Designed positives | 25% | 25.000% | 18.306% |
| Designed negatives | 10% | 9.375% | 13.730% |
| Kizz microphone positives | 15% | 14.063% | 10.297% |
| Targeted negatives | 20% | 20.313% | 23.343% |

Across 400 steps with batches of 64, the ledger recorded 25,600 samples. The
realized sample mix was 54.688% positive and 45.313% negative. Because Piper and
designed hard negatives used a `2.0` error penalty and Kizz hard negatives also
received extra within-group weight, negatives represented about 59.96% of
weighted pressure. This is why reports must show both numbers.

Within the Piper groups, phrase-separated feature sources and the
one-clip-per-speaker/phrase cap prevented the raw 67,150-file corpus from setting
the distribution. Adult and teen ElevenLabs positives had separate feature
sources. Designed negatives also had adult and teen sources. Targeted negatives
combined Kizz hard negatives, ordinary speech, dinner-party speech, and
background-only features under declared within-group weights.

## V19 training and export

V19 used framework commit `16628e90338a197b8a53ddce3af4af6ff24079ad`
and continued from v18a weights with:

- deterministic seed `231`;
- 200 steps at learning rate `5e-6`, then 200 at `1e-6`;
- batch size 64;
- binary focal cross-entropy, gamma `2.0`, without automatic class balancing;
- frozen batch-normalization statistics;
- one time mask of up to four frames and one frequency mask of up to three bins;
- evaluation every 100 steps.

The MixedNet architecture used:

- stride `3`;
- 48 first-convolution filters with kernel `5`;
- four pointwise blocks with 96 filters;
- MixConv kernels `[5]`, `[7,11]`, `[9,15]`, and `[23]`.

Checkpoint 100 retained the strongest replay recall. Checkpoint 200 reduced
some confusable accepts but missed more positives; the later checkpoint was
worse again. Exporting “the final model” would therefore have selected the wrong
artifact. Every retained checkpoint must be converted and challenged
independently.

## Evaluation ladder

Use the following gates in order. Each catches a different form of leakage or
non-transfer.

1. **Training diagnostics:** loss and checkpoint metrics. Useful for detecting
   collapse, never for qualification.
2. **Voice-held-out validation:** choose the cutoff while reporting every
   pronunciation, provider, voice, and age cohort.
3. **Frozen synthetic test:** measure independent identities after model and
   cutoff selection.
4. **Unseen pronunciation probes:** test readings absent from training. Keep
   them outside future training or replace them with new probes.
5. **Device test, reset per capture:** isolate acoustic classification.
6. **Device test, carry until detection:** approximate streaming accumulation
   and reset behavior.
7. **Fresh current-room speaker challenge:** reject candidates that improve old
   aggregate captures but fail new voices or sessions.
8. **Physical speaker replay:** play held-out WAVs through the actual room and
   inspect wake probability, detection, and re-arming.
9. **Natural human acceptance:** test adults and children at ordinary distance,
   phrasing, and volume.
10. **Long ambient guard:** measure false accepts during conversation, music,
    TV, podcasts, quiet, and device movement.

The physical test must record the model hash, cutoff, sliding window, firmware,
microphone profile, gain, distance, playback level when applicable, detections,
peak probabilities, and whether the detector re-armed.

## Separate detector failures from downstream failures

A spoken command crosses several systems. Label the failed stage before changing
the wake model:

| Observation | Likely layer to inspect |
| --- | --- |
| Microphone peak stays near zero | Mic enablement, gain, frontend, power, or hardware |
| Mic sees speech but wake probability stays low | Model recall, preprocessing mismatch, or acoustic domain |
| Wake probability is close to the cutoff | Cutoff trade-off; inspect false-negative and false-positive distributions together |
| Wake fires but the device leaves listening early | Voice state machine, post-wake VAD, capture duration, or timeout |
| Wake fires and listening completes, but STT receives little or no audio | Command buffering and network transport |
| Transcript appears but no action occurs | LLM/tool routing or command execution |
| Detector works once after boot but not again | Re-arming, streaming-state reset, task health, or connection state |

During this effort, “speech recognition unavailable” followed valid wake
detections because only a fraction of the multi-second command audio reached the
server. That was not evidence for lower mic gain or a worse wake model. Likewise,
live tests sometimes became unreliable after body movement; the cause was not
isolated. Preserve the observation and capture mic, state, and probability
telemetry rather than folding it into model recall.

## Experiment families and what they contributed

| Phase | Main question | Durable result |
| --- | --- | --- |
| Early combined and fee-family models | Can synthetic spellings separate the wake phrase from nearby words? | Bare-word and conjunction shortcuts must be explicit negatives; aggregate synthetic recall is misleading. |
| Device adaptation and capacity search | Can a broad model learn the first Kizz recordings? | Capture misses, align phrases, and preserve the target frontend; more capacity alone did not solve transfer. |
| Quality mask and independent splits | Were poor sources and leaked speakers inflating results? | Human-referenced screening and voice-identity splits became required. |
| V10–v12 stratification | Does deliberate source balance beat directory-size sampling? | Per-batch quotas, source caps, and designed negatives became reusable framework features. |
| V13–v19 | Which loss, capacity, device playback, labels, and fine-tuning choices improve live recall? | Focal loss, frozen BN, multi-condition Kizz playback, and checkpoint 100 produced the v19 control. |
| V20–v26 counterexample adaptation | Can mined room false wakes be patched into v19? | Invalid checkpoint selection and over-concentrated counterexamples can create flattering but unusable models. |
| V27–v28 diversified separation | Can a broader rebuild reduce hard-negative accepts? | Aggregate device metrics improved, but a fresh-speaker gate rejected v28. |
| V29–v31 controlled continuation | Can gentle negative pressure repair v19? | Unfrozen BN collapsed recall; frozen BN restored positives but also false wakes; v31 still lacked separation. |

Detailed measurements and rejection reasons are in [Experiments](EXPERIMENTS.md).

## Tools and what each preserves

| Tool | Purpose | Durable output |
| --- | --- | --- |
| [`generate_recipe_samples.py`](../../tools/generate_recipe_samples.py) | Generate Piper phrases with disjoint speaker ranges and resumable reuse | Generation manifest and per-WAV synthesis metadata |
| [`design_elevenlabs_voice_catalog.py`](../../tools/design_elevenlabs_voice_catalog.py) | Design adult/teen voice identities and save previews | Split-bound voice catalog |
| [`add_labeled_voice_samples.py`](../../tools/add_labeled_voice_samples.py) | Render seeded ElevenLabs positive and negative variants | PCM WAVs plus voice/settings provenance |
| [`apply_phrase_spans.py`](../../tools/apply_phrase_spans.py) | Attach reviewed phrase timing to immutable captures | Validated device manifest update |
| [`build_synthetic_quality_mask.py`](../../tools/build_synthetic_quality_mask.py) | Compare generated sources with human timing and objective audio limits | Hash-bound accepted/rejected source mask |
| [`prepare_background_corpus.py`](../../tools/prepare_background_corpus.py) | Separate training backgrounds from stress evidence | Licensed, hashed background manifest |
| [`build_recipe_features.py`](../../tools/build_recipe_features.py) | Select speakers/phrases/providers, augment training, and extract `micro_speech` features | Feature build manifest |
| [`run_enrollment_service.py`](../../tools/run_enrollment_service.py) | Direct a configured microphone to capture every attempt | Device WAVs and corpus entries, including misses |
| [`simulate_enrollment_device.py`](../../tools/simulate_enrollment_device.py) | Test enrollment without hardware | End-to-end protocol evidence |
| [`validate_device_corpus.py`](../../tools/validate_device_corpus.py) | Enforce format, hashes, profiles, identity, and split integrity | Validation result |
| [`build_device_corpus_features.py`](../../tools/build_device_corpus_features.py) | Align and extract manifest-assigned device splits | Device feature archives |
| [`write_stratified_training_config.py`](../../tools/write_stratified_training_config.py) | Turn source shares into deterministic batch quotas | Training YAML with balance report |
| [`audit_training_ablation.py`](../../tools/audit_training_ablation.py) | Reject undeclared configuration differences | Paired-run audit |
| [`audit_source_ablation.py`](../../tools/audit_source_ablation.py) | Prove that only source diversity changed while class exposure stayed fixed | Source comparison audit |
| [`select_recipe_cutoff.py`](../../tools/select_recipe_cutoff.py) | Choose a cutoff from validation only | Artifact-bound cutoff frontier |
| [`evaluate_recipe_model.py`](../../tools/evaluate_recipe_model.py) | Report per-phrase synthetic performance | Pronunciation and foil report |
| [`evaluate_device_corpus_model.py`](../../tools/evaluate_device_corpus_model.py) | Score real captures in isolated and runtime-like state modes | Cohort and qualification report |

The macOS `say`, `afplay`, `osascript`, Deepgram timestamping, and custom replay
scripts were useful experiment fixtures. They are not yet generalized repository
tools. If reused, promote them into provenance-aware commands rather than copying
their temporary scripts.

## What remains missing

- independent natural human adult and child recordings in validation and test;
- multiple recording sessions and normal household distances for each speaker;
- more natural human hard negatives and long connected speech;
- labeled playback-while-Kizz-is-speaking cases if barge-in is required;
- a frozen, representative long-ambient acceptance duration and false-accept
  budget;
- the checked-in or externally published v18a initial weights, source manifests,
  and device corpus needed to reproduce v19 byte for byte;
- physical qualification across other microphone profiles before claiming a
  shared model.

## Fresh-start recommendation

Treat v19 as the control, not as the base for an endless sequence of patches.
Freeze its artifact and reports. Build a new corpus version with the corrected
`High Five Kizz` label, independent natural speakers, retained live failures,
time-matched hard negatives, normal and challenging acoustic banks, and a
published sampling plan. Train a clean baseline and one declared variant. Select
the cutoff from validation, open test once, then run the physical ladder. If the
new model loses live recall, compare its weakest cohorts with v19 before changing
architecture, balance, and augmentation at the same time.
