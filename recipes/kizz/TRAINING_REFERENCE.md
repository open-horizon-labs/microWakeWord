# Training a wake word: the Kizz case study

This is a practical guide to training a custom **microWakeWord** detector. It
uses **HiPhi Kizz** as the example because it exposes the problems a real
product must solve: people pronounce “HiPhi” differently; `Kizz` sounds close
to `kids`, `kiss`, `quiz`, `his`, and `this`; and the device has to work through
a small microphone while life is happening around it.

You do not need previous model-training experience to use this guide. Read the
first four sections in order. The later sections preserve the complete Kizz
recipe so that someone can repeat, inspect, or improve it without relying on
temporary scripts or memory. Generic commands live in
[Usage](../../documentation/USAGE.md); the detailed run ledger is in
[Experiments](EXPERIMENTS.md).

## Start here: what a wake-word model does

The detector repeatedly listens to a short slice of sound and produces a score
between `0` and `1`:

- a score near `1` means “this sounds like the wake phrase”;
- a score near `0` means “this does not sound like the wake phrase.”

We choose a **cutoff** (also called a threshold). If the score reaches that
number, the device wakes. At a cutoff of `.70`, a score of `.82` wakes Kizz and
a score of `.41` does not.

There are two ways to fail:

- **False negative:** someone says “HiPhi Kizz” and Kizz does not wake.
- **False positive** or **false wake:** Kizz wakes for something else, such as
  “Hey Kizz,” a TV line, or music.

Lowering the cutoff catches more true wakes, but can also create more false
wakes. That trade only works when the two kinds of audio have different scores.
For example, a true wake at `.72` and a confusing phrase at `.68` are close;
we can test cutoffs around that boundary. A true wake at `.03` and a confusing
phrase at `.74` are in the wrong order. No cutoff can accept `.03` while
rejecting `.74`. The fix is better training evidence, not threshold tuning.

## The four things a useful model must prove

1. **It hears the phrase.** Test several natural pronunciations, volumes, and
   distances. The measure is recall: how many real wake attempts it catches.
2. **It ignores nearby sounds.** Test close words, ordinary conversation, TV,
   music, and room noise. Count the false wakes rather than calling a few lucky
   minutes “good.”
3. **It works outside its practice set.** Keep some voices, sessions, and
   acoustic conditions completely out of training until evaluation.
4. **It works in the product.** A WAV on a laptop is not Kizz. The final test
   uses the actual microphone, firmware, speaker, room, and streaming state.

These are different questions. A model can pass the first two on familiar
recordings and still fail a new person in the living room.

## A small vocabulary before the recipe

| Term | Plain meaning |
| --- | --- |
| **Positive** | Audio that should wake the device: a full “HiPhi Kizz” phrase. |
| **Negative** | Audio that must not wake it: `Hey Kizz`, conversation, TV, room noise, and so on. |
| **Hard negative** | A negative that sounds unusually close to the target phrase, such as `Hi-Fi Kiss` or `Wi-Fi Kizz`. |
| **Training set** | The examples the model is allowed to learn from. |
| **Validation set** | Separate examples used to choose settings such as the cutoff. The model must not train on them. |
| **Test set** | A final, untouched exam. Open it only after choosing the model and cutoff. |
| **Voice identity** | The person or persistent TTS voice that made a clip. It is not the filename. |
| **Augmentation** | A reproducible variation of a clean recording: room echo, noise, level, timing, and similar conditions. |
| **Device recording** | Audio recorded by the target product, including its mic, gain, room, and electronics. |
| **Checkpoint** | A saved version of the model partway through training. The last one is not always the best one. |
| **Frontend** | The device’s audio preparation before the model sees it: microphone input, gain, sampling, and processing. |

## What “independent evaluation” actually means

Changing the words in a clip is not enough. If the same TTS voice says
`Hi-Fi Kizz` in training and `Hi Phi Kizz` in validation, the model may benefit
from that familiar voice rather than learning the phrase. Likewise, two WAV
files made by the same person in the same session are more alike than they look.

Give each source voice to one split before generating audio. Keep a fresh human
speaker or later recording session out of all earlier work as a final reality
check.

| Evaluation set | What it proves | What it does **not** prove |
| --- | --- | --- |
| New file, same voice and session | The model did not memorize one WAV | It generalizes to another person or room |
| New phrase, same voice | It handles phrase variation | It generalizes beyond that voice |
| New voice, same room | It handles another speaker | It works in another acoustic setting |
| New voice and new session/room | It has a chance of working in normal use | It will work universally |

## Current Kizz baseline

V19 checkpoint 100 is the best live control from this effort. It used a cutoff
of `.70` and a one-frame sliding window. It is **not** a qualified release: its
held-out and physical results still show too many misses and false wakes.

Its quantized TensorFlow Lite model file is preserved so later work has a stable
comparison point:

```text
76250d0cef49f893df4724ea6cce0e87b8a8d0d63cf10fbe23c0e624298871ff
```

### Current live observation

Kizz has produced some false wake triggers with no speech following them. The
post-wake timeout prevented those events from becoming accidental music
commands, so they were harmless to the listener but still failures of the wake
detector. We do not yet know whether these events come from the model confusing
ambient sound, a microphone level that is too high, movement, or a runtime state
transition. A gain change without an audio capture would be a guess.

The next capture path must retain a short ring buffer around every wake. If no
speech follows before the post-wake timeout, label the reviewed recording
`false_wake_no_command` and preserve the peak wake score, cutoff, gain, device
profile, room condition, and detector state. Do not add these clips to training
automatically: an abandoned but legitimate request is different evidence from
an ambient false wake.

## What this work taught us

1. A large pile of synthetic clips can look good in a report and still fail on
   the device. TTS voice, TTS provider, room, speaker, microphone, and recording
   session each change the sound the model hears.
2. Keep every source voice in one split: training, validation, or test. Splitting
   only by phrase or filename lets familiar voices leak into evaluation.
3. Piper supplies broad synthetic variety, but it does not provide trustworthy
   adult or child labels. We added designed adult and teen ElevenLabs voices to
   get labeled cohorts and another TTS family.
4. A clean TTS file and that same file played through a laptop speaker into Kizz
   are different examples. The second includes the room and the real microphone.
5. Save wake attempts that the detector misses. A dataset made only from successes
   teaches the model to repeat its old successes.
6. Do not use an average score as proof that the model works for everyone. A
   model can score well across old recordings while missing a new person. Before
   flashing, run a small challenge from a new speaker or a new recording session.
7. Test both isolated clips and continuous audio. Isolated clips test sound
   classification; continuous audio also tests the device’s re-arming behavior.
8. “Balanced data” means deciding what each batch contains. A small group can
   still have outsized influence if its mistakes are given extra loss weight.
9. Adding more negatives can make the model worse if they crowd out positive
   examples or make the two score distributions move together. V29–v31 showed
   this failure mode.

## How our understanding changed

| What we first assumed | What the evidence required | What changed our mind |
| --- | --- | --- |
| “Make enough TTS and adjust the cutoff” | Build separate voice, room, and device sources, then check that correct and incorrect phrases receive different scores | Natural positives scored near zero while some room speech exceeded `.70`. |
| “Split filenames or phrases” | Assign each voice and recording session to one split before generating clips | One apparent held-out device positive came from the training speaker under a second ID. |
| “Synthetic speech stands in for device speech” | Treat a clean WAV and the same WAV replayed into Kizz as different evidence | A model that accepted 17/17 clean device-training clips accepted only 1/8 when those phrases were replayed into Kizz. |
| “The largest folder should count most” | Decide the share of each source and pronunciation in every batch | Tens of thousands of Piper clips otherwise drowned out designed voices and scarce device evidence. |
| “A failed voice turn means the wake model failed” | Check wake detection, listening state, command capture, network transport, transcription, and command execution separately | Several turns woke correctly but later produced “speech recognition unavailable” because command audio did not reach the server. |
| “Training needs the physical product online” | Test enrollment with a simulated device; use hardware to test real sound | The enrollment protocol can prove that it retains misses without Kizz attached. |
| “The trainer belongs inside UHC” | Give enrollment its own configured endpoint | Training and production voice have different lifecycles and deployment needs. |
| “Write Kizz-only training code” | Store device facts in a profile and reuse the training framework | Future microphone products can reuse the same recording rules and evaluate a shared model first. |

## The repeatable training loop

Use this loop for a new wake phrase. Change one declared variable at a time;
otherwise a better or worse result will not tell you why it changed.

1. **Write down what should wake and what must not.** Include pronunciation
   variants, close-sounding phrases, ordinary speech, music, and room noise.
2. **Create clean source audio.** Use more than one voice family. Keep the
   original WAVs and record who or which persistent TTS voice made each one.
3. **Assign splits before training.** A voice belongs to train, validation, or
   test—not to more than one. Reserve a later speaker/session for reality checks.
4. **Record the target device.** Play reserved clips through a speaker and ask
   real people to speak naturally. Save wakes, misses, and false wakes.
5. **Create controlled variations.** Derive noise, room echo, level, and timing
   changes from clean audio. Keep the recipe so each variation can be rebuilt.
6. **Train with deliberate source shares.** Do not let the largest folder decide
   what the model sees most often.
7. **Choose the cutoff on validation data.** Freeze the model and cutoff, then
   open the untouched test set once.
8. **Run the actual device.** Test continuous listening, re-arming, normal
   distances, and a long false-wake guard. Record the model hash and settings.

### Evidence we should have collected earlier

- independent natural adult and child speakers assigned to fixed splits;
- multiple sessions and normal-distance recordings for each speaker;
- an inventory of every temporary generator, fixture, model, manifest, and
  report before summarizing the recipe;
- a physical-test record binding model hash, threshold, sliding window,
  microphone gain, firmware, and active server instance;
- separate telemetry for microphone input, wake probability, detector state,
  captured command bytes, transmitted bytes, STT result, and command result;
- the source and availability of the v18a weights used to initialize v19.

### Process failures worth avoiding

- Important knowledge lived in temporary scripts and generated catalogs while
  the repository described only the happy path. Put anything needed to repeat a
  result in a checked-in tool, manifest, or case-study record.
- We debugged wake detection, post-wake capture, speech-to-text, and commands in
  one live loop. That made it too easy to blame the wrong layer. Give each layer
  its own result and recovery telemetry.
- Model, cutoff, gain, and server process changed during physical tests without
  one run record. Print and save the active configuration before the first
  utterance.
- The first documentation pass recorded the winning settings but not every
  source and setup. Inventory each workspace file as documented, promoted,
  or intentionally discarded.

### Reusable parts of the framework

The reusable parts are the speaker-split generators, voice catalog, source
quality screen, background inventory, device recording rules, phrase
alignment, batch sampler, one-change-at-a-time audits, cutoff selector, and
two-mode evaluator. The [tool ledger](#tools-and-what-each-preserves) maps each
one to code.

### Rules we now follow

- Preserve clean WAVs; derive masks, aligned clips, acoustic variants, and
  features reproducibly.
- Bind every generated or derived corpus to its recipe, source model, voice
  identity, settings, split, and hashes.
- Assign a human or synthetic voice identity to one split before generation.
- Never treat Piper speaker IDs as adult or child labels.
- Cap large homogeneous sources by speaker and phrase instead of feeding every
  available clip into each run.
- Keep sampling groups single-class and report both how many examples each group
  supplied and how strongly its errors were weighted during learning.
- Treat validation as the only cutoff-selection source. Open test after the
  candidate and cutoff are frozen.
- Retain live misses, false wakes, and ambient audio as labeled evidence.
- Keep Kizz-specific labels and results in this recipe; keep enrollment,
  source records, balancing, and evaluation code device-neutral.
- Compare one shared model across microphone profiles before creating
  device-specific models.
- Flash only the quantized model file that passed the declared offline checks, then
  record physical results against its hash.

## Building the Kizz corpus

### Audio that should wake Kizz

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

### Audio that must not wake Kizz

The negative set was designed in layers:

| Kind of negative | Examples | What it checks |
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

## Where the audio came from

No one source is enough. Each source below contributes a different kind of
evidence. “Fed v19” means it helped train the preserved v19 control; it does not
mean it is automatically good enough for a future model.

| Source | How we made it | What it teaches | Fed v19? |
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

## Piper: a large synthetic pool

Piper provides many generated voices cheaply. It is useful for breadth, but it
is still one synthesis family. We use a diverse subset rather than letting its
file count dominate a training run.

The Open Horizon Labs
[`piper-sample-generator`](https://github.com/open-horizon-labs/piper-sample-generator)
fork was pinned at commit `35d2f2d`. It adds speaker-range isolation, seeded
generation, and per-WAV provenance to the upstream generator.

### Keeping Piper voices separate

The LibriTTS-R generator does not use one fixed voice per WAV. It blends two
base speaker representations. V19 used
the first 700 base speakers because later LibriTTS-R identities were more prone
to artifacts:

| Split | Half-open base-speaker range | Planned sample share |
| --- | ---: | ---: |
| Train | `0..560` | 80% |
| Validation | `560..630` | 10% |
| Test | `630..700` | 10% |

The ranges do not overlap before synthesis. Each WAV records both base speaker
IDs, blend amount, phrase, and synthesis settings. The IDs do **not** tell us a
speaker’s age, so they cannot justify adult/child claims.

### How we varied a Piper voice

The generator cycled through:

| Parameter | Values |
| --- | --- |
| Speaker interpolation | `0.2`, `0.4`, `0.6`, `0.8` |
| Length scale | `0.72`, `0.85`, `1.0`, `1.15`, `1.32` |
| Synthesis noise | `0.55`, `0.667`, `0.8`, `0.95`, `1.15` |
| Duration noise | `0.65`, `0.8`, `0.95` |
| Seed | Recipe seed `231`, advanced deterministically by phrase and split |

The v19 generation manifest contained 67,150 Piper WAVs: 38,300 positives and
28,850 close-sounding negatives. We did **not** train on all of them each time.
Feature building chose at most one clip for each participating speaker/phrase
pair, then the batch sampler gave Piper a fixed share.

The large corpus remains useful as a pool for repeatable, diverse subsets.
Feeding it all into every run would drown out ElevenLabs and device recordings.

## ElevenLabs: a second synthetic voice family

We used ElevenLabs for two separate jobs:

1. **Voice Design** created persistent voice identities from written acoustic
   descriptions.
2. **Text to Speech** rendered wake phrases and hard negatives with those fixed
   identities.

The distinction matters because a fresh preview is not necessarily a new voice.
The persistent provider voice ID defines the voice identity and therefore its
train/validation/test split.

### The eight designed voices

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

We asked for natural household speech, not announcer or character voices. Voice
Design used `eleven_ttv_v3`, guidance `5`, seeds `231` through `238`, and an
ordinary conversational preview paragraph. Enhancement was disabled. We saved
every preview, selected preview zero for each catalog entry, and promoted it to
a persistent voice.

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

The voice catalog records the provider, TTS model, voice IDs, split, age cohort,
design description, and seed. The generation manifest records the voice and
settings for each WAV. Provider voice IDs and generated audio remain workspace
artifacts rather than repository content.

### Close phrases in the same voices

The first designed-voice pass concentrated on positives. A later pass rendered
all 28 short confusable phrases in the same adult and teen voice families.
Without this, the model could take a shortcut: ElevenLabs would usually mean a
positive while Piper supplied most close phrases. These negatives remove that
shortcut. V19 gave them their own sampling group.

### Longer sentences that caused false wakes

Observed room false wakes were expanded into connected sentences: kids/five,
Wi-Fi, “if he is,” quiz/kiss, ordinary commands, and unrelated household speech.
These were rendered as a separate ElevenLabs corpus so a later experiment could
add them without changing the original phrase sources. That work informed
v20–v31; it did not create the v19 artifact.

## Other voice generators we explored

### Kokoro

A local Kokoro pilot rendered the wake phrase with `af_heart`, `am_fenrir`,
`bf_emma`, and `bm_fable`. The same voices rendered four collision probes:
`Hee Fee Kiss`, `Hey Kizz`, `Hi-Fi Kiss`, and `High Fee Kiss`.

It helped us hear how another model family pronounced the name. But the pilot
had no checked-in generator, identity/split record, or full phrase set. We did
not use it for v19 or count it as independent evaluation. If we add it later,
it needs the same catalog and provenance as every other source.

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
This checked the enrollment path and exposed microphone/transport problems. We
did not use it to train or qualify v19: system voices were a small related
family, and the early record-keeping was weaker.

## Bringing synthetic speech into the real room

A clean WAV does not include the laptop speaker, room, distance, Kizz mic, or
gain. We therefore played split-reserved ElevenLabs training voices from the
laptop into Kizz and recorded what Kizz actually heard through enrollment.

### First playback setup

- four ElevenLabs training identities: two adult and two teen;
- three positives: `Hi-Fi Kizz`, `Hi Phi Kizz`, `Hiffy Kizz`;
- three hard negatives: `Hi-Fi Kiss`, `high five kids`, `Wi-Fi Kizz`;
- laptop system volume 50%;
- approximately 15 cm speaker-to-device distance;
- 2.5-second Kizz capture window;
- the middle clean source WAV from each phrase/voice directory.

### A second playback setup

A second pass synthesized new slow and fast examples with the same four
training identities, played them at 50%, and captured 2.6 seconds. This added two
more examples for each positive phrase and one more for each selected negative.
It recorded the source WAV hash, voice identity, age cohort, render settings,
volume, distance, and session.

### V19’s two playback conditions

The final v19 adaptation pass replayed one existing clean source per phrase and
voice under two physical conditions:

| Condition | Laptop volume | Playback delay inside capture | Purpose |
| --- | ---: | ---: | --- |
| `quiet-early` | 42% | 180 ms | Quieter speech near the start of the feature window |
| `loud-late` | 60% | 700 ms | Louder speech at a different window phase |

Both used a 2-second recording at roughly 15 cm. The clean source was retained;
the Kizz recording became a separate `synthetic_playback` capture. This pass was
the practical bridge between designed voices and the Kizz microphone channel.

Playback recordings are useful training evidence, but they are not human
validation. The source voice and split still apply, and a laptop speaker cannot
stand in for a person at normal household distance.

## Human attempts and room sound

The enrollment service tells a named device to record a short attempt whether or
not the current detector fires. Each entry records what the speaker intended,
who spoke, the session, device profile, firmware, conditions, provisional
`detected` result, WAV hash, and fixed split.

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

## Finding the wake phrase inside a longer recording

A model sees fixed-length audio windows. It can miss the wake phrase when a
recording is longer or speech begins late. We used three ways to find where the
phrase occurs:

1. **Human review:** record `start_ms` and `end_ms` for the intended phrase.
2. **Deepgram Nova-3 timestamps:** use utterance/word timing as a proposed span,
   then review it. Transcription was often wrong—one `Hi-Fi Kizz` became “I
   fight his”—while the time interval was still useful.
3. **Normalized cross-correlation:** for known speaker-playback fixtures, locate
   the voiced part of the clean source WAV inside the Kizz recording and record
   the correlation score.

Feature building keeps the measured phrase plus 250 ms of surrounding sound.
The original recording and hash stay unchanged. For a long capture without a
reliable phrase location, random cropping is the fallback. Cropping from the
start or end is safe only when the recording protocol guarantees where speech
will land.

## Checking sources and creating realistic variations

### Rejecting unusable source clips

Eight reviewed human wake spans initially established a 560–880 ms observed
range. Adding a 25% margin produced a 483–1,100 ms accepted positive span. The
v19 screen also required:

- RMS above `-50 dBFS`;
- clipped-sample fraction no greater than `0.001`;
- source duration no greater than 1,700 ms, leaving placement room inside the
  two-second training window;
- non-silent audio.

The v19 combined manifest contained 68,750 clean WAVs. The mask accepted 58,175
and rejected 10,575. Positive timing limits did not apply to hard negatives.
Passing this screen did not mean TTS sounded human or matched Kizz. It only
meant the source was usable for the declared training recipe.

### Background sound

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

### Variations created during feature building

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

Speech was 3 to 20 dB louder than background sound, and timing moved by 150 to
300 ms. Validation and test clips stayed clean so they measured the model rather
than random training noise. Later tooling added a `challenging` profile where
noise can exceed speech (-6 to 6 dB SNR). It was not part of v19; test it as a
declared one-change experiment rather than silently replacing the baseline.

## What v19 showed the model in each batch

The source folders were very different sizes, so v19 used **stratified batches**:
each batch received a planned share from each source instead of simply pulling
from the largest folder. The first percentage says how many examples the model
saw. The final percentage also accounts for error penalties: examples whose
mistakes cost more during training push the model more strongly.

| Group | Planned share | Realized sample share | Weighted-pressure share |
| --- | ---: | ---: | ---: |
| Piper positives | 15% | 15.625% | 11.441% |
| Piper hard negatives | 15% | 15.625% | 22.883% |
| Designed positives | 25% | 25.000% | 18.306% |
| Designed negatives | 10% | 9.375% | 13.730% |
| Kizz microphone positives | 15% | 14.063% | 10.297% |
| Targeted negatives | 20% | 20.313% | 23.343% |

Across 400 batches of 64, the ledger recorded 25,600 samples. The literal mix
was 54.688% positive and 45.313% negative. Piper and designed close negatives
used a `2.0` error penalty, and Kizz hard negatives received extra weight inside
their group. As a result, negatives supplied about 59.96% of the learning
pressure. Both numbers matter: a dataset can look balanced by file count while
the loss function emphasizes one side.

Within the Piper groups, phrase-separated feature sources and the
one-clip-per-speaker/phrase cap prevented the raw 67,150-file corpus from setting
the distribution. Adult and teen ElevenLabs positives had separate feature
sources. Designed negatives also had adult and teen sources. Targeted negatives
combined Kizz hard negatives, ordinary speech, dinner-party speech, and
background-only features under declared within-group weights.

## The exact v19 training settings

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

Checkpoint 100 heard the most replayed wake attempts. Checkpoint 200 rejected
some close phrases but missed more real wakes; the later checkpoint was worse
again. The last checkpoint is not automatically the best model. Convert and test
every promising checkpoint independently.

## How to evaluate a candidate model

Run these checks in order. Each catches a different way a model can look better
on paper than it feels in use.

1. **Training diagnostics:** loss and checkpoint metrics. They reveal a broken
   run, but do not prove product quality.
2. **Validation with held-out voices:** choose the cutoff while reporting every
   pronunciation, provider, voice, and age group.
3. **Untouched synthetic test:** measure new voices only after choosing the
   model and cutoff.
4. **Unseen pronunciation probes:** test readings absent from training. Keep
   them outside future training or replace them with new probes.
5. **Device test, reset per capture:** checks audio classification without
   leftover streaming state.
6. **Device test, carry until detection:** checks continuous listening,
   accumulation, and reset behavior.
7. **Fresh speaker/session challenge:** do not accept a model that improves old
   averages but fails a new person or later recording session.
8. **Physical speaker replay:** play held-out WAVs through the actual room and
   inspect wake probability, detection, and re-arming.
9. **Natural human acceptance:** test adults and children at ordinary distance,
   phrasing, and volume.
10. **Long ambient guard:** measure false accepts during conversation, music,
    TV, podcasts, quiet, and device movement.

The physical test must record the model hash, cutoff, sliding window, firmware,
microphone profile, gain, distance, playback level when applicable, detections,
peak probabilities, and whether the detector re-armed.

## Do not blame the wake model for every voice failure

A spoken command crosses several systems. Identify the failed stage before
changing the wake model:

| Observation | Likely layer to inspect |
| --- | --- |
| Microphone peak stays near zero | Mic enablement, gain, audio preparation, power, or hardware |
| Mic sees speech but wake probability stays low | Model recall, a mismatch in audio preparation, or a mismatch between training sound and the room/device |
| Wake probability is close to the cutoff | Test the cutoff trade-off against both missed wakes and false wakes |
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

## What the experiment families taught us

| Phase | Main question | Durable result |
| --- | --- | --- |
| Early combined and fee-family models | Can synthetic spellings separate the wake phrase from nearby words? | Include bare words and joined-word traps as negatives; broad synthetic recall can mislead. |
| Device adaptation and capacity search | Can a broad model learn the first Kizz recordings? | Save misses, locate phrases, and preserve the device’s audio preparation; a bigger model alone did not transfer. |
| Quality screen and independent splits | Were poor sources and leaked speakers inflating results? | Use human-referenced screening and voice-identity splits. |
| V10–v12 batch balance | Does deliberate source balance beat directory-size sampling? | Per-batch shares, source caps, and designed negatives became reusable framework features. |
| V13–v19 | Which loss, capacity, device playback, labels, and fine-tuning choices improve live recall? | Focal loss, frozen BN, multi-condition Kizz playback, and checkpoint 100 produced the v19 control. |
| V20–v26 counterexample adaptation | Can mined room false wakes be patched into v19? | Invalid checkpoint selection and over-concentrated counterexamples can create flattering but unusable models. |
| V27–v28 diversified separation | Can a broader rebuild reduce close-phrase accepts? | Device averages improved, but a fresh-speaker check rejected v28. |
| V29–v31 controlled continuation | Can gentle extra emphasis on negatives repair v19? | Unfrozen BN collapsed recall; frozen BN restored positives but also false wakes; v31 still did not separate the two score groups. |
| V32 canonical reboot | Can corrected labels, connected speech, and broad hard mining create a clean binary boundary? | Rejection improved sharply, but the best remine candidate retained only 13.1% validation recall and 6/17 raw-device positive recall. |
| V33–v34 retention and verifier probes | Can v19's useful recall survive tiny updates, classifier-only retraining, or broader hard-negative pressure? | No: v33a emitted no usable point, v33b retained 6.4% validation recall, and v34 stayed below 40% at near-saturated cutoffs. |
| Ordered-state v1 | Can one local phone-state encoder plus deterministic temporal ordering create the missing invariant? | The best fixed endpoint reached 50% recall at 0.24963 FAPH; later checkpoints regressed, so the run stopped without opening test or flashing. |

Detailed measurements and rejection reasons are in [Experiments](EXPERIMENTS.md).

## Tools and the records they create

| Tool | Purpose | Durable output |
| --- | --- | --- |
| [`generate_recipe_samples.py`](../../tools/generate_recipe_samples.py) | Generate Piper phrases with disjoint speaker ranges and resumable reuse | Generation manifest and per-WAV synthesis metadata |
| [`design_elevenlabs_voice_catalog.py`](../../tools/design_elevenlabs_voice_catalog.py) | Design adult/teen voice identities and save previews | Split-bound voice catalog |
| [`add_labeled_voice_samples.py`](../../tools/add_labeled_voice_samples.py) | Render seeded ElevenLabs positive and negative variants | PCM WAVs plus a record of the voice and settings |
| [`apply_phrase_spans.py`](../../tools/apply_phrase_spans.py) | Attach reviewed phrase timing to source captures | Validated device manifest update |
| [`build_synthetic_quality_mask.py`](../../tools/build_synthetic_quality_mask.py) | Compare generated sources with human timing and objective audio limits | Hash-bound list of accepted and rejected sources |
| [`prepare_background_corpus.py`](../../tools/prepare_background_corpus.py) | Separate training backgrounds from stress evidence | Licensed, hashed background manifest |
| [`build_recipe_features.py`](../../tools/build_recipe_features.py) | Select speakers/phrases/providers, augment training, and extract `micro_speech` features | Feature build manifest |
| [`run_enrollment_service.py`](../../tools/run_enrollment_service.py) | Direct a configured microphone to capture every attempt | Device WAVs and corpus entries, including misses |
| [`simulate_enrollment_device.py`](../../tools/simulate_enrollment_device.py) | Test enrollment without hardware | End-to-end protocol evidence |
| [`validate_device_corpus.py`](../../tools/validate_device_corpus.py) | Enforce format, hashes, profiles, identity, and split integrity | Validation result |
| [`build_device_corpus_features.py`](../../tools/build_device_corpus_features.py) | Align and extract manifest-assigned device splits | Device feature archives |
| [`write_stratified_training_config.py`](../../tools/write_stratified_training_config.py) | Turn source shares into deterministic batch quotas | Training YAML with balance report |
| [`audit_training_ablation.py`](../../tools/audit_training_ablation.py) | Reject undeclared configuration differences | Paired-run audit |
| [`audit_source_ablation.py`](../../tools/audit_source_ablation.py) | Prove that only source diversity changed while class exposure stayed fixed | Source comparison audit |
| [`select_recipe_cutoff.py`](../../tools/select_recipe_cutoff.py) | Choose a cutoff from validation only | Model-bound cutoff comparison |
| [`evaluate_recipe_model.py`](../../tools/evaluate_recipe_model.py) | Report per-phrase synthetic performance | Pronunciation and close-phrase report |
| [`evaluate_device_corpus_model.py`](../../tools/evaluate_device_corpus_model.py) | Score real captures in isolated and runtime-like state modes | Group-by-group device test report |

The macOS `say`, `afplay`, `osascript`, Deepgram timestamping, and custom replay
scripts were useful experiment fixtures. They are not yet generalized repository
tools. If reused, promote them into provenance-aware commands rather than copying
their temporary scripts.

## Evidence still missing

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

## Historical recommended next training pass

The following recommendation was written before the clean-slate teacher/student
experiment and is preserved as history. It is not the current plan.

Treat v19 as the control. The next iteration is a narrow continuation from its
checkpoint, not a replacement based on one encouraging live session:

1. Add reviewed `false_wake_no_command` captures and matched ordinary audio as a
   dedicated negative source.
2. Keep all v19 positive sources, pronunciation variants, held-out voices, and
   device positives in the sampler. The new negatives must not crowd them out.
3. Use a small learning rate, preserve the v19 model settings, and save several
   checkpoints. Change only the declared negative source and sampling pressure.
4. Compare every checkpoint with v19 on held-out real positives, the new false
   wake set, hard negatives, long ambient audio, and physical Kizz replay.
5. Inspect microphone peak and gain alongside the scores. If false wakes cluster
   at saturation or movement conditions, fix the frontend or gain path instead
   of asking training to learn a hardware fault.
6. Select the cutoff only after the score distributions improve, then repeat the
   physical fresh-speaker challenge before flashing.

The success condition recorded at that time was fewer reviewed false wakes with
no follow-up speech while held-out and real-speaker recall stayed at least as
good as v19. A lower false-wake count without that recall guard was not an
improvement.

## Current disposition

The clean-slate teacher/student experiment followed this historical
recommendation but did not produce a deployable model. Its student looked clean
on the short offline evaluation and then produced frequent false positives on
StackChan. Deployment therefore reverts to the ESP-IDF wake-word model.

The next training run must add representative natural human positives and
long-form household/TV negatives, then qualify the exact firmware-shaped
artifact in live use. Do not tune the discarded student into deployment by
threshold alone.
