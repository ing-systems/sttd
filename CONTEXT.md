# sttd

Real-time Polish speech-to-text for a live microphone. Audio is cut into
utterances locally so that only speech-bearing stretches ever reach the speech
model, which prints its transcripts to stdout.

## Language

### Utterance and its stages

**Utterance**:
One stretch of speech, from its onset through the trailing silence that ends
it, treated as a single unit of transcription. It is what the model is handed,
and it yields at most one line of output.
_Avoid_: clip, chunk, segment, phrase

**Utterance buffer**:
The in-progress accumulation of an utterance that is still open — its audio and
its measurements, gathering until it is closed and either accepted or dropped.
It is the builder that produces an utterance, not an utterance itself.
_Avoid_: utterance (for the in-progress form), accumulator, window

**Pre-roll**:
Audio from just before an utterance opens, held in reserve and prepended to it
so a quietly-started word isn't clipped by the delay before speech is noticed.
It spans utterances on purpose: it always holds the run-up to whatever opens
next.
_Avoid_: lookback, ring buffer (as a name for the audio itself)

### What the acoustics observed

**Acoustic evidence**:
The measurements taken over an utterance's speech content while it was
buffered: how much of it was judged speech, and how loud that speech was. It
describes the sound heard, never the transcript produced.
_Avoid_: audio features, signal stats, clip metadata

**Evidence strength**:
The two-band verdict — weak or strong — that acoustic evidence earns, which
decides how much the resulting transcript must be trusted. It summarises the
evidence rather than measuring anything new.
_Avoid_: confidence, quality score

**VAD verdict**:
The per-frame judgement of whether that frame holds speech. It is evidence, not
a fact: a click, a breath or mains hum can carry it, and real speech can fail
it.
_Avoid_: speech detection, voice activity (as a verdict)

### Deciding not to emit

**Gate**:
A check that can stop an utterance, or a transcript, from being emitted.
Acoustic gates run on the sound before the model sees it; transcript gates run
on the text after.
_Avoid_: filter (for the decision), validation

**Confidence floor**:
The minimum average text-token probability a transcript must reach to be
emitted. The floor is strict when evidence strength is weak and relaxed when
it is strong, because one fixed value cannot separate hard-but-real speech from
an invention over silence.
_Avoid_: threshold, cutoff, score

**Text token**:
A token of the transcript itself, as opposed to a timestamp token, which marks
when a segment started or ended and says nothing about what was said. Only text
tokens inform confidence.
_Avoid_: word, output token

**Hallucination**:
A fluent transcript produced for audio that contains no such speech — most
often a line of subtitle boilerplate recalled from training data once there is
nothing to transcribe. Recognised by exact match against known boilerplate,
after folding Polish diacritics so a single mis-accented letter can't slip one
through.
_Avoid_: false positive, noise, garbage

### Signal paths

**VAD branch**:
The signal path whose only job is to judge speech from not-speech:
denoised and resampled down to the judged rate. Nothing on this branch is ever
transcribed.
_Avoid_: detection path, analysis branch

**Whisper branch**:
The signal path that gets transcribed, kept raw and undenoised, because the
model was trained on raw audio. Its samples are what an utterance is made of.
_Avoid_: audio branch, main path

**Direct hop / via the 48 kHz intermediate**:
The whisper branch reaches the model's rate in either one resampling step or
two, the longer route passing through the rate the denoiser demands. The
two-step form is the fallback for hardware rates where the one-step form cannot
stay lined up in time with the VAD branch.
_Avoid_: upsampling, downsampling, conversion chain
