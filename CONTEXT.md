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
it. Produced by `earshot`, whose neural predictions land on a fixed 16 ms
window rather than the pipeline's 10 ms frame cadence; the verdict for a frame
that falls between windows re-reports the last prediction rather than guessing.
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
resampled down to the judged rate. Nothing on this branch is ever
transcribed.
_Avoid_: detection path, analysis branch

**Denoiser**:
The model that cleans the VAD branch's 48 kHz frame before it's resampled
down and handed to `earshot`. Opt-in via `STTD_DENOISER=deepfilter`
(default `none`: no denoising, no 48 kHz intermediate — see `direct hop`);
only ever runs on the VAD branch, never on the whisper branch. DeepFilterNet3
adds a 20 ms lookahead ahead of its own output, absorbed by pre-roll
(300 ms) and silence hangover (800 ms) the same way `earshot`'s own 16 ms
prediction window is.
_Avoid_: noise suppression (generic), the denoiser (as if only one exists)

**Whisper branch**:
The signal path that gets transcribed, kept raw and undenoised, because the
model was trained on raw audio. Its samples are what an utterance is made of.
_Avoid_: audio branch, main path

**Direct hop**:
The default path to the judged/transcribed rate: hw rate straight to 16 kHz
in one resampling step, feeding both the VAD verdict and the utterance
buffer. Only `STTD_DENOISER=deepfilter` routes the VAD branch through the
48 kHz rate the denoiser demands — that intermediate exists only for the
denoised VAD branch.
_Avoid_: upsampling, downsampling, conversion chain
