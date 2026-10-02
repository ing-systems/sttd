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

**Speech density**:
The share of an utterance's transmitted frames that were VAD-flagged as
speech. Judged post-trim, so a short real word with a full silence hangover
still passes while scattered VAD flicker spread over a long span does not.
Long utterances skip the gate since genuine pauses lower their density.
_Avoid_: speech ratio, fill factor

**Hallucination**:
A fluent transcript produced for audio that contains no such speech — most
often a line of subtitle boilerplate recalled from training data once there is
nothing to transcribe. Recognised by exact match against known boilerplate,
after folding Polish diacritics so a single mis-accented letter can't slip one
through.
_Avoid_: false positive, noise, garbage

**Emission**:
What happens after the transcript gate emits: opt-in stdout print
(`--stdout`), clipboard copy, auto-paste, and WAV/TXT dump. Owned by the emitter module behind a
seam; the ASR worker hands it one emission per utterance — tag, raw audio,
trimmed text, verdict, acoustic evidence — and gets back nothing.
_Avoid_: output, sink, side effects (as a name)

### Speech models

**Speech model**:
The loaded, ready-to-run engine an utterance is handed to for transcription —
chosen once at startup by the model spec and fixed for the process's lifetime.
Two engines sit behind the same seam: Parakeet-TDT (direct `ort` sessions, a
Rust mel frontend, dictionary phrase-boosting on its greedy decode, and per-word
confidences for the dictionary gate) and Whisper large-v3-turbo (three `ort`
sessions, a Rust log-mel frontend, and a KV-cached greedy decode with the same
per-word confidences but no phrase boosting). Parakeet auto-detects the spoken
language per utterance; Whisper is told which to decode in.
(Sherpa-onnx support lives on branch `sherpa-onnx`.)
_Avoid_: model (bare, when the engine — as opposed to the weights — is meant), backend

**Engine**:
Which of the two speech models a run uses, settled by the model spec (or, for a
model directory named on the command line, by the files it holds). Everything
past that point — the worker loop, the gates, the metrics, the emission — is
engine-agnostic.
_Avoid_: backend, model kind

**Model spec**:
A `--model` value that names a catalog entry rather than a filesystem path —
`parakeet`, `parakeet-ultra-int8`, `parakeet-redux-int8`, `whisper` or
`whisper-large-v3-turbo-int8`. Resolved into the models directory baked into the
binary at build time (`STTD_MODELS_DIR`, default `models`; the Arch package
uses `/usr/share/sttd/models`) and, for a catalog spec, downloaded if missing;
an existing directory passes through untouched instead, and the engine is read
off its files.
_Avoid_: model name, model id

**Decoding language**:
The language Whisper is told to decode in, fixed into its prompt as a language
token for the process's lifetime — `pl` unless `--language` says otherwise,
with `auto` detecting per utterance from the prompt's own first-token logits.
Parakeet has no such thing: it detects the language itself, every utterance.
_Avoid_: language detection (that is what Parakeet does, and what `--language
auto` asks Whisper for), locale

**Padded window**:
The span of audio a Whisper utterance is decoded in, padded (and truncated) to
length before it reaches the encoder. The exported encoder takes exactly 3000 mel
frames (30 s) because its input axes are constants in the ONNX graph, so by default
Whisper's cost per utterance does not scale with how much was said: a two-second
utterance costs the same encoder pass as a thirty-second one, and the RTF column
reports against the window rather than the utterance. Under **streaming mode** the
window is chosen per utterance instead (the audio rounded up to a second, plus a
second, never under 15 s or over 30 s).
_Avoid_: chunk (that is a piece of an utterance, cut by the VAD), clip

**Streaming mode**:
What `--stream` turns on: **partial transcripts** while an utterance is open and,
for Whisper, a **padded window** that follows the audio, made possible by
patching the encoder graph in memory when the model loads. It changes when and how
fast text appears in the log, never what the final transcript of an utterance goes
through.
_Avoid_: live mode, real-time mode (the whole tool is real-time), sliding window
(there is no overlap between decodes)

**Partial transcript**:
A decode of an utterance that has not closed yet: its audio so far, cut like a
close would cut it (up to the last speech plus the minimum tail), decoded about
once a second and logged. It is a preview only. It takes no utterance number,
feeds no split-seam dedup, reaches no emitter, metric row or autopause, and it
loses to any closed utterance waiting in the queue; only the latest one is kept.
_Avoid_: interim result (that is the interim split, which cuts a real utterance),
hypothesis, draft

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
only ever runs on the VAD branch, never on the model branch. DeepFilterNet3
adds a 20 ms lookahead ahead of its own output, absorbed by pre-roll
(300 ms) and silence hangover (800 ms) the same way `earshot`'s own 16 ms
prediction window is.
_Avoid_: noise suppression (generic), the denoiser (as if only one exists)

**Speech cache**:
Whisper's key/value tensors for one utterance: the cross-attention half, which
depends only on the encoder output and is reused unchanged for every generated
token, and the self-attention half, which grows by one token per step. Two
graphs keep this split — one runs the prompt and produces it, the other runs
each later token off it. Parakeet has no counterpart (it decodes frame by frame
with no attention cache).
_Avoid_: KV cache (fine as a term, but the split is the point), hidden state

**Model branch**:
The signal path that gets transcribed, kept raw and undenoised, because the
speech model was trained on raw audio. Its samples are what an utterance is
made of.
_Avoid_: audio branch, main path

**Direct hop**:
The default path to the judged/transcribed rate: hw rate straight to 16 kHz
in one resampling step, feeding both the VAD verdict and the utterance
buffer. Only `STTD_DENOISER=deepfilter` routes the VAD branch through the
48 kHz rate the denoiser demands — that intermediate exists only for the
denoised VAD branch.
_Avoid_: upsampling, downsampling, conversion chain
