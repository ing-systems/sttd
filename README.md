# sttd

Real-time Polish-English speech-to-text for a live microphone.

`sttd` captures microphone audio, cuts it into utterances locally, and sends only speech-bearing audio to the speech model. It prints or pastes one line per utterance.

Single Rust binary. No server. No network at runtime except first model download.

## Features

- Live capture through `cpal` (ALSA / PipeWire) with device pick and channel `mix|auto|N`.
- Two-branch signal path: VAD branch judges speech, model branch stays raw for transcription.
- Utterance buffering with 300 ms pre-roll, 800 ms hangover, interim and early splits.
- Acoustic gates (RMS, speech density) and transcript gates (hallucination, repetition).
- Parakeet-TDT-0.6B-v3 (`int8` default, `fp32` optional) through `ort`, with Rust mel frontend.
- Optional translation of each accepted transcript before emission (`--translate`, ONNX: a MarianMT `opus-mt` pair, or one multilingual NLLB-200 for every direction).
- Emission to stdout, Wayland clipboard (`wl-copy`), auto-paste (`ydotool`), spoken transcript (`--speak`, piper), WAV+TXT dump, last-text file, JSONL metrics.
- Offline file mode, daemon mode, systemd user unit, pause/resume signals.
- Optional VAD-branch denoiser (DeepFilterNet3, `denoise` feature).
- Custom dictionary for post-gate correction.

## Pipeline

```text
cpal (hw rate, N ch) -> channel pick|mix -> 10 ms frames
  -> VAD branch: resample to 16 kHz -> earshot -> verdict per frame
  -> model branch: resample to 16 kHz (raw) -> utterance buffer
  -> rtrb queue -> ASR worker -> conditioning -> Parakeet -> gates -> [translation] -> emitter
```

Details:

- Default path resamples hardware rate straight to 16 kHz (`direct hop`). With `--denoiser deepfilter`, the VAD branch detours through 48 kHz, denoise, then 16 kHz. The model branch always stays raw.
- An utterance opens after 3 consecutive speech frames. Pre-roll prepends run-up audio so quiet word starts survive.
- An open utterance closes after `--hangover-ms` of silence (default 800 ms). Only trimmed audio plus a short tail reaches the model. Long speech emits interim splits past 8 s and force-splits past 28-30 s.
- Conditioning removes DC, applies 5 ms fades, and guards peaks. `Hp` removes 80 Hz hum. `Agc` lifts quiet speech toward the target RMS.
- Drops never emit. Emits go to the configured emitter.
- The gates see the source text. With `--translate`, the step between the gates and the emitter rewrites the text, so only the emitter gets the translation (see *Translation*).

See `src/main.rs:1-51` and `CONTEXT.md` for vocabulary (`utterance`, `VAD verdict`, `gate`, `hallucination`, `emission`, `translation`, `translator`, `model spec`, `VAD/model branch`, `direct hop`).

## Requirements

- Linux x86_64, Rust stable, `alsa-lib`.
- ~638 MiB disk for the default Parakeet `int8` model, or ~1443 MiB for Whisper.
- Optional: ~114 MiB more for the translator pair that `--translate` downloads on first use, or ~870 MiB for the multilingual `nllb-200-distilled-600M`.
- Optional: `wl-clipboard` (`wl-copy`) for `--copy`, `ydotool` + `ydotoold` for `--paste` (Wayland only), `espeak-ng` (+ its voice data package) for `--speak`, PipeWire/WirePlumber for capture, Python 3 stdlib for `tools/`.

## Models

Two engines ship behind one `--model` flag. Catalog (`src/config.rs`):

| Spec | Size | Contents |
| --- | --- | --- |
| `parakeet` (alias for `int8`) | 638 MiB | encoder int8 + decoder_joint int8 + vocab + tokenizer |
| `parakeet-ultra-int8` | 638 MiB | same as above |
| `whisper` (alias for `int8`) | 1443 MiB | encoder int8 + decoder int8 + decoder_with_past int8 + tokenizer |
| `whisper-large-v3-turbo-int8` | 1443 MiB | same as above |

Parakeet weights come from `huggingface.co/Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx` (ONNX weights from its `int8/` folder, `vocab.txt` from the repo root) and `huggingface.co/nvidia/parakeet-tdt-0.6b-v3` (tokenizer only). The export keeps the base architecture and tokenizer, so the Rust mel frontend and dictionary phrase-boosting are unchanged.

Whisper weights come from `huggingface.co/onnx-community/whisper-large-v3-turbo` (graphs from its `onnx/` folder, `tokenizer.json` from its root).

### Engines

| | Parakeet-TDT-0.6B-v3 | Whisper large-v3-turbo |
| --- | --- | --- |
| Window | grows with the utterance | fixed 30 s; follows the audio (15 s at least, `--audio-ctx`) |
| Inference cost | scales with speech length | one full encoder pass per utterance, whatever its length (about 2x less with the dynamic window) |
| Language | auto-detects per utterance | `--language` (any Whisper code; defaults to the system locale's language, or `auto`) |
| Dictionary phrase boosting | yes (logit-level) | no; the post-gate correction still applies |
| Per-word confidences | yes | yes |

Whisper's encoder takes a **fixed** 30 s window: its `input_features` axes are constants in the ONNX graph, not symbolic. A two-second utterance therefore costs the same encoder pass as a thirty-second one, and `--metrics` reports RTF against that window rather than the utterance. Measured on a 16-core Ryzen 9 9950X, that is ~1.1 s per utterance (rtf 0.04) against Parakeet's 64 ms on the same audio — usable for `--file` work and slow dictation. Live streaming (below) is the way to use it live: it patches the encoder so the window follows the audio. The single-graph `decoder_model_merged` is deliberately not used: it re-runs cross-attention over all 1500 encoder frames per generated token, so the prompt decoder plus the cached-token decoder are shipped instead.

Roughly 97 % of Whisper's inference time is inside ONNX Runtime's own kernels (GEMM, quantized GEMM, attention softmax), and about 80 % of it is the one encoder pass — the exported graph pins that window, so without the in-memory patch no shorter window can be fed. Each generated token adds ~8 ms on top. ORT session settings were measured against this (memory pattern, denormals-as-zero, fewer threads for the per-token session) and none of them moved the total; see the table on `init_session` in `src/whisper_core/engine.rs`.

### Live streaming

Streaming is always on. It does two things:

1. **Partial transcripts (microphone mode).** While you speak, sttd decodes the still-open utterance about once a second and logs it at `info` level (`utt_000012: partial (3.0 s audio, 303 ms): ...`). The final transcript, the gates, the clipboard and paste are unchanged: only the final reaches them. The worker keeps only the newest snapshot, so a model slower than one step skips steps instead of falling behind, and a closed utterance always goes before a waiting partial (it waits at most one decode). Works with both engines.
2. **A short Whisper encoder window.** whisper.cpp does this with `audio_ctx`, and so does sttd, as `--audio-ctx`. The exported encoder pins 30 s in two places (128 `Reshape` constants and the 1500-row position table), so sttd patches the graph in memory when it loads (`src/whisper_core/dyn_encoder.rs`; nothing is written to the model directory) and then picks the window per utterance: the audio rounded up to a whole second, plus 1 s, never below the `--audio-ctx` floor (**1500** frames = 15 s by default) and never above 30 s. If the graph does not match what the patch expects, sttd warns and keeps the fixed 30 s window. `--file` gets the short window only (no partials offline).

Measured on a 16-core Ryzen 9 9950X with three espeak Polish clips (1.4 s, 6 s and 10 s of speech; this is synthetic speech, so check it on your own voice with `tools/eval.py`):

| Encoder window | Per-utterance time | Finals identical to the fixed 30 s run |
| --- | --- | --- |
| 30 s (patch failed) | 1.07 - 1.46 s | reference |
| 15 s (default) | 0.42 - 0.71 s | 3 of 3 |
| 10 s (tried, not used) | 0.25 - 0.60 s | 1 of 3 (`dziękuję` became `jankruje`) |

A window cut close to the audio is worse than a short one: with no margin a 6 s clip repeated itself and under ~3 s of speech the decode looped, so the floor and the margin are not tuning knobs to shave. With the short window the decode also gets a token budget tied to the audio length (20 + 15 tokens per second, at most 448), which bounds a looping decode.

`--audio-ctx` / `STTD_AUDIO_CTX` moves that floor: it is an even mel-frame count (100 frames = 1 s of 16 kHz audio) in `100..=3000`, and the window is never shorter than it. Lower is cheaper per utterance and worse per word: the 10 s row above is what `--audio-ctx 1000` gives, and `--audio-ctx 100` on a 1.2 s utterance leaves a 3 s window in which the decode loops (the repetition gate then drops the utterance rather than emitting the loop). Whisper only; Parakeet sizes its own window and reports the flag as ignored.

In a simulated live run (a 24 s file with pauses, streamed through the real VAD and model, 15 s floor) partials came one per second, each 0.41 - 0.69 s to decode, and the three finals matched the fixed-window run. A machine that needs more than a second per decode simply skips steps. Known limit: a partial is cut mid-word, so it can misspell words that the final gets right (`po zągiem` for `pociągiem`).

Without the `cuda` feature both engines run on CPU. With it, the Parakeet and Whisper encoders run on the CUDA execution provider and the decoders stay on CPU, TF32 is off, and transcripts are therefore not bit-exact against a CPU run. `--gpu` selects the CUDA path and is accepted without a warning in such a build; in a CPU-only build it is accepted and warned about for both engines.

Resolution (`src/model_store.rs`):

- `--model SPEC|PATH`. A catalog spec resolves into the models directory and downloads if missing. A path to an existing directory passes through untouched, and which engine it holds is decided by its files (`decoder_joint-model.int8.onnx` vs `decoder_with_past_model_int8.onnx`); a directory with neither, or with both, is rejected.
- Models directory is baked at compile time from `STTD_MODELS_DIR` (see `build.rs:10-18`). Default is `models` (dev checkout). Packaged builds use `/usr/share/sttd/models`.
- Setting `STTD_MODELS_DIR` at runtime does nothing. Use `--model <path>` to override.

List models, voices and devices:

```sh
sttd --list-models
sttd --list-devices
sttd --list-devices --device usb
sttd --list-voices              # every piper voice (~1300 rows)
sttd --list-voices pl_PL        # filtered by key or language
sttd --list-voices german
```

## Spoken emission

`--speak` reads every accepted transcript aloud through the default output device, using a [piper](https://github.com/rhasspy/piper) voice. It is a fourth emission channel next to stdout, the clipboard and the paste, and it runs on its own thread so the emit queue behind it keeps draining.

```sh
sttd --speak --autopause                 # single-shot: speak, wait, speak
sttd --speak --voice en_US-amy-medium
sttd --speak --voice /opt/voices/my-voice   # a voice directory already on disk
```

The voice is fetched on first use from `huggingface.co/rhasspy/piper-voices` into `<models>/voices/<key>/` (the `.onnx` and its `.onnx.json`, ~60 MiB for a `medium`). `--voice` takes any key that repository publishes, and any of its older aliases (`pl-mls_6892-low` for `pl_PL-mls_6892-low`); `--list-voices` prints them all with sizes and an `I` marker for the ones already downloaded. Two requirements and one limit:

- **`espeak-ng` must be installed** (with its voice data package). sttd shells out to `espeak-ng -v <voice> --ipa -q`, which is what piper's own bundled phonemizer calls; keeping the binary outside the process leaves espeak-ng's GPL-3.0 out of this MIT binary and keeps cmake and bindgen off the build. A missing binary is a startup error, not a silent no-op.
- **`phoneme_type: "text"` voices are refused.** They take raw text with no espeak voice, so there is nothing to phonemize. Multi-speaker voices are refused too (they need a speaker choice `--voice` does not offer); `--list-voices` marks them with `*` after the speaker count.
- **The microphone is held while sttd speaks.** Otherwise it transcribes itself, and on a laptop mic that becomes a loop. The hold is the same pipeline hold as a pause, on its own flag (the pause flag is yours — setting it would trigger an unload or an autopause that has nothing to do with this). The cost is explicit: **what you say during playback is discarded, not queued.** So `--speak` pairs with `--autopause` (each resume arms one sentence), or with headphones. Without either, dictation and playback eat each other — sttd warns about this at startup.

Implementation (`src/piper.rs`, `src/voice_store.rs`): one `ort` session per voice, run directly, the same pattern as `parakeet_core` and `whisper_core`. No TTS framework sits in between, because none fits this build — `piper-rs` and `dengjen-piper-rs` pin `ort` to an exact older release, `piper-tts-rs` is GPL-3.0, and `piper-plus` is a fork with its own phonemizer and model extensions. The graph is small and fixed (`input`, `input_lengths`, `scales` = noise/length/noise_w, plus `sid` on multi-speaker models), so the work is the phoneme string and the contract, both of which follow upstream piper exactly: one clause per espeak line, the clause terminator re-attached (espeak's `--ipa` drops it, and it is what carries the intonation), `(lang)` switches stripped, the NFD code-point split, the `[BOS, PAD, (ids…, PAD)*, EOS]` id layout, and piper's peak normalize with its `0.01` floor.

## Translation

`--translate <LANG>` runs each accepted transcript through an ONNX translation model before emission, so stdout, the clipboard, auto-paste and `--speak` all receive the translated text. The gates, the dictionary and `clean` keep seeing the source text.

```sh
sttd --translate en                          # English output, Polish source (the primary path)
sttd --language auto --translate en          # English output, any source language
sttd --translate de --translate-model /opt/translators/mine
```

Two kinds of model sit behind `--translate-model`, and they are told apart by each export's own `config.json` rather than by its name — so a directory you built yourself works either way.

### Pair translators (the default)

[MarianMT](https://huggingface.co/docs/transformers/model_doc/marian) `opus-mt` exports, one repository per language pair, fetched on first use from `huggingface.co/Xenova/opus-mt-<src>-<tgt>` into `<models>/translators/<spec>/` (two int8 graphs, the tokenizer and `config.json`, ~114 MiB per pair).

**The pair comes from `--language` plus `--translate`, and coverage is thin where you might expect it to be wide.** These are per-pair models, so a direction that nobody exported does not exist:

| | Languages |
|---|---|
| `--language pl --translate en` | works — the primary path |
| `--language <code> --translate en` | 25 source languages, including `de`, `fr`, `es`, `ru`, `uk`, `zh` |
| `--language en --translate <code>` | 20 target languages |
| `--language pl --translate <anything else>` | **fails**, naming what Polish can reach (`en`) and the multilingual option below |
| `--language auto --translate en` | uses the many-sources-into-English model |
| `--language auto --translate <not en>` | fails: no many-source model exists for that target |

Polish is the sore spot: `Helsinki-NLP` publishes ten `opus-mt-pl-*` models (`pl-de`, `pl-fr`, `pl-es`, `pl-uk`, `pl-sv`, `pl-no`, `pl-ar`, `pl-eo`, `pl-lt`, `pl-en`) but only `pl-en` has been converted to ONNX, and there is no `X-pl` pair in either direction. sttd refuses rather than mistranslating, and the error names the alternative.

### Multilingual translators

`--translate-model nllb-200-distilled-600M` is one model for **every** direction, which removes the table above. It is ~870 MiB (both graphs are about four times a pair's) and translates roughly 8x slower per utterance.

```sh
sttd --language pl --translate de --translate-model nllb-200-distilled-600M
sttd --language pl --translate ja --translate-model nllb-200-distilled-600M
```

| | Languages |
|---|---|
| `--language <code> --translate <code>` | any of ~200, in both directions, including `pl → de/fr/es/uk/...` |
| `--language auto` | **fails**: its source is an input, and `auto` does not name one |

Three things to know before using one:

- **It needs a named `--language`,** unlike a pair export. The source language is something the model is *told*, not something its weights decide, so `--language auto` has nothing to tell it.
- **It is slower, and much bigger.** ~800 ms per utterance on four threads against ~100 ms for a pair, with the same uncached O(n²) decode over a decoder four times the width — which is why its token cap is 48 against a pair's 96. If you translate continuously, the latency is the cost of the coverage.
- **Under Parakeet, `--language` does not name what it is transcribed in.** Parakeet detects the spoken language per utterance and ignores `--language` for transcription; for a multilingual translator the flag only tells the *translator* what to assume. sttd warns about this at startup, because speech in another language will then be mistranslated.

NLLB spells languages the FLORES way (`pol_Latn`, `eng_Latn`, `deu_Latn`). `--language` and `--translate` take the ISO-639-1 codes sttd uses everywhere else and sttd maps them; a FLORES code also passes through verbatim, so all ~200 are reachable. A code neither form matches is refused with the examples — deliberately, since NLLB's 256k-entry BPE vocabulary contains ordinary subword pieces that look like language codes, and passing a typo through would mistranslate silently.

### Both

Four properties are worth knowing:

- **The step runs after the gates.** A dropped utterance is never translated, and a hallucination or repetition gate judges the source text, never the translation.
- **A failure emits the source.** If the model returns nothing or the decode fails, sttd warns and emits the source text unchanged, so a translation fault never costs you the sentence.
- **Translation runs on its own thread**, behind a 4-deep FIFO, so it never adds its 60–150 ms (or ~800 ms) to the gap between utterances. A full queue drops the *oldest* waiting transcript and says so, rather than the newest.
- **512 tokens is a quality window, not a shape limit.** Text is split on sentence terminators (`.` `!` `?` `;` newline, not `:`) and each piece is truncated to 512 tokens, so quality falls off past that. Text with no terminator stays one piece.

**Eval runs must not use `--translate`.** `tools/eval.py` scores sttd's `--stdout` lines against each reference `.txt`, and with `--translate` those lines hold the translation. The WER then measures the translation model, not the speech model. If you do run with `--translate`, the source transcript is in the `src_text` column of the `--metrics` rows, not in the scored text.

With `--speak`, a `--voice` whose key language differs from the target warns too (`pl_PL-*` reads English badly). The comparison uses the code you typed, so a multilingual run into `eng_Latn` is matched against a `pl_PL-*` voice correctly.

Implementation (`src/translate.rs`, `src/translator_store.rs`): two `ort` sessions (encoder plus the merged decoder, the same pattern as `parakeet_core` and `whisper_core`) with a greedy decode and no prompt. The decode cannot use the KV cache (the export's cached branch is broken), so each step re-runs the decoder over the whole sequence; that makes it O(n²) in output length, which is what the token cap bounds. Measured on four threads: 100 ms for a `opus-mt-pl-en` pair, 763 ms for NLLB `pl → en`, 817 ms for `pl → de`.

Four details the models decide rather than the code, all read from each export's own `config.json`:

- **Which kind of model it is,** from `model_type` (`marian` vs `m2m_100`). This is why a `--translate-model` *directory* works for either family: the family comes from the file, not from the path.
- **The decoder geometry** — 6 layers of 8 heads for a Marian pair, 12 of 16 for NLLB — and with it the cache shape, 24 slots against 48.
- **The opening token, which differs per export.** A pair opens on its own `decoder_start_token_id` (`pl-en` 63429, `mul-en` 64171, both the pad id). A multilingual export opens on `[start, target code]`; measured against the weights, opening on the target code alone decodes `"is"`, and omitting the start decodes nothing at all. This is also why NLLB declaring `decoder_start_token_id == eos_token_id == 2` is legal: the start token *is* the `</s>` decoder prefix.
- **The pad id must be suppressed** — the int8 `mul-en` export pins its pad logit at `0.000` while every real token sits near `-40`, so a plain argmax emits `<pad>` forever and the translation comes out empty. A multilingual export suppresses every added token except `</s>`, which is all 202 language codes among them: a language code mid-sentence is the model switching language on its own.

## Build

```sh
cargo build --release
cargo build --release --features denoise
cargo build --release --features cuda
STTD_MODELS_DIR=/usr/share/sttd/models cargo build --release
```

Notes:

- `.cargo/config.toml:1-4` sets `-C target-cpu=native`.
- `ort` downloads ONNX Runtime binaries at build time (`download-binaries`).
- `denoise` pulls DeepFilterNet3 (`tract` backend).
- `cuda` needs CUDA 13 and cuDNN 9 at runtime, not only at build time. The ONNX Runtime provider library must sit beside the binary, so copy it in and run the linker step that adds an `$ORIGIN` rpath; a provider found on the system loader path alone does not survive a move of the binary. The TensorRT execution provider ships in the same distribution but is never registered.
- The vendored ONNX Runtime 1.28 CUDA provider ships no kernel for Blackwell `sm_120` GPUs, such as the RTX 5070, so `--gpu` fails there with CUDA error 209 and falls back only when no CUDA device is visible. Tested on an RTX 5070.
- Under CUDA, `--threads` sets ORT `intra_op` for the CPU-resident nodes only (the decoders); the CUDA nodes ignore it.

Test:

```sh
cargo test
cargo test --release --frozen
```

No clippy/rustfmt config. No CI config. Tests live inline (`#[cfg(test)]`, 392 tests).

Five Whisper tests read the downloaded model's `tokenizer.json` and skip themselves when it is absent, so they cover the byte-level BPE handling and the compiled prompt token ids on a machine that has the weights, and leave `cargo test` runnable on a clean checkout.

The piper tests are the same kind: pure over their inputs. The espeak phoneme string is asserted against output read off this machine (espeak 1.52.0), the id layout against upstream's, and the voice index against a trimmed fixture of the real `voices.json`. The `espeak-ng` process and the ORT session are the untested glue, as the speech engines' device I/O is.

## Use

Show help (full reference; every flag has `STTD_*` env fallback, CLI wins):

```sh
cargo run --release -- --help
sttd --help
```

Live microphone (starts paused by default; resume with signal):

```sh
sttd --stdout
sttd --stdout --no-start-paused
sttd --device usb --channel auto --stdout
sttd --copy --paste --stdout
```

Offline file (same pipeline, fixed WAV input; ignores pause flags):

```sh
sttd --file speech.wav --stdout
sttd --file speech.wav --speak              # transcribe a recording aloud
```

Daemon:

```sh
sttd --start
sttd --status
sttd --stop
sttd --start --copy --paste --autopause
```

Paste last emission again:

```sh
sttd --paste-last
```

Tuning example:

```sh
sttd --file a.wav --stdout \
  --hangover-ms 500 \
  --vad-threshold 0.6 \
  --min-rms 0.005 \
  --min-density 0.3
```

Eval and regression corpus (stdlib Python only):

```sh
tools/eval.py --corpus corpus/ --output eval-out/
tools/eval.py --corpus corpus/ --output eval-out/ --sweep "STTD_MIN_DENSITY=0.2,0.3"
tools/dump2corpus.py --dumps dumps/2026-09-22 --corpus corpus/regression/
tools/eval.py --self-test
tools/dump2corpus.py --self-test
```

Arch package (local build from this checkout):

```sh
cd packaging && makepkg
```

## Configuration

Precedence: CLI flag > `STTD_*` env > compiled default. See `src/cli.rs:26-351`.

Key tunables:

| Flag / Env | Default | Effect |
| --- | --- | --- |
| `--model` / `STTD_MODEL` | `parakeet` | Catalog spec or model directory path. |
| `--language` / `STTD_LANGUAGE` | system locale language | Any language code Whisper supports (`pl`, `en`, `de`, `yue`, ...), or `auto` (per-utterance detection). Unset: the system locale's language (`LC_ALL`, `LC_MESSAGES` or `LANG`), or `auto` where that names none. A code the model has no token for fails the load; a system locale the model cannot decode warns and detects instead. Whisper only; Parakeet auto-detects and warns that the flag is ignored. |
| `--audio-ctx` / `STTD_AUDIO_CTX` | `1500` | Floor of that window, in mel frames (100 = 1 s), even, `100..=3000`. Whisper only; Parakeet sizes its own window and reports the flag as ignored. See *Live streaming*. |
| `--device` / `STTD_DEVICE` | unset | Case-insensitive substring match on input device. |
| `--channel` / `STTD_CHANNEL` | `mix` | `mix`, `auto` (loudest), or channel index `N`. |
| `--denoiser` / `STTD_DENOISER` | `none` | `none` or `deepfilter` (needs `denoise` feature). VAD branch only. |
| `--threads` / `STTD_THREADS` | `0` (auto) | Decoder threads; `0` = physical cores. |
| `--gpu` / `STTD_GPU` | off | Load the model on the CUDA EP. Needs a `cuda`-feature build: encoders run on CUDA, decoders stay on CPU, TF32 is off, so transcripts are not bit-exact against CPU. Without that feature the flag is accepted and warned about, because both engines run on CPU. Not usable on Blackwell `sm_120` GPUs (RTX 5070), which the vendored ONNX Runtime 1.28 CUDA provider does not support. |
| `--vad-threshold` / `STTD_VAD_THRESHOLD` | `0.5` | Min earshot score for speech. |
| `--vad-adapt` / `STTD_VAD_ADAPT` | on | Adapt threshold to noise floor with hysteresis. |
| `--min-rms` / `STTD_MIN_RMS` | `0.003` | Min speech-only RMS to keep utterance. `0` disables. |
| `--min-density` / `STTD_MIN_DENSITY` | `0.25` | Min VAD-speech share; utterances over ~3 s skip it. `0` disables. |
| `--hangover-ms` / `STTD_HANGOVER_MS` | `800` | Silence before close, 200-2000 ms, step 10. |
| `--target-rms` / `STTD_TARGET_RMS` | `0.04` | AGC target; `0` disables gain. |
| `--dictionary` / `STTD_DICTIONARY` | unset | Custom vocabulary file, one phrase per line. Applies only to accepted transcripts. |
| `--remove-hesitations` / `STTD_REMOVE_HESITATIONS` | off | Strip fillers (`yyy`, `eee`, `um`, ...). Language-agnostic. |
| `--stdout` / `STTD_STDOUT` | off | Print one line per accepted utterance. |
| `--speak` / `STTD_SPEAK` | off | Read each accepted transcript aloud through the default output device (needs `espeak-ng`). Holds the microphone while it speaks, so what you say during playback is discarded — pair it with `--autopause`, or use headphones. Independent of `--sound`. |
| `--voice` / `STTD_VOICE` | `pl_PL-gosia-medium` | Piper voice key or a voice directory. Downloaded on first use (~60 MiB). Ignored, and not downloaded, without `--speak`. |
| `--translate` / `STTD_TRANSLATE` | unset | Translate each accepted transcript into `<LANG>` before emission. Unset: no translation step and no translator download. Every emission channel then receives the translation. The direction comes from `--language` plus this flag; with the default pair translator only exported pairs exist (see *Translation*). |
| `--translate-model` / `STTD_TRANSLATE_MODEL` | unset | Override the resolved direction: a catalog spec (`opus-mt-mul-en`, or `nllb-200-distilled-600M` for every direction at ~870 MiB), or a directory holding `encoder_model_quantized.onnx`, `decoder_model_merged_quantized.onnx`, `tokenizer.json` and `config.json`. Downloaded on first use. Ignored, and not downloaded, without `--translate`. |
| `--copy` / `STTD_COPY` | off | Copy each emission via `wl-copy`. |
| `--paste` / `STTD_PASTE` | off | Paste via `ydotool`. Runs only when copy runs on fresh text. |
| `--paste-mode` / `STTD_PASTE_MODE` | `ctrl-v` | `ctrl-v`, `ctrl-shift-v`, or `type`. |
| `--dump-dir` / `STTD_DUMP_DIR` (`-o`) | unset | Write WAV+TXT per utterance; keeps newest 100 pairs. TXT for drops starts with `DROPPED(<reason>)`. |
| `--metrics` / `STTD_METRICS` | unset | Append JSONL per utterance (timing, RTF, evidence, verdict, text). With `--translate`, `text` holds the translation and `src_text` the source. |
| `--start-paused` / `STTD_START_PAUSED` | on | Start paused; file mode ignores it. |
| `--autopause` / `STTD_AUTOPAUSE` | off | Pause again after each emission; each resume arms one sentence. The pause flushes the utterance buffer first, so speech made while the model was decoding is not lost. |
| `--no-dbus` / `STTD_NO_DBUS` | off | Skip the session-bus service (both interfaces). Signals and `--stop` keep working. File mode never starts it. |
| `--auto-unload` / `STTD_AUTO_UNLOAD` | off | Unload model while paused to free VRAM. |
| `--auto-unload-after` / `STTD_AUTO_UNLOAD_AFTER` | `180` s | Pause duration before unload; `0` = immediate. |
| `--sound` / `STTD_SOUND` | on | Pause/resume/ready tones on default output. |
| `--start` / `STTD_START` | off | Fork to background daemon. |
| `--pid-file` / `STTD_PID_FILE` | `<tmp>/sttd.pid` | Daemon pid file. |
| `--log-file` / `STTD_LOG_FILE` | `<tmp>/sttd.log` | Daemon combined log. |
| `RUST_LOG` | `info` | Log level. |

Bool parser accepts `1/0`, `true/false`, `yes/no`, `on/off`.

## Runtime control

Signals (microphone mode, `src/main.rs:458-486`):

| Signal | Effect |
| --- | --- |
| `SIGUSR1` | Pause; the utterance buffer is flushed first (see below). |
| `SIGUSR2` | Resume; with `--autopause` arms one sentence. |
| `SIGHUP` | Toggle pause/run; pausing flushes the utterance buffer first. |
| `SIGTERM` | Graceful stop, flush trailing speech. |
| `Ctrl-C` | Same as `SIGTERM`. |

Pausing never loses the last sentence: the pause edge cuts the open utterance
buffer and transcribes it while the run is held, so it reaches stdout, the
clipboard and `--dump-dir` right away. Anything else waiting in the model queue
at that moment is dropped unless it closed within the last 2 s (`asr.rs`'s
`keep_while_paused`): a queue that far behind is a real backlog, and transcribing
it later would paste stale text into whatever window has focus.

Daemon files default to `<tmp>/sttd.pid`, `<tmp>/sttd.log`, `<tmp>/sttd-last.txt`.

systemd user unit (`packaging/sttd.service`):

```sh
systemctl --user enable --now sttd
systemctl --user kill -s SIGUSR2 sttd   # arm one sentence
systemctl --user kill -s SIGHUP sttd    # toggle
systemctl --user kill -s SIGUSR1 sttd   # pause
```

The unit runs foreground (`ExecStart=/usr/bin/sttd --autopause --copy 1 --paste 1`). It starts paused by default. It needs `WAYLAND_DISPLAY` for copy/paste (`systemctl --user import-environment WAYLAND_DISPLAY`).

D-Bus (microphone mode, unless `--no-dbus`): one bus name `org.sttd.Daemon` with two interfaces on separate object paths. A missing session bus is not fatal: the service retries in the background and the signals above keep working. File mode (`--file`) never starts the service.

| Interface | Object path | Members |
| --- | --- | --- |
| `org.sttd.Daemon` | `/org/sttd/Daemon` | `Pause`, `Resume`, `Toggle`, property `Paused`, signal `StateChanged` |
| `org.sttd.Dictionary` | `/org/sttd/Dictionary` | `LoadFile`, `ReplaceWords`, `AddWords`, `RemoveWords`, `GetWords`, `Contains`, properties `Count` and `Dirty`, signal `DictionaryChanged` |

Pause control (`Pause` and `Toggle` flush the utterance buffer, same as
`SIGUSR1`/`SIGHUP` above):

```sh
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Pause
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Resume
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Toggle
busctl --user get-property org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Paused
```

Live dictionary:

```sh
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary LoadFile s /path/to/words.txt
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary ReplaceWords as 2 AppImage Syntocinon
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary AddWords as 1 Zoxide
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary RemoveWords as 1 zoxide
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary Contains s appimage
busctl --user call org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary GetWords
busctl --user get-property org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary Count
busctl --user get-property org.sttd.Daemon /org/sttd/Dictionary org.sttd.Dictionary Dirty
busctl --user monitor org.sttd.Daemon
```

`LoadFile` takes a real path. An empty or whitespace-only path is rejected with `(false, count, "empty path")`; it never reloads anything. To reload the current file, read the path from `GetWords` and pass it back to `LoadFile`. A failed load keeps the previous words and returns `(false, old_count, message)`.

`ReplaceWords` swaps the whole list in one step. `AddWords` merges and returns `(count, added)`; a normalized collision with a live entry keeps the live entry, so `added` is 0 for a pure re-case such as `appimage` against a loaded `AppImage`. `RemoveWords` drops by normalized form and returns `(count, removed)`. `Contains` compares case- and accent-insensitively, so `ZOLC`, `żółć` and `ŻÓŁĆ` are one word. Lowercasing runs before diacritic folding, so an uppercase Polish spelling is a duplicate of the lowercase one rather than a second entry.

`Dirty` reports whether the live list still matches the source file. `LoadFile` clears it. `ReplaceWords` always sets it, because the server cannot know that a caller-supplied list matches the file. `AddWords` and `RemoveWords` set it only when they actually changed the list. A reload therefore discards earlier edits, and `Dirty` is how a client detects that in advance.

`Count` and `Dirty` are point reads: they return the current value, and they do not push. A client that tracks changes subscribes to `DictionaryChanged(count, ok, message)`, whose first argument is the count after every mutation.

`ReplaceWords`, `AddWords`, `RemoveWords` and `LoadFile` sanitize with the same rules as the file path: entries with digits and entries of more than two words are skipped, duplicates (case/accent-insensitive) are dropped. `DictionaryChanged` fires after each of them, including a failed load.

Any process running as the same user can call these methods, so the dictionary is only as trustworthy as the session. `Count` is uncapped, because the caps apply in post-correction and not at swap time. Dictionary rules and caps are unchanged from `--dictionary` (see `src/dictionary.rs`): above `POSTCORR_WARN_DICT_WORDS` (50) false positives grow, and at `POSTCORR_MAX_DICT_WORDS` (100) fuzzy correction is off while exact casing restore stays on.

## Gates

Acoustic gates run before the model (`src/gate.rs`, `src/utterance.rs`):

- Minimum 200 ms VAD speech, minimum RMS, minimum density (short utterances only).
- Trim to last speech plus adaptive 120-300 ms tail.

Transcript gates run after the model:

- Known hallucination boilerplate (exact plus fuzzy for long patterns, Polish diacritics folded).
- Repetition loops (unique-ratio plus bigram loops).
- Optional dictionary correction and hesitation stripping (accepted transcripts only).

## Project layout

```text
src/main.rs          pipeline doc, dispatch, mic/file modes
src/cli.rs           clap flags + STTD_* env
src/config.rs        constants, Parakeet catalog, parsers
src/capture.rs       cpal input, device list
src/signal.rs        VAD/model branch topology
src/frame.rs         FrameProcessor: Hp/Agc + VAD + buffering
src/vad.rs           earshot 16 ms windows to 10 ms frames
src/noise.rs         noise floor + hysteresis
src/utterance.rs     pre-roll, UtteranceBuffer, splits
src/asr.rs           worker thread, conditioning, metrics
src/parakeet.rs      SpeechModel impl (Parakeet)
src/parakeet_core/   ONNX engine, mel frontend, booster
src/whisper.rs       SpeechModel impl (Whisper), decoding language
src/whisper_core/    ONNX engine, log-mel frontend, KV-cached decode, in-memory encoder patch (dyn_encoder)
src/mel.rs           Slaney mel filterbank, shared by both frontends
src/decode.rs        argmax/softmax/word confidences, shared by both engines
src/gate.rs          acoustic/transcript verdicts
src/emit.rs          stdout/clipboard/paste/speak/dump/last-text
src/piper.rs         piper voice: config, espeak phonemes, ort VITS run
src/voice_store.rs   voice index, --voice resolution, voice download
src/translate.rs     seq2seq translator (both families) + the translation thread
src/translator_store.rs  translator catalog and families, resolution, download, config read
src/dictionary.rs    custom vocabulary
src/clean.rs         filler stripping
src/daemon.rs        start/stop/status
src/dsp.rs, src/hp.rs, src/agc.rs, src/denoise.rs, src/sound.rs, src/wav.rs
tools/eval.py        WER batch eval + threshold sweep
tools/dump2corpus.py build eval corpus from --dump-dir
packaging/           PKGBUILD, sttd.service
models/              gitignored weights (default int8 + silero legacy file)
models/voices/       downloaded piper voices + the cached voices.json index
models/translators/  downloaded translators (--translate)
CONTEXT.md           domain vocabulary
```

## License

MIT. See `LICENSE`.
