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
- Emission to stdout, Wayland clipboard (`wl-copy`), auto-paste (`ydotool`), WAV+TXT dump, last-text file, JSONL metrics.
- Offline file mode, daemon mode, systemd user unit, pause/resume signals.
- Optional VAD-branch denoiser (DeepFilterNet3, `denoise` feature).
- Custom dictionary for post-gate correction.

## Pipeline

```text
cpal (hw rate, N ch) -> channel pick|mix -> 10 ms frames
  -> VAD branch: resample to 16 kHz -> earshot -> verdict per frame
  -> model branch: resample to 16 kHz (raw) -> utterance buffer
  -> rtrb queue -> ASR worker -> conditioning -> Parakeet -> gates -> emitter
```

Details:

- Default path resamples hardware rate straight to 16 kHz (`direct hop`). With `--denoiser deepfilter`, the VAD branch detours through 48 kHz, denoise, then 16 kHz. The model branch always stays raw.
- An utterance opens after 3 consecutive speech frames. Pre-roll prepends run-up audio so quiet word starts survive.
- An open utterance closes after `--hangover-ms` of silence (default 800 ms). Only trimmed audio plus a short tail reaches the model. Long speech emits interim splits past 8 s and force-splits past 28-30 s.
- Conditioning removes DC, applies 5 ms fades, and guards peaks. `Hp` removes 80 Hz hum. `Agc` lifts quiet speech toward the target RMS.
- Drops never emit. Emits go to the configured emitter.

See `src/main.rs:1-51` and `CONTEXT.md` for vocabulary (`utterance`, `VAD verdict`, `gate`, `hallucination`, `emission`, `model spec`, `VAD/model branch`, `direct hop`).

## Requirements

- Linux x86_64, Rust stable, `alsa-lib`.
- ~638 MiB disk for the default Parakeet `int8` model, or ~1443 MiB for Whisper.
- Optional: `wl-clipboard` (`wl-copy`) for `--copy`, `ydotool` + `ydotoold` for `--paste` (Wayland only), PipeWire/WirePlumber for capture, Python 3 stdlib for `tools/`.

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

List models and devices:

```sh
sttd --list-models
sttd --list-devices
sttd --list-devices --device usb
```

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

No clippy/rustfmt config. No CI config. Tests live inline (`#[cfg(test)]`, 316 tests).

Five Whisper tests read the downloaded model's `tokenizer.json` and skip themselves when it is absent, so they cover the byte-level BPE handling and the compiled prompt token ids on a machine that has the weights, and leave `cargo test` runnable on a clean checkout.

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
| `--copy` / `STTD_COPY` | off | Copy each emission via `wl-copy`. |
| `--paste` / `STTD_PASTE` | off | Paste via `ydotool`. Runs only when copy runs on fresh text. |
| `--paste-mode` / `STTD_PASTE_MODE` | `ctrl-v` | `ctrl-v`, `ctrl-shift-v`, or `type`. |
| `--dump-dir` / `STTD_DUMP_DIR` (`-o`) | unset | Write WAV+TXT per utterance; keeps newest 100 pairs. TXT for drops starts with `DROPPED(<reason>)`. |
| `--metrics` / `STTD_METRICS` | unset | Append JSONL per utterance (timing, RTF, evidence, verdict, text). |
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
src/emit.rs          stdout/clipboard/paste/dump/last-text
src/dictionary.rs    custom vocabulary
src/clean.rs         filler stripping
src/daemon.rs        start/stop/status
src/dsp.rs, src/hp.rs, src/agc.rs, src/denoise.rs, src/sound.rs, src/wav.rs
tools/eval.py        WER batch eval + threshold sweep
tools/dump2corpus.py build eval corpus from --dump-dir
packaging/           PKGBUILD, sttd.service
models/              gitignored weights (default int8 + silero legacy file)
CONTEXT.md           domain vocabulary
```

## License

MIT. See `LICENSE`.
