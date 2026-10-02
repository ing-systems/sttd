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
- ~640 MiB disk for the default `int8` model, ~2.4 GiB for `fp32`.
- Optional: `wl-clipboard` (`wl-copy`) for `--copy`, `ydotool` + `ydotoold` for `--paste` (Wayland only), PipeWire/WirePlumber for capture, Python 3 stdlib for `tools/`.

## Models

Catalog (`src/config.rs:253-277`):

| Spec | Size | Contents |
| --- | --- | --- |
| `parakeet` (alias for `int8`) | 640 MiB | encoder int8 + decoder_joint int8 + vocab + tokenizer |
| `parakeet-tdt-0.6b-v3-int8` | 640 MiB | same as above |
| `parakeet-tdt-0.6b-v3-fp32` | 2.4 GiB | encoder fp32 + data + decoder + vocab + tokenizer |

Weights come from `huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx` and `huggingface.co/nvidia/parakeet-tdt-0.6b-v3` (tokenizer only).

Resolution (`src/config.rs:210-221`, `src/model_store.rs`):

- `--model SPEC|PATH`. A catalog spec resolves into the models directory and downloads if missing. A path to an existing directory passes through untouched.
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
STTD_MODELS_DIR=/usr/share/sttd/models cargo build --release
```

Notes:

- `.cargo/config.toml:1-4` sets `-C target-cpu=native`.
- `ort` downloads ONNX Runtime binaries at build time (`download-binaries`).
- `denoise` pulls DeepFilterNet3 (`tract` backend).

Test:

```sh
cargo test
cargo test --release --frozen
```

No clippy/rustfmt config. No CI config. Tests live inline (`#[cfg(test)]`, 267 tests).

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
| `--device` / `STTD_DEVICE` | unset | Case-insensitive substring match on input device. |
| `--channel` / `STTD_CHANNEL` | `mix` | `mix`, `auto` (loudest), or channel index `N`. |
| `--denoiser` / `STTD_DENOISER` | `none` | `none` or `deepfilter` (needs `denoise` feature). VAD branch only. |
| `--threads` / `STTD_THREADS` | `0` (auto) | Decoder threads; `0` = physical cores. |
| `--gpu` / `STTD_GPU` | off | Load model on GPU. |
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
| `--autopause` / `STTD_AUTOPAUSE` | off | Pause again after each emission; each resume arms one sentence. |
| `--no-dbus` / `STTD_NO_DBUS` | off | Skip the session-bus service. Signals and `--stop` keep working. File mode never starts it. |
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
| `SIGUSR1` | Pause. |
| `SIGUSR2` | Resume; with `--autopause` arms one sentence. |
| `SIGHUP` | Toggle pause/run. |
| `SIGTERM` | Graceful stop, flush trailing speech. |
| `Ctrl-C` | Same as `SIGTERM`. |

Daemon files default to `<tmp>/sttd.pid`, `<tmp>/sttd.log`, `<tmp>/sttd-last.txt`.

systemd user unit (`packaging/sttd.service`):

```sh
systemctl --user enable --now sttd
systemctl --user kill -s SIGUSR2 sttd   # arm one sentence
systemctl --user kill -s SIGHUP sttd    # toggle
systemctl --user kill -s SIGUSR1 sttd   # pause
```

The unit runs foreground (`ExecStart=/usr/bin/sttd --autopause --copy 1 --paste 1`). It starts paused by default. It needs `WAYLAND_DISPLAY` for copy/paste (`systemctl --user import-environment WAYLAND_DISPLAY`).

D-Bus (microphone mode, unless `--no-dbus`): bus name `org.sttd.Daemon`, object path `/org/sttd/Daemon`, interface `org.sttd.Daemon`. A missing session bus is not fatal: the service retries in the background and the signals above keep working. File mode (`--file`) never starts the service.

```sh
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Pause
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Resume
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Toggle
busctl --user get-property org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon Paused
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon ReloadDictionary s ""
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon ReloadDictionary s /path/to/words.txt
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon SetWords as 2 AppImage Syntocinon
busctl --user call org.sttd.Daemon /org/sttd/Daemon org.sttd.Daemon GetDictionary
busctl --user monitor org.sttd.Daemon
```

`ReloadDictionary` with an empty path reloads the startup `--dictionary` file. A failed load keeps the previous words and returns `(false, old_count, message)`. `SetWords` sanitizes with the same rules as the file path: entries with digits and entries of more than two words are skipped, duplicates (case/accent-insensitive) are dropped.

Signals: `StateChanged(paused)` on every pause state change (D-Bus call or signal handler), `DictionaryChanged(count, ok, message)` after `ReloadDictionary` and `SetWords`, including a failed reload.

Any process running as the same user can call these methods, so the dictionary is only as trustworthy as the session. Dictionary rules and caps are unchanged from `--dictionary` (see `src/dictionary.rs`): above `POSTCORR_WARN_DICT_WORDS` (50) false positives grow, and at `POSTCORR_MAX_DICT_WORDS` (100) fuzzy correction is off while exact casing restore stays on.

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
src/parakeet.rs      SpeechModel impl
src/parakeet_core/   ONNX engine, mel frontend, booster
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
