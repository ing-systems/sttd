# sttd pipeline

Real-time Polish speech-to-text for live microphone. Audio is cut into utterances locally so only speech-bearing stretches reach the speech model.

Data flow:

```
cpal (hw rate, N ch) -> channel pick/mix -> raw rtrb<f32> ring
  -> accumulator (~10 ms hw chunks)
  -> SignalPath (Fir resample hw -> 16 kHz, optional 48 kHz denoise for VAD
     only, then 80 Hz Hp biquad per 16 kHz branch)
  -> (vad16, whisper16) pairs (already HP'd)
  -> Agc shared-gain (VAD-gated, 1-frame delayed, speech-only) on both frames
  -> SpeechDetector (earshot VAD verdict per 10 ms, no internal HP)
  -> UtteranceBuffer (open / append / split / close + acoustic gates)
  -> Utterance{pcm, evidence} -> rtrb<Utterance> queue (cap 16, weakest-first shed to 4)
  -> ASR worker thread (condition -> inference -> transcript gates; engine is
     whisper.cpp or Parakeet, picked once at startup by --model)
  -> Emitter (async queue cap 512 -> WAV/TXT dump, stdout, wl-copy, ydotool paste)
```

Mic and file modes share `FrameProcessor`; only the source differs.

Source files: `src/main.rs:1`, `src/capture.rs:1`, `src/signal.rs:1`, `src/dsp.rs:1`, `src/denoise.rs:1`, `src/vad.rs:1`, `src/noise.rs:1`, `src/utterance.rs:1`, `src/gate.rs:1`, `src/asr.rs:1`, `src/whisper.rs:1`, `src/parakeet.rs:1`, `src/parakeet_burn/mod.rs:1`, `src/sherpa.rs:1`, `src/emit.rs:1`, `src/config.rs:1`, `src/cli.rs:1`, `src/wav.rs:1`, `src/weights.rs:1`, `src/sound.rs:1`, `src/daemon.rs:1`.

Terminology (`CONTEXT.md`): utterance, utterance buffer, pre-roll, acoustic evidence, evidence strength, VAD verdict, gate, confidence floor, speech density, text token, hallucination, emission, VAD branch, model branch, denoiser, direct hop.

---

## 1. Capture

* Device: `src/capture.rs:214` `select_input_device`. `--device`/`STTD_DEVICE` substring case-insensitive, else system default. `src/capture.rs:124` `--list-devices` table.
* Channel: `src/dsp.rs:125` `ChannelSel::Mix|Pick|Auto`, `src/cli.rs:50` `--channel` default `mix`. `Auto` probes once per open (~500 ms) via `src/capture.rs:15` `probe_loudest_channel` + `src/dsp.rs:168` `pick_louder_channel` (highest RMS); silence/mono falls back to mix. Validation `src/dsp.rs:207` `resolve_channel`.
* Stream: `src/capture.rs:252` `build_input_stream` for F32/I16/U16. Callback does per-sample mono mix/convert + `rtrb` push only. No alloc/log/block. Full ring = counted drop (`StreamTelemetry::dropped_chunks`). `err_fn` counts `errors`, sets `device_gone` on `DeviceNotAvailable`.
* Ring: `src/main.rs:236` `RAW_RING_SECS=4`, capacity `hw_rate*4` in `src/main.rs:581` `open_input_path`. Created per open (follows hw rate); utterance ring survives rebuilds.
* Main loop: `src/main.rs:266` `run_microphone_mode` drains `raw_rx` via `drain_raw_into:242` into `accumulator`, parks 2 ms when short, feeds `hw_input_frames:250` (`hw_rate/100`) chunks to `FrameProcessor::process_frame`. 5 s throttled queue warn covers `raw_rx`, `whisper`, `signal_vad/whisper`, drops. Shutdown: `drop(stream)`, drain, `flush_trailing_speech`, drop `whisper_tx` so worker sees `is_abandoned`, join.
* Rebuild: `src/main.rs:367` on `device_gone`, sleep 2 s, `open_input_path` again, clear accumulator, `discard_pending`, keep warm VAD floor.

## 2. Signal path (resampling + optional denoiser)

Owner: `src/signal.rs:46` `SignalPath`. Callers feed hw chunks, get `Vec<PairedFrames>` where `PairedFrames=(SharedFrame,SharedFrame)`, each 160 samples @16 kHz.

* `Direct` (default `--denoiser none`): single `FirStream(hw->16k, Sample32/Db90)` + single `Hp` in `src/signal.rs`. One allocation shared by both branches (`Arc::new(frame)` cloned), so `Arc::ptr_eq(vad,whisper)`.
* `Denoised` (`--denoiser deepfilter`, needs `denoise` Cargo feature): `src/signal.rs:77` `hw->48k -> Denoiser::denoise_frame -> 48k->16k` for VAD + independent `hw->16k` raw for whisper. Whisper branch stays raw by design (model trained raw); 48 kHz exists only for denoiser. `src/denoise.rs:50` `Denoiser::new`, `src/denoise.rs:108` `denoise_frame` (never panics, passthrough on error), `src/denoise.rs:94` `reset` on pause.
* Adapter: `src/dsp.rs` `FirStream` over `ResamplerFir` (64 taps / 32-sample delay, Db90; arbitrary rates via `new_from_hz`). `feed_into` accepts any length, returns complete `frame_out` frames; remainder buffered. Equal rates = bit-exact passthrough. `flush_into` zero-pads trailing partial FFT chunk at shutdown. `clear` drops partial state on pause. Pairing by order; both branches average 100 frames/s, queues stay 0-2 (`SIGNAL_PENDING_WARN:44`=8).
* Constants: `src/config.rs:15` `WHISPER_RATE=16000`, `FRAME_16K=160`, `DENOISE_RATE=48000`, `FRAME_48K=480`.

## 3. VAD + background tracker

* Bridge: `src/vad.rs:53` `SpeechDetector`. Pipeline 10 ms cadence vs earshot fixed 16 ms (256-sample) window.
* Per `verdict`: input already 80 Hz HP'd upstream (`Hp`) -> stage -> while `>=256`, `predict_f32` per 128-step (50% overlap, ~10 preds/8 frames) -> hysteresis -> 3-window majority `push_window:196`. Frames without new window re-report latch. Adds ~1 window + median latency, absorbed by pre-roll/hangover.
* Hysteresis: enter `noise.enter_threshold(base)`, exit `enter-0.1`. `adapt=false` (`STTD_VAD_ADAPT=0`) = single fixed threshold.
* Median: `MEDIAN_WINDOWS:30`=3 pre-filled `false`. 1-of-3 never flips; needs 2 consecutive to open, rides single dropout.
* HP: `highpass_frame:173` zero-state start; `reset:222` clears staging/latch/median/HP/earshot RNN, keeps noise floor.
* Noise: `src/noise.rs:28` `NoiseTracker{ floor init 0.001 }`. `frame_rms:43`, `update_background:54` asymmetric EMA (`NOISE_EMA_ALPHA=0.005` ~2 s, 4x faster down), clamp 0.0001-0.05. `enter=base+min(floor*10,0.2)`, `exit=enter-0.1`. Live: only non-speech updates (speech never raises floor). Pause: `observe_background:211` (RMS only, no earshot, no latch change) keeps resamplers continuous.

## 4. Utterance buffering + acoustic gates

* Owner: `src/utterance.rs:70` `UtteranceBuffer`, driven by `src/frame.rs:67` `FrameProcessor::process_frame`. `FrameProcessor` also owns `observe_pause:103`, `discard_pending:94`, `discard_speech_keep_warm:113`, `flush_trailing_speech:120`, `signal_pending_depths:59`.
* `push_frame:164`: `absorb` then `record_preroll`. Pre-roll ring `RING_FRAMES:20`=30 (300 ms) spans utterances; `reset:425` keeps it, `discard:440` clears it for pause.
* Open: only after `ONSET_SPEECH_FRAMES:64`=3 consecutive speech frames. Onset frames not lost (already in pre-roll). Adaptive trim `preroll_trim_skip:275`: onset-burst RMS (ring tail + current) * `PREROLL_TRIM_FACTOR=0.1` floored at `0.002`; scan ring for first frame >= threshold, keep `PREROLL_SAFETY_FRAMES=10` (100 ms) before it. Loud onset -> safety + onset only; quiet onset -> full ring.
* Open behavior `absorb:170`: append every frame regardless of verdict; track only from speech frames: `speech_frames`, `samples_at_last_speech`, `speech_energy_sum_sq` + `speech_sample_count` (speech-only RMS), `speech_frame_rms` (weighted density), `quiet_run` (low-energy split).
* Splits:
  * hangover `--hangover-ms` default 800 (`SILENCE_FRAMES_THRESHOLD=80`), `with_hangover_frames:137`, flush when `silence_frames > hangover`.
  * interim `INTERIM_SPLIT_SAMPLES:123`=8 s + `INTERIM_SPLIT_SILENCE_FRAMES=30` (300 ms) emits prefix; continuation starts fresh, pre-roll spans boundary.
  * early `EARLY_SPLIT_SAMPLES:106`=28 s + `EARLY_SPLIT_SILENCE_FRAMES=10` (100 ms VAD) or `EARLY_SPLIT_QUIET_FRAMES=5` (50 ms `RMS<0.003`) cuts in quiet.
  * hard `MAX_UTTERANCE_SAMPLES:101`=30 s flush.
* Close: `close:344` computes `adaptive_tail_end:481` (min `TAIL_MIN=1920` 120 ms, max `TAIL_MAX=4800` 300 ms while tail RMS>=`min_rms`; legacy fixed `TAIL_KEEP=3200` when `min_rms<=0`), `total_frames`, `density=speech/total`, `weighted=rms/(min_rms*3)` capped 0-1 (`weighted_density:462`), speech RMS, `effective_density` (raw if `density>=0.35` else weighted). Then `gate.check_acoustic` (`src/gate.rs:72`): `<MIN_SPEECH_FRAMES=20` -> `TooFewFrames`; `density<min_density` (default 0.25, exempt `>DENSITY_LONG_EXEMPT=48000` ~3 s, `0` disables) -> `LowDensity`; `rms<min_speech_rms` (default 0.003) -> `LowEnergy`. Fail = log + `reset`, no whisper. Pass = truncate to `tail_end`, mic advisor (`MIC_LOW_PEAK=0.02`, `MIC_CLIP>=0.99` ratio `>0.005` warn-only), return `Utterance{pcm, evidence{speech_frames,total_frames,rms}}`.

## 5. Queue / backpressure

* `ASR_QUEUE_CAP:175`=16 (`src/asr.rs:175`). `push_utterance:...` (`src/main.rs`) never blocks; full = warn + drop newest. Dead worker = warn.
* Worker-side shed (`shed_stale=true` mic, false file): if `rx.slots()>ASR_SHED_KEEP=4` (`src/asr.rs:169`), drain all, `shed_survivors` keeps top-4 by `(density,rms)`, newer wins ties, transcribes chronologically so prompt context stays ordered.

## 6. ASR worker + speech model engines

Three engine families exist, picked once at startup by `--model`/`STTD_MODEL` and fixed
for the process's lifetime: whisper.cpp (`src/whisper.rs`, optional), NVIDIA
Parakeet-TDT-0.6B-v3 via `parakeet-rs` (`src/parakeet.rs`, default), and sherpa-onnx
(`src/sherpa.rs`, only with the `sherpa` Cargo feature: offline Nemo-transducer
`sherpa-parakeet-tdt-0.6b-v3-int8`, sherpa-whisper
`sherpa-whisper-large-v3-turbo-int8`, Canary-1B-v2
`sherpa-canary-1b-v2-int8` — Polish Canary needs a native lib newer than
sherpa-onnx 1.13.8, see `SherpaKind`).
A fourth Parakeet runtime — the native Burn port without ONNX Runtime
(`src/parakeet_burn/mod.rs:1`, `--parakeet-engine burn`, `parakeet-burn` Cargo
feature) — is in bring-up: frontend/decode/quant/vocab plus one Conformer block
are implemented and unit-tested, but `transcribe` still bails after mel
extraction until the encoder/joint land, so ORT remains the transcribing path.
See `docs/parakeet-burn.md` for the module-by-module status and next steps.
Whisper, both Parakeet runtimes (ORT + Burn), plus sherpa-offline implement `asr::SpeechModel` (`src/asr.rs:69`);
everything engine-agnostic — the worker loop, gates, metrics, prompt-context
tail, audio conditioning — lives in `src/asr.rs` and never branches on which
engine is loaded.

* `spawn_asr_worker:196` -> `asr_loop:238` -> `asr_init_and_run:314` on
  thread `asr`. `ready` channel fail-fast on first load; later reloads log +
  retry 5 s. `asr::load_model` dispatches on `weights::ModelRef` to
  `WhisperModel::load` (`src/whisper.rs:96`), `ParakeetModel::load`
  (`src/parakeet.rs:39`), or `ParakeetBurnModel::load`
  (`src/parakeet_burn/mod.rs:78`) when `--parakeet-engine burn` is set, each
  wrapped in `asr::with_loading_ticker` for a
  per-second "loading... Ns" log on models that take tens of seconds.
* Outer load/unload loop: one `Box<dyn SpeechModel>` per iteration; scope
  exit (`drop(model)`) frees VRAM/ORT sessions. Loaded loop polls `rx` 20 ms,
  transcribes, checks `should_unload:278` (enabled + paused >=
  `auto_unload_after`, default 180 s, `0`=immediate). Unloaded wait stashes
  arrivals, reloads eagerly on resume. `stash` replayed after load. Shutdown
  when `is_abandoned` (mic) / end (file).
* Per utterance `transcribe_and_emit:934` (`utt_NNNNNN` monotonic across
  reloads):
  * `condition_utterance:1032` (`src/asr.rs`, shared by both engines): DC
    subtract, 5 ms fades (`CONDITION_FADE_SAMPLES`), peak safety 0.99. No
    per-utterance gain here — levels are already normalized frame-wise,
    upstream, by the VAD-gated shared-gain `Agc`.
  * `model.transcribe(conditioned, effective_prompt, running)`: the only
    engine-specific step. Whisper (`src/whisper.rs`): pads to
    `WHISPER_MIN_SAMPLES=16400` (mel `1+(n+200-400)/160 >=100`; 16000 alone =
    990 ms rejected), then `BeamSearch{beam_size 5, patience -1}`,
    `n_threads`, lang (`pl/en/auto` -> `set_language`) + `translate`
    (`--translate` -> `set_translate`), `initial_prompt`,
    `suppress_blank/non_speech`, `temperature 0.0 / inc 0.2 / entropy 2.4 /
    logprob -1.0` ladder (re-try only, final temp always accepted), abort
    callback on `running==false`; scores confidence via
    `average_text_confidence` (average lexical text tokens, `id<token_beg`,
    excluding timestamp tokens and trailing punctuation on isolated words).
    Parakeet (`src/parakeet.rs`): pads to `PARAKEET_MIN_SAMPLES` (0.5 s,
    encoder does 8x subsampling), decodes greedily via `parakeet-rs`
    (auto-detects language per utterance, `--language` not honored), and
    reports **no per-token probability** — `Transcript.confidence` is
    `None`.
  * `SpeechModel::uses_prompt()` gates the conversational tail: `false` for
    Parakeet (no prompt support) and for whisper under `--translate` (a
    prior would bias the next translation, or drag output back to the
    source language). When `true`, an emitted transcript feeds
    `push_prompt_context` (last N chars, default 150, max 2000, snap
    forward to sentence `.!?…` else word, `0`=clear/disable); drops never
    feed it.
  * `gate.decide_transcript` (`src/gate.rs:123`): empty -> `DropEmpty`;
    `floor_for` (strong if `frames>=STRONG_SPEECH_FRAMES=20 &&
    rms>=min_rms*STRONG_RMS_FACTOR=3` -> `min_conf_strong` 0.35 else
    `min_conf` 0.6) -> `LowConfidence`, but only when the engine reported a
    `prob` at all — Parakeet's `None` skips this arm entirely and falls
    straight to the patterns below; `is_hallucination` (normalize
    lowercase+fold PL diacritics+punct->space; exact vs
    `HALLUCINATION_PATTERNS`, fuzzy Levenshtein<=2 if len>=10 vs long
    patterns) -> `Hallucination`; `is_repetition` (6+ words, unique<=50% or
    bigram x4 `REPETITION_BIGRAM_COUNT=4`) -> `Repetition`; else `Emit`.
  * Metrics: `infer_rtf`, `format_metrics_row`, `write_metrics_row`
    (`src/asr.rs`) JSONL (`tag,samples,audio_sec,infer_ms,rtf,speech_frames,
    total_frames,density,rms,prob,logprob,verdict,text`; `prob`/`logprob`
    are JSON `null` when the engine reports none), flush per row; RTF>1.0
    warns. `should_autopause:...` re-pauses only on emit when enabled.
* Sherpa streaming (`--sherpa-mode streaming`, `src/sherpa.rs`
  `spawn_sherpa_streaming_worker`): bypasses `FrameProcessor`'s utterance
  buffer — 16 kHz samples (same resample/Hp/AGC path via
  `process_frame_samples`) queue into a 30 s sample ring; sherpa's Silero VAD
  owns the cuts (`--sherpa-vad-threshold`, `--sherpa-min-silence`,
  `--sherpa-min-speech`). Each VAD segment decodes through the same
  `OfflineRecognizer` (sherpa's endorsed streaming path for NeMo models —
  they ship no true online encoder), then the shared transcript gates +
  emitter + `--metrics` JSONL (evidence synthesized at density 1.0, `prob`
  null like Parakeet). Interim partials (`--sherpa-partials`, 1 s cadence,
  20 s cap) only `log::info`, never clipboard/paste. Pause/unload/reload and
  `--autopause` mirror the utterance worker; sub-`min_speech_duration`
  blips at file end are skipped, same call as the `MIN_SPEECH_FRAMES`
  acoustic gate.

* Model resolution: `src/weights.rs` `resolve_model` maps a `--model` value
  to `weights::ModelRef::Whisper(path)` (spec -> `models/ggml-<spec>.bin`,
  else path passthrough) or `ModelRef::Parakeet{dir, assets}` (spec/alias ->
  `models/<dir>`, or an existing directory passed directly), or
  `ModelRef::Sherpa{dir, assets, kind}` (spec/alias -> `models/sherpa-<dir>`,
  or an existing directory containing `tokens.txt`).
  `ensure_available` then downloads whatever's missing: `ggml-*.bin` from
  `MODEL_BASE_URL` (HF whisper.cpp repo) for whisper, loose ONNX files from
  `PARAKEET_BASE_URL` (HF `istupakov/parakeet-tdt-0.6b-v3-onnx`) for Parakeet,
  or sherpa-layout ONNX + `tokens.txt` from `SHERPA_PARAKEET_BASE_URL` /
  `SHERPA_CANARY_BASE_URL` for sherpa (plus `silero_vad.onnx` from
  `SHERPA_VAD_URL` when `--sherpa-mode streaming`). Default is `parakeet`
  (int8 Parakeet-TDT-v3 via parakeet-rs, 2-spec `PARAKEET_CATALOG` with an
  `fp32` sibling); whisper's 21-spec catalog is `MODEL_CATALOG` in
  `src/config.rs:288` (only with the `whisper` Cargo feature); sherpa's
  catalog (`sherpa-parakeet-tdt-0.6b-v3-int8`, `sherpa-whisper-large-v3-turbo-int8`
  and `sherpa-canary-1b-v2-int8`, plus the `sherpa-parakeet` / `sherpa-whisper`
  / `sherpa-canary` aliases) is `SHERPA_CATALOG` (only with the `sherpa` Cargo
  feature). Canary-1B-v2 is the community Sarphix int8 export of
  `nvidia/canary-1b-v2` (25 languages incl. Polish); Polish decode needs a
  native lib newer than sherpa-onnx 1.13.8 (see `SherpaKind`).

## 7. Emission

Seam: `src/emit.rs:73` `Emitter::emit(Emission{tag,pcm,text,verdict,prob,logprob,evidence})`.

* `SystemEmitter:88` async: `emit:191` clones to `OwnedEmission` + bounded `rtrb:512` push (full/worker-gone = warn, never blocks inference). Worker thread `EmitterWorker:102` FIFO handles side effects; `Drop:176` closes + joins so shutdown flushes.
* `handle:208`: WAV dump first for every utterance (`write_wav_mono16` via `hound`, `WHISPER_RATE`); `DropEmpty` -> `DROPPED(empty)` TXT; `Drop{kind}` -> log + `dump_reason` TXT (`DROPPED(low-conf/hallucination/repetition)`); `Emit` -> log `avg_p/logprob/threshold/strength/density`, opt-in stdout (`--stdout`, default off), `wl-copy --sensitive` if `--copy` (default off), `ydotool key 29:1 47:1 47:0 29:0` Ctrl+V only if copied and `--paste` (default off), TXT dump if `--dump-dir`. Missing `wl-copy`/`ydotool` disables once. Non-Wayland (`WAYLAND_DISPLAY`/`XDG_SESSION_TYPE`) disables copy+paste once. `RecordingEmitter:403` in-memory for tests.
* Prompt default `src/config.rs:401` short PL/EN (long keyword lists removed: `initial_prompt` as prior leaks into low-signal decodes; whisper-only — Parakeet ignores it, see §6).

## 8. Runtime extras

* Signals `src/main.rs:613`: `SIGUSR1` pause, `SIGUSR2` resume, `SIGHUP` toggle, `SIGTERM` graceful shutdown (same `running` flag as Ctrl-C, flushes trailing). Handler thread flips flags + optional `src/sound.rs` cues (440 Hz paused, 660 Hz resumed, 660+880 ready); main loop does discarding. `start_paused` default true.
* Pause: entry edge `discard_pending` (signal queues+resamplers+VAD+utterance incl. pre-roll); while paused `discard_speech_keep_warm` (VAD reset + utterance discard, floor kept, resamplers continuous) + `observe_pause` + 10 ms park for fast resume. Resume starts clean, threshold adapted.
* Sounds `src/sound.rs:41`: `tone_buffer`/`sequence_buffer` pure synth + fire-and-forget cpal playback; failures warn only.
* Daemon `src/daemon.rs:31`: `--start` double-fork via `daemonize` (keeps cwd, pid/log files default `<tmp>/sttd.pid/log`), `--stop` TERM/wait 10 s/KILL, `--status` `kill(pid,0)`.
* CLI `src/cli.rs:31`: every tunable flag + `STTD_*` env, CLI>env>default; `--model` also `WHISPER_MODEL_PATH` legacy; `--file`, `--list-devices/models`, `--metrics`, `--beam/temperature/inc/entropy/logprob/threads/gpu`, `--language/translate`, `--vad-threshold/adapt`, `--min-rms/density/conf/conf-strong`, `--target-rms`, `--hangover-ms` 200-2000 step 10, `--prompt/context`, `--copy/paste/stdout/dump-dir`, `--start-paused/autopause/debug-tokens`, `--auto-unload/after`, `--sound`, daemon paths.

## 9. File mode

`run_file_mode:666`: read WAV (`hound`, int 8/16/24/32 or float32, channel honoring), resolve model, spawn worker (`shed_stale=false`, no pause/unload/sound/autopause), blocking `ready` wait 600 s, feed `input_frames` chunks, drop trailing partial (<~30 ms), `flush_trailing_speech`, join. Deterministic offline tuning against fixed recording (`tools/eval.py`, `tools/dump2corpus.py`).

## 10. Change entry points

* Faster/slower splits: `--hangover-ms`, `INTERIM_*`, `EARLY_*`, `MAX_*` in `src/config.rs:42`, logic `src/utterance.rs:170`.
* Fewer hallucinations: `GateConfig` thresholds `src/config.rs:130`, patterns `HALLUCINATION_PATTERNS:340`, `is_hallucination/is_repetition` `src/gate.rs:272`, whisper confidence scoring `average_text_confidence` `src/whisper.rs` (n/a for Parakeet, which reports no per-token probability).
* VAD sensitivity: `--vad-threshold/adapt`, `NoiseTracker` gains `src/noise.rs:14`, median `MEDIAN_WINDOWS` `src/vad.rs:30`, `ONSET/MIN_SPEECH_FRAMES` `src/config.rs:57`.
* Quality vs speed: `--model` (whisper spec/path, or `parakeet`/`parakeet-tdt-0.6b-v3-{int8,fp32}`, or sherpa `sherpa-parakeet-tdt-0.6b-v3-int8`), `--beam-size`, `--temperature*` (whisper only — ignored with a warning under Parakeet, see §6), `--threads`, `--gpu`, `--target-rms`, `--prompt/context` (whisper only).
* Switching/adding an ASR engine: `weights::ModelRef`/`resolve_model`/`ensure_available` (`src/weights.rs`) for model resolution + download, `asr::SpeechModel` (`src/asr.rs:48`) for the trait a new engine implements, `main::unsupported_parakeet_flags`/`validate_parakeet_model_flags` (`src/main.rs`) for the pattern that warns/rejects flags an engine doesn't honor.
