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
  -> whisper worker thread (condition -> gain -> inference -> transcript gates)
  -> Emitter (async queue cap 512 -> WAV/TXT dump, stdout, wl-copy, ydotool paste)
```

Mic and file modes share `FrameProcessor`; only the source differs.

Source files: `src/main.rs:1`, `src/capture.rs:1`, `src/signal.rs:1`, `src/dsp.rs:1`, `src/denoise.rs:1`, `src/vad.rs:1`, `src/noise.rs:1`, `src/utterance.rs:1`, `src/gate.rs:1`, `src/whisper.rs:1`, `src/emit.rs:1`, `src/config.rs:1`, `src/cli.rs:1`, `src/wav.rs:1`, `src/weights.rs:1`, `src/sound.rs:1`, `src/daemon.rs:1`.

Terminology (`CONTEXT.md`): utterance, utterance buffer, pre-roll, acoustic evidence, evidence strength, VAD verdict, gate, confidence floor, speech density, text token, hallucination, emission, VAD branch, whisper branch, denoiser, direct hop.

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

* `WHISPER_QUEUE_CAP:99`=16 (`src/whisper.rs:99`). `push_utterance:251` never blocks; full = warn + drop newest. Dead worker = warn.
* Worker-side shed (`shed_stale=true` mic, false file): if `rx.slots()>WHISPER_SHED_KEEP=4`, drain all, `shed_survivors:787` keeps top-4 by `(density,rms)`, newer wins ties, transcribes chronologically so prompt context stays ordered.

## 6. Whisper worker

* `spawn_whisper_worker:120` -> `whisper_loop:171` -> `whisper_init_and_run:232` on thread `whisper`. `ready` channel fail-fast on first load; later reloads log + retry 5 s. `load_whisper_context:544` with per-second tick; `use_gpu`, `n_threads` (`resolve_n_threads:111`, `0`=auto via `auto_threads:103`).
* Outer load/unload loop: one `ctx+state` per iteration; scope exit frees VRAM. Loaded loop polls `rx` 20 ms, transcribes, checks `should_unload:211` (enabled + paused >= `auto_unload_after`, default 180 s, `0`=immediate). Unloaded wait stashes arrivals, reloads eagerly on resume. `stash` replayed after load. Shutdown when `is_abandoned` (mic) / end (file).
* Per utterance `transcribe_and_emit:882` (`utt_NNNNNN` monotonic across reloads):
  * `condition_utterance:994`: DC subtract, 1-pole HP 60 Hz (`CONDITION_HP_CUTOFF_HZ=60`), 5 ms fades (`CONDITION_FADE_SAMPLES=80`), peak safety 0.99.
  * `conditioned_speech_rms:1062`: scale speech RMS by `conditioned/raw` overall ratio so hum/DC removal lowers gain estimate; fallback raw on zero/non-finite.
  * `apply_gain:1086` toward `--target-rms` 0.05 (`0` disables), only amplify (`>=1.0`), cap `MAX_GAIN=8.0`, clip-backoff to 0.99. Borrowed when no gain (loud common case no copy).
  * `pad_with_dither:1109` to `WHISPER_MIN_SAMPLES=16400` (mel `1+(n+200-400)/160 >=100`; 16000 alone = 990 ms rejected). Deterministic xorshift ±1e-4.
  * `transcribe:1227`: `BeamSearch{beam_size 5, patience -1}`, `n_threads`, lang (`pl/en/auto` -> `set_language`), `initial_prompt=build_effective_prompt(base,tail)`, `suppress_blank/non_speech`, `temperature 0.0 / inc 0.2 / entropy 2.4 / logprob -1.0` ladder (re-try only, final temp always accepted), abort callback on `running==false`. Joins segment texts, collects `ScoredToken{id,p,plog,is_lexical}` (`is_lexical=text has alphanumeric`).
  * `average_text_confidence:1174`: average lexical text tokens (`id<token_beg`); fallback all text tokens if none lexical; 0/0 if none. Excludes timestamp tokens + trailing punct on words (low LM prob on isolated words).
  * `gate.decide_transcript:123`: empty -> `DropEmpty`; `floor_for:111` (strong if `frames>=STRONG_SPEECH_FRAMES=20 && rms>=min_rms*STRONG_RMS_FACTOR=3` -> `min_conf_strong` 0.35 else `min_conf` 0.6) -> `LowConfidence`; `is_hallucination:329` (normalize lowercase+fold PL diacritics+punct->space; exact vs `HALLUCINATION_PATTERNS`, fuzzy Levenshtein<=2 if len>=10 vs long patterns) -> `Hallucination`; `is_repetition:272` (6+ words, unique<=50% or bigram x4 `REPETITION_BIGRAM_COUNT=4`) -> `Repetition`; else `Emit`. Emit pushes to `prompt_tail` via `push_prompt_context:613` (last N chars, default 150, max 2000, snap forward to sentence `.!?…` else word, `0`=clear/disable). Drops never feed context.
  * Metrics: `infer_rtf:773`, `format_metrics_row:807`, `write_metrics_row:843` JSONL (`tag,samples,audio_sec,infer_ms,rtf,speech_frames,total_frames,density,rms,prob,logprob,verdict,text`), flush per row; RTF>1.0 warns. `should_autopause:227` re-pauses only on emit when enabled.
* Model: `src/weights.rs:11` `resolve_model_path` (spec -> `models/ggml-<spec>.bin`, else path passthrough), `ensure_available:30` downloads `ggml-*.bin` from `MODEL_BASE_URL` (HF whisper.cpp). Default `large-v3-turbo-q8_0` (21-spec catalog `src/config.rs:264`).

## 7. Emission

Seam: `src/emit.rs:73` `Emitter::emit(Emission{tag,pcm,text,verdict,prob,logprob,evidence})`.

* `SystemEmitter:88` async: `emit:191` clones to `OwnedEmission` + bounded `rtrb:512` push (full/worker-gone = warn, never blocks inference). Worker thread `EmitterWorker:102` FIFO handles side effects; `Drop:176` closes + joins so shutdown flushes.
* `handle:208`: WAV dump first for every utterance (`write_wav_mono16` via `hound`, `WHISPER_RATE`); `DropEmpty` -> `DROPPED(empty)` TXT; `Drop{kind}` -> log + `dump_reason` TXT (`DROPPED(low-conf/hallucination/repetition)`); `Emit` -> log `avg_p/logprob/threshold/strength/density`, opt-in stdout (`--stdout`, default off), `wl-copy --sensitive` if `--copy` (default on), `ydotool key 29:1 47:1 47:0 29:0` Ctrl+V only if copied and `--paste` (default on), TXT dump if `--dump-dir`. Missing `wl-copy`/`ydotool` disables once. Non-Wayland (`WAYLAND_DISPLAY`/`XDG_SESSION_TYPE`) disables copy+paste once. `RecordingEmitter:403` in-memory for tests.
* Prompt default `src/config.rs:321` short PL/EN (long keyword lists removed: `initial_prompt` as prior leaks into low-signal decodes).

## 8. Runtime extras

* Signals `src/main.rs:613`: `SIGUSR1` pause, `SIGUSR2` resume, `SIGHUP` toggle, `SIGTERM` graceful shutdown (same `running` flag as Ctrl-C, flushes trailing). Handler thread flips flags + optional `src/sound.rs` cues (440 Hz paused, 660 Hz resumed, 660+880 ready); main loop does discarding. `start_paused` default true.
* Pause: entry edge `discard_pending` (signal queues+resamplers+VAD+utterance incl. pre-roll); while paused `discard_speech_keep_warm` (VAD reset + utterance discard, floor kept, resamplers continuous) + `observe_pause` + 10 ms park for fast resume. Resume starts clean, threshold adapted.
* Sounds `src/sound.rs:41`: `tone_buffer`/`sequence_buffer` pure synth + fire-and-forget cpal playback; failures warn only.
* Daemon `src/daemon.rs:31`: `--start` double-fork via `daemonize` (keeps cwd, pid/log files default `<tmp>/sttd.pid/log`), `--stop` TERM/wait 10 s/KILL, `--status` `kill(pid,0)`.
* CLI `src/cli.rs:31`: every tunable flag + `STTD_*` env, CLI>env>default; `--model` also `WHISPER_MODEL_PATH` legacy; `--file`, `--list-devices/models`, `--metrics`, `--beam/temperature/inc/entropy/logprob/threads/gpu`, `--vad-threshold/adapt`, `--min-rms/density/conf/conf-strong`, `--target-rms`, `--hangover-ms` 200-2000 step 10, `--prompt/context`, `--copy/paste/stdout/dump-dir`, `--start-paused/autopause/debug-tokens`, `--auto-unload/after`, `--sound`, daemon paths.

## 9. File mode

`run_file_mode:666`: read WAV (`hound`, int 8/16/24/32 or float32, channel honoring), resolve model, spawn worker (`shed_stale=false`, no pause/unload/sound/autopause), blocking `ready` wait 600 s, feed `input_frames` chunks, drop trailing partial (<~30 ms), `flush_trailing_speech`, join. Deterministic offline tuning against fixed recording (`tools/eval.py`, `tools/dump2corpus.py`).

## 10. Change entry points

* Faster/slower splits: `--hangover-ms`, `INTERIM_*`, `EARLY_*`, `MAX_*` in `src/config.rs:42`, logic `src/utterance.rs:170`.
* Fewer hallucinations: `GateConfig` thresholds `src/config.rs:130`, patterns `HALLUCINATION_PATTERNS:340`, `is_hallucination/is_repetition` `src/gate.rs:272`, confidence `average_text_confidence` `src/whisper.rs:1174`.
* VAD sensitivity: `--vad-threshold/adapt`, `NoiseTracker` gains `src/noise.rs:14`, median `MEDIAN_WINDOWS` `src/vad.rs:30`, `ONSET/MIN_SPEECH_FRAMES` `src/config.rs:57`.
* Quality vs speed: `--model`, `--beam-size`, `--temperature*`, `--threads`, `--gpu`, `--target-rms`, `--prompt/context`.
