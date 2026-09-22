# sttd pipeline — review request

Real-time Polish speech-to-text for live mic. Audio cut into utterances locally; only speech-bearing stretches reach whisper; transcripts to stdout/clipboard/paste/dump.

## Data flow

```
cpal (hw rate, N ch) -> channel pick/mix -> raw rtrb<f32> ring (4s)
  -> accumulator (~10ms hw chunks)
  -> SignalPath (Fir hw->16k, opt 48k denoise for VAD only, 80Hz Hp per 16k branch)
  -> (vad16, whisper16) pairs (160 samp @16k)
  -> Agc shared-gain (VAD-gated, 1-frame delayed, speech-only)
  -> SpeechDetector (earshot VAD verdict per 10ms)
  -> UtteranceBuffer (open/append/split/close + acoustic gates)
  -> Utterance{pcm, evidence} -> rtrb<Utterance> cap 16, weakest-first shed to 4
  -> whisper worker thread (condition -> gain -> inference -> transcript gates)
  -> Emitter (async queue cap 512 -> WAV/TXT dump, stdout, wl-copy, ydotool paste)
```

Mic and file modes share `FrameProcessor`; only source differs.

Sources: `src/main.rs:1`, `src/capture.rs:1`, `src/signal.rs:1`, `src/dsp.rs:1`, `src/denoise.rs:1`, `src/vad.rs:1`, `src/noise.rs:1`, `src/frame.rs:1`, `src/utterance.rs:1`, `src/gate.rs:1`, `src/whisper.rs:1`, `src/emit.rs:1`, `src/config.rs:1`, `src/cli.rs:1`, `src/agc.rs:1`, `src/hp.rs:1`.

Terminology (`CONTEXT.md`): utterance, utterance buffer, pre-roll, acoustic evidence, evidence strength, VAD verdict, gate, confidence floor, speech density, text token, hallucination, emission, VAD branch, whisper branch, denoiser, direct hop.

## 1. Capture — `src/capture.rs:214`, `src/main.rs:267`

* Device select substring case-insensitive else default; `--list-devices` table.
* Channel `src/dsp.rs:125` `Mix|Pick|Auto` (`--channel` default `mix`). Auto probes ~500ms loudest RMS; fallback mix.
* Callback: mono mix/convert + `rtrb` push only. Full = counted drop. `DeviceNotAvailable` -> `device_gone`.
* Raw ring `src/main.rs:236` `RAW_RING_SECS=4`, cap `hw_rate*4`, recreated per open.
* Main loop `src/main.rs:266`: drain `drain_raw_into:242` -> accumulator, 2ms park when short, feed `hw_rate/100` chunks to `FrameProcessor::process_frame`. 5s throttled queue warn (raw, whisper, signal depths + drops). Shutdown: drop stream, drain, `flush_trailing_speech`, drop `whisper_tx`, join.
* Rebuild `src/main.rs:401`: on `device_gone`, sleep 2s, `open_input_path`, clear accumulator, `discard_pending`. Warm VAD floor carried across explicitly (`FrameProcessor::noise_floor`/`adopt_noise_floor`) — the rebuild constructs a brand-new `FrameProcessor`/`SpeechDetector`, so this doesn't fall out of `discard_pending`'s "keep the floor" behavior for free; it used to silently reset to the cold default (review finding M4, fixed).

## 2. SignalPath — `src/signal.rs:46`

* `Direct` (default `--denoiser none`): single `FirStream(hw->16k)` + `Hp`. One `Arc` shared (`Arc::ptr_eq(vad,whisper)`).
* `Denoised` (`--denoiser deepfilter` + `denoise` feature): VAD `hw->48k->denoise->48k->16k`, whisper independent `hw->16k` raw. Whisper never denoised. `src/denoise.rs:108` never panics, passthrough on error.
* `FirStream` over `ResamplerFir` (64 taps, Db90). `feed_into` any length, remainder buffered; equal rates passthrough. `flush_into` zero-pads at shutdown. `clear` on pause. Pairing by order; steady 0-2 pending (`SIGNAL_PENDING_WARN:44`=8).
* Consts `src/config.rs:15`: `WHISPER_RATE=16000`, `FRAME_16K=160`, `DENOISE_RATE=48000`, `FRAME_48K=480`. 80Hz Hp per 16k branch.

## 3. AGC — `src/agc.rs:22`, `src/frame.rs:82`

* VAD-gated shared-gain, `prev_speech` 1-frame delay breaks VAD<->gain loop. Silence unity, no adapt. Speech-only RMS toward `--target-rms` 0.04 (0 disables), amplify-only, cap `MAX_GAIN=8.0`, attack 10ms / release 200ms, clip-backoff 0.99. Activation floor `AGC_MIN_RMS=0.003`. Reset on pause.

## 4. VAD + noise — `src/vad.rs:53`, `src/noise.rs:28`

* Bridge 10ms cadence vs earshot fixed 16ms/256-samp window, step 128 (50% overlap, ~10 preds/8 frames). `while >=256 predict_f32`, hysteresis, 3-window majority `push_window:196`. No fresh window -> re-report latch. Latency ~1 window+median, absorbed by pre-roll/hangover.
* Hysteresis: `enter=base+min(floor*10,0.2)`, `exit=enter-0.1`. `STTD_VAD_ADAPT=0` = fixed threshold.
* Median 3 pre-filled false: needs 2 consecutive to open, rides single dropout.
* `NoiseTracker{floor init 0.001}`: RMS EMA `ALPHA=0.005` (~2s), 4x faster down, clamp 0.0001-0.05. Live: only non-speech updates. Pause: RMS-only observe, resamplers continuous.
* Defaults: `--vad-threshold` 0.5, `--vad-adapt` on.

## 5. UtteranceBuffer — `src/utterance.rs:70`, driven `src/frame.rs:67`

* Pre-roll ring `RING_FRAMES:20`=30 (300ms) spans utterances; `reset:425` keeps it, `discard:440` clears on pause. `push_frame:164` absorb then record.
* Open: only after `ONSET_SPEECH_FRAMES:64`=3 consecutive speech. Adaptive trim `preroll_trim_skip:275`: onset-burst RMS*`PREROLL_TRIM_FACTOR=0.1` floor `0.002`, scan ring for first >= thresh, keep `PREROLL_SAFETY_FRAMES=10` (100ms) before it.
* While open: append every frame; track only speech frames: `speech_frames`, `samples_at_last_speech`, `speech_energy_sum_sq`+count (speech-only RMS), `speech_frame_rms` (weighted density), `quiet_run`.
* Splits: hangover `--hangover-ms` default 800 (`SILENCE_FRAMES_THRESHOLD=80`) flush when `silence>hangover`; interim `INTERIM_SPLIT_SAMPLES:123`=8s + 30 frames (300ms) emits prefix; early `EARLY_SPLIT_SAMPLES:106`=28s + 10 frames VAD (100ms) or 5 frames quiet RMS<0.003 (50ms); hard `MAX_UTTERANCE_SAMPLES:101`=30s.
* Close `close:344`: `adaptive_tail_end:481` (min 1920/120ms, max 4800/300ms while tail RMS>=min_rms; legacy 3200 when min_rms<=0), `density=speech/total` post-trim, `weighted=rms/(min_rms*3)` 0-1, `effective_density` (raw if density>=0.35 else weighted), speech RMS. Then `gate.check_acoustic` (`src/gate.rs:72`): `<MIN_SPEECH_FRAMES=20` TooFew; `density<min_density` default 0.25 (exempt `>48000` ~3s, 0 disables) LowDensity; `rms<min_speech_rms` default 0.003 LowEnergy. Fail=log+reset, no whisper. Pass=truncate to tail_end, mic advisor (`MIC_LOW_PEAK=0.02`, clip >=0.99 ratio>0.005 warn-only), return `Utterance{pcm, evidence}`.

## 6. Queue — `src/whisper.rs:99`, `src/main.rs:252`

* `WHISPER_QUEUE_CAP`=16. `push_utterance:251` never blocks; full drops newest + warn; dead worker warn.
* Worker shed (`shed_stale=true` mic, false file): if slots>4, drain all, keep top-4 by `(density,rms)` newer wins ties, transcribe chronologically.

## 7. Whisper worker — `src/whisper.rs:120`

* `spawn_whisper_worker:120` -> `whisper_loop:171` -> `whisper_init_and_run:232` on `whisper` thread. `ready` channel fail-fast first load; later reloads log+retry 5s. `load_whisper_context:544`, `use_gpu`, `n_threads` (0=auto).
* Load/unload loop: one ctx+state per iter; scope exit frees VRAM. Poll rx 20ms, check `should_unload:211` (paused>=auto_unload_after default 180s, 0=immediate). Unloaded stashes arrivals, reloads on resume. Shutdown on `is_abandoned` (mic) / end (file).
* Per utterance `transcribe_and_emit` (`utt_NNNNNN` monotonic): condition `condition_utterance` (DC subtract, 5ms fades 80 samp, peak 0.99 — hum already removed upstream by `Hp`, levels already normalized upstream by the VAD-gated shared-gain `Agc` in `frame.rs`, not here) -> `transcribe` pads to `WHISPER_MIN_SAMPLES=16400` via `pad_with_dither` (mel 100 frames; 16000=990ms rejected) xorshift ±1e-4 -> BeamSearch{5, -1}, lang pl/en/auto, `initial_prompt=build_effective_prompt(base,tail)`, `no_context=true` (pinned explicitly, not left at the upstream default), suppress_blank/non_speech, temp 0.0/inc 0.2/entropy 2.4/logprob -1.0 ladder, abort on !running. Joins segments, collects `ScoredToken`.
  * The `conditioned_speech_rms`/`apply_gain` step this bullet used to describe was removed when per-frame `Agc` replaced per-utterance static gain (see §3); `gate::floor_for`'s strong/weak split now scores `Evidence.rms` from the *pre-AGC* whisper-frame RMS (`Agc::process_pair`'s return value, threaded through `UtteranceBuffer::push_frame_evidenced`) rather than the post-gain buffer, which used to make almost every utterance read as strong evidence (review finding M1, fixed).
* `average_text_confidence:1174`: avg lexical text tokens (id<token_beg); fallback all text; 0/0 if none. Excludes timestamps + trailing punct.
* `gate.decide_transcript:123`: empty DropEmpty; `floor_for:111` (strong if frames>=20 && rms>=min_rms*3 -> 0.35 else 0.6) LowConfidence; `is_hallucination:329` (lower+fold PL diacritics+punct->space; exact vs HALLUCINATION_PATTERNS, fuzzy Levenshtein<=2 if len>=10 vs long) Hallucination; `is_repetition:272` (6+ words unique<=50% or bigram x4) Repetition; else Emit. Emit pushes `push_prompt_context:613` (last N chars default 150 max 2000 snap to sentence else word, 0=off). Drops never feed context.
* Metrics JSONL (tag,samples,audio_sec,infer_ms,rtf,...), flush per row; RTF>1.0 warns. `should_autopause:227` re-pauses only on emit.
* Model `src/weights.rs:11`: spec->`models/ggml-<spec>.bin` else passthrough, `ensure_available:30` downloads from HF whisper.cpp. Default `large-v3-turbo-q8_0` (21-spec catalog `src/config.rs:264`).

## 8. Emission — `src/emit.rs:73`

* `Emitter::emit(Emission{tag,pcm,text,verdict,prob,logprob,evidence})`.
* `SystemEmitter:88` async: `emit:191` clone to `OwnedEmission` + `rtrb:512` (full/gone=warn, never blocks). Worker FIFO; Drop joins so shutdown flushes.
* `handle:208`: WAV first every utterance (`hound`, 16k); DropEmpty->DROPPED(empty) TXT; Drop->log+dump_reason TXT; Emit->log, opt-in stdout (`--stdout` default off), `wl-copy --sensitive` if `--copy` (default on), `ydotool Ctrl+V` only if copied and `--paste` (default on), TXT if `--dump-dir`. Missing binaries / non-Wayland disables once.
* Prompt default `src/config.rs:321` short PL/EN (long keyword lists removed — leaks into low-signal decodes).

## 9. Runtime

* Signals `src/main.rs:613`: SIGUSR1 pause, SIGUSR2 resume, SIGHUP toggle, SIGTERM graceful (flushes trailing). Handler flips flags + `src/sound.rs` cues (440 paused, 660 resumed, 660+880 ready); main does discarding. `start_paused` default true.
* Pause: entry `discard_pending` (queues+resamplers+VAD+utterance incl pre-roll); while paused drain-and-drop + 50ms park. Resume clean, floor kept.
* Daemon `src/daemon.rs:31`: `--start` double-fork (pid/log default `<tmp>/sttd.pid/log`), `--stop` TERM/wait 10s/KILL, `--status` kill 0.
* CLI `src/cli.rs:31`: all tunables CLI>env(`STTD_*`)>default; `--model` also `WHISPER_MODEL_PATH` legacy; `--file`, `--list-devices/models`, `--metrics`, `--beam/temperature/inc/entropy/logprob/threads/gpu`, `--vad-threshold/adapt`, `--min-rms/density/conf/conf-strong`, `--target-rms`, `--hangover-ms` 200-2000 step10, `--prompt/context`, `--copy/paste/stdout/dump-dir`, `--start-paused/autopause/debug-tokens`, `--auto-unload/after`, `--sound`.
* File mode `run_file_mode:666`: read WAV (hound int/float, channel honoring), same FrameProcessor, worker shed_stale=false no pause/unload, blocking ready 600s, feed input_frames chunks, drop trailing partial <~30ms, flush. Tools `tools/eval.py`, `tools/dump2corpus.py` for tuning.

## 10. Change entry points

* Splits: `--hangover-ms`, `INTERIM_*`, `EARLY_*`, `MAX_*` `src/config.rs:42`, logic `src/utterance.rs:170`.
* Hallucinations: `GateConfig` `src/config.rs:130`, patterns `HALLUCINATION_PATTERNS:340`, `is_hallucination/is_repetition` `src/gate.rs:272`, `average_text_confidence` `src/whisper.rs:1174`.
* VAD: `--vad-threshold/adapt`, `NoiseTracker` `src/noise.rs:14`, median `src/vad.rs:30`, `ONSET/MIN_SPEECH_FRAMES` `src/config.rs:57`.
* Quality/speed: `--model/beam/temperature*/threads/gpu/target-rms/prompt/context`.

---

## Review request

Please review the pipeline above for correctness, robustness, and tuning risks. Focus on:

1. Gate interaction (acoustic vs transcript confidence floor) — can hard-but-real speech pass while silence hallucinations are blocked? Are `0.6/0.35`, `0.25` density, `0.003` RMS, `20` frames consistent?
2. Pre-roll trim + adaptive tail — does trimming preserve onset without dragging silence? Does tail logic preserve plosives/reverb without hurting density?
3. VAD bridge (16ms earshot vs 10ms frames) + median/hysteresis/noise floor — onset/offset latency, flicker rejection, warm-floor behavior across pause.
4. AGC shared-gain feedback — 1-frame delay sufficient? Freeze-on-silence + caps safe?
5. Queue shed ordering — weakest-first to 4 + drop-newest on full: can real speech starve? Is chronological replay after shed correct for prompt context?
6. Whisper conditioning/gain/padding — DC/HP/fades/peak, RMS rescaling, dither pad to 16400.
7. Emission async guarantees — no inference blocking, flush on shutdown, WAV-first ordering, Wayland/clipboard fallbacks.
8. Pause/resume state clearing — `discard_pending` completeness, floor preservation, rebuild path on device loss.
9. File vs mic parity — same `FrameProcessor` behavior, shed/pause/unload differences justified?
10. Missing failure modes — resampler divergence, RTF>1 backlog, model download/load errors, daemon/signal races.

Report each finding as: location (`file:line`), severity (blocking/major/minor), what is wrong, why it matters, suggested fix. Call out what is correct as well as what is not.
