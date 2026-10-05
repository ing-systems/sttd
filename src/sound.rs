// ---------------------------------------------------------------------------
// Audible pause/resume/ready cues, and spoken playback, behind one seam.
// ---------------------------------------------------------------------------
//
// Mic mode starts paused by default and the model load (plus the reload
// after a long pause) takes tens of seconds with only a log line to mark
// readiness — invisible in daemon mode. Three short synthesized tones mark
// the transitions instead: paused (low ack), resumed (mid ack), ready
// (rising two-tone = "speak now").
//
// The same output path now also carries spoken transcripts (`--speak`, see
// `piper`): `play_mono` takes a waveform at whatever rate it was synthesized
// at and puts it on the default output, resampling to the device rate when the
// two differ — a voice runs at its own rate (22.05 kHz for most piper exports)
// while the device usually does not. So the device half is shared and only the
// buffer in front of it differs: tones are synthesized at the device rate,
// speech is resampled into it.
//
// Zero new dependencies: `cpal` (already used for capture) opens the default
// output device and plays a synthesized sine. Playback is fire-and-forget on
// a detached thread so signal/ASR threads never block; failures (no
// output device, busy) log once per cue and never propagate. Tone synthesis
// (`tone_buffer`, `sequence_buffer`) and the resampling (`resample_mono`) are
// pure and unit-tested; device I/O is untested glue by design.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SupportedStreamConfig;
use resampler::{Attenuation, Latency, ResamplerFir};

/// Peak amplitude of the cue tones. Conservative so a loud desktop mix
/// doesn't clip.
const CUE_GAIN: f32 = 0.2;
/// Linear fade at each end of a tone to avoid click transients.
const FADE_MS: u64 = 10;
/// Silence between the two `ready` tones.
const READY_GAP_MS: u64 = 50;
/// Quiet window the capture side discards around cue playback so cue tones
/// are never transcribed (see `main::cue_guard`). Covers the longest cue
/// (ready: 100 + 50 gap + 150 = 300 ms) plus output-device open latency on
/// the detached cue thread.
pub(crate) const CUE_QUIET_MS: u64 = 500;

/// Synthesize one sine tone at `sample_rate`: `freq_hz` for `duration_ms`,
/// with short linear fades at both ends to avoid clicks. Pure over its
/// inputs so tests don't touch audio hardware.
pub(crate) fn tone_buffer(freq_hz: f32, duration_ms: u64, sample_rate: u32) -> Vec<f32> {
    if duration_ms == 0 || sample_rate == 0 {
        return Vec::new();
    }
    let total = (sample_rate as u64 * duration_ms / 1000) as usize;
    if total == 0 {
        return Vec::new();
    }
    let fade = ((sample_rate as u64 * FADE_MS / 1000) as usize).min(total / 2);
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let t = i as f32 / sample_rate as f32;
        let mut s = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * CUE_GAIN;
        if fade > 0 {
            if i < fade {
                s *= (i + 1) as f32 / (fade + 1) as f32;
            } else if i >= total - fade {
                let j = total - 1 - i;
                s *= (j + 1) as f32 / (fade + 1) as f32;
            }
        }
        out.push(s);
    }
    out
}

/// Concatenate `notes` with `gap_ms` of silence between them. Pure over its
/// inputs so tests don't touch audio hardware.
pub(crate) fn sequence_buffer(notes: &[(f32, u64)], gap_ms: u64, sample_rate: u32) -> Vec<f32> {
    let mut out = Vec::new();
    let gap = if sample_rate == 0 {
        0
    } else {
        (sample_rate as u64 * gap_ms / 1000) as usize
    };
    for (i, &(freq, dur)) in notes.iter().enumerate() {
        if i > 0 {
            out.extend(std::iter::repeat_n(0.0f32, gap));
        }
        out.extend(tone_buffer(freq, dur, sample_rate));
    }
    out
}

/// Paused ack: single low beep (440 Hz, 150 ms).
pub(crate) fn play_paused() {
    play_sequence(&[(440.0, 150)]);
}

/// Resumed ack: single mid blip (660 Hz, 100 ms). Fires on signal receipt,
/// before any reload — `play_ready` follows when transcription can resume.
pub(crate) fn play_resumed() {
    play_sequence(&[(660.0, 100)]);
}

/// Ready jingle: rising two-tone (660 Hz / 880 Hz) = "speak now". Played on
/// first load (when unpaused), on reload after a long pause, and on
/// short-pause resume where the model never unloaded.
pub(crate) fn play_ready() {
    play_sequence(&[(660.0, 100), (880.0, 150)]);
}

/// Play `notes` (`(freq_hz, duration_ms)` per tone) on a detached thread.
/// Returns immediately; device failures log and never propagate.
fn play_sequence(notes: &[(f32, u64)]) {
    let spec = notes.to_vec();
    std::thread::Builder::new()
        .name("sound-cue".into())
        .spawn(move || play_blocking(&spec))
        .ok();
}

/// Play a synthesized waveform on the default output device, blocking.
///
/// Used for a spoken transcript, on the speak thread, which has nothing else to
/// do while it speaks — so unlike the cues this one is not fire-and-forget, and
/// `finish_by_ms` is what lets a shutdown cut it off instead of waiting out the
/// audio. Returns whether it played to the end.
///
/// Any failure is a `log::warn`: speech must never break transcription, and the
/// emit queue behind it keeps draining either way.
pub(crate) fn play_mono(samples: &[f32], sample_rate: u32, finish_by_ms: &AtomicU64) -> bool {
    if samples.is_empty() {
        return true;
    }
    let Some((device, config, device_rate)) = default_output() else {
        return false;
    };
    let mono = resample_mono(samples, sample_rate, device_rate);
    stream_and_play(&device, &config, mono, Some(finish_by_ms))
}

/// Whether `finish_by_ms` holds a deadline that has already passed. Zero means
/// no deadline, so a caller that never set one is never cut off.
///
/// Takes the clock as an argument so the wait itself is testable without a
/// device: `now_ms` is a process-wide monotonic counter, not wall time.
pub(crate) fn past_deadline(finish_by_ms: &AtomicU64, now_ms: u64) -> bool {
    let deadline = finish_by_ms.load(Ordering::Acquire);
    deadline != 0 && now_ms >= deadline
}

/// Blocking playback on the cue thread: open the default output, render the
/// sequence at its native rate, play, wait, drop. Any failure is a `log::warn`
/// — cues must never break transcription.
fn play_blocking(spec: &[(f32, u64)]) {
    let Some((device, config, sample_rate)) = default_output() else {
        return;
    };
    let mono = sequence_buffer(spec, READY_GAP_MS, sample_rate);
    stream_and_play(&device, &config, mono, None);
}

/// The default output device, its stream config and its rate. `None` with a
/// warning for either of the two ways there is no usable output.
fn default_output() -> Option<(cpal::Device, SupportedStreamConfig, u32)> {
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            log::warn!("playback skipped: no default output device");
            return None;
        }
    };
    let supported = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("playback skipped: no output config ({e:?})");
            return None;
        }
    };
    let sample_rate = supported.sample_rate().0;
    Some((device, supported, sample_rate))
}

/// Resample mono `input` from `in_rate` to `out_rate`, once, as a whole buffer.
///
/// The same `ResamplerFir` the capture path uses (`dsp::FirStream`), so speech
/// and mic audio pass one filter: a pass-through when the rates already agree,
/// and one zero-drain at the end so the FIR's own tail is not truncated. Pure
/// over its inputs, so it is unit-tested; device I/O is not.
pub(crate) fn resample_mono(input: &[f32], in_rate: u32, out_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == out_rate || in_rate == 0 || out_rate == 0 {
        return input.to_vec();
    }
    let mut resampler =
        ResamplerFir::new_from_hz(1, in_rate, out_rate, Latency::Sample32, Attenuation::Db90);
    let out: Vec<f32> = resample_all(&mut resampler, input);
    // Drain the FIR delay and the buffered tail, as `FirStream::flush_into`
    // does for the capture path. Bounded: zeros in produce zeros out, so a
    // fixed drain cannot synthesize indefinite silence.
    let flush = vec![0.0f32; (resampler.delay() + 2048).max(2048)];
    let drained = resample_all(&mut resampler, &flush);
    if drained.iter().any(|s| *s != 0.0) {
        log::trace!(
            "playback resampler drain produced {} samples",
            drained.len()
        );
    }
    out
}

/// Feed `input` through `resampler` until it is consumed, appending every output
/// sample.
///
/// One `resample` call consumes only what fits its output slice and buffers the
/// rest, so this loops — the same reason `dsp::FirStream::feed_into` does, and
/// without it a 1 s buffer would come back a fraction of its length.
fn resample_all(resampler: &mut ResamplerFir, input: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(input.len() * 2 + 4096);
    // A generous fixed scratch: big enough that the loop is one or two passes,
    // and no allocation happens inside it.
    let mut scratch = vec![0.0f32; 16 * 1024];
    let mut offset = 0usize;
    while offset < input.len() {
        let (consumed, produced) = match resampler.resample(&input[offset..], &mut scratch) {
            Ok(pair) => pair,
            Err(e) => {
                log::warn!("playback resampling failed ({e:?}); playing what was converted");
                break;
            }
        };
        out.extend_from_slice(&scratch[..produced]);
        offset += consumed;
        if consumed == 0 && produced == 0 {
            break;
        }
    }
    out
}

/// Play `mono` (already at the device rate) to the open device and wait for it.
///
/// The wait honors `stop` in 20 ms steps so a shutdown can cut the audio off
/// instead of blocking teardown for its full length; dropping the stream at
/// return is what silences it.
/// Stream `mono` (at the device rate) to the open device and wait for it.
/// Returns whether it played to the end.
///
/// `finish_by_ms`, when given, bounds the wait: the shutdown deadline is checked
/// every 20 ms and the stream is dropped at that point, which is what silences
/// it. `None` (the cue path) waits the whole buffer out.
fn stream_and_play(
    device: &cpal::Device,
    supported: &SupportedStreamConfig,
    mono: Vec<f32>,
    finish_by_ms: Option<&AtomicU64>,
) -> bool {
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    // `StreamConfig` carries no sample format (the format is the closure's
    // sample type), so the supported config converts to the concrete one the
    // stream builders want.
    let config: cpal::StreamConfig = supported.clone().into();
    if mono.is_empty() {
        return true;
    }
    // De-interleave position shared with the callback.
    let pos = Arc::new(AtomicUsize::new(0));
    let total_frames = mono.len();
    let frames = Arc::new(mono);
    let err_fn = |err| log::warn!("playback stream error: {err}");

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let pos = Arc::clone(&pos);
            let frames = Arc::clone(&frames);
            device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    for frame in data.chunks_mut(channels) {
                        let i = pos.fetch_add(1, Ordering::Relaxed);
                        let s = if i < total_frames { frames[i] } else { 0.0 };
                        for out in frame.iter_mut() {
                            *out = s;
                        }
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let pos = Arc::clone(&pos);
            let frames = Arc::clone(&frames);
            device.build_output_stream(
                &config,
                move |data: &mut [i16], _| {
                    for frame in data.chunks_mut(channels) {
                        let i = pos.fetch_add(1, Ordering::Relaxed);
                        let s = if i < total_frames { frames[i] } else { 0.0 };
                        let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                        for out in frame.iter_mut() {
                            *out = v;
                        }
                    }
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let pos = Arc::clone(&pos);
            let frames = Arc::clone(&frames);
            device.build_output_stream(
                &config,
                move |data: &mut [u16], _| {
                    for frame in data.chunks_mut(channels) {
                        let i = pos.fetch_add(1, Ordering::Relaxed);
                        let s = if i < total_frames { frames[i] } else { 0.0 };
                        let v = ((s + 1.0) * 32767.5).clamp(0.0, 65535.0) as u16;
                        for out in frame.iter_mut() {
                            *out = v;
                        }
                    }
                },
                err_fn,
                None,
            )
        }
        fmt => {
            log::warn!("playback skipped: unsupported sample format {fmt:?}");
            return false;
        }
    };
    let stream = match stream {
        Ok(s) => s,
        Err(e) => {
            log::warn!("playback skipped: failed to build output stream ({e:?})");
            return false;
        }
    };
    if let Err(e) = stream.play() {
        log::warn!("playback skipped: failed to start output stream ({e:?})");
        return false;
    }
    // Buffer duration plus margin so the tail isn't cut by the drop.
    let ms = total_frames as u64 * 1000 / sample_rate as u64 + 100;
    let Some(finish_by_ms) = finish_by_ms else {
        std::thread::sleep(Duration::from_millis(ms));
        return true;
    };
    let done = Instant::now() + Duration::from_millis(ms);
    while Instant::now() < done {
        // The clock is read per iteration rather than threaded through, so this
        // loop cannot be the thing that hangs: a deadline that never arrives
        // ends it when the buffer has played out.
        if past_deadline(finish_by_ms, crate::emit::now_ms()) {
            return false;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_length_matches_duration_and_rate() {
        assert_eq!(tone_buffer(440.0, 150, 48_000).len(), 7200);
        assert_eq!(tone_buffer(660.0, 100, 16_000).len(), 1600);
        assert!(tone_buffer(440.0, 0, 48_000).is_empty());
        assert!(tone_buffer(440.0, 150, 0).is_empty());
    }

    #[test]
    fn tone_fades_edges_and_limits_peak() {
        let buf = tone_buffer(440.0, 150, 48_000);
        assert!(buf.iter().all(|&s| s.abs() <= CUE_GAIN + 1e-6));
        // Fades pull both ends toward zero (first sample well below peak).
        assert!(buf[0].abs() < CUE_GAIN * 0.1);
        assert!(buf[buf.len() - 1].abs() < CUE_GAIN * 0.1);
        // Middle reaches near full gain (sine peak over a full period).
        let mid_peak = buf[buf.len() / 4..buf.len() * 3 / 4]
            .iter()
            .fold(0.0f32, |m, &s| m.max(s.abs()));
        assert!(mid_peak > CUE_GAIN * 0.9, "mid_peak={mid_peak}");
    }

    #[test]
    fn sequence_joins_with_silence_gap() {
        let rate = 16_000;
        let seq = sequence_buffer(&[(660.0, 100), (880.0, 150)], READY_GAP_MS, rate);
        let first = tone_buffer(660.0, 100, rate).len();
        let gap = (rate as u64 * READY_GAP_MS / 1000) as usize;
        let second = tone_buffer(880.0, 150, rate).len();
        assert_eq!(seq.len(), first + gap + second);
        assert!(seq[first..first + gap].iter().all(|&s| s == 0.0));
    }

    #[test]
    fn sequence_single_note_has_no_gap() {
        let rate = 48_000;
        let seq = sequence_buffer(&[(440.0, 150)], READY_GAP_MS, rate);
        assert_eq!(seq, tone_buffer(440.0, 150, rate));
    }

    #[test]
    fn cue_quiet_window_covers_longest_cue() {
        // Longest cue is ready: 100 + 50 gap + 150 = 300 ms of audible tone
        // (plus device-open latency on the detached cue thread).
        assert!(CUE_QUIET_MS >= 300, "CUE_QUIET_MS={CUE_QUIET_MS}");
    }

    #[test]
    fn resampling_matching_rates_is_a_copy() {
        let input = vec![0.25f32, -0.5, 1.0];
        assert_eq!(resample_mono(&input, 48_000, 48_000), input);
        // A degenerate rate must not divide by zero.
        assert_eq!(resample_mono(&input, 0, 48_000), input);
        assert_eq!(resample_mono(&input, 48_000, 0), input);
        assert!(resample_mono(&[], 22_050, 48_000).is_empty());
    }

    #[test]
    fn resampling_up_and_down_changes_length_by_the_rate_ratio() {
        // A second of a 1 kHz tone, so the length change is unambiguous and the
        // signal itself is checkable.
        let in_rate = 22_050;
        let tone: Vec<f32> = (0..in_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / in_rate as f32).sin())
            .collect();
        for out_rate in [44_100u32, 48_000] {
            let up = resample_mono(&tone, in_rate, out_rate);
            // Within one filter's rounding, not exact: the resampler emits whole
            // output frames plus its own delay tail.
            let want = in_rate as f64 * out_rate as f64 / in_rate as f64;
            let slack = want * 0.05 + 4096.0;
            assert!(
                (up.len() as f64 - want).abs() < slack,
                "{in_rate}->{out_rate}: {} samples, wanted ~{want}",
                up.len()
            );
            // The tone survived: it is still loud, and it is still a 1 kHz tone
            // (energy near that frequency dominates).
            let peak = up.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak > 0.5, "{in_rate}->{out_rate}: peak={peak}");
            let crossings = zero_crossings(&up);
            let hz = crossings as f64 * out_rate as f64 / up.len() as f64;
            assert!(
                (hz - 1000.0).abs() < 50.0,
                "{in_rate}->{out_rate}: tone came out at {hz:.0} Hz"
            );
        }
        // Downwards too: one second of 48 kHz into 22.05 kHz.
        let down_in: Vec<f32> = (0..48_000)
            .map(|i| ((i as f32 * 0.05).sin()) * ((i as f32 * 0.013).cos()))
            .collect();
        let down = resample_mono(&down_in, 48_000, 22_050);
        // 22 050 is one second at the target rate; the FIR's own delay eats the
        // first few samples of it, so this is "about a second", not "over".
        let want = 22_050.0;
        assert!(
            (down.len() as f64 - want).abs() < want * 0.05,
            "48k->22.05k: {} samples, wanted ~{want}",
            down.len()
        );
        assert!(down.iter().all(|s| s.abs() <= 1.5), "no runaway gain");
        assert!(
            down.iter().fold(0.0f32, |m, s| m.max(s.abs())) > 0.5,
            "the signal survived"
        );
    }

    /// Sign changes over the whole buffer, a cheap frequency proxy.
    fn zero_crossings(buf: &[f32]) -> usize {
        buf.windows(2).filter(|w| w[0] <= 0.0 && w[1] > 0.0).count()
    }
}
