// ---------------------------------------------------------------------------
// Audible pause/resume/ready cues, behind one seam.
// ---------------------------------------------------------------------------
//
// Mic mode starts paused by default and the model load (plus the reload
// after a long pause) takes tens of seconds with only a log line to mark
// readiness — invisible in daemon mode. Three short synthesized tones mark
// the transitions instead: paused (low ack), resumed (mid ack), ready
// (rising two-tone = "speak now").
//
// Zero new dependencies: `cpal` (already used for capture) opens the default
// output device and plays a synthesized sine. Playback is fire-and-forget on
// a detached thread so signal/ASR threads never block; failures (no
// output device, busy) log once per cue and never propagate. Tone synthesis
// (`tone_buffer`, `sequence_buffer`) is pure and unit-tested; device I/O is
// untested glue by design.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

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

/// Blocking playback on the cue thread: open the default output, render the
/// sequence at its native rate, play, wait, drop. Any failure is a `log::warn`
/// — cues must never break transcription.
fn play_blocking(spec: &[(f32, u64)]) {
    let host = cpal::default_host();
    let device = match host.default_output_device() {
        Some(d) => d,
        None => {
            log::warn!("sound cue skipped: no default output device");
            return;
        }
    };
    let supported = match device.default_output_config() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("sound cue skipped: no output config ({e:?})");
            return;
        }
    };
    let sample_rate = supported.sample_rate().0;
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();

    let mono = sequence_buffer(spec, READY_GAP_MS, sample_rate);
    if mono.is_empty() {
        return;
    }
    // De-interleave position shared with the callback.
    let pos = Arc::new(AtomicUsize::new(0));
    let total_frames = mono.len();
    let frames = Arc::new(mono);
    let err_fn = |err| log::warn!("sound cue stream error: {err}");

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
            log::warn!("sound cue skipped: unsupported sample format {fmt:?}");
            return;
        }
    };
    let stream = match stream {
        Ok(s) => s,
        Err(e) => {
            log::warn!("sound cue skipped: failed to build output stream ({e:?})");
            return;
        }
    };
    if let Err(e) = stream.play() {
        log::warn!("sound cue skipped: failed to start output stream ({e:?})");
        return;
    }
    // Buffer duration plus margin so the tail isn't cut by the drop.
    let ms = total_frames as u64 * 1000 / sample_rate as u64 + 100;
    std::thread::sleep(Duration::from_millis(ms));
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
}
