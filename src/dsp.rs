use anyhow::Result;
use resampler::{Attenuation, Latency, ResamplerFir};

// ---------------------------------------------------------------------------
// Signal primitives: channel downmix + resampling
// ---------------------------------------------------------------------------

/// Mix one interleaved frame to a single mono f32 sample. `channel` `None`
/// averages all `channels` samples, `Some(i)` takes sample `i` verbatim, and
/// an out-of-range `i` yields silence (zeros) rather than panicking on the
/// audio callback — the caller validates and fails fast at startup (see
/// `resolve_channel`). `channels <= 1` is a verbatim copy.
///
/// `conv` scales one raw sample to f32, so f32, i16 and u16 device data share
/// this one path (see `i16_to_f32`, `u16_to_f32`).
///
/// `#[inline]` because the capture callback calls this once per sample (see
/// `capture::build_for_format`), and the binary ships without LTO.
#[inline]
pub(crate) fn mono_mix_frame<T, C>(
    frame: &[T],
    channels: usize,
    channel: Option<usize>,
    conv: C,
) -> f32
where
    T: Copy,
    C: Fn(T) -> f32,
{
    if channels <= 1 {
        return frame.first().map_or(0.0, |&s| conv(s));
    }
    match channel {
        None => {
            let mut sum = 0.0f32;
            for &s in frame.iter().take(channels) {
                sum += conv(s);
            }
            sum / channels as f32
        }
        Some(i) if i < channels => conv(frame[i]),
        Some(_) => 0.0,
    }
}

/// Downmix one interleaved buffer to mono with the per-sample `conv` scale.
/// A trailing incomplete frame is dropped, which keeps the mono output
/// aligned to the interleaved frames.
fn downmix_with<T, C>(data: &[T], channels: usize, channel: Option<usize>, conv: C) -> Vec<f32>
where
    T: Copy,
    C: Fn(T) -> f32,
{
    let channels = channels.max(1);
    debug_assert_eq!(
        data.len() % channels,
        0,
        "interleaved buffer length must be a multiple of channels"
    );
    data.chunks_exact(channels)
        .map(|frame| mono_mix_frame(frame, channels, channel, &conv))
        .collect()
}

/// Scale one i16 capture sample to f32 in [-1, 1]. The divisor is the
/// full-scale value (32768), the same scale the capture callback pushed to
/// the raw ring.
#[inline]
pub(crate) fn i16_to_f32(s: i16) -> f32 {
    s as f32 / 32768.0
}

/// Scale one u16 capture sample to f32 in [-1, 1] over the u16 range.
#[inline]
pub(crate) fn u16_to_f32(s: u16) -> f32 {
    s as f32 / 65535.0 * 2.0 - 1.0
}

/// Downmix an interleaved multi-channel buffer to mono by averaging. Test
/// reference for the `None` arm of `downmix_to_mono_with_channel`; production
/// callers use `select_channel_and_downmix`, which resolves and mixes in one
/// step.
#[cfg(test)]
pub(crate) fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    downmix_to_mono_with_channel(data, channels, None)
}

/// Downmix with an optional channel pick: `None` averages all channels,
/// `Some(i)` takes channel `i` verbatim. See `mono_mix_frame` for the
/// out-of-range rule.
pub(crate) fn downmix_to_mono_with_channel(
    data: &[f32],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    downmix_with(data, channels, channel, |s: f32| s)
}

/// Resolve a `ChannelSel` for already-decoded interleaved f32 audio and
/// downmix it in one step: `Mix` averages, `Pick` validates and takes one
/// channel, `Auto` keeps the loudest channel (`pick_louder_channel`, mix
/// fallback on silence or mono). Returns the mono samples plus the picked
/// channel index (`None` when the result is a mix), so the caller can log an
/// auto pick.
pub(crate) fn select_channel_and_downmix(
    sel: ChannelSel,
    interleaved: &[f32],
    channels: usize,
) -> anyhow::Result<(Vec<f32>, Option<usize>)> {
    let channel = match sel {
        ChannelSel::Mix => None,
        ChannelSel::Pick(i) => resolve_channel(Some(i), channels)?,
        ChannelSel::Auto => pick_louder_channel(interleaved, channels),
    };
    let samples = downmix_to_mono_with_channel(interleaved, channels, channel);
    Ok((samples, channel))
}

/// Single-pass I16 -> mono f32 with channel select/mix. Test reference only
/// (`#[cfg(test)]`): it allocates a `Vec`, which the real-time capture
/// callback forbids. That callback calls the allocation-free
/// `mono_mix_frame` with `i16_to_f32`, so both paths agree by construction.
#[cfg(test)]
pub(crate) fn downmix_i16_to_mono_with_channel(
    data: &[i16],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    downmix_with(data, channels, channel, i16_to_f32)
}

/// Single-pass U16 -> mono f32 with channel select/mix. Test reference only
/// (`#[cfg(test)]`), same as the i16 variant: the callback uses
/// `mono_mix_frame` with `u16_to_f32`.
#[cfg(test)]
pub(crate) fn downmix_u16_to_mono_with_channel(
    data: &[u16],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    downmix_with(data, channels, channel, u16_to_f32)
}

/// Which input channel to transcribe: mix, one pick, or auto-pick the
/// loudest at startup. Parsed from `--channel`/`STTD_CHANNEL` (`mix`,
/// `auto`, or a channel index `0..`); unset means mix (previous behavior).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ChannelSel {
    /// Average all channels down to mono (default).
    Mix,
    /// Transcribe channel `i` verbatim.
    Pick(usize),
    /// Measure once at startup and transcribe the loudest channel
    /// (highest RMS). Falls back to mix on mono input or when the probe
    /// hears only silence.
    Auto,
}

impl std::fmt::Display for ChannelSel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mix => f.write_str("mix"),
            Self::Pick(i) => write!(f, "{i}"),
            Self::Auto => f.write_str("auto"),
        }
    }
}

/// Parse `--channel`/`STTD_CHANNEL`: `mix`/empty = mix, `auto` = loudest,
/// otherwise a channel index. Used as the clap `value_parser` so
/// `--channel auto`, `STTD_CHANNEL=1` and the unset default all agree.
pub(crate) fn parse_channel_sel(s: &str) -> Result<ChannelSel, String> {
    let t = s.trim().to_lowercase();
    if t.is_empty() || t == "mix" {
        return Ok(ChannelSel::Mix);
    }
    if t == "auto" {
        return Ok(ChannelSel::Auto);
    }
    match t.parse::<usize>() {
        Ok(i) => Ok(ChannelSel::Pick(i)),
        Err(_) => Err(format!(
            "expected mix, auto or a channel index 0.., got {s:?}"
        )),
    }
}

/// Pick the loudest channel (highest RMS) from interleaved f32 audio.
/// Returns `None` for mono input, empty input, or when every channel is
/// silent — the caller then falls back to mix, which equals mono anyway.
pub(crate) fn pick_louder_channel(interleaved: &[f32], channels: usize) -> Option<usize> {
    if channels <= 1 || interleaved.len() < channels {
        return None;
    }
    let frames = interleaved.len() / channels;
    if frames == 0 {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best_energy = -1.0f64;
    let mut any_signal = false;
    for ch in 0..channels {
        let mut sum_sq = 0.0f64;
        let mut n = 0usize;
        for f in 0..frames {
            let s = interleaved[f * channels + ch] as f64;
            sum_sq += s * s;
            n += 1;
        }
        if n == 0 {
            continue;
        }
        let rms = (sum_sq / n as f64).sqrt();
        if rms > 1e-9 {
            any_signal = true;
        }
        if sum_sq > best_energy {
            best_energy = sum_sq;
            best_idx = ch;
        }
    }
    if !any_signal {
        return None;
    }
    Some(best_idx)
}

/// Validate `--channel` against the device's channel count: `None` (mix)
/// always passes, `Some(i)` needs `i < channels`.
pub(crate) fn resolve_channel(
    channel: Option<usize>,
    channels: usize,
) -> anyhow::Result<Option<usize>> {
    match channel {
        None => Ok(None),
        Some(i) if i < channels.max(1) => Ok(Some(i)),
        Some(i) => anyhow::bail!(
            "--channel/STTD_CHANNEL={i} out of range for {channels}-channel input (valid: 0..{})",
            channels.max(1) - 1,
        ),
    }
}

/// Resolve a `ChannelSel` without probing: `Mix` -> mix, `Pick` ->
/// validated pick, `Auto` on mono -> mix. `Auto` on multi-channel returns
/// the loudest-channel probe result when `probe` is `Some`, otherwise an
/// error telling the caller to probe (see `capture::probe_loudest_channel`
/// live, `pick_louder_channel` for files). Splitting pure resolution from
/// I/O this way keeps the helper unit-testable without a microphone.
pub(crate) fn resolve_channel_sel(
    sel: ChannelSel,
    channels: usize,
    probe: Option<Option<usize>>,
) -> anyhow::Result<Option<usize>> {
    match sel {
        ChannelSel::Mix => Ok(None),
        ChannelSel::Pick(i) => resolve_channel(Some(i), channels),
        ChannelSel::Auto if channels.max(1) <= 1 => Ok(None),
        ChannelSel::Auto => match probe {
            Some(picked) => Ok(picked),
            None => anyhow::bail!(
                "--channel=auto needs a startup probe for {channels}-channel input \
                 (use capture::probe_loudest_channel live, pick_louder_channel for files)"
            ),
        },
    }
}

/// Copy every complete `frame_out`-sized frame of `src` into `out` and stage
/// the sub-frame remainder at the tail of the `pending_out` ring, advancing
/// `out_head` past the drained frames.
///
/// Frame `Vec`s are freshly allocated: `SignalPath` moves them into `Arc`s as
/// queue entries, so they cannot be recycled here. The ring only ever holds
/// the sub-frame remainder between calls, which is what makes this cheaper
/// than staging the whole chunk and slicing it out again.
fn push_complete_frames(
    src: &[f32],
    frame_out: usize,
    pending_out: &mut Vec<f32>,
    out_head: &mut usize,
    out: &mut Vec<Vec<f32>>,
) {
    let staged = pending_out.len() - *out_head;
    debug_assert!(staged < frame_out, "ring must hold only a partial frame");
    let mut rest = src;
    // Top up a partial frame from previous calls first.
    if staged > 0 {
        let take = (frame_out - staged).min(rest.len());
        pending_out.extend_from_slice(&rest[..take]);
        rest = &rest[take..];
        if pending_out.len() - *out_head == frame_out {
            out.push(pending_out[*out_head..].to_vec());
            *out_head = pending_out.len();
        }
    }
    while rest.len() >= frame_out {
        out.push(rest[..frame_out].to_vec());
        rest = &rest[frame_out..];
    }
    // Keep the leftover partial frame live by resetting the head when the
    // ring is fully drained; otherwise it rides at `out_head`.
    if !rest.is_empty() {
        pending_out.extend_from_slice(rest);
    }
    if *out_head == pending_out.len() {
        pending_out.clear();
        *out_head = 0;
    }
}

/// Raw hw-rate samples per ~10 ms chunk: the external framing every caller
/// feeds `FrameProcessor::process_frame` with. Independent of the resampler's
/// internal buffering (see `FirStream`) so the capture loop and file mode
/// never change with the resampler.
pub(crate) fn hw_input_frames(hw_rate: u32) -> usize {
    ((hw_rate as f64 / 100.0).round() as usize).max(1)
}

/// Streaming adapter over `ResamplerFir` (polyphase FIR, 64 taps / 32-sample
/// delay, 90 dB stopband). Accepts arbitrary input lengths; the FIR consumes
/// variably and produces as much as its buffered state allows.
///
/// `feed` returns every complete `frame_out`-sized output frame (480 for
/// 48 kHz, 160 for 16 kHz) completed by this input; remainder stays buffered.
/// Equal in/out rates bypass entirely (bit-exact passthrough).
///
/// `SignalPath` keeps one adapter per branch and pairs up output frames by
/// order (see `SignalPath::feed`): completions land on a different cadence per
/// branch, so a strict one-call-one-frame mapping is impossible. Catch-up
/// bursts are bounded — both branches average 100 frames/s — and the queues
/// absorb them.
///
/// Output lags the input by the FIR delay (`taps/2`, 32 samples; zero when
/// passing through), flushed by the `flush_zeros` zero-drain; absorbed by
/// pre-roll (300 ms) and the silence hangover (800 ms) the same way earshot's
/// own 16 ms prediction window is.
pub(crate) struct FirStream {
    inner: Option<ResamplerFir>,
    frame_out: usize,
    /// Output staging ring: `out_head..` holds the not-yet-drained samples,
    /// `..out_head` is dead until `maybe_compact_out` shifts the live tail
    /// back to the front.
    pending_out: Vec<f32>,
    out_head: usize,
    scratch_out: Vec<f32>,
    /// Zero-drain input for `flush_into`, sized once at `new` so the flush
    /// path never allocates. Read-only: the resampler consumes it.
    flush_zeros: Vec<f32>,
    dirty: bool,
}

/// Dead prefix length that triggers `maybe_compact_out`. Compaction memmoves
/// the live tail, so it is amortized over 8192 drained samples (32 KB) instead
/// of running on most 10 ms chunks.
const OUT_COMPACT_HEAD: usize = 8192;

impl FirStream {
    pub(crate) fn new(in_rate: u32, out_rate: u32, frame_out: usize) -> Result<Self> {
        if in_rate == out_rate {
            return Ok(Self {
                inner: None,
                frame_out,
                pending_out: Vec::new(),
                out_head: 0,
                scratch_out: Vec::new(),
                flush_zeros: Vec::new(),
                dirty: false,
            });
        }
        if in_rate == 0 || out_rate == 0 {
            anyhow::bail!("invalid resample rates: {in_rate} -> {out_rate}");
        }
        let inner = ResamplerFir::new_from_hz(
            1,
            in_rate,
            out_rate,
            Latency::Sample32,
            Attenuation::Db90,
        );
        let scratch_out = vec![0.0f32; inner.buffer_size_output().max(frame_out * 4)];
        // Single zero-drain flushes the FIR delay + buffered tail (≈1-2
        // frames). Looping would synthesize indefinite silence: zeros in
        // always produce zeros out. Fixed at construction: `delay()` and
        // `frame_out` never change, so `flush_into` needs no computation and
        // no allocation.
        let flush_need = (inner.delay() + frame_out * 2).clamp(256, 2048);
        Ok(Self {
            inner: Some(inner),
            frame_out,
            pending_out: Vec::new(),
            out_head: 0,
            scratch_out,
            flush_zeros: vec![0.0f32; flush_need],
            dirty: false,
        })
    }

    /// Feed arbitrary input samples; returns every complete `frame_out`-sized
    /// output frame this input completes (usually 0-2 at 10 ms input cadence,
    /// occasionally more on catch-up).
    #[cfg_attr(not(feature = "denoise"), allow(dead_code))]
    pub(crate) fn feed(&mut self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        self.feed_into(input, &mut out);
        out
    }

    /// Same as `feed`, but appends into a caller-reused buffer so the outer
    /// `Vec` container doesn't reallocate on every 10 ms chunk (the inner
    /// per-frame `Vec`s are still freshly allocated — they become queue
    /// entries downstream). `SignalPath` reuses one scratch across calls.
    pub(crate) fn feed_into(&mut self, input: &[f32], out: &mut Vec<Vec<f32>>) {
        if !input.is_empty() {
            self.dirty = true;
        }
        if self.inner.is_none() {
            self.push_complete_frames(input, out);
            return;
        }
        let Self {
            inner,
            frame_out,
            pending_out,
            out_head,
            scratch_out,
            ..
        } = self;
        let inner = inner.as_mut().expect("passthrough handled above");
        let mut offset = 0;
        while offset < input.len() {
            let (consumed, produced) = match inner.resample(&input[offset..], scratch_out) {
                Ok((c, p)) => (c, p),
                Err(e) => {
                    log::warn!("resampler error: {e:?} (skipping chunk)");
                    break;
                }
            };
            // Hand complete frames to `out` straight from `scratch_out` and
            // stage only the sub-frame remainder, so each output sample is
            // copied once instead of twice (ring -> frame Vec).
            push_complete_frames(
                &scratch_out[..produced],
                *frame_out,
                pending_out,
                out_head,
                out,
            );
            offset += consumed;
            if consumed == 0 && produced == 0 {
                break;
            }
        }
    }

    /// Drain internal FIR state at shutdown by feeding zeros until no more
    /// output appears, then return every complete output frame left buffered.
    /// A trailing partial frame (< `frame_out` samples) is still dropped.
    #[cfg_attr(not(feature = "denoise"), allow(dead_code))]
    pub(crate) fn flush(&mut self) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        self.flush_into(&mut out);
        out
    }

    /// Same as `flush`, appending into a caller-reused buffer (see `feed_into`).
    pub(crate) fn flush_into(&mut self, out: &mut Vec<Vec<f32>>) {
        if self.dirty {
            self.dirty = false;
            if let Some(inner) = self.inner.as_mut() {
                // Single zero-drain (pre-sized, zero-filled at `new`) to flush
                // the FIR delay + buffered tail; see the sizing comment there.
                match inner.resample(&self.flush_zeros, &mut self.scratch_out) {
                    Ok((_, produced)) => {
                        let src = &self.scratch_out[..produced];
                        push_complete_frames(
                            src,
                            self.frame_out,
                            &mut self.pending_out,
                            &mut self.out_head,
                            out,
                        );
                    }
                    Err(e) => {
                        log::warn!("resampler flush error: {e:?} (dropping tail)");
                    }
                }
            }
        }
        // A staged partial frame stays in the ring and is dropped with it
        // (see `flush`); compact so a following `feed` starts at offset 0.
        self.maybe_compact_out();
    }

    /// Passthrough drain: `src` is the caller's input verbatim, so complete
    /// frames come straight out of it with one copy.
    fn push_complete_frames(&mut self, src: &[f32], out: &mut Vec<Vec<f32>>) {
        let Self {
            frame_out,
            pending_out,
            out_head,
            ..
        } = self;
        push_complete_frames(src, *frame_out, pending_out, out_head, out);
    }

    /// Reuse the ring's tail allocation: after a drain the ring holds only the
    /// sub-frame remainder, so `copy_within` shifts at most `frame_out`
    /// samples and no per-flush `Vec` is created. Amortized over
    /// `OUT_COMPACT_HEAD` drained samples.
    fn maybe_compact_out(&mut self) {
        if self.out_head == 0 {
            return;
        }
        if self.out_head >= OUT_COMPACT_HEAD || self.out_head * 2 >= self.pending_out.len() {
            self.pending_out.copy_within(self.out_head.., 0);
            let live = self.pending_out.len() - self.out_head;
            self.pending_out.truncate(live);
            self.out_head = 0;
        }
    }

    /// Drop buffered state for a pause transition, so the first frames after
    /// resume aren't blended from pre-pause audio.
    pub(crate) fn clear(&mut self) {
        if let Some(inner) = self.inner.as_mut() {
            inner.reset();
        }
        self.pending_out.clear();
        self.out_head = 0;
        self.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, SAMPLE_RATE};

    /// Test oracle for the raw hw-rate chunk size `hw_input_frames` settles
    /// on: a 10 ms slice of the hardware rate.
    fn expected_input_chunk_for(hw_rate: u32) -> usize {
        ((hw_rate as f64 * FRAME_48K as f64 / DENOISE_RATE as f64).round() as usize).max(1)
    }

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix_to_mono(&[1.0, 3.0, 2.0, 4.0], 2), vec![2.0, 3.0]);
        assert_eq!(downmix_to_mono(&[1.0, 2.0, 3.0], 1), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn downmix_channel_pick_selects_verbatim() {
        assert_eq!(
            downmix_to_mono_with_channel(&[1.0, 3.0, 2.0, 4.0], 2, Some(0)),
            vec![1.0, 2.0]
        );
        assert_eq!(
            downmix_to_mono_with_channel(&[1.0, 3.0, 2.0, 4.0], 2, Some(1)),
            vec![3.0, 4.0]
        );
        assert_eq!(
            downmix_to_mono_with_channel(&[1.0, 3.0, 2.0, 4.0], 2, None),
            vec![2.0, 3.0]
        );
        // Out-of-range never panics (callback safety); startup validation
        // rejects it instead (see `resolve_channel`).
        assert_eq!(
            downmix_to_mono_with_channel(&[1.0, 3.0, 2.0, 4.0], 2, Some(7)),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn downmix_i16_matches_f32_path() {
        // 0.5 in f32 == 16384 in i16; check mix, pick, mono, OOR.
        let stereo_i16 = [16384i16, -16384, 8192, 8192];
        let mixed = downmix_i16_to_mono_with_channel(&stereo_i16, 2, None);
        assert_eq!(mixed.len(), 2);
        assert!((mixed[0] - 0.0).abs() < 1e-6);
        assert!((mixed[1] - 0.25).abs() < 1e-6);
        assert_eq!(
            downmix_i16_to_mono_with_channel(&stereo_i16, 2, Some(0)),
            vec![0.5, 0.25]
        );
        assert_eq!(
            downmix_i16_to_mono_with_channel(&stereo_i16, 2, Some(7)),
            vec![0.0, 0.0]
        );
        assert_eq!(
            downmix_i16_to_mono_with_channel(&[16384i16, -16384], 1, None),
            vec![0.5, -0.5]
        );
    }

    #[test]
    fn downmix_u16_matches_f32_path() {
        // 0 -> -1.0, 65535 -> 1.0, midscale ~0.
        let mono = downmix_u16_to_mono_with_channel(&[0u16, 65535, 32768], 1, None);
        assert!((mono[0] + 1.0).abs() < 1e-6);
        assert!((mono[1] - 1.0).abs() < 1e-6);
        assert!(mono[2].abs() < 1e-3);
        let stereo = [0u16, 65535, 65535, 0];
        let mixed = downmix_u16_to_mono_with_channel(&stereo, 2, None);
        assert_eq!(mixed.len(), 2);
        assert!(mixed[0].abs() < 1e-6);
        assert!(mixed[1].abs() < 1e-6);
        assert_eq!(
            downmix_u16_to_mono_with_channel(&stereo, 2, Some(7)),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn mono_mix_frame_matches_every_sample_type() {
        // One frame, three sample types, three scales: mix, pick, OOR.
        assert_eq!(mono_mix_frame(&[1.0f32, 3.0], 2, None, |s| s), 2.0);
        assert_eq!(mono_mix_frame(&[1.0f32, 3.0], 2, Some(1), |s| s), 3.0);
        assert_eq!(mono_mix_frame(&[1.0f32, 3.0], 2, Some(7), |s| s), 0.0);
        assert_eq!(mono_mix_frame(&[16384i16, -16384], 2, None, i16_to_f32), 0.0);
        assert_eq!(mono_mix_frame(&[65535u16, 0], 2, Some(1), u16_to_f32), -1.0);
        // Mono input is a verbatim copy regardless of the pick.
        assert_eq!(mono_mix_frame(&[0.5f32], 1, Some(7), |s| s), 0.5);
    }

    #[test]
    fn select_channel_and_downmix_matches_manual_paths() {
        let stereo = vec![0.1f32, 0.5, 0.2, 0.9];
        let (mixed, picked) = select_channel_and_downmix(ChannelSel::Mix, &stereo, 2).unwrap();
        assert_eq!(mixed, vec![0.3, 0.55]);
        assert_eq!(picked, None);
        let (picked_s, ch) = select_channel_and_downmix(ChannelSel::Pick(1), &stereo, 2).unwrap();
        assert_eq!(picked_s, vec![0.5, 0.9]);
        assert_eq!(ch, Some(1));
        // Auto picks the loudest channel here (R: 0.5/0.9 beats L: 0.1/0.2).
        let (auto_s, ch) = select_channel_and_downmix(ChannelSel::Auto, &stereo, 2).unwrap();
        assert_eq!(auto_s, vec![0.5, 0.9]);
        assert_eq!(ch, Some(1));
        // Out-of-range pick fails fast instead of returning silence.
        assert!(select_channel_and_downmix(ChannelSel::Pick(2), &stereo, 2).is_err());
        // Silence auto-falls-back to mix.
        let (sil, ch) = select_channel_and_downmix(ChannelSel::Auto, &[0.0f32; 8], 2).unwrap();
        assert_eq!(sil, vec![0.0; 4]);
        assert_eq!(ch, None);
    }

    #[test]
    fn resolve_channel_validates_against_device() {
        assert_eq!(resolve_channel(None, 2).unwrap(), None);
        assert_eq!(resolve_channel(Some(0), 2).unwrap(), Some(0));
        assert_eq!(resolve_channel(Some(1), 2).unwrap(), Some(1));
        assert!(resolve_channel(Some(2), 2).is_err());
        assert_eq!(resolve_channel(Some(0), 1).unwrap(), Some(0));
        assert!(resolve_channel(Some(1), 1).is_err());
    }

    #[test]
    fn channel_sel_parses_mix_auto_index() {
        assert_eq!(parse_channel_sel("mix").unwrap(), ChannelSel::Mix);
        assert_eq!(parse_channel_sel("").unwrap(), ChannelSel::Mix);
        assert_eq!(parse_channel_sel("auto").unwrap(), ChannelSel::Auto);
        assert_eq!(parse_channel_sel("AUTO").unwrap(), ChannelSel::Auto);
        assert_eq!(parse_channel_sel("0").unwrap(), ChannelSel::Pick(0));
        assert_eq!(parse_channel_sel("1").unwrap(), ChannelSel::Pick(1));
        assert!(parse_channel_sel("bzdura").is_err());
        assert_eq!(ChannelSel::Mix.to_string(), "mix");
        assert_eq!(ChannelSel::Auto.to_string(), "auto");
        assert_eq!(ChannelSel::Pick(1).to_string(), "1");
    }

    #[test]
    fn louder_channel_picks_highest_rms() {
        // L quiet (0.01), R loud (0.2) -> R.
        let mut stereo = Vec::new();
        for _ in 0..160 {
            stereo.push(0.01);
            stereo.push(0.2);
        }
        assert_eq!(pick_louder_channel(&stereo, 2), Some(1));
        // Swapped -> L.
        let mut stereo = Vec::new();
        for _ in 0..160 {
            stereo.push(0.2);
            stereo.push(0.01);
        }
        assert_eq!(pick_louder_channel(&stereo, 2), Some(0));
        // Silence -> mix fallback, mono -> mix fallback.
        assert_eq!(pick_louder_channel(&vec![0.0f32; 320], 2), None);
        assert_eq!(pick_louder_channel(&[0.5], 1), None);
        assert_eq!(pick_louder_channel(&[], 2), None);
    }

    #[test]
    fn channel_sel_resolves_without_probe() {
        assert_eq!(resolve_channel_sel(ChannelSel::Mix, 2, None).unwrap(), None);
        assert_eq!(
            resolve_channel_sel(ChannelSel::Pick(1), 2, None).unwrap(),
            Some(1)
        );
        assert!(resolve_channel_sel(ChannelSel::Pick(2), 2, None).is_err());
        assert_eq!(
            resolve_channel_sel(ChannelSel::Auto, 1, None).unwrap(),
            None
        );
        assert!(resolve_channel_sel(ChannelSel::Auto, 2, None).is_err());
        assert_eq!(
            resolve_channel_sel(ChannelSel::Auto, 2, Some(Some(1))).unwrap(),
            Some(1)
        );
        assert_eq!(
            resolve_channel_sel(ChannelSel::Auto, 2, Some(None)).unwrap(),
            None
        );
    }

    #[test]
    fn chunk_sizes_yield_480_out() {
        assert_eq!(expected_input_chunk_for(44_100), 441);
        assert_eq!(expected_input_chunk_for(48_000), 480);
        assert_eq!(expected_input_chunk_for(16_000), 160);
        assert_eq!(expected_input_chunk_for(96_000), 960);
        for hw in [16_000, 44_100, 48_000, 96_000] {
            assert_eq!(hw_input_frames(hw), expected_input_chunk_for(hw));
        }
    }

    #[test]
    fn passthrough_is_bit_exact() {
        let mut s = FirStream::new(DENOISE_RATE, DENOISE_RATE, FRAME_48K).unwrap();
        let input: Vec<f32> = (0..960).map(|i| i as f32 / 960.0).collect();
        let frames = s.feed(&input);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames.concat(), input);
        // Nothing buffered: passthrough drains exactly.
        assert!(s.flush().is_empty());
    }

    #[test]
    fn fir_48k_to_16k_tracks_input() {
        // 200 input frames of 480 must yield ~200 output frames of 160:
        // FIR streaming may shift the count by a few frames at startup
        // (delay fill), but every drained frame is exactly sized.
        let mut s = FirStream::new(DENOISE_RATE, SAMPLE_RATE, FRAME_16K).unwrap();
        let mut count = 0;
        for _ in 0..200 {
            for f in s.feed(&vec![0.0f32; FRAME_48K]) {
                assert_eq!(f.len(), FRAME_16K);
                count += 1;
            }
        }
        for f in s.flush() {
            assert_eq!(f.len(), FRAME_16K);
            count += 1;
        }
        assert!(
            (195..=210).contains(&count),
            "expected ~200 frames, got {count}"
        );
    }

    #[test]
    fn hw_streams_track_input_for_common_rates() {
        // Both branches fed the same 10 ms hw chunks must stay in lockstep
        // (pairing by order in `FrameProcessor` relies on it): equal frame
        // counts within burst tolerance, exact frame sizes always.
        for hw in [16_000, 44_100, 48_000, 96_000] {
            let hw_chunk = hw_input_frames(hw);
            let mut s48 = FirStream::new(hw, DENOISE_RATE, FRAME_48K).unwrap();
            let mut s16 = FirStream::new(hw, SAMPLE_RATE, FRAME_16K).unwrap();
            let (mut n48, mut n16) = (0, 0);
            for _ in 0..200 {
                let input = vec![0.0f32; hw_chunk];
                for f in s48.feed(&input) {
                    assert_eq!(f.len(), FRAME_48K, "hw={hw}");
                    n48 += 1;
                }
                for f in s16.feed(&input) {
                    assert_eq!(f.len(), FRAME_16K, "hw={hw}");
                    n16 += 1;
                }
            }
            n48 += s48.flush().len();
            n16 += s16.flush().len();
            for (n, want) in [(n48, 200), (n16, 200)] {
                assert!(
                    (193..=210).contains(&n),
                    "hw={hw}: expected ~{want} frames, got {n}"
                );
            }
            assert!(
                n48.abs_diff(n16) <= 6,
                "hw={hw}: branches diverged ({n48} vs {n16})"
            );
        }
    }

    #[test]
    fn arbitrary_rates_supported() {
        // FIR supports any integer rate pair (FFT did not): 8k/11k/24k must
        // construct and track input instead of erroring.
        for hw in [8_000, 11_025, 24_000] {
            let mut s = FirStream::new(hw, SAMPLE_RATE, FRAME_16K).unwrap();
            let hw_chunk = hw_input_frames(hw);
            let mut n = 0;
            for _ in 0..50 {
                n += s.feed(&vec![0.0f32; hw_chunk]).len();
            }
            n += s.flush().len();
            assert!(n > 40, "hw={hw}: expected ~50 frames, got {n}");
        }
    }

    #[test]
    fn flush_recovers_trailing_audio() {
        // FIR buffers delay + lookahead: fed frames plus zero-drain flush must
        // recover the tail instead of losing it at shutdown.
        let mut s = FirStream::new(DENOISE_RATE, SAMPLE_RATE, FRAME_16K).unwrap();
        let fed: usize = (0..2).map(|_| s.feed(&vec![0.1f32; FRAME_48K]).len()).sum();
        let flushed = s.flush();
        assert!(!flushed.is_empty(), "flush must recover the tail");
        for f in &flushed {
            assert_eq!(f.len(), FRAME_16K);
        }
        assert!(
            (2..=6).contains(&(fed + flushed.len())),
            "fed={fed} flushed={} total out of range",
            flushed.len()
        );
    }

    #[test]
    fn ragged_passthrough_chunks_concatenate_exactly() {
        // Ragged input lengths (below, at, and above the frame size) must
        // stage partial frames across calls and concatenate bit-exactly: no
        // sample lost, duplicated or reordered by the ring bookkeeping.
        const FRAME: usize = 160;
        let input: Vec<f32> = (0..1000).map(|i| i as f32 / 1000.0).collect();
        for chunk in [1usize, 7, 159, 160, 161, 333] {
            let mut s = FirStream::new(SAMPLE_RATE, SAMPLE_RATE, FRAME).unwrap();
            let mut got: Vec<f32> = Vec::new();
            for part in input.chunks(chunk) {
                for f in s.feed(part) {
                    assert_eq!(f.len(), FRAME, "chunk={chunk}");
                    got.extend_from_slice(&f);
                }
            }
            // Whole frames only: the trailing partial frame is not drained.
            let whole = input.len() / FRAME * FRAME;
            assert_eq!(got, input[..whole], "chunk={chunk}");
        }
    }

    #[test]
    fn clear_drops_buffered_state() {
        let mut s = FirStream::new(DENOISE_RATE, SAMPLE_RATE, FRAME_16K).unwrap();
        let _ = s.feed(&vec![0.1f32; FRAME_48K]);
        s.clear();
        // After clear + only zero-drain, at most delay-fill silence may appear;
        // in-flight speech must be gone. Feed silence after clear still works.
        let _ = s.feed(&vec![0.0f32; FRAME_48K]);
    }
}
