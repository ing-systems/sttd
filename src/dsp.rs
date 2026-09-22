use anyhow::{Context, Result};
use resampler::{ResamplerFft, SampleRate};

// ---------------------------------------------------------------------------
// Signal primitives: channel downmix + resampling
// ---------------------------------------------------------------------------

/// Downmix an interleaved multi-channel buffer to mono by averaging.
pub(crate) fn downmix_to_mono(data: &[f32], channels: usize) -> Vec<f32> {
    downmix_to_mono_with_channel(data, channels, None)
}

/// Downmix with an optional channel pick: `None` averages all channels (see
/// `downmix_to_mono`), `Some(i)` takes channel `i` verbatim. An out-of-range
/// `i` falls back to silence (zeros) rather than panicking on the audio
/// callback — the caller validates and fails fast at startup (see
/// `resolve_channel`).
pub(crate) fn downmix_to_mono_with_channel(
    data: &[f32],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    debug_assert!(channels > 0, "channels must be >= 1 (cpal guarantees this)");
    debug_assert_eq!(
        data.len() % channels.max(1),
        0,
        "interleaved buffer length must be a multiple of channels"
    );
    if channels <= 1 {
        return data.to_vec();
    }
    match channel {
        None => data
            .chunks_exact(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect(),
        Some(i) if i < channels => data
            .chunks_exact(channels)
            .map(|frame| frame[i])
            .collect(),
        Some(_) => vec![0.0f32; data.len() / channels.max(1)],
    }
}

/// Single-pass I16 -> mono f32 with channel select/mix. Avoids the
/// intermediate f32 buffer the callback used to build before downmixing,
/// halving per-callback allocs on I16 devices. Same scale as before:
/// `s / 32768.0`. Same out-of-range rule as the f32 variant: silence.
pub(crate) fn downmix_i16_to_mono_with_channel(
    data: &[i16],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    let frames = data.len() / channels.max(1);
    if channels <= 1 {
        return data.iter().map(|&s| s as f32 / 32768.0).collect();
    }
    match channel {
        None => {
            let inv = 1.0 / channels as f32;
            let mut out = Vec::with_capacity(frames);
            for frame in data.chunks_exact(channels) {
                let mut sum = 0.0f32;
                for &s in frame {
                    sum += s as f32 / 32768.0;
                }
                out.push(sum * inv);
            }
            out
        }
        Some(i) if i < channels => {
            let mut out = Vec::with_capacity(frames);
            for frame in data.chunks_exact(channels) {
                out.push(frame[i] as f32 / 32768.0);
            }
            out
        }
        Some(_) => vec![0.0f32; frames],
    }
}

/// Single-pass U16 -> mono f32 with channel select/mix. Same scale as
/// before: `s / 65535.0 * 2.0 - 1.0`. Same out-of-range rule: silence.
pub(crate) fn downmix_u16_to_mono_with_channel(
    data: &[u16],
    channels: usize,
    channel: Option<usize>,
) -> Vec<f32> {
    let frames = data.len() / channels.max(1);
    if channels <= 1 {
        return data
            .iter()
            .map(|&s| s as f32 / 65535.0 * 2.0 - 1.0)
            .collect();
    }
    match channel {
        None => {
            let inv = 1.0 / channels as f32;
            let mut out = Vec::with_capacity(frames);
            for frame in data.chunks_exact(channels) {
                let mut sum = 0.0f32;
                for &s in frame {
                    sum += s as f32 / 65535.0 * 2.0 - 1.0;
                }
                out.push(sum * inv);
            }
            out
        }
        Some(i) if i < channels => {
            let mut out = Vec::with_capacity(frames);
            for frame in data.chunks_exact(channels) {
                out.push(frame[i] as f32 / 65535.0 * 2.0 - 1.0);
            }
            out
        }
        Some(_) => vec![0.0f32; frames],
    }
}

/// Validate `--channel` against the device's channel count: `None` (mix)
/// always passes, `Some(i)` needs `i < channels`.
pub(crate) fn resolve_channel(channel: Option<usize>, channels: usize) -> anyhow::Result<Option<usize>> {
    match channel {
        None => Ok(None),
        Some(i) if i < channels.max(1) => Ok(Some(i)),
        Some(i) => anyhow::bail!(
            "--channel/STTD_CHANNEL={i} out of range for {channels}-channel input (valid: 0..{})",
            channels.max(1) - 1,
        ),
    }
}

/// Raw hw-rate samples per ~10 ms chunk: the external framing every caller
/// feeds `FrameProcessor::process_frame` with. Deliberately independent of
/// `ResamplerFft`'s own fixed chunk sizes (see `FftStream`) so the capture
/// loop and file mode never change with the resampler.
pub(crate) fn hw_input_frames(hw_rate: u32) -> usize {
    ((hw_rate as f64 / 100.0).round() as usize).max(1)
}

/// Streaming adapter over `ResamplerFft`, whose chunk sizes are fixed
/// internally (e.g. 1176 in / 1280 out for 44.1 kHz -> 48 kHz) instead of
/// caller-chosen like rubato's used to be.
///
/// `feed` accepts any input length and returns the complete `frame_out`-sized
/// output frames (480 for 48 kHz, 160 for 16 kHz) it completes; any remainder
/// stays buffered for the next call. Equal in/out rates bypass the FFT
/// entirely (bit-exact passthrough) instead of smearing the signal through a
/// needless round trip.
///
/// Both branches keep their own adapter and pair up output frames by order
/// (see `FrameProcessor::process_frame`): the fixed chunks complete on a
/// different cadence per branch, so a strict one-call-one-frame mapping is
/// impossible. Catch-up bursts are bounded — both branches average 100
/// frames/s — and the queues absorb them.
///
/// The first output chunk carries the FFT overlap-add startup transient
/// (overlaps start at zero) and every output lags the input by `delay()`
/// input samples; both are absorbed by pre-roll (300 ms) and the silence
/// hangover (800 ms) the same way earshot's own 16 ms prediction window is.
pub(crate) struct FftStream {
    inner: Option<ResamplerFft>,
    chunk_in: usize,
    chunk_out: usize,
    frame_out: usize,
    pending_in: Vec<f32>,
    /// Consumed prefix of `pending_in` (see `feed_into`): advancing a head
    /// index avoids an O(n) `drain(..chunk_in)` memmove per FFT chunk on the
    /// 100 fps hot path; the buffer is compacted only when the head grows
    /// large.
    in_head: usize,
    pending_out: Vec<f32>,
    /// Consumed prefix of `pending_out` (see `drain_frames_into`): same
    /// rationale as `in_head`, one memmove per ~50 frames instead of one per
    /// frame.
    out_head: usize,
    scratch_in: Vec<f32>,
    scratch_out: Vec<f32>,
}

impl FftStream {
    pub(crate) fn new(in_rate: u32, out_rate: u32, frame_out: usize) -> Result<Self> {
        if in_rate == out_rate {
            return Ok(Self {
                inner: None,
                chunk_in: frame_out,
                chunk_out: frame_out,
                frame_out,
                pending_in: Vec::new(),
                in_head: 0,
                pending_out: Vec::new(),
                out_head: 0,
                scratch_in: Vec::new(),
                scratch_out: Vec::new(),
            });
        }
        let in_sr = SampleRate::try_from(in_rate)
            .map_err(|_| supported_rates_error(in_rate))
            .with_context(|| format!("unsupported input sample rate for ResamplerFft: {in_rate} Hz"))?;
        let out_sr = SampleRate::try_from(out_rate)
            .map_err(|_| supported_rates_error(out_rate))
            .with_context(|| {
                format!("unsupported output sample rate for ResamplerFft: {out_rate} Hz")
            })?;
        let inner = ResamplerFft::new(1, in_sr, out_sr);
        let chunk_in = inner.chunk_size_input();
        let chunk_out = inner.chunk_size_output();
        let scratch_in = vec![0.0f32; chunk_in];
        let scratch_out = vec![0.0f32; chunk_out];
        Ok(Self {
            inner: Some(inner),
            chunk_in,
            chunk_out,
            frame_out,
            pending_in: Vec::new(),
            in_head: 0,
            pending_out: Vec::new(),
            out_head: 0,
            scratch_in,
            scratch_out,
        })
    }

    /// Fixed FFT chunk sizes `(in, out)`; `(frame_out, frame_out)` when
    /// passing through. For startup logging only.
    pub(crate) fn chunk_sizes(&self) -> (usize, usize) {
        (self.chunk_in, self.chunk_out)
    }

    /// Feed arbitrary input samples; returns every complete `frame_out`-sized
    /// output frame this input completes (usually 0-2 at 10 ms input
    /// cadence, occasionally more on catch-up).
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
        match self.inner.as_mut() {
            Some(inner) => {
                self.pending_in.extend_from_slice(input);
                while self.pending_in.len() - self.in_head >= self.chunk_in {
                    self.scratch_in
                        .copy_from_slice(&self.pending_in[self.in_head..self.in_head + self.chunk_in]);
                    // Scratch buffers are exactly chunk-sized, the only error
                    // condition on mono input. Never panic on the hot path:
                    // skip the chunk and keep the pipeline running.
                    if let Err(e) = inner.resample(&self.scratch_in, &mut self.scratch_out) {
                        log::warn!("resampler error: {e:?} (skipping chunk)");
                        self.in_head += self.chunk_in;
                        continue;
                    }
                    self.pending_out.extend_from_slice(&self.scratch_out);
                    self.in_head += self.chunk_in;
                }
                self.maybe_compact_in();
            }
            None => self.pending_out.extend_from_slice(input),
        }
        self.drain_frames_into(out)
    }

    /// Zero-pad a trailing partial input chunk through one final resample so
    /// tail audio (up to one FFT chunk) isn't lost at shutdown, then return
    /// every complete output frame left buffered. A trailing partial frame
    /// (< `frame_out` samples) is still dropped, same as before.
    pub(crate) fn flush(&mut self) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        self.flush_into(&mut out);
        out
    }

    /// Same as `flush`, appending into a caller-reused buffer (see
    /// `feed_into`).
    pub(crate) fn flush_into(&mut self, out: &mut Vec<Vec<f32>>) {
        if self.inner.is_some() && self.pending_in.len() > self.in_head {
            let remaining = self.pending_in.len() - self.in_head;
            debug_assert!(remaining < self.chunk_in);
            self.scratch_in[..remaining]
                .copy_from_slice(&self.pending_in[self.in_head..]);
            self.scratch_in[remaining..].fill(0.0);
            self.pending_in.clear();
            self.in_head = 0;
            // Same guarantee as in `feed_into`: exact chunk size, cannot fail.
            // Never panic on shutdown either; drop the tail on error.
            if let Some(inner) = self.inner.as_mut() {
                if let Err(e) = inner.resample(&self.scratch_in, &mut self.scratch_out) {
                    log::warn!("resampler flush error: {e:?} (dropping tail)");
                } else {
                    self.pending_out.extend_from_slice(&self.scratch_out);
                }
            }
        }
        self.drain_frames_into(out)
    }

    fn drain_frames_into(&mut self, out: &mut Vec<Vec<f32>>) {
        while self.pending_out.len() - self.out_head >= self.frame_out {
            out.push(
                self.pending_out[self.out_head..self.out_head + self.frame_out].to_vec(),
            );
            self.out_head += self.frame_out;
        }
        self.maybe_compact_out();
    }

    /// Drop the consumed prefix once it grows large, so the buffers stay
    /// bounded without a memmove per chunk/frame.
    fn maybe_compact_in(&mut self) {
        if self.in_head >= 8192 || (self.in_head > 0 && self.in_head * 2 >= self.pending_in.len()) {
            self.pending_in.drain(..self.in_head);
            self.in_head = 0;
        }
    }

    fn maybe_compact_out(&mut self) {
        if self.out_head >= 8192
            || (self.out_head > 0 && self.out_head * 2 >= self.pending_out.len())
        {
            self.pending_out.drain(..self.out_head);
            self.out_head = 0;
        }
    }

    /// Drop buffered partial-chunk/partial-frame state for a pause
    /// transition, so the first frames after resume aren't blended from
    /// pre-pause audio.
    pub(crate) fn clear(&mut self) {
        self.pending_in.clear();
        self.in_head = 0;
        self.pending_out.clear();
        self.out_head = 0;
    }
}

fn supported_rates_error(rate: u32) -> anyhow::Error {
    anyhow::anyhow!(
        "{rate} Hz is not a ResamplerFft rate (supported: 16000, 22050, 32000, 44100, \
         48000, 88200, 96000, 176400, 192000, 384000)"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{DENOISE_RATE, FRAME_16K, FRAME_48K, WHISPER_RATE};

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
    fn resolve_channel_validates_against_device() {
        assert_eq!(resolve_channel(None, 2).unwrap(), None);
        assert_eq!(resolve_channel(Some(0), 2).unwrap(), Some(0));
        assert_eq!(resolve_channel(Some(1), 2).unwrap(), Some(1));
        assert!(resolve_channel(Some(2), 2).is_err());
        assert_eq!(resolve_channel(Some(0), 1).unwrap(), Some(0));
        assert!(resolve_channel(Some(1), 1).is_err());
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
        let mut s = FftStream::new(DENOISE_RATE, DENOISE_RATE, FRAME_48K).unwrap();
        let input: Vec<f32> = (0..960).map(|i| i as f32 / 960.0).collect();
        let frames = s.feed(&input);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames.concat(), input);
        // Nothing buffered: passthrough drains exactly.
        assert!(s.flush().is_empty());
    }

    #[test]
    fn fft_48k_to_16k_tracks_input() {
        // 200 input frames of 480 must yield ~200 output frames of 160:
        // buffering/short bursts may shift the count by a frame or two, but
        // every drained frame is exactly sized.
        let mut s = FftStream::new(DENOISE_RATE, WHISPER_RATE, FRAME_16K).unwrap();
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
            (198..=203).contains(&count),
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
            let mut s48 = FftStream::new(hw, DENOISE_RATE, FRAME_48K).unwrap();
            let mut s16 = FftStream::new(hw, WHISPER_RATE, FRAME_16K).unwrap();
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
                assert!((198..=203).contains(&n), "hw={hw}: expected ~{want} frames, got {n}");
            }
            assert!(
                n48.abs_diff(n16) <= 2,
                "hw={hw}: branches diverged ({n48} vs {n16})"
            );
        }
    }

    #[test]
    fn unsupported_rates_error() {
        assert!(FftStream::new(8_000, DENOISE_RATE, FRAME_48K).is_err());
        assert!(FftStream::new(11_025, WHISPER_RATE, FRAME_16K).is_err());
        assert!(FftStream::new(24_000, DENOISE_RATE, FRAME_48K).is_err());
    }

    #[test]
    fn flush_recovers_trailing_partial_chunk() {
        // 48k -> 16k needs 768 samples per FFT chunk: two 480-frame feeds
        // complete one chunk (256 out), leaving 192 buffered. Without a
        // flush that tail audio would be lost at shutdown.
        let mut s = FftStream::new(DENOISE_RATE, WHISPER_RATE, FRAME_16K).unwrap();
        let fed: usize = (0..2)
            .map(|_| s.feed(&vec![0.1f32; FRAME_48K]).len())
            .sum();
        assert_eq!(fed, 1);
        let flushed = s.flush();
        assert!(!flushed.is_empty(), "flush must recover the tail");
        for f in &flushed {
            assert_eq!(f.len(), FRAME_16K);
        }
        // Steady state plus the padded tail: 960 in -> one 768-chunk (256
        // out) fed, then the 192-sample remainder padded to a full chunk
        // (256 more out) = 512 samples = 3 frames; the last 32 samples are
        // a partial frame and stay dropped.
        assert_eq!(fed + flushed.len(), 3);
    }

    #[test]
    fn clear_drops_partial_chunk_state() {
        let mut s = FftStream::new(DENOISE_RATE, WHISPER_RATE, FRAME_16K).unwrap();
        // One 480-frame feed completes no full chunk (768 needed), so the
        // input sits in pending_in; after clear, flush must find nothing.
        assert!(s.feed(&vec![0.1f32; FRAME_48K]).is_empty());
        s.clear();
        assert!(s.flush().is_empty());
    }
}
