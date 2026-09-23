// TDT greedy decode state machine, mirroring `parakeet-rs` `model_tdt.rs`:
// per-frame joint logits `[vocab + n_durations]`, argmax token + argmax
// duration, blank = vocab-1, duration>0 skips frames, max 10 tokens/step.
// Pure over logits so it tests without weights or Burn.

/// One joint step input: token logits + duration logits.
pub(crate) struct JointStep {
    pub(crate) token_logits: Vec<f32>,
    pub(crate) duration_logits: Vec<f32>,
}

pub(crate) struct TdtGreedy {
    vocab_size: usize,
    blank_id: usize,
    max_tokens_per_step: usize,
}

impl TdtGreedy {
    pub(crate) fn new(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            blank_id: vocab_size.saturating_sub(1),
            max_tokens_per_step: 10,
        }
    }

    pub(crate) fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Decode `T` frames. `joint(t, last_token)` runs the decoder+joint
    /// for frame `t` (Burn path) or returns canned logits (tests).
    pub(crate) fn decode(
        &self,
        n_frames: usize,
        mut joint: impl FnMut(usize, usize) -> JointStep,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        let mut tokens = Vec::new();
        let mut frames = Vec::new();
        let mut durations = Vec::new();
        let mut t = 0usize;
        let mut emitted = 0usize;
        let mut last = self.blank_id;
        while t < n_frames {
            let step = joint(t, last);
            let tok = Self::argmax(&step.token_logits);
            let dur = if step.duration_logits.is_empty() {
                0
            } else {
                Self::argmax(&step.duration_logits)
            };
            if tok != self.blank_id {
                tokens.push(tok);
                frames.push(t);
                durations.push(dur);
                last = tok;
                emitted += 1;
            }
            if dur > 0 {
                t += dur;
                emitted = 0;
            } else if tok == self.blank_id || emitted >= self.max_tokens_per_step {
                t += 1;
                emitted = 0;
            }
        }
        (tokens, frames, durations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step_with(tok: usize, dur: usize, vocab: usize, n_dur: usize) -> JointStep {
        let mut tl = vec![-10.0; vocab];
        tl[tok] = 10.0;
        let mut dl = vec![-10.0; n_dur];
        dl[dur.min(n_dur - 1)] = 10.0;
        JointStep {
            token_logits: tl,
            duration_logits: dl,
        }
    }

    #[test]
    fn blank_advances_one_frame() {
        let d = TdtGreedy::new(8);
        let (toks, frames, _) = d.decode(4, |_, _| step_with(7, 0, 8, 5));
        assert!(toks.is_empty());
        assert!(frames.is_empty());
    }

    #[test]
    fn token_then_duration_skip() {
        // Frame 0 emits token 3 with duration 2 -> jump to frame 2, which
        // is blank -> frame 3 emits token 5 with duration 0 -> stays.
        let d = TdtGreedy::new(8);
        let calls = std::cell::RefCell::new(Vec::new());
        let (toks, frames, durs) = d.decode(4, |t, _| {
            calls.borrow_mut().push(t);
            match t {
                0 => step_with(3, 2, 8, 5),
                _ => step_with(7, 0, 8, 5),
            }
        });
        assert_eq!(toks, vec![3]);
        assert_eq!(frames, vec![0]);
        assert_eq!(durs, vec![2]);
        // Duration skip means frame 1 was never evaluated.
        assert!(!calls.borrow().contains(&1), "calls={calls:?}");
    }

    #[test]
    fn repeated_tokens_without_duration_stay_on_frame() {
        // Same-frame multi-emit (up to the cap): with a single frame the
        // loop emits the cap then steps past the end.
        let d = TdtGreedy::new(8);
        let (toks, frames, _) = d.decode(1, |_, _| step_with(2, 0, 8, 5));
        assert_eq!(toks.len(), 10, "capped at 10/frame");
        assert!(frames.iter().all(|&f| f == 0));
    }
}
