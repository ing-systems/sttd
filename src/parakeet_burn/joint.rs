// Prediction + joint network (Burn port).
//
// Graph facts read off `decoder_joint-model.int8.onnx` (55 nodes,
// verified with probe scripts, not assumed):
// - Per step: `targets [1,1]` (int32 last-emitted token) -> Cast ->
//   Gather one row of `decoder.prediction.embed.weight [8193,640]` (U8)
//   -> DequantizeLinear -> `[640]`.
// - Two LSTM layers, hidden 640, plain ONNX semantics: `iofc` gate
//   order, sigmoid i/o/f, tanh cell, no peephole, no clip. W/R are
//   stored `[1,640,2560]` = (in, 4H)/(H, 4H), transposed against the
//   ONNX LSTM spec's `[1,4H,in]` — the same (K, N) convention as the
//   encoder's `MatMulInteger` weights — so `forward` reads them as-is
//   (`x @ stored`, no transpose at load). B `[1,5120]` F32 is the
//   standard `[Wb(2560), Rb(2560)]` pair.
// - Joint: enc frame `[1024]` @ `[1024,640]` + bias, pred state `[640]`
//   @ `[640,640]` + bias, add, Relu, @ `[640,8198]` + bias ->
//   `[8198]` = 8193 token logits + 5 duration logits (the split
//   `model_tdt.rs` makes when it slices `logits_data`).
// - `prednet_lengths` just echoes `target_length`; not reproduced.
// - States commit only when a non-blank token emits (see `step_for`);
//   `model_tdt.rs::greedy_decode` runs the joint on every visited frame
//   but only copies `output_states_*` back on emit.
//
// Like the encoder, weights stay 8-bit in RAM (scalar scale/zp in the
// converted file) and dequantize inside the fp32 forward.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};

use super::backend::{CpuBackend, CpuDevice};
use super::decode::{JointStep, TdtGreedy};
use super::quant::Uint8Linear;
use super::tensor::TensorCtx;
use super::weights::RawTensor;

/// Joint config resolved from converted weights at load.
#[derive(Debug, Clone)]
pub(crate) struct JointConfig {
    pub(crate) encoder_dim: usize,
    pub(crate) pred_hidden: usize,
    pub(crate) pred_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) n_durations: usize,
}

impl Default for JointConfig {
    fn default() -> Self {
        Self {
            encoder_dim: 1024,
            pred_hidden: 640,
            pred_layers: 2,
            vocab_size: 0, // filled from vocab.txt at load
            n_durations: 5,
        }
    }
}

/// Output dimension of the joint: token logits + duration logits.
pub(crate) fn joint_out_dim(vocab_size: usize, n_durations: usize) -> usize {
    vocab_size + n_durations
}

/// Token embedding table: `[rows][cols]` U8 + scalar scale/zp.
pub(crate) struct EmbedTable {
    pub(crate) data: Vec<u8>,
    pub(crate) scale: f32,
    pub(crate) zero_point: f32,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

impl EmbedTable {
    pub(crate) fn lookup(&self, token: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.lookup_into(token, &mut out);
        out
    }

    /// [`Self::lookup`] into a reused buffer (per-step, so the row Vec is
    /// not reallocated every decode step); same per-element arithmetic.
    pub(crate) fn lookup_into(&self, token: usize, out: &mut Vec<f32>) {
        assert!(token < self.rows, "token {token} >= {}", self.rows);
        let base = token * self.cols;
        out.clear();
        out.extend(
            self.data[base..base + self.cols]
                .iter()
                .map(|&q| (q as f32 - self.zero_point) * self.scale),
        );
    }
}

/// One LSTM layer with stored-orientation weights (`[in][4H]`, `[H][4H]`).
pub(crate) struct LstmLayer {
    pub(crate) w: Vec<i8>, // [in_dim][4*hid], row-major
    pub(crate) w_scale: f32,
    pub(crate) w_zp: f32,
    pub(crate) r: Vec<i8>, // [hid][4*hid], row-major
    pub(crate) r_scale: f32,
    pub(crate) r_zp: f32,
    pub(crate) b: Vec<f32>, // [8*hid] = [Wb(4H), Rb(4H)]
    pub(crate) in_dim: usize,
    pub(crate) hid: usize,
    /// Lazily-dequantized f32 twins of `w`/`r` (same layout). `step`
    /// dequantized on the fly per MAC (`(q - zp) * scale` per inner
    /// iteration — ~3 M redundant dequants per layer per decode step);
    /// the cached values are bit-identical (same per-element arithmetic),
    /// so only the repeated work disappears.
    w_f32: std::sync::OnceLock<Vec<f32>>,
    r_f32: std::sync::OnceLock<Vec<f32>>,
    /// Tensor twins for the tensorized gate MACs: dequantized `w`/`r` as
    /// Flex matrices plus the folded bias, built once (see `step`).
    t_cache: std::sync::OnceLock<LstmT>,
}

/// Dequantized Flex twin of an LSTM layer's gate matrices: `w`/`r`
/// stacked as one `[in+hid,4H]` f32 matrix plus the folded bias `[1,4H]`
/// (`Wb+Rb`, the same sum `step` adds per gate). One matmul per step
/// instead of two (`x@W + h@R` needs two dispatches plus an add); the
/// single 1280-wide dot differs from the split pair by f32 summation
/// order only (~1e-7 relative — the same class as the existing
/// tensor-vs-scalar residual, and the sigmoid/tanh gates are
/// contractive, so it cannot amplify).
#[derive(Clone, Debug)]
struct LstmT {
    wr: burn::tensor::Tensor<CpuBackend, 2>,
    b: burn::tensor::Tensor<CpuBackend, 2>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LstmLayer {
    /// Dequantized `[in_dim][4*hid]` twin of `w`, built once
    /// (bit-identical to the old inline per-MAC dequant: same per-element
    /// `(q - zp) * scale`, so the accumulation order in `step` decides
    /// the result exactly as before).
    fn wf(&self) -> &Vec<f32> {
        self.w_f32.get_or_init(|| {
            self.w
                .iter()
                .map(|&q| (q as f32 - self.w_zp) * self.w_scale)
                .collect()
        })
    }

    /// Dequantized `[hid][4*hid]` twin of `r`, built once (same terms).
    fn rf(&self) -> &Vec<f32> {
        self.r_f32.get_or_init(|| {
            self.r
                .iter()
                .map(|&q| (q as f32 - self.r_zp) * self.r_scale)
                .collect()
        })
    }

    /// Build the dequantized twins + Flex gate tensors now (load-time
    /// warming). The gate MACs always run on Flex when the tensor path
    /// is enabled (their f32 inputs are never quantized, so `igemm`
    /// does not apply — but at 61 µs per matmul Flex is already fast
    /// enough here that no integer path is needed).
    pub(crate) fn warm(&self, device: &CpuDevice) {
        let _ = (self.wf(), self.rf());
        let _ = self.t_cached(device);
    }

    /// Flex gate-matrix twin, built once from the dequantized twins
    /// (same values stacked — `[W; R]` row-blocks, so no transpose: the
    /// stores are already `[in,4H]`/`[H,4H]` row-major).
    fn t_cached(&self, device: &CpuDevice) -> &LstmT {
        self.t_cache.get_or_init(|| {
            use burn::tensor::{Tensor, TensorData};
            let g4 = 4 * self.hid;
            let mut wrf = Vec::with_capacity((self.in_dim + self.hid) * g4);
            wrf.extend_from_slice(self.wf());
            wrf.extend_from_slice(self.rf());
            let wr: Tensor<CpuBackend, 2> = Tensor::from_data(
                TensorData::new(wrf, [self.in_dim + self.hid, g4]),
                device,
            );
            let bf: Vec<f32> = (0..g4).map(|j| self.b[j] + self.b[g4 + j]).collect();
            let b: Tensor<CpuBackend, 2> =
                Tensor::from_data(TensorData::new(bf, [1, g4]), device);
            LstmT { wr, b }
        })
    }
}

/// Reusable per-step LSTM buffers (see [`LstmLayer::step_into`]): the
/// greedy loop runs tens to hundreds of steps per utterance, and the
/// `xh`/gates/candidate allocations were pure churn.
#[derive(Default)]
pub(crate) struct LstmStep {
    xh: Vec<f32>,
    gates: Vec<f32>,
    pub(crate) h: Vec<f32>,
    pub(crate) c: Vec<f32>,
}

impl LstmLayer {
    /// One step: `(h_new, c_new)` from `(x, h_prev, c_prev)`, allocating
    /// fresh outputs (tests/goldens; the greedy loop uses `step_into`).
    pub(crate) fn step(
        &self,
        x: &[f32],
        h: &[f32],
        c: &[f32],
        ctx: &TensorCtx,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut buf = LstmStep::default();
        self.step_into(x, h, c, ctx, &mut buf);
        (buf.h, buf.c)
    }

    /// [`Self::step`] writing into caller-owned buffers (`buf.xh`/gates/
    /// h/c are resized as needed and reused across steps).
    /// Gate MACs dispatch on `ctx`: the tensor path runs them as one Flex
    /// matmul over the concatenated `[x, h]` input against the stacked
    /// gate matrix (one dispatch + one input tensor per step instead of
    /// two; scalar nonlinearities unchanged). Gate nonlinearities and the
    /// state update stay scalar on both paths (same libm calls, same
    /// order — exact), now parallel over the hidden dimension (per-element
    /// independent, so bit-identical to the serial loop). The tensor gates
    /// differ from scalar by f32 summation order only (~1e-7 relative, and
    /// the sigmoid/tanh gates are contractive, so it cannot amplify); the
    /// embed lookup stays scalar on both (one row gather).
    pub(crate) fn step_into(
        &self,
        x: &[f32],
        h: &[f32],
        c: &[f32],
        ctx: &TensorCtx,
        buf: &mut LstmStep,
    ) {
        assert_eq!(x.len(), self.in_dim);
        assert_eq!(h.len(), self.hid);
        assert_eq!(c.len(), self.hid);
        let g4 = 4 * self.hid;
        buf.xh.clear();
        buf.xh.reserve(self.in_dim + self.hid);
        buf.xh.extend_from_slice(x);
        buf.xh.extend_from_slice(h);
        if ctx.enabled {
            use burn::tensor::{Tensor, TensorData};
            let tc = self.t_cached(&ctx.device);
            let xt: Tensor<CpuBackend, 2> = Tensor::from_data(
                TensorData::new(buf.xh.clone(), [1, self.in_dim + self.hid]),
                &ctx.device,
            );
            let g = xt.matmul(tc.wr.clone()) + tc.b.clone();
            buf.gates = g.into_data().to_vec::<f32>().expect("lstm gates: to_vec");
        } else {
            let wf = self.wf();
            let rf = self.rf();
            buf.gates.resize(g4, 0.0);
            // Gate columns are independent: parallelize over chunks,
            // keeping the per-gate accumulation order identical
            // (bit-exact vs serial).
            use rayon::prelude::*;
            buf.gates
                .par_chunks_mut(256)
                .enumerate()
                .for_each(|(chunk, gchunk)| {
                    let j0 = chunk * 256;
                    for (k, g) in gchunk.iter_mut().enumerate() {
                        let j = j0 + k;
                        // Bias pair first (matches the graph's Add order
                        // trivially — addition commutes; kept explicit for
                        // readability).
                        let mut acc = self.b[j] + self.b[g4 + j];
                        for (i, xi) in x.iter().enumerate() {
                            acc += wf[i * g4 + j] * xi;
                        }
                        for (i, hi) in h.iter().enumerate() {
                            acc += rf[i * g4 + j] * hi;
                        }
                        *g = acc;
                    }
                });
        }
        // ONNX gate order: i, o, f, c. Each hidden element is independent:
        // rayon only spreads the loop, so this is bit-exact vs serial.
        let (gi, rest) = buf.gates.split_at(self.hid);
        let (go, rest) = rest.split_at(self.hid);
        let (gf, gc) = rest.split_at(self.hid);
        buf.h.resize(self.hid, 0.0);
        buf.c.resize(self.hid, 0.0);
        use rayon::prelude::*;
        buf.h
            .par_iter_mut()
            .zip(buf.c.par_iter_mut())
            .enumerate()
            .for_each(|(k, (h_new, c_new))| {
                let i = sigmoid(gi[k]);
                let o = sigmoid(go[k]);
                let f = sigmoid(gf[k]);
                let g = gc[k].tanh();
                let cc = f * c[k] + i * g;
                *c_new = cc;
                *h_new = o * cc.tanh();
            });
    }
}

/// Candidate states from one `step_for` call (committed to `self` only
/// when the step emits a non-blank token).
pub(crate) struct StepStates {
    pub(crate) h: [Vec<f32>; 2],
    pub(crate) c: [Vec<f32>; 2],
}

pub(crate) struct BurnJoint {
    embed: EmbedTable,
    l0: LstmLayer,
    l1: LstmLayer,
    enc: Uint8Linear,  // 1024 -> 640
    pred: Uint8Linear, // 640 -> 640
    out: Uint8Linear,  // 640 -> vocab + durations
    vocab_size: usize,
    blank_id: usize,
    h: [Vec<f32>; 2],
    c: [Vec<f32>; 2],
    /// Per-step reusable buffers (embed row + both LSTM layers): the
    /// greedy loop runs tens to hundreds of steps per utterance.
    scratch: JointScratch,
}

#[derive(Default)]
struct JointScratch {
    embed: Vec<f32>,
    l0: LstmStep,
    l1: LstmStep,
}

fn take_f32(tensors: &HashMap<String, RawTensor>, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    match tensors.get(name) {
        Some(RawTensor::F32 { shape, data }) => Ok((shape.clone(), data.clone())),
        Some(other) => anyhow::bail!("{name}: expected F32, got {}", other.kind()),
        None => anyhow::bail!("{name}: missing tensor"),
    }
}

fn scalar(v: &[f32], name: &str) -> Result<f32> {
    if v.len() == 1 {
        Ok(v[0])
    } else {
        anyhow::bail!("{name}: expected scalar scale/zp, got len {}", v.len())
    }
}

fn take_matmul(
    tensors: &HashMap<String, RawTensor>,
    wname: &str,
    bias_name: &str,
    in_dim: usize,
    out_dim: usize,
) -> Result<Uint8Linear> {
    let raw = tensors
        .get(wname)
        .with_context(|| format!("{wname}: missing tensor"))?;
    let (shape, data, scale, zp) = match raw {
        RawTensor::U8 {
            shape,
            data,
            scale,
            zp,
        } => (shape, data, scale, zp),
        other => anyhow::bail!("{wname}: expected U8, got {}", other.kind()),
    };
    if shape.as_slice() != [in_dim, out_dim] {
        anyhow::bail!("{wname}: shape {shape:?} != [{in_dim}, {out_dim}]");
    }
    let (_, bias) = take_f32(tensors, bias_name)?;
    if bias.len() != out_dim {
        anyhow::bail!("{bias_name}: len {} != {out_dim}", bias.len());
    }
    Ok(Uint8Linear {
        // ONNX MatMul weight (K, N) = (in, out): transpose to the
        // `[out][in]` store `forward` reads (see `Uint8Linear` docs).
        weight_u8: super::quant::transpose_u8(data, in_dim, out_dim),
        scale: vec![scalar(scale, &format!("{wname}.scale"))?; out_dim],
        zero_point: vec![scalar(zp, &format!("{wname}.zero_point"))?; out_dim],
        bias: Some(bias),
        in_dim,
        out_dim,
        cache: std::sync::OnceLock::new(),
        w_i16: std::sync::OnceLock::new(),
    })
}

fn take_lstm(
    tensors: &HashMap<String, RawTensor>,
    wname: &str,
    rname: &str,
    bname: &str,
    in_dim: usize,
    hid: usize,
) -> Result<LstmLayer> {
    let get_i8 = |name: &str| -> Result<(Vec<usize>, Vec<i8>, f32, f32)> {
        match tensors.get(name) {
            Some(RawTensor::I8 {
                shape,
                data,
                scale,
                zp,
            }) => Ok((
                shape.clone(),
                data.clone(),
                scalar(scale, &format!("{name}.scale"))?,
                zp.first().copied().unwrap_or(0.0),
            )),
            Some(other) => anyhow::bail!("{name}: expected I8, got {}", other.kind()),
            None => anyhow::bail!("{name}: missing tensor"),
        }
    };
    let (ws, w, w_sc, w_zp) = get_i8(wname)?;
    let (rs, r, r_sc, r_zp) = get_i8(rname)?;
    // Stored orientation is (in, 4H): no transpose (see module docs).
    if ws.as_slice() != [1, in_dim, 4 * hid] {
        anyhow::bail!("{wname}: shape {ws:?} != [1, {in_dim}, {}]", 4 * hid);
    }
    if rs.as_slice() != [1, hid, 4 * hid] {
        anyhow::bail!("{rname}: shape {rs:?} != [1, {hid}, {}]", 4 * hid);
    }
    let (bs, b) = take_f32(tensors, bname)?;
    if bs.as_slice() != [1, 8 * hid] {
        anyhow::bail!("{bname}: shape {bs:?} != [1, {}]", 8 * hid);
    }
    Ok(LstmLayer {
        w,
        w_scale: w_sc,
        w_zp,
        r,
        r_scale: r_sc,
        r_zp,
        b,
        in_dim,
        hid,
        w_f32: std::sync::OnceLock::new(),
        r_f32: std::sync::OnceLock::new(),
        t_cache: std::sync::OnceLock::new(),
    })
}

/// Exact tensor names `weights::load_selected` must fetch for `load`.
pub(crate) fn joint_want() -> HashSet<String> {
    [
        "decoder.prediction.embed.weight",
        "onnx::LSTM_205",
        "onnx::LSTM_206",
        "onnx::LSTM_207",
        "onnx::LSTM_225",
        "onnx::LSTM_226",
        "onnx::LSTM_227",
        "onnx::MatMul_228",
        "onnx::MatMul_229",
        "onnx::MatMul_230",
        "joint.enc.bias",
        "joint.pred.bias",
        "joint.joint_net.2.bias",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

impl BurnJoint {
    /// Load the prediction + joint weights. Shapes are asserted here so
    /// a corrupt file fails at load, never mid-utterance.
    pub(crate) fn load(
        tensors: &HashMap<String, RawTensor>,
        vocab_size: usize,
    ) -> Result<Self> {
        let (es, edata, escale, ezp) = match tensors.get("decoder.prediction.embed.weight") {
            Some(RawTensor::U8 {
                shape,
                data,
                scale,
                zp,
            }) => (
                shape.clone(),
                data.clone(),
                scalar(scale, "decoder.prediction.embed.weight.scale")?,
                scalar(zp, "decoder.prediction.embed.weight.zero_point")?,
            ),
            Some(other) => anyhow::bail!("embed: expected U8, got {}", other.kind()),
            None => anyhow::bail!("embed: missing tensor"),
        };
        if es != vec![vocab_size, 640] {
            anyhow::bail!("embed shape {es:?} != [{vocab_size}, 640]");
        }
        let embed = EmbedTable {
            data: edata,
            scale: escale,
            zero_point: ezp,
            rows: vocab_size,
            cols: 640,
        };
        let l0 = take_lstm(tensors, "onnx::LSTM_205", "onnx::LSTM_206", "onnx::LSTM_207", 640, 640)?;
        let l1 = take_lstm(tensors, "onnx::LSTM_225", "onnx::LSTM_226", "onnx::LSTM_227", 640, 640)?;
        let enc = take_matmul(tensors, "onnx::MatMul_228", "joint.enc.bias", 1024, 640)?;
        let pred = take_matmul(tensors, "onnx::MatMul_229", "joint.pred.bias", 640, 640)?;
        let out = take_matmul(tensors, "onnx::MatMul_230", "joint.joint_net.2.bias", 640, vocab_size + 5)?;
        let blank_id = vocab_size.saturating_sub(1);
        Ok(Self {
            embed,
            l0,
            l1,
            enc,
            pred,
            out,
            vocab_size,
            blank_id,
            h: [vec![0.0; 640], vec![0.0; 640]],
            c: [vec![0.0; 640], vec![0.0; 640]],
            scratch: JointScratch::default(),
        })
    }

    pub(crate) fn reset(&mut self) {
        for v in self.h.iter_mut().chain(self.c.iter_mut()) {
            v.fill(0.0);
        }
    }

    /// Build every cached tensor / dequantized twin now (load-time
    /// warming): the 3 joint projections plus both LSTM layers.
    pub(crate) fn warm(&self, device: &super::backend::CpuDevice) {
        self.enc.warm(device);
        self.pred.warm(device);
        self.out.warm(device);
        self.l0.warm(device);
        self.l1.warm(device);
    }

    /// Overwrite the running states (tests seed ORT states to isolate
    /// one step; the greedy driver only ever uses `reset` + commits).
    #[cfg(test)]
    pub(crate) fn set_states(&mut self, h: [Vec<f32>; 2], c: [Vec<f32>; 2]) {
        self.h = h;
        self.c = c;
    }

    #[cfg(test)]
    pub(crate) fn states(&self) -> (&[Vec<f32>; 2], &[Vec<f32>; 2]) {
        (&self.h, &self.c)
    }

    /// One joint evaluation for `enc_frame` with prediction input
    /// `token`. Returns the logits plus the candidate new states; the
    /// candidates commit to `self` only when the step emits a non-blank
    /// token — exactly the `output_states_*` copy-back rule in
    /// `model_tdt.rs::greedy_decode`.
    pub(crate) fn step_for(
        &mut self,
        enc_frame: &[f32],
        token: usize,
        ctx: &TensorCtx,
    ) -> (JointStep, StepStates) {
        assert_eq!(enc_frame.len(), 1024);
        // Per-step buffers are reused (no fresh embed/xh/gates/candidate
        // allocations); only the returned logits/states are owned.
        let JointScratch { embed, l0, l1 } = &mut self.scratch;
        self.embed.lookup_into(token, embed);
        self.l0.step_into(embed, &self.h[0], &self.c[0], ctx, l0);
        self.l1.step_into(&l0.h, &self.h[1], &self.c[1], ctx, l1);
        let pe = self.enc.forward_dispatch(enc_frame, ctx);
        let pp = self.pred.forward_dispatch(&l1.h, ctx);
        let mut hsum: Vec<f32> = pe.iter().zip(pp.iter()).map(|(a, b)| a + b).collect();
        for v in hsum.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let logits = self.out.forward_dispatch(&hsum, ctx);
        let tok = TdtGreedy::argmax(&logits[..self.vocab_size]);
        if tok != self.blank_id {
            self.h = [l0.h.clone(), l1.h.clone()];
            self.c = [l0.c.clone(), l1.c.clone()];
        }
        let step = JointStep {
            token_logits: logits[..self.vocab_size].to_vec(),
            duration_logits: logits[self.vocab_size..].to_vec(),
        };
        (
            step,
            StepStates {
                h: [l0.h.clone(), l1.h.clone()],
                c: [l0.c.clone(), l1.c.clone()],
            },
        )
    }

    /// Greedy-decode `enc` frames (`[T'][1024]` rows from `BurnEncoder`).
    /// Owns the LSTM states across the utterance: reset on entry, commit
    /// on emit inside `step_for`; the frame-advance/duration machine is
    /// `TdtGreedy`'s (shared with the pure-logits unit tests).
    pub(crate) fn decode(
        &mut self,
        enc: &[Vec<f32>],
        ctx: &TensorCtx,
    ) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        self.reset();
        let greedy = TdtGreedy::new(self.vocab_size);
        greedy.decode(enc.len(), |t, last| self.step_for(&enc[t], last, ctx).0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_dim_matches_onnx_contract() {
        // v3 vocab 8193 (incl. blank) + 5 durations.
        assert_eq!(joint_out_dim(8193, 5), 8198);
    }

    #[test]
    fn default_shapes_match_onnx_io() {
        let c = JointConfig::default();
        assert_eq!(c.encoder_dim, 1024);
        assert_eq!(c.pred_hidden, 640);
        assert_eq!(c.pred_layers, 2);
    }

    #[test]
    fn joint_want_lists_fixed_names() {
        let want = joint_want();
        // 13 base names; `.scale`/`.zero_point` companions ride along
        // via `load_selected`.
        assert_eq!(want.len(), 13);
        assert!(want.contains("decoder.prediction.embed.weight"));
        assert!(want.contains("onnx::LSTM_205"));
        assert!(want.contains("joint.joint_net.2.bias"));
    }

    #[test]
    fn embed_lookup_dequantizes_row() {
        // 2 rows x 3 cols, zp=10, scale=0.5: row1 = [10,12,8] -> [0,1,-1].
        let e = EmbedTable {
            data: vec![0, 0, 0, 10, 12, 8],
            scale: 0.5,
            zero_point: 10.0,
            rows: 2,
            cols: 3,
        };
        assert_eq!(e.lookup(1), vec![0.0, 1.0, -1.0]);
    }

    #[test]
    #[should_panic]
    fn embed_lookup_rejects_oob_token() {
        let e = EmbedTable {
            data: vec![0, 0],
            scale: 1.0,
            zero_point: 0.0,
            rows: 1,
            cols: 2,
        };
        let _ = e.lookup(1);
    }

    fn zero_lstm(in_dim: usize, hid: usize) -> LstmLayer {
        LstmLayer {
            w: vec![0; in_dim * 4 * hid],
            w_scale: 1.0,
            w_zp: 0.0,
            r: vec![0; hid * 4 * hid],
            r_scale: 1.0,
            r_zp: 0.0,
            b: vec![0.0; 8 * hid],
            in_dim,
            hid,
            w_f32: std::sync::OnceLock::new(),
            r_f32: std::sync::OnceLock::new(),
            t_cache: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn lstm_zero_step_stays_at_origin() {
        // Zero weights/bias/state: all gates see 0 -> i=o=f=0.5,
        // g~ = 0 -> c = 0, h = 0.
        let l = zero_lstm(2, 2);
        let (h, c) = l.step(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0], &TensorCtx::scalar());
        assert_eq!((h, c), (vec![0.0, 0.0], vec![0.0, 0.0]));
    }

    #[test]
    fn lstm_cell_gates_follow_iofc_order() {
        // hid=1, in=1: bias Wb = [i, o, f, g] picks one gate at a time
        // (+10 >> sigmoid ~= 1, others 0 -> sigmoid(0) = 0.5).
        // Case forget-open, rest shut... i(0.5),o(0.5),f(~1),g(0):
        // c = 1*c_prev + 0.5*0 = c_prev; h = 0.5*tanh(c).
        let mut l = zero_lstm(1, 1);
        l.b = vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (h, c) = l.step(&[0.0], &[0.0], &[2.0], &TensorCtx::scalar());
        assert!((c[0] - 2.0).abs() < 1e-3, "c={c:?}");
        assert!((h[0] - 0.5 * 2.0f32.tanh()).abs() < 1e-3, "h={h:?}");
        // Case input-open, forget-shut: c = 0*prev + 1*tanh(10)... g
        // must be large: set g bias 10 instead.
        l.b = vec![10.0, 0.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let (h2, c2) = l.step(&[0.0], &[0.0], &[2.0], &TensorCtx::scalar());
        // i~=1, f~=0, g~=tanh(10)~=1 -> c ~= 1.
        assert!((c2[0] - 1.0).abs() < 1e-2, "c2={c2:?}");
        // o = 0.5 -> h = 0.5*tanh(c).
        assert!((h2[0] - 0.5 * c2[0].tanh()).abs() < 1e-5, "h2={h2:?}");
    }

    /// ORT parity: golden steps from onnxruntime-CPU runs of
    /// `decoder_joint-model.int8.onnx` (`joint_golden.safetensors`).
    /// Step 0 is (noise encoder frame, blank, zero states); step 1
    /// chains ORT's own step-0 outputs; the non-blank step pins a
    /// non-blank embed row. Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn ort_joint_parity() {
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let want = joint_want();
        let tensors =
            load_selected(&dir.join("decoder_joint-burn.safetensors"), &want).unwrap();
        let mut joint = BurnJoint::load(&tensors, 8193).unwrap();

        let bytes = std::fs::read("src/parakeet_burn/testdata/joint_golden.safetensors").unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let f32vec = |name: &str, len: usize| -> Vec<f32> {
            let v = st.tensor(name).unwrap();
            assert_eq!(v.shape(), &[len], "{name}");
            v.data()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let f32mat = |name: &str, rows: usize, cols: usize| -> Vec<Vec<f32>> {
            let v = st.tensor(name).unwrap();
            assert_eq!(v.shape(), &[rows, cols], "{name}");
            v.data()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<f32>>()
                .chunks(cols)
                .map(|c| c.to_vec())
                .collect()
        };
        let frame = f32vec("enc_frame", 1024);
        let sctx = TensorCtx::scalar();
        let cmp = |got: &[f32], want: &[f32], tag: &str| {
            assert_eq!(got.len(), want.len(), "{tag} len");
            let mut max_abs: f32 = 0.0;
            let mut sum_abs = 0.0f32;
            let mut ref_max = 0.0f32;
            for (a, b) in got.iter().zip(want.iter()) {
                max_abs = max_abs.max((a - b).abs());
                sum_abs += (a - b).abs();
                ref_max = ref_max.max(b.abs());
            }
            eprintln!(
                "{tag}: max_abs={max_abs:.4} mean_abs={:.4} ref_max={ref_max:.4}",
                sum_abs / got.len() as f32
            );
            (max_abs, ref_max)
        };

        // Step 0: blank in, zero states. Blank emits nothing, so the
        // running states must stay zero while the candidates match ORT.
        let (s0, cand0) = joint.step_for(&frame, 8192, &sctx);
        let want0 = f32vec("logits0", 8198);
        let all0: Vec<f32> = s0
            .token_logits
            .iter()
            .chain(s0.duration_logits.iter())
            .copied()
            .collect();
        let (m0, r0) = cmp(&all0, &want0, "step0 logits");
        assert!(m0 < 0.01 * r0.max(1.0), "step0 logits m0={m0} r0={r0}");
        assert_eq!(s0.duration_logits.len(), 5);
        let (h, c) = joint.states();
        assert!(h.iter().all(|r| r.iter().all(|&v| v == 0.0)), "blank must not commit");
        assert!(c.iter().all(|r| r.iter().all(|&v| v == 0.0)), "blank must not commit");
        let h1 = f32mat("h1", 2, 640);
        let c1 = f32mat("c1", 2, 640);
        for layer in 0..2 {
            let (mh, rh) = cmp(&cand0.h[layer], &h1[layer], &format!("step0 h{layer}"));
            assert!(mh < 0.10 * rh.max(1.0), "h{layer} mh={mh} rh={rh}");
            let (mc, rc) = cmp(&cand0.c[layer], &c1[layer], &format!("step0 c{layer}"));
            assert!(mc < 0.10 * rc.max(1.0), "c{layer} mc={mc} rc={rc}");
        }

        // Step 1: ORT's token + states as input (fully isolated).
        joint.set_states([h1[0].clone(), h1[1].clone()], [c1[0].clone(), c1[1].clone()]);
        let tok_v = st.tensor("token0").unwrap();
        let tok0 = i64::from_le_bytes(tok_v.data().try_into().unwrap()) as usize;
        assert_eq!(tok0, 8192);
        let (s1, cand1) = joint.step_for(&frame, tok0, &sctx);
        let want1 = f32vec("logits1", 8198);
        let all1: Vec<f32> = s1
            .token_logits
            .iter()
            .chain(s1.duration_logits.iter())
            .copied()
            .collect();
        let (m1, r1) = cmp(&all1, &want1, "step1 logits");
        assert!(m1 < 0.01 * r1.max(1.0), "step1 logits m1={m1} r1={r1}");
        let h2 = f32mat("h2", 2, 640);
        let c2 = f32mat("c2", 2, 640);
        for layer in 0..2 {
            let (mh, rh) = cmp(&cand1.h[layer], &h2[layer], &format!("step1 h{layer}"));
            assert!(mh < 0.10 * rh.max(1.0), "h{layer} mh={mh} rh={rh}");
            let (mc, rc) = cmp(&cand1.c[layer], &c2[layer], &format!("step1 c{layer}"));
            assert!(mc < 0.10 * rc.max(1.0), "c{layer} mc={mc} rc={rc}");
        }

        // Non-blank step: pins a non-blank embed row and the commit path.
        joint.set_states([h1[0].clone(), h1[1].clone()], [c1[0].clone(), c1[1].clone()]);
        let (snb, candnb) = joint.step_for(&frame, 0, &sctx);
        let wantnb = f32vec("logits_nb", 8198);
        let allnb: Vec<f32> = snb
            .token_logits
            .iter()
            .chain(snb.duration_logits.iter())
            .copied()
            .collect();
        let (mnb, rnb) = cmp(&allnb, &wantnb, "step-nb logits");
        assert!(mnb < 0.01 * rnb.max(1.0), "nb logits mnb={mnb} rnb={rnb}");
        let hnb = f32mat("h_nb", 2, 640);
        let cnb = f32mat("c_nb", 2, 640);
        for layer in 0..2 {
            let (mh, rh) = cmp(&candnb.h[layer], &hnb[layer], &format!("nb h{layer}"));
            assert!(mh < 0.10 * rh.max(1.0), "nb h{layer}");
            let (mc, rc) = cmp(&candnb.c[layer], &cnb[layer], &format!("nb c{layer}"));
            assert!(mc < 0.10 * rc.max(1.0), "nb c{layer}");
        }
        // Token 0 may or may not be blank-adjacent... it is a real token
        // (blank is 8192), so the commit must have happened.
        let (h, _) = joint.states();
        assert!(
            h.iter().any(|r| r.iter().any(|&v| v != 0.0)),
            "non-blank must commit"
        );
    }

    /// Tensor-vs-scalar dispatch parity on real joint weights: same step
    /// through both paths, logits compared tightly (f32 order only —
    /// the LSTM/embed stay scalar on both sides, only the three
    /// projections dispatch). Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn tensor_joint_step_matches_scalar() {
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let want = joint_want();
        let tensors =
            load_selected(&dir.join("decoder_joint-burn.safetensors"), &want).unwrap();
        let mut joint = BurnJoint::load(&tensors, 8193).unwrap();
        // Deterministic pseudo-random encoder frame.
        let mut seed: u64 = 0xB17;
        let frame: Vec<f32> = (0..1024)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 2.0
            })
            .collect();
        let tctx = TensorCtx {
            device: super::super::backend::cpu_device(),
            enabled: true,
        };
        let (ss, _) = {
            joint.reset();
            joint.step_for(&frame, 0, &TensorCtx::scalar())
        };
        let (st, _) = {
            joint.reset();
            joint.step_for(&frame, 0, &tctx)
        };
        let all_s: Vec<f32> = ss.token_logits.iter().chain(ss.duration_logits.iter()).copied().collect();
        let all_t: Vec<f32> = st.token_logits.iter().chain(st.duration_logits.iter()).copied().collect();
        let (mut mx, mut rm) = (0.0f32, 0.0f32);
        for (a, b) in all_s.iter().zip(all_t.iter()) {
            mx = mx.max((a - b).abs());
            rm = rm.max(b.abs());
        }
        eprintln!("joint tensor-vs-scalar: max_abs={mx:.2e} ref_max={rm:.3}");
        assert!(mx < 1e-3 * rm.max(1.0), "mx={mx} rm={rm}");
    }

    }
