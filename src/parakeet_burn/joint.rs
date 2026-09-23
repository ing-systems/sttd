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

use super::tdt_greedy::{JointStep, TdtGreedy};
use super::quant_linear::Uint8Linear;
use super::tensor_ctx::TensorCtx;
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
///
/// Gate math mirrors the graph's `DynamicQuantizeLSTM` (verified node by
/// node off `decoder_joint-model.int8.onnx`: F32 X in, I8 W/R + F32 B):
/// x and h are dynamically quantized per-tensor (same window/zp/q as the
/// encoder's `DynamicQuantizeLinear`), the gates run as exact integer
/// MACs `Σ(qx-zpx)(w-zpw)` / `Σ(qh-zph)(r-zpr)` with one f32 scale-up +
/// folded bias each — the same formula on the scalar, i16 (`vpmaddwd`)
/// and VNNI (`vpdpbusd` + exact zpw correction) paths, so all three are
/// bit-identical (integer arithmetic is exact regardless of blocking).
/// Gate nonlinearities and the state update stay f32 like the graph.
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
    /// Lazily-widened i16 twins, transposed to `[4H][in]`/`[4H][hid]`
    /// (the int-GEMM layout): the i16 path's read-only weights. Built at
    /// prime_cache, on first use otherwise.
    w_i16: std::sync::OnceLock<Vec<i16>>,
    r_i16: std::sync::OnceLock<Vec<i16>>,
    /// Lazily-transposed raw s8 twins (`[4H][in]`, `[4H][hid]`): the VNNI
    /// path's read-only weights (the store is `[in][4H]`).
    w_tr: std::sync::OnceLock<Vec<i8>>,
    r_tr: std::sync::OnceLock<Vec<i8>>,
    /// Lazily-built raw s8 row sums over the TRANSPOSED twins (VNNI
    /// correction term's `rowsum`).
    w_sum: std::sync::OnceLock<Vec<i32>>,
    r_sum: std::sync::OnceLock<Vec<i32>>,
    /// Lazily-folded bias `Wb + Rb` (the same sum the graph's Add applies
    /// per gate). Built once; the per-step pair add disappears.
    b_fold: std::sync::OnceLock<Vec<f32>>,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl LstmLayer {
    /// Widened i16 twin TRANSPOSED to `[4H][in]` (the int-GEMM layout):
    /// `T[o,k] = (w[k*4H+o] - zpw)`. The stored orientation is
    /// `[in][4H]` (see `take_lstm`); the old f32 path read it as-is
    /// (`x @ stored`), but every `int_gemm` entry reads `[N][K]`
    /// row-major, so the twin transposes once here. Exact (reorder +
    /// integral zp).
    fn w_wide(&self) -> &Vec<i16> {
        self.w_i16.get_or_init(|| {
            transpose_widen_s8(&self.w, 4 * self.hid, self.in_dim, self.w_zp.round() as i32)
        })
    }

    /// Widened i16 twin of `r`, transposed to `[4H][hid]` (same terms).
    fn r_wide(&self) -> &Vec<i16> {
        self.r_i16.get_or_init(|| {
            transpose_widen_s8(&self.r, 4 * self.hid, self.hid, self.r_zp.round() as i32)
        })
    }

    /// Transposed s8 twin of `w` (`[4H][in]`, raw values): the VNNI
    /// path's read-only weights. Built once; the row sums below run over
    /// these rows.
    fn w_tr(&self) -> &Vec<i8> {
        self.w_tr.get_or_init(|| transpose_s8(&self.w, 4 * self.hid, self.in_dim))
    }

    /// Transposed s8 twin of `r` (`[4H][hid]`, same terms).
    fn r_tr(&self) -> &Vec<i8> {
        self.r_tr.get_or_init(|| transpose_s8(&self.r, 4 * self.hid, self.hid))
    }

    /// Raw s8 row sums of the TRANSPOSED `w` (VNNI correction term),
    /// built once.
    fn w_rows(&self) -> &Vec<i32> {
        self.w_sum
            .get_or_init(|| super::int_gemm::row_sums_s8(self.w_tr(), 4 * self.hid, self.in_dim))
    }

    /// Raw s8 row sums of the transposed `r` (same terms).
    fn r_rows(&self) -> &Vec<i32> {
        self.r_sum
            .get_or_init(|| super::int_gemm::row_sums_s8(self.r_tr(), 4 * self.hid, self.hid))
    }

    /// Folded bias `Wb + Rb` per gate, built once.
    fn bfold(&self) -> &Vec<f32> {
        self.b_fold.get_or_init(|| {
            let g4 = 4 * self.hid;
            (0..g4).map(|j| self.b[j] + self.b[g4 + j]).collect()
        })
    }

    /// Build every integer twin now (load-time warming): widened weights,
    /// transposed s8 stores, row sums and folded bias, so the first
    /// utterance never pays the cold build.
    pub(crate) fn prime_cache(&self) {
        let _ = (self.w_wide(), self.r_wide());
        let _ = (self.w_tr(), self.r_tr());
        let _ = (self.w_rows(), self.r_rows());
        let _ = self.bfold();
    }
}

/// Transpose `[in][4H]` s8 to `[4H][in]` row-major (pure reorder, exact).
/// Rows (outputs) are independent, so parallel.
fn transpose_s8(w: &[i8], out_dim: usize, in_dim: usize) -> Vec<i8> {
    use rayon::prelude::*;
    assert_eq!(w.len(), out_dim * in_dim);
    let mut t = vec![0i8; w.len()];
    t.par_chunks_mut(in_dim)
        .enumerate()
        .for_each(|(o, row)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = w[k * out_dim + o];
            }
        });
    t
}

/// Transposed + widened twin: `T[o,k] = (w[k*out+o] - zpw)` as i16.
/// Exact for integral zp (the stored I8 zero-points are).
fn transpose_widen_s8(w: &[i8], out_dim: usize, in_dim: usize, zpw: i32) -> Vec<i16> {
    use rayon::prelude::*;
    assert_eq!(w.len(), out_dim * in_dim);
    let mut t = vec![0i16; w.len()];
    t.par_chunks_mut(in_dim)
        .enumerate()
        .for_each(|(o, row)| {
            for (k, v) in row.iter_mut().enumerate() {
                *v = w[k * out_dim + o] as i16 - zpw as i16;
            }
        });
    t
}

/// Reusable per-step LSTM buffers (see [`LstmLayer::step_into`]): the
/// greedy loop runs tens to hundreds of steps per utterance, and the
/// gates/candidate allocations were pure churn.
#[derive(Default)]
pub(crate) struct LstmStep {
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

    /// [`Self::step`] writing into caller-owned buffers (`buf.gates/h/c
    /// are resized as needed and reused across steps).
    /// Gate MACs run as exact integer products on every path: x and h are
    /// dynamically quantized per-tensor (same window/zp/q the encoder's
    /// `DynamicQuantizeLinear` produces), then `y = mac_x*sx*sw +
    /// mac_h*sh*sr + (Wb+Rb)` with the bias folded at load. VNNI
    /// (`vpdpbusd` + exact zpw correction) and i16 (`vpmaddwd` over
    /// widened rows) compute the same integers, so dispatch on `ctx`
    /// changes speed, never values. Gate nonlinearities and the state
    /// update stay scalar (same libm calls, same order — exact), parallel
    /// over the hidden dimension (per-element independent, so bit-identical
    /// to the serial loop). The embed lookup stays scalar (one row gather).
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
        buf.gates.resize(g4, 0.0);
        if ctx.use_vnni() {
            // Raw u8 activations + exact zpw-aware VNNI gates. Two calls:
            // the x-part writes (with the folded bias), the h-part
            // accumulates via `AddScale` — same f32 order on every path.
            let (qx, sx_x, zpx_x) = super::quant_linear::dynamic_quantize_u8(x);
            let (qh, sx_h, zpx_h) = super::quant_linear::dynamic_quantize_u8(h);
            let sumx = super::int_gemm::sum_u8(&qx);
            let sumh = super::int_gemm::sum_u8(&qh);
            super::int_gemm::vnni_gemm_zp_epi(
                &qx,
                &[sumx],
                1,
                self.in_dim,
                self.w_tr(),
                self.w_rows(),
                g4,
                &[self.w_scale],
                Some(self.bfold()),
                sx_x,
                zpx_x as i32,
                self.w_zp.round() as i32,
                &mut buf.gates,
                super::int_gemm::Epilogue::None,
            );
            super::int_gemm::vnni_gemm_zp_epi(
                &qh,
                &[sumh],
                1,
                self.hid,
                self.r_tr(),
                self.r_rows(),
                g4,
                &[self.r_scale],
                None,
                sx_h,
                zpx_h as i32,
                self.r_zp.round() as i32,
                &mut buf.gates,
                super::int_gemm::Epilogue::AddScale(1.0),
            );
        } else {
            // i16 twins of the same products (bit-identical; machines
            // without AVX-512 run the drivers' scalar integer fallback).
            let (ax, sx_x, _) = super::int_gemm::quantize_rows_i16(x);
            let (ah, sx_h, _) = super::int_gemm::quantize_rows_i16(h);
            super::int_gemm::int_gemm_epi(
                &ax,
                1,
                self.in_dim,
                self.w_wide(),
                g4,
                &[self.w_scale],
                Some(self.bfold()),
                sx_x,
                &mut buf.gates,
                super::int_gemm::Epilogue::None,
            );
            super::int_gemm::int_gemm_epi(
                &ah,
                1,
                self.hid,
                self.r_wide(),
                g4,
                &[self.r_scale],
                None,
                sx_h,
                &mut buf.gates,
                super::int_gemm::Epilogue::AddScale(1.0),
            );
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

/// Memoized prediction-side work for one `(token, states-version,
/// ctx-mode)` visit (see `BurnJoint::step_core`): both LSTM candidates +
/// the pred projection. Consecutive blank frames share inputs (same last
/// token, uncommitted states), so all but the first skip the embed lookup,
/// both LSTM layers and the pred projection entirely.
#[derive(Default)]
struct PredCache {
    key: Option<(usize, u64, u8)>,
    l0h: Vec<f32>,
    l0c: Vec<f32>,
    l1h: Vec<f32>,
    l1c: Vec<f32>,
    pp: Vec<f32>,
}

/// Memoized encoder projection for one `(frame-index, ctx-mode)` visit
/// (see `BurnJoint::step_for_frame`): TDT re-evaluates the same frame on
/// multi-emit (`dur == 0`), where only the token side changes.
#[derive(Default)]
struct EncCache {
    key: Option<(usize, u8)>,
    pe: Vec<f32>,
}

/// Per-process cache observability (read by tests; two integer increments
/// per step — noise next to the ~0.5 ms of math).
#[derive(Default, Debug)]
pub(crate) struct JointCacheStats {
    pub(crate) pred_hits: u64,
    pub(crate) pred_misses: u64,
    pub(crate) enc_hits: u64,
    pub(crate) enc_misses: u64,
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
    /// Committed-states version: bumped on every non-blank commit (and on
    /// test-only `set_states`), so `pred_cache` never serves numbers
    /// computed from older states. Reset per utterance (see `reset`).
    version: u64,
    /// Memoized prediction side + encoder projection (see `step_core` /
    /// `step_for_frame`). Keyed by inputs, so reuse is exact.
    pred_cache: PredCache,
    enc_cache: EncCache,
    /// Hit/miss observability for the caches above.
    pub(crate) stats: JointCacheStats,
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
        weight_u8: super::quant_linear::transpose_u8(data, in_dim, out_dim),
        scale: vec![scalar(scale, &format!("{wname}.scale"))?; out_dim],
        zero_point: vec![scalar(zp, &format!("{wname}.zero_point"))?; out_dim],
        bias: Some(bias),
        in_dim,
        out_dim,
        cache: std::sync::OnceLock::new(),
        w_i16: std::sync::OnceLock::new(),
            w_vnni: std::sync::OnceLock::new(),
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
        w_i16: std::sync::OnceLock::new(),
        r_i16: std::sync::OnceLock::new(),
        w_tr: std::sync::OnceLock::new(),
        r_tr: std::sync::OnceLock::new(),
        w_sum: std::sync::OnceLock::new(),
        r_sum: std::sync::OnceLock::new(),
        b_fold: std::sync::OnceLock::new(),
    })
}

/// Exact tensor names `weights::load_selected` must fetch for `load`.
pub(crate) fn required_tensors() -> HashSet<String> {
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
            version: 0,
            pred_cache: PredCache::default(),
            enc_cache: EncCache::default(),
            stats: JointCacheStats::default(),
        })
    }

    pub(crate) fn reset(&mut self) {
        for v in self.h.iter_mut().chain(self.c.iter_mut()) {
            v.fill(0.0);
        }
        // A new utterance restarts the frame index and the committed
        // states: entries memoized under either must not survive.
        self.version = 0;
        self.pred_cache.key = None;
        self.enc_cache.key = None;
    }

    /// Build every cached tensor / dequantized twin now (load-time
    /// warming): the 3 joint projections plus both LSTM layers.
    pub(crate) fn prime_cache(&self, device: &super::runtime::CpuDevice) {
        self.enc.prime_cache(device);
        self.pred.prime_cache(device);
        self.out.prime_cache(device);
        self.l0.prime_cache();
        self.l1.prime_cache();
    }

    /// Overwrite the running states (tests seed ORT states to isolate
    /// one step; the greedy driver only ever uses `reset` + commits).
    #[cfg(test)]
    pub(crate) fn set_states(&mut self, h: [Vec<f32>; 2], c: [Vec<f32>; 2]) {
        self.h = h;
        self.c = c;
        // Arbitrary states break the version chain: drop memoization.
        self.version = self.version.wrapping_add(1);
        self.pred_cache.key = None;
        self.enc_cache.key = None;
    }

    #[cfg(test)]
    pub(crate) fn states(&self) -> (&[Vec<f32>; 2], &[Vec<f32>; 2]) {
        (&self.h, &self.c)
    }

    /// Tensor-mode discriminant for cache keys: scalar, i16 and VNNI
    /// compute different values (VNNI is approximate by construction;
    /// i16-vs-Flex differ by f32 summation order), so a hit must never
    /// serve another mode's numbers.
    fn ctx_mode(ctx: &TensorCtx) -> u8 {
        if !ctx.enabled {
            0
        } else if ctx.use_vnni() {
            2
        } else {
            1
        }
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
        let mode = Self::ctx_mode(ctx);
        let pe = self.enc.forward_dispatch_auto(enc_frame, ctx);
        self.step_core(pe, token, ctx, mode)
    }

    /// `decode`'s frame-aware entry: `pe` memoized per `(t, mode)` (the
    /// same frame is re-evaluated on multi-emit, where only the token
    /// side changes), then the shared `step_core`.
    fn step_for_frame(
        &mut self,
        t: usize,
        enc_frame: &[f32],
        token: usize,
        ctx: &TensorCtx,
        mode: u8,
    ) -> (JointStep, StepStates) {
        assert_eq!(enc_frame.len(), 1024);
        let pe = if self.enc_cache.key == Some((t, mode)) {
            self.stats.enc_hits += 1;
            self.enc_cache.pe.clone()
        } else {
            self.stats.enc_misses += 1;
            let pe = self.enc.forward_dispatch_auto(enc_frame, ctx);
            self.enc_cache.key = Some((t, mode));
            self.enc_cache.pe = pe.clone();
            pe
        };
        self.step_core(pe, token, ctx, mode)
    }

    /// Tail of `step_for` past the encoder projection: memoized LSTM +
    /// pred side, then add + ReLU + out projection + commit rule.
    /// `pe` is the encoder projection for the visited frame (cached or
    /// fresh — `step_core` never recomputes it).
    fn step_core(
        &mut self,
        pe: Vec<f32>,
        token: usize,
        ctx: &TensorCtx,
        mode: u8,
    ) -> (JointStep, StepStates) {
        assert_eq!(pe.len(), self.l1.hid);
        // Prediction side: memoized on (token, committed-states version,
        // ctx mode). Blank runs hit: same last token, states uncommitted.
        // A commit bumps `version`, so the next step always recomputes.
        let key = Some((token, self.version, mode));
        let (l0h, l0c, l1h, l1c, pp) = if self.pred_cache.key == key {
            self.stats.pred_hits += 1;
            let pc = &self.pred_cache;
            (
                pc.l0h.clone(),
                pc.l0c.clone(),
                pc.l1h.clone(),
                pc.l1c.clone(),
                pc.pp.clone(),
            )
        } else {
            self.stats.pred_misses += 1;
            // Per-step buffers are reused (no fresh embed/xh/gates/candidate
            // allocations); only the returned logits/states are owned.
            let JointScratch { embed, l0, l1 } = &mut self.scratch;
            self.embed.lookup_into(token, embed);
            self.l0.step_into(embed, &self.h[0], &self.c[0], ctx, l0);
            self.l1.step_into(&l0.h, &self.h[1], &self.c[1], ctx, l1);
            let pp = self.pred.forward_dispatch_auto(&l1.h, ctx);
            self.pred_cache.key = key;
            self.pred_cache.l0h = l0.h.clone();
            self.pred_cache.l0c = l0.c.clone();
            self.pred_cache.l1h = l1.h.clone();
            self.pred_cache.l1c = l1.c.clone();
            self.pred_cache.pp = pp.clone();
            (l0.h.clone(), l0.c.clone(), l1.h.clone(), l1.c.clone(), pp)
        };
        let mut hsum: Vec<f32> = pe.iter().zip(pp.iter()).map(|(a, b)| a + b).collect();
        for v in hsum.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        let logits = self.out.forward_dispatch_auto(&hsum, ctx);
        let tok = TdtGreedy::argmax(&logits[..self.vocab_size]);
        if tok != self.blank_id {
            // Commit from the step's own values (cached or fresh — never
            // from `scratch`, which a cache hit leaves untouched).
            self.h = [l0h.clone(), l1h.clone()];
            self.c = [l0c.clone(), l1c.clone()];
            self.version = self.version.wrapping_add(1);
        }
        let step = JointStep {
            token_logits: logits[..self.vocab_size].to_vec(),
            duration_logits: logits[self.vocab_size..].to_vec(),
        };
        (
            step,
            StepStates {
                h: [l0h, l1h],
                c: [l0c, l1c],
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
        let mode = Self::ctx_mode(ctx);
        let greedy = TdtGreedy::new(self.vocab_size);
        greedy.decode(enc.len(), |t, last| {
            self.step_for_frame(t, &enc[t], last, ctx, mode).0
        })
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
        let want = required_tensors();
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
            w_i16: std::sync::OnceLock::new(),
            r_i16: std::sync::OnceLock::new(),
            w_tr: std::sync::OnceLock::new(),
            r_tr: std::sync::OnceLock::new(),
            w_sum: std::sync::OnceLock::new(),
            r_sum: std::sync::OnceLock::new(),
            b_fold: std::sync::OnceLock::new(),
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
        // Fresh layers per case: weights are immutable after load in
        // production, and the folded-bias twin is built once.
        // Case forget-open, rest shut... i(0.5),o(0.5),f(~1),g(0):
        // c = 1*c_prev + 0.5*0 = c_prev; h = 0.5*tanh(c).
        let mut l = zero_lstm(1, 1);
        l.b = vec![0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (h, c) = l.step(&[0.0], &[0.0], &[2.0], &TensorCtx::scalar());
        assert!((c[0] - 2.0).abs() < 1e-3, "c={c:?}");
        assert!((h[0] - 0.5 * 2.0f32.tanh()).abs() < 1e-3, "h={h:?}");
        // Case input-open, forget-shut: c = 0*prev + 1*tanh(10)... g
        // must be large: set g bias 10 instead.
        let mut l2 = zero_lstm(1, 1);
        l2.b = vec![10.0, 0.0, -10.0, 10.0, 0.0, 0.0, 0.0, 0.0];
        let (h2, c2) = l2.step(&[0.0], &[0.0], &[2.0], &TensorCtx::scalar());
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
        let want = required_tensors();
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

    /// All-paths dispatch parity on real joint weights: scalar and i16
    /// must agree BIT-EXACTLY (same integer gate MACs everywhere; the
    /// VNNI zpw correction is exact integer arithmetic, not a
    /// requantization). The VNNI leg is compared loosely: the enc/pred/out
    /// projections still run Phase-11 requantized kernels (approximate by
    /// design, ~1-3%), while the LSTM gates themselves are exact across
    /// modes (pinned bit-exact at gate level by the probe/`vnni_zp`
    /// kernel test). Ignored: needs converted weights on disk.
    /// `cargo test --features parakeet-burn -- --ignored`.
    #[test]
    #[ignore]
    fn joint_paths_agree_exactly() {
        use super::super::weights::load_selected;
        let dir = std::path::Path::new("models/parakeet-tdt-0.6b-v3-int8");
        let want = required_tensors();
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
        let dev = super::super::runtime::cpu_device();
        let run = |joint: &mut BurnJoint, ctx: &TensorCtx| {
            // Fresh states per mode (no pred-cache aliasing: reset clears
            // it). Token 0 on this frame may emit or blank — either way
            // every mode must compute the same-shape step.
            joint.reset();
            let (s, _) = joint.step_for(&frame, 0, ctx);
            s.token_logits
                .iter()
                .chain(s.duration_logits.iter())
                .copied()
                .collect::<Vec<f32>>()
        };
        let sctx = TensorCtx::scalar();
        let all_s = run(&mut joint, &sctx);
        let all_i = run(
            &mut joint,
            &TensorCtx { device: dev, enabled: true, vnni: false },
        );
        assert_eq!(all_s, all_i, "scalar vs i16 must be bit-exact");
        eprintln!("joint scalar-vs-i16: exact match");
        let all_v = run(&mut joint, &TensorCtx::with_vnni(dev, true));
        let (mut mx, mut rm) = (0.0f32, 0.0f32);
        for (a, b) in all_v.iter().zip(all_i.iter()) {
            mx = mx.max((a - b).abs());
            rm = rm.max(b.abs());
        }
        eprintln!("joint vnni-vs-i16: max_abs={mx:.4} ref_max={rm:.3}");
        assert!(mx < 0.05 * rm.max(1.0), "mx={mx} rm={rm}");
    }

    /// Hand-built tiny joint for cache-logic tests (no weights file):
    /// hid=4, vocab=8 (blank 7). All weights are exact zeros, so every
    /// projection returns its bias exactly on every path and `out_bias`
    /// steers argmax directly. `token_bias` (len 8) + `dur_bias` (len 5)
    /// form the out projection.
    fn tiny_u8(in_dim: usize, out_dim: usize, bias: Vec<f32>) -> Uint8Linear {
        Uint8Linear {
            weight_u8: vec![0u8; in_dim * out_dim],
            scale: vec![1.0],
            zero_point: vec![0.0],
            bias: Some(bias),
            in_dim,
            out_dim,
            cache: std::sync::OnceLock::new(),
            w_i16: std::sync::OnceLock::new(),
            w_vnni: std::sync::OnceLock::new(),
        }
    }

    fn tiny_lstm() -> LstmLayer {
        LstmLayer {
            w: vec![0i8; 4 * 16],
            w_scale: 1.0,
            w_zp: 0.0,
            r: vec![0i8; 4 * 16],
            r_scale: 1.0,
            r_zp: 0.0,
            b: vec![0.0; 32],
            in_dim: 4,
            hid: 4,
            w_i16: std::sync::OnceLock::new(),
            r_i16: std::sync::OnceLock::new(),
            w_tr: std::sync::OnceLock::new(),
            r_tr: std::sync::OnceLock::new(),
            w_sum: std::sync::OnceLock::new(),
            r_sum: std::sync::OnceLock::new(),
            b_fold: std::sync::OnceLock::new(),
        }
    }

    fn tiny_joint(token_bias: Vec<f32>, dur_bias: Vec<f32>) -> BurnJoint {
        assert_eq!(token_bias.len(), 8);
        assert_eq!(dur_bias.len(), 5);
        let mut bias = token_bias;
        bias.extend(dur_bias);
        BurnJoint {
            embed: EmbedTable {
                data: vec![0u8; 8 * 4],
                scale: 1.0,
                zero_point: 0.0,
                rows: 8,
                cols: 4,
            },
            l0: tiny_lstm(),
            l1: tiny_lstm(),
            enc: tiny_u8(1024, 4, vec![0.0; 4]),
            pred: tiny_u8(4, 4, vec![0.0; 4]),
            out: tiny_u8(4, 13, bias),
            vocab_size: 8,
            blank_id: 7,
            h: [vec![0.0; 4], vec![0.0; 4]],
            c: [vec![0.0; 4], vec![0.0; 4]],
            scratch: JointScratch::default(),
            version: 0,
            pred_cache: PredCache::default(),
            enc_cache: EncCache::default(),
            stats: JointCacheStats::default(),
        }
    }

    fn blank_token_bias() -> Vec<f32> {
        let mut b = vec![0.0; 8];
        b[7] = 5.0;
        b
    }

    fn token0_bias() -> Vec<f32> {
        let mut b = vec![0.0; 8];
        b[0] = 5.0;
        b
    }

    #[test]
    fn blank_repeats_hit_pred_cache() {
        // Same blank token, uncommitted states: the repeat must return
        // identical logits from the cache (1 miss + 1 hit), with states
        // still zero (blank never commits) and no enc-cache traffic
        // (`step_for` has no frame index).
        let mut j = tiny_joint(blank_token_bias(), vec![0.0; 5]);
        let frame = vec![0.0; 1024];
        let ctx = TensorCtx::scalar();
        let (s0, _) = j.step_for(&frame, 7, &ctx);
        let (s1, _) = j.step_for(&frame, 7, &ctx);
        assert_eq!(s0.token_logits, s1.token_logits);
        assert_eq!(s0.duration_logits, s1.duration_logits);
        assert!(j.h.iter().all(|r| r.iter().all(|&v| v == 0.0)));
        assert_eq!((j.stats.pred_hits, j.stats.pred_misses), (1, 1));
        assert_eq!((j.stats.enc_hits, j.stats.enc_misses), (0, 0));
    }

    #[test]
    fn commit_bumps_version_and_misses() {
        // Every emit commits, so the same token on changed states must
        // recompute (2 misses, no hits) — a stale pred entry would both
        // corrupt the commit chain and hide here as a hit.
        let mut j = tiny_joint(token0_bias(), vec![0.0; 5]);
        let frame = vec![0.0; 1024];
        let ctx = TensorCtx::scalar();
        let (s0, _) = j.step_for(&frame, 0, &ctx);
        assert_eq!(TdtGreedy::argmax(&s0.token_logits), 0);
        let (s1, _) = j.step_for(&frame, 0, &ctx);
        assert_eq!(s0.token_logits, s1.token_logits);
        assert_eq!((j.stats.pred_hits, j.stats.pred_misses), (0, 2));
        assert_eq!(j.version, 2);
    }

    #[test]
    fn cache_key_isolates_ctx_modes() {
        // Same inputs through scalar then tensor must not alias: the
        // modes compute different values on real weights (VNNI
        // approximate, i16-vs-Flex f32 order), so the key carries the mode
        // even though this zero-weight toy agrees exactly.
        let mut j = tiny_joint(blank_token_bias(), vec![0.0; 5]);
        let frame = vec![0.0; 1024];
        let sctx = TensorCtx::scalar();
        let tctx = TensorCtx {
            device: super::super::runtime::cpu_device(),
            enabled: true,
            vnni: false,
        };
        let (ss, _) = j.step_for(&frame, 7, &sctx);
        let (st, _) = j.step_for(&frame, 7, &tctx);
        assert_eq!(ss.token_logits, st.token_logits);
        assert_eq!((j.stats.pred_hits, j.stats.pred_misses), (0, 2));
    }

    #[test]
    fn decode_blank_run_reuses_pred_side() {
        // 4 blank frames advancing one per step (dur bias peaks at 1):
        // distinct frames miss enc each time, identical prediction inputs
        // hit 3 of 4.
        let mut dur = vec![0.0; 5];
        dur[1] = 5.0;
        let mut j = tiny_joint(blank_token_bias(), dur);
        let enc = vec![vec![0.0; 1024]; 4];
        let (toks, _, _) = j.decode(&enc, &TensorCtx::scalar());
        assert!(toks.is_empty());
        assert_eq!((j.stats.pred_hits, j.stats.pred_misses), (3, 1));
        assert_eq!((j.stats.enc_hits, j.stats.enc_misses), (0, 4));
    }

    #[test]
    fn decode_multiemit_reuses_encoder_projection() {
        // Token 0 emits with duration 0: TDT stays on the one frame up to
        // the 10-token cap. Every emit commits (new version → pred always
        // misses) while the encoder projection hits 9 of 10.
        let mut dur = vec![0.0; 5];
        dur[0] = 5.0;
        let mut j = tiny_joint(token0_bias(), dur);
        let enc = vec![vec![0.0; 1024]; 1];
        let (toks, _, durs) = j.decode(&enc, &TensorCtx::scalar());
        assert_eq!(toks, vec![0; 10]);
        assert!(durs.iter().all(|&d| d == 0));
        assert_eq!((j.stats.pred_hits, j.stats.pred_misses), (0, 10));
        assert_eq!((j.stats.enc_hits, j.stats.enc_misses), (9, 1));
    }

    }
