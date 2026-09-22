# Parakeet Burn port — bring-up status

Native int8 Parakeet-TDT-0.6B-v3 without ONNX Runtime (`src/parakeet_burn/mod.rs:1`).
`parakeet-rs` (ORT) stays the default engine; this port is selected per run and
fails fast until it can transcribe end to end, so a half-wired model can never
silently mistranscribe.

## Selection

* Runtime flag: `--parakeet-engine ort|burn` (`STTD_PARAKEET_ENGINE`, `src/cli.rs:186`), default `ort`. Only meaningful with a Parakeet `--model`; other engines ignore it. Non-Parakeet `--model` + `burn` warns and has no effect (`src/main.rs:1377`).
* Build gate: `burn` needs `--features parakeet-burn`; `burn` + `--gpu` additionally needs `--features parakeet-burn-cuda` (`src/cli.rs:186-189`, `Cargo.toml:68-79`). Without the feature, `burn` bails with the rebuild command (`src/asr.rs:122-128`).
* Dispatch: `asr::load_model` (`src/asr.rs:114-131`) routes Parakeet `ModelRef` to `ParakeetBurnModel::load` or the ORT path. Everything engine-agnostic (worker loop, gates, metrics, conditioning) is shared.
* CUDA note: `burn-cuda` 0.21 (cudarc 0.19.9) rejects the CUDA 13.4 toolkit on this host (`Cargo.toml:73-77`); CPU Burn is unaffected, GPU bring-up waits on a cudarc release with CUDA 13.x support.

## Converted weights

`burn-import` cannot ingest the quantized graph (`DynamicQuantizeLinear`, `MatMulInteger`, `ConvInteger`, `DynamicQuantizeLSTM`), so the converter extracts static int8 weights + scales into safetensors (`src/parakeet_burn/weights.rs:1-11`):

```
python3 tools/convert-parakeet-int8-to-burn.py --model-dir models/parakeet-tdt-0.6b-v3-int8
python3 tools/convert-parakeet-int8-to-burn.py --model-dir models/parakeet-tdt-0.6b-v3-int8 --check-only  # inspect, don't write
```

Outputs next to the ONNX files: `encoder-burn.safetensors` + `decoder_joint-burn.safetensors` + `encoder-manifest.json` (existing `vocab.txt` is reused). Converter keeps `<base>` (U8/I8) + `<base>.scale`/`<base>.zero_point` pairs, plain F32 biases, the frozen relative-position table (`pos_enc.table`, `(9999, 1024)` from the encoder `Constant`), and the per-layer depthwise-bias map (`dw_bias`, traced through `Reshape` to the anonymous `onnx::Conv_*` initializer). `load` refuses to start without the two safetensors and prints the exact command (`src/parakeet_burn/weights.rs:22-44`).

## Module status

| Module | State | Notes |
|---|---|---|
| `frontend.rs` | done, tested | mel parity with `parakeet-rs` TDT-v3: preemphasis 0.97, reflect-pad STFT (512/160/400 Hann), 128-bin Slaney mel, `ln(x + 2^-24)`, per-feature norm. CPU/CUDA share bit-identical features. |
| `vocab.rs` | done, tested | loads same `vocab.txt`, `▁` → space + digit spacing + `<…>` skip mirrors `decoder_tdt.rs`; blank = last id. |
| `decode.rs` | done, tested | TDT greedy state machine over joint logits (blank = vocab-1, duration skip, max 10 tokens/step). Pure over logits, no weights needed. |
| `quant.rs` | done, tested | `Uint8Linear` (encoder MatMuls/Convs) + `Int8Linear` (decoder LSTMs) store 8-bit + scale/zp and dequantize at use; swap to native int8 GEMM later changes only `forward`. Store is always `[out][in]` — ONNX `MatMul` `(K, N)` weights are transposed at load (`transpose_u8`); missing that silently computes `W @ x` instead of `W^T @ x` for square shapes. |
| `backend.rs` | done, tested | `CpuBackend = burn_ndarray::NdArray`, `CudaBackend` with `parakeet-burn-cuda`; `select_device` is log-only until the encoder threads it through. |
| `encoder.rs` | stack done, ORT-verified (isolated) | `BurnEncoder` (`PreEncodeWeights` + 24 `BlockWeights` + frozen pos table) loads via `encoder_want` and forwards mel → `[T'][1024]` rows; `load` shape-asserts per layer. Isolated per-layer parity is flat at ~2-15% (see Tests); chained full-depth drift (~40% relative at the tiny final states) is activation-requantization accumulation, matched exactly only once `quant::forward` replicates `DynamicQuantizeLinear` (step 4). |
| `block.rs` | single block done, ORT-verified | full Conformer block matching the int8 graph op-for-op in fp32: half-step FFN sandwich, XL relative attention, GLU + depthwise k=9 + Swish conv, 5 norms (eps 1e-5), `pos_window` slicing. `BlockWeights::load` shape-asserts; layer-0 output matches onnxruntime to ~1% (see Tests). |
| `pre_encode.rs` | done, ORT-verified | mel `[T][128]` → `[T'][1024]`: 1→256 k3s2 +Relu, depthwise k3s2, pointwise 1x1 +Relu, depthwise k3s2, pointwise 1x1 +Relu, freq-fold to 4096, `MatMul` + bias. Exact staged length `(t+1)/2 × 3` (not `t/8` for small `t`). Output matches onnxruntime to ~2% (see Tests). |
| `weights.rs` | gate + loaders done | `require_converted_weights` gate, `load_selected` (per-layer subset so bring-up never loads the full 582 MB / 2.4 GB f32; companions ride along via `aux_pair`), `load_safetensors_f32` inspection path. |
| `joint.rs` | config only | `JointConfig` (1024 / 2×640 / vocab + 5 durations) matches `decoder_joint-model.int8.onnx` IO; per-step joint compute lands with the decoder. |
| `model` (`mod.rs`) | stub | `load` builds vocab + frontend then hits the weights gate; `transcribe` extracts mel then bails telling the operator to use `--parakeet-engine ort` today. `uses_prompt() = false`, like the ORT path. |

ONNX reference (read off `encoder-model.int8.onnx`, not assumed): 5654 nodes, 8x subsampling, output `[1, 1024, T/8]`; `DynamicQuantizeLinear` ×223, `MatMulInteger` ×217, `ConvInteger` ×77, `LayerNormalization` ×120 (uniform eps 1e-5, biased variance), `Softmax` ×24; quantized linears carry no bias; FFN activation is Swish; depthwise pads are `[4, 4]`.

## Tests

* Unit (no weights): `cargo test --features parakeet-burn` — frontend, greedy, quant (incl. MatMul transpose), vocab, backend matmul, manifest validation (incl. conv groups), layernorm/Swish/GLU/attention/pos-window, conv stride/pad geometry, depthwise routing, exact staged length, full-stack `encoder_want` coverage.
* Real weights (ignored): `cargo test --features parakeet-burn -- --ignored` — needs `encoder-burn.safetensors` + `encoder-manifest.json` on disk. Covers layer-0 `block_forward` finiteness + norm invariant, pre-encode shape/finiteness/determinism, full-`BurnEncoder` load + forward shape/finiteness/determinism, and three ORT golden parity tests.
* ORT parity goldens (`src/parakeet_burn/testdata/`): `pre_encode_golden.safetensors` (T=64 seed-1234 mel → `/pre_encode/out/Add_output_0`) and `layer0_golden.safetensors` (that output → `/layers.0/norm_out/...`), both from onnxruntime-CPU runs of the int8 encoder, plus `encoder_full_golden.safetensors` (same mel → every `/layers.N/norm_out` + final `outputs` transposed to `[8, 1024]` rows; 24 intermediates via a temp extracted model since the shipped ONNX exposes only `outputs`/`encoded_lengths`). fp32-vs-int8 residual is activation-requantization noise — cross-checked against an independent numpy fp32 replication agreeing to 3 decimals (bias 0.10): pre-encode max 85.6 at ref_max 3921 (bound 5%), layer-0 max 0.85 at ref_max 75.8 (bound 3%). Isolated per-block parity (each block on ORT's exact input for its depth) is flat at ~2-15% relative (bounds 20%, 0.02 absolute for the tiny final layer with ref_max ~0.12) — no depth spikes, so no mis-mapped layer. Chained full-depth drift (max 0.05 at ref_max 0.12) is reported, not asserted: per-block noise amplifies down the residual stack and only exact `DynamicQuantizeLinear` replication (step 4) removes it. The layer-0 golden pins the MatMul transpose: without it every Conformer linear silently transposes.

## What's next (ordered)

1. ~~Pre-encode + subsampling forward (2× strided conv + norms, 4x + frame stacking = 8x) with ORT parity test.~~ Done (`pre_encode.rs`, golden-verified).
2. ~~24-layer stack loop over `block_forward` + per-layer parity vs ORT.~~ Done (`BurnEncoder::load`/`forward` + `encoder_want` in `encoder.rs`, `encoder_full_golden.safetensors` with all 24 intermediates, isolated per-layer parity flat at ~2-15%; chained drift documented, exact match waits on step 4's activation quantization).
3. Joint/LSTM per-step compute (`joint.rs`) wired into the `decode.rs` greedy loop, then into `ParakeetBurnModel::transcribe` (replacing the current bail).
4. Swap the naive `quant::forward` inner matmul for Burn tensor ops (NdArray first, no call-site changes); thread `BurnDevice` through to allocation.
5. CUDA path (`parakeet-burn-cuda` + `--gpu`) once cudarc supports the installed toolkit; RTF check vs ORT.
6. End-to-end file-mode parity (`--parakeet-engine burn` vs `ort` on a fixed WAV) before any mic default change.
