# Parakeet-Burn performance findings (Phase 9, 2026-09-23)

How the native Burn engine went 153 → 113 ms per 2.5 s utterance, what was
tried and rejected (with numbers), and why ORT (~51 ms) is still faster.
All timings release-mode on the Ryzen 9 9950X host (16C/32T, AVX-512 +
VNNI, 1 MB L2/core), `--threads 0` (= 16 physical) unless noted.

## Baselines (synthetic 2.5 s WAV, `bench_2s5.wav`)

| Engine | transcribe | rtf |
|---|---|---|
| burn (pre-Phase-9) | 153 ms | 0.06 |
| burn (post-Phase-9) | 113–116 ms | 0.045 |
| ort | 51–54 ms | 0.02 |

## Where time goes (post-Phase-9, realistic T'=32 shapes)

Release `profile_breakdown` (`cargo test --release --features
parakeet-burn -- --ignored profile_breakdown --nocapture`), min-of-5:

| Stage | ms / 2.5 s utt |
|---|---|
| 24 Conformer blocks (~4.9 ms/layer: attn ~1.2, FFN ~0.9 x2, conv ~0.65, norms/adds rest) | ~100–117 |
| pre-encode (5 convs + out-proj) | ~12 |
| joint decode (~40 steps) | ~16 |
| frontend (251 frames) | ~1–2 |
| Per-utterance GEMM math total | ~20 GMACs |

Per-layer GEMM count (T=32): FFN 4 x 134M + attn/out proj 8 x 33M +
conv pointwise ~100M ≈ 0.8 GMACs/layer ≈ 20 GMACs/utterance.
Effective GEMM rate: ~400 GMAC/s machine-wide (~26 GMAC/s/core).

## What worked (all bit-exact, all gated by exact-equality tests + ORT goldens)

| Change | Effect |
|---|---|
| `igemm` 2D tiling (row-chunks x L2 column blocks, direct writes, no stitch) + serial quantize (`quantize_from_rows`, no `flat` copy) | FFN linears −25–45% per call (0.467→0.355, 0.554→0.394, wout 0.281→0.151 ms at T32); quantize 0.08→0.02 ms/call |
| Exact parallel integer depthwise conv (`forward_par`) replaces Flex conv1d | blocks −8%; tensor-vs-scalar residual 3.8e-6 (drift removed) |
| Exact parallel integer pre-encode convs (`conv2d_par`, 1x1s as integer GEMM) | pre −17%; ORT golden max_abs = 0.00000 (was 5e-6) |
| Fused LSTM gates (`[x,h]@[W;R]+b`, one matmul/step) | joint 17.1→16.0 ms |
| Frontend: cached FFT plan/window, parallel frames | 4.6→~1 ms |
| Side effect: removing Flex f32 drift fixed the espeak blank-collapse — burn is now word-exact vs ORT on robotic speech where it previously emitted empty | accuracy win, free |

## What was tried and rejected (measured)

| Idea | Result |
|---|---|
| Scalar attention instead of Flex (hypothesis: dispatch dominates) | Flex 10x FASTER (T=8: 0.9 vs 9.4 ms). Kept Flex |
| 64 B-aligned i16 buffers (split `vpmaddwd` loads) | 8% on microbench; not worth custom-allocator risk. Dropped |
| AVX-512 VNNI (`vpdpbusd`, 64 MACs/instr) | Inapplicable: `(q-zpx)(w-zpw)` spans ±255 (9-bit); decompositions net to zero gain at equal bytes. Blocked by the bit-exact contract |
| Thread count (binary) | 4T=140, 8T=115, 16T=119, 32T=143 ms. Default 16 is fine (±3%) |
| Thin LTO | No gain (123 vs 118 = noise). Reverted to keep builds fast |

Supporting: single-thread GEMM = 63 GMAC/s (kernel OK); 16-thread
scaling only 6.7x (power/contention-bound AVX-512 + migration, no
pinning, per-call `Vec` allocs instead of arenas).

## Why ORT is faster (same 20 GMACs of work)

1. **VNNI math (~2x):** `vpdpbusd` does 64 int8 MACs/instr on u8
   weights vs `vpmaddwd`'s 32 on i16 (half throughput, double traffic).
   ORT needs ~650+ GMAC/s effective; Burn does ~400.
2. **Fusion:** ORT fuses quantize→GEMM→dequant→bias(→ReLU) in one
   cache-resident pass. Burn makes ~5 separate passes per linear.
3. **Dispatch:** ~288 linear calls (alloc + quantize + 2 rayon spawns +
   re-chunk each) + ~80 tiny Flex ops/attention-layer + per-step LSTM
   allocs, vs ORT's pre-planned packed kernels.
4. **Threading/memory:** pinned threads, arenas, block-major packed
   weights vs migration-prone pool + per-call allocs + row-major.

## Options to close the remaining ~2.3x gap

- (a) Accept it: rtf 0.045 is 22x realtime; gap is VNNI + fusion.
- (b) VNNI-friendly requantization: explicit exactness-policy change,
  needs ORT-golden re-baselining under policy (b).
- (c) Fused forward kernels + arenas + pinning: major refactor,
  exactness preserved, estimated ceiling still ~1.5x (math-bound).
- (d) GPU path when cudarc supports CUDA 13.x (blocked, unchanged).

## Phase 10: fused kernels (2026-09-23, same day)

Option (c), minus the arena refactor (deferred): SIMD quantize, GEMM
epilogues, shared-A Q/K/V, a fused attention core, joint per-step
buffers, and thread pinning. Everything stays bit-exact — the ORT
goldens were re-run unchanged and the tensor path is now *bit-identical*
to the scalar reference (`vs_scalar=0.00e0` on the layer-0 golden; all
24 isolated per-layer residuals identical to the scalar run), so the
chained drift is unchanged (0.0086).

| Change | Effect (T'=32; quantize numbers are single-call/serial, so thread-count independent) |
|---|---|
| Exact AVX-512 quantize twins: `vdivps` (IEEE, like `/`) + `trunc(x + copysign(0.5,x))` reproduces Rust's half-away `round` for the bounded ±255 window; plus `quantize_u8_3d` so pre-encode never builds the 8 MB flat copy | quantize 32×1024: 0.089 → 0.003 ms; 32×4096: 0.107 → 0.011 ms |
| `Epilogue::{Swish, Relu, AddScale, BiasRelu, Glu}` applied where the GEMM's f32 scale-up lands, writing directly into caller-owned rows (`linear_i16_rows`) | FFNs: l1 swish → l2 `h += 0.5·y` in two calls; conv: pw1 GLU (2048-wide pre-activation never materializes) → dw swish → pw2 `h += y`; attention out-proj `h += y`. No branch buffers, no swish/GLU/scale/add passes, no row copies |
| Shared-A Q/K/V: attention quantizes its input once for the three projections that read the same `x` (same `DynamicQuantizeLinear` window, exact) | −2 quantize passes + dispatches per layer |
| Fused relative-attention core: one rayon task per query row (AC + shifted BD + softmax + AV) in the *scalar* path's exact accumulation order, replacing ~100 tiny Flex ops/layer | attn stage 1.21 → 0.75 ms/layer (32T), 0.55 (16T pinned); `fused_attention_matches_scalar_core` pins bit-exactness |
| Joint: per-step buffers reused (embed row, `xh`, gates, candidates), gate nonlinearities parallel over hidden (per-element, exact) | fewer allocations; the Flex gate matmul stays |
| Rayon workers pinned to physical cores (sysfs topology; `STTD_BURN_PIN=0` opts out; skipped if `--threads` > physical cores) | ~95 → ~86 ms e2e |

### Numbers (2.5 s synthetic WAV, `--threads 0` = 16 pinned physical)

| Engine | transcribe | rtf |
|---|---|---|
| burn (Phase 9 binary, re-measured pre-Phase-10 on this host) | 116–124 ms | 0.046–0.050 |
| burn (Phase 10) | 82–96 ms (typical ~86) | 0.033–0.038 |
| ort | 52–60 ms | 0.021–0.024 |

Single runs dip to 82 ms; the 96 ms runs line up with background load
from concurrent builds/tests, so treat ~86 ms as the comparable number.

`profile_breakdown` now sizes/pins the pool like the binary, so its
numbers match production. Same-harness comparisons (the old harness ran
unpinned on rayon's default 32 threads, i.e. two workers per physical
core) and the production-config rerun:

| Stage (T'=32) | Phase 9, 32T | Phase 10, 32T | Phase 10, 16T pinned |
|---|---|---|---|
| pre-encode (T=256 mel) | 12.1 ms | 8.9 ms | 7.8 ms |
| 24 blocks | 117.4 ms (4.89/layer) | 88.2 ms (3.67/layer) | 62.9 ms (2.62/layer) |
| attn stage / layer | 1.21 ms | 0.75 ms | 0.55 ms |
| conv stage / layer | 0.61 ms | 0.51 ms | 0.37 ms |
| ffn stage / layer | 0.84 ms | 0.71 ms | 0.50 ms |
| joint decode (T=32 random) | ~16 ms | ~16.8 ms | 16.8 ms (0.47 ms/step) |
| frontend (2.5 s) | 0.8 ms | 0.7 ms | 0.7 ms |

(The 32T column isolates the Phase 10 kernel work; the 16T pinned column
is what the binary actually runs. The Phase 9 baseline re-measured today
for the e2e table is 116–124 ms vs the 113–116 ms recorded then — host
load variance, same 16-thread config.)

### Tried and rejected in Phase 10

| Idea | Result |
|---|---|
| Scalar LSTM gate matvec instead of the Flex matmul | 1.52 vs 0.47 ms/step — Flex stays |
| Pairing the GLU halves inside the GEMM micro-kernel | not needed: the post-pass writes only the gated 1024-wide rows; its cost is inside the epilogue noise |
| Moving the attention projections into one concatenated QKV GEMM | superseded by shared-A (same quantize savings, no weight re-layout) |

### What's left

The remaining gap is the GEMM math itself (~2.4 ms/layer of linears,
weight-streaming bound) plus VNNI; the elementwise bucket is now
layernorm + softmax + residual adds, and per-layer `Vec<Vec<f32>>`
scratch still allocates (the deferred arena refactor). Options (a), (b)
and (d) from Phase 9 stand unchanged.
