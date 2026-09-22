// FastConformer encoder stack (Burn port, bring-up).
//
// ONNX reference (`encoder-model.int8.onnx`): 5654 nodes, 8x subsampling,
// encoder output `[1, 1024, T/8]`, inputs `audio_signal [1, 128, T]` +
// `length`. Quantized ops: `DynamicQuantizeLinear` x223,
// `MatMulInteger` x217, `ConvInteger` x77, `LayerNormalization` x120,
// `Softmax` x24.
//
// Bring-up order (each step lands with a parity test against ORT):
//  1. subsampling conv block (2 x Conv1d stride 2 + norms)
//  2. one Conformer block (FFN + MHSA + conv + norms, int8 Linear)
//  3. full depth + length masking + 8x output projection to 1024
//
// Until step 3 completes, `forward` is unreachable: `load` fails fast
// (see `weights.rs`) so no partial encoder can mistranscribe.

use super::quant::Int8Linear;

/// One Conformer block config (FastConformer-TDT-0.6B-v3 shapes, read off
/// the int8 ONNX: 24 `layers.N` blocks, FFN 1024 -> 4096 -> 1024).
#[derive(Debug, Clone)]
pub(crate) struct ConformerConfig {
    pub(crate) d_model: usize,
    pub(crate) n_heads: usize,
    pub(crate) ffn_expansion: usize,
    pub(crate) conv_kernel: usize,
}

impl Default for ConformerConfig {
    fn default() -> Self {
        Self {
            d_model: 1024,
            // Unverified: head count is structural (attention Reshape),
            // not recoverable from weight shapes (QKV is one MatMul);
            // confirm against the graph during encoder bring-up.
            n_heads: 8,
            ffn_expansion: 4,
            conv_kernel: 9,
        }
    }
}

/// Encoder config: depth is resolved from converted weights at load.
#[derive(Debug, Clone)]
pub(crate) struct EncoderConfig {
    pub(crate) in_mels: usize,
    pub(crate) d_model: usize,
    pub(crate) out_dim: usize,
    pub(crate) n_layers: usize,
    pub(crate) block: ConformerConfig,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            in_mels: 128,
            d_model: 1024,
            out_dim: 1024,
            n_layers: 24, // `layers.0`..`layers.23` in the int8 ONNX
            block: ConformerConfig::default(),
        }
    }
}

/// Subsampling: 2 x strided conv = 4x, + frame stacking = 8x total
/// (matches the TDT `encoder_stride = 8` in `decoder_tdt.rs`).
pub(crate) fn subsampled_frames(n_mel_frames: usize) -> usize {
    n_mel_frames / 8
}

/// Placeholder encoder holding int8 linears once converted.
pub(crate) struct BurnEncoder {
    #[allow(dead_code)]
    config: EncoderConfig,
    #[allow(dead_code)]
    linears: Vec<Int8Linear>,
}

impl BurnEncoder {
    pub(crate) fn new(config: EncoderConfig) -> Self {
        Self {
            config,
            linears: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subsampling_matches_tdt_stride_8() {
        // 101 mel frames (1 s) -> 12 encoder frames (floor).
        assert_eq!(subsampled_frames(101), 12);
        assert_eq!(subsampled_frames(800), 100);
    }

    #[test]
    fn default_config_shapes_match_onnx_io() {
        let c = EncoderConfig::default();
        assert_eq!(c.in_mels, 128);
        assert_eq!(c.out_dim, 1024);
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.d_model, 1024);
    }
}
