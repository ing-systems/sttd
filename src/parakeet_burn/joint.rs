// Prediction + joint network (Burn port, bring-up).
//
// ONNX reference (`decoder_joint-model.int8.onnx`): inputs
// `encoder_outputs [1, 1024, 1]` + `targets [1, 1]` + `target_length` +
// LSTM states `[2, 1, 640]` x2; outputs joint logits
// `[vocab + n_durations]` + `prednet_lengths` + updated states.
// `DynamicQuantizeLSTM` x2, `MatMulInteger` x3, `Gemm/MatMul` for the
// joint projection.
//
// The greedy loop lives in `decode.rs`; this module owns the per-step
// joint computation once weights are converted.

use super::quant::Int8Linear;

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

#[allow(dead_code)]
pub(crate) struct BurnJoint {
    config: JointConfig,
    linears: Vec<Int8Linear>,
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
}
