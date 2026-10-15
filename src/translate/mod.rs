// ---------------------------------------------------------------------------
// Translation step: text in, text out, with an ONNX seq2seq model as the
// translator.
// ---------------------------------------------------------------------------
//
// The step runs after the gates and before the emitter, so everything that
// decides *whether* to speak still reads the source text. Only what the user
// hears changes. A failure here never drops an utterance: the translation thread
// runs the result through `fallback_or_translated`, which warns and returns the
// source.
//
// The parts, one submodule each:
//
// * `text` and `tokenizer` are the pure half (`split_sentences`,
//   `fallback_or_translated`, `patched_value`): no weights, so they are
//   unit-tested without a model.
// * `engine` is the loaded `Translator` — the three `ort` sessions, the tokenizer
//   and the greedy decode — and `kv_cache` the decoder cache it threads through.
// * `plan` is the run's translation settings and the load for one target;
//   `state` is the switch that decides which translator is live at runtime.
// * `lane` is `TranslationOut`, the thread the translator runs on, with a
//   bounded FIFO in front of it so a translation never stalls the next decode.
//
// **The decode is two graphs, and why is the one thing here worth reading.** The
// export ships a merged decoder and a `decoder_with_past` one. A single graph
// cannot do both halves of a seq2seq decode, because the graph that computes the
// cross-attention keys is not the graph that reads them back:
//
// * The **prompt step** runs the merged graph. It takes `encoder_hidden_states`,
//   opens the decoder on the family's opening ids, and — this is the part that is
//   measured rather than assumed — returns the key/value cache under `present.`
//   even with `use_cache_branch` false. `true` cannot: that branch fails on an
//   empty past with `encoder_attn/Reshape_4 ... The dimension with value zero
//   exceeds the dimension size of the input tensor`.
// * Every **later step** runs the cached graph, one token at a time against the
//   cache. It takes no `encoder_hidden_states` and no cache flag, because the
//   cross-attention half travels inside the cache, and it returns the self half
//   only.
//
// So the decode is **O(n) in output length** instead of the O(n²) a single graph
// re-running its own prefix costs, and the cross-attention keys and values are
// computed once per utterance rather than once per token.
//
// Two model families run through those same graphs, and they differ in exactly
// two places, both of them **inputs**:
//
// * A MarianMT pair export takes raw text and opens its decoder on
//   `decoder_start_token_id`, because the weights already only do one direction.
// * A multilingual NLLB-200 export is told which way to go: the source code goes
//   in front of the encoder's ids and closes them with `</s>`, and the decoder
//   opens on the *target* code. `decoder_start_token_id` is never used, which is
//   why that export declaring it equal to `eos_token_id` is fine.
//
// [`Feed`] is that difference, as a value the loaded translator holds; everything
// after it is shared. The decoder's *shape* also differs, but that is read out of
// each export's own `config.json` rather than compiled in — see
// `translator_store::GraphShape` — so a re-exported model describes itself.
// ---------------------------------------------------------------------------

mod engine;
mod kv_cache;
mod lane;
mod plan;
mod state;
mod text;
mod tokenizer;

pub(crate) use lane::{TranslationJob, TranslationOut};
pub(crate) use plan::{PlanFlags, TranslationPlan};
pub(crate) use state::{RealTranslator, TranslationState};
#[cfg(test)]
pub(crate) use text::fallback_or_translated;
