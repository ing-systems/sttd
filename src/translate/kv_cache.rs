//! The decoder cache: which inputs a session declares, the empty tensors the
//! prompt step sends, and which slots a cached step replaces.

use anyhow::{Context, Result};
use ndarray17::ArrayD;
use ort::session::Session;
use ort::session::SessionInputValue;
use ort::value::{TensorRef, ValueType};

use crate::translator_store::GraphShape;

/// The name prefix of the decoder's cache inputs. The matching outputs are the
/// same strings under `present.`, which [`present_name`] derives from this.
///
/// A `past_key_values.N.{decoder,encoder}.{key,value}` shape, which is what
/// names a slot: `N` walks layers and the two halves are self- and
/// cross-attention. How many there are is the export's own
/// [`GraphShape::cache_slots`], not a constant here.
const PAST_PREFIX: &str = "past_key_values.";

/// The two halves of a cache slot, as the export spells them inside a slot name.
///
/// Which attention a slot belongs to is not something sttd decides: the export
/// names it, and reading the name is what keeps a cross-attention tensor out of a
/// self-attention slot. Both halves are the same shape, so nothing else would.
const SELF_SLOT: &str = ".decoder.";
const CROSS_SLOT: &str = ".encoder.";

/// What one decoder session declares about its key/value cache, read once at load.
///
/// Both halves of the answer, because a decode needs both: the empty tensors the
/// prompt step sends, and which of them a cached step replaces.
pub(super) struct CacheSlots {
    /// `(input name, empty tensor)` in the order the graph declares its inputs.
    /// Built at load because the graph's metadata does not change and a step that
    /// re-walked it would rebuild 24 or 48 names per generated token.
    pub(super) empty: Vec<(String, ArrayD<f32>)>,
    /// `(index into empty, output name)` for the self-attention slots only — the
    /// ones a step replaces. The output name is the input name without its
    /// `past_key_values.` prefix, so a name and the tensor read into it cannot
    /// drift apart.
    self_attention: Vec<(usize, String)>,
}

impl CacheSlots {
    /// The slot names, sorted, for comparing two graphs' cache contracts.
    pub(super) fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.empty.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names
    }

    /// The empty slots as named inputs, borrowed from the tensors built at load.
    ///
    /// Views rather than fresh tensors: a zero-length slot holds no elements, so
    /// the only per-step cost is naming it.
    pub(super) fn empty_inputs(&self) -> impl Iterator<Item = (&str, SessionInputValue<'_>)> {
        self.empty.iter().map(|(name, tensor)| {
            (
                name.as_str(),
                TensorRef::from_array_view(tensor.view())
                    .expect("a cache slot built here must be convertible")
                    .into(),
            )
        })
    }
}

/// The key/value cache one utterance's decode runs against: what the prompt step
/// produced, fed back to every step after it.
///
/// **Two halves, and only one of them moves.** A slot is
/// `[batch, heads, sequence, head_dim]`, and the export's own name says which
/// attention a slot belongs to: `past_key_values.N.decoder.*` is self-attention
/// and grows by one token per step, `past_key_values.N.encoder.*` is
/// cross-attention and is a function of the encoder output alone. The cached
/// decoder returns **only the self half** under `present.` — measured on both
/// exports — because the cross half is an input it never rewrites. So the cross
/// tensors the prompt filled are handed back unchanged for the whole decode, and
/// reading them out of every step's output would copy the encoder's keys and
/// values once per token for no change at all.
///
/// Slot order is the cached graph's own, read at load and shared with the
/// [`CacheSlots`] that names them, so a cross-attention tensor cannot end up in a
/// self-attention slot. That mistake would be no error: both halves are
/// `[1, heads, n, head_dim]`, and only the transcript would show it.
pub(super) struct KvCache {
    /// `(input name, tensor)` in slot order: per layer, the self-attention key
    /// and value, then the cross-attention key and value.
    slots: Vec<(String, ArrayD<f32>)>,
    /// `(index into slots, output name)` for the self-attention slots. See
    /// [`CacheSlots::self_attention`]; the cross half is not in it because a step
    /// never rewrites it.
    self_attention: Vec<(usize, String)>,
}

impl KvCache {
    /// Both halves, read out of a prompt run's outputs by the names
    /// [`CacheSlots`] derived from the cached graph.
    ///
    /// Owned rather than borrowed: the tensors belong to the session's output
    /// borrow, and the decode outlives it. The copy is the point of caching — one
    /// copy of the encoder's keys and values, against re-running the whole prefix
    /// on every step.
    pub(super) fn capture(
        slots: &CacheSlots,
        outputs: &ort::session::SessionOutputs<'_>,
    ) -> Result<Self> {
        let read = |name: &str| -> Result<ArrayD<f32>> {
            outputs
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("the prompt step returned no {name}"))?
                .try_extract_array::<f32>()
                .map(|view| view.to_owned())
                .with_context(|| format!("{name} is not an f32 array"))
        };
        let mut cache = KvCache {
            slots: Vec::with_capacity(slots.empty.len()),
            self_attention: slots.self_attention.clone(),
        };
        for (name, _) in &slots.empty {
            cache.slots.push((name.clone(), read(&present_name(name))?));
        }
        Ok(cache)
    }

    /// How many slots this cache feeds, for sizing the step's input list.
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Every slot as a named input, in slot order, borrowed rather than rebuilt,
    /// for the caller to append to its own input list.
    pub(super) fn named_inputs(
        &self,
    ) -> impl Iterator<Item = Result<(&str, SessionInputValue<'_>)>> {
        self.slots.iter().map(|(name, tensor)| {
            Ok((
                name.as_str(),
                TensorRef::from_array_view(tensor.view())
                    .with_context(|| format!("{name} must be convertible"))?
                    .into(),
            ))
        })
    }

    /// Replace the self-attention half from one cached run, leaving the
    /// cross-attention half as the prompt left it.
    ///
    /// Only the self half is read out, and that is the whole reason this is not
    /// `cache.slots = capture(...)`: the cross tensors are megabytes that would
    /// be copied out of the output and back on every single token, unchanged.
    pub(super) fn update_self(&mut self, outputs: &ort::session::SessionOutputs<'_>) -> Result<()> {
        for (index, name) in &self.self_attention {
            self.slots[*index].1 = outputs
                .get(name.as_str())
                .ok_or_else(|| anyhow::anyhow!("a cached step returned no {name}"))?
                .try_extract_array::<f32>()
                .with_context(|| format!("{name} is not an f32 array"))?
                .to_owned();
        }
        Ok(())
    }
}

/// The name a `past_key_values.N.*` cache input is returned under: the same
/// string without its prefix (`present.N.decoder.key`).
///
/// Derived from the input name rather than read off the graph's output list, which
/// is what makes a name and the tensor stored under it one fact instead of two.
fn present_name(input_name: &str) -> String {
    input_name.replacen(PAST_PREFIX, "present.", 1)
}

/// Whether a cache slot is the self-attention half — the only one a cached step
/// rewrites, and the only one a slot's name has to be read to tell apart.
///
/// The cross half is named too, and that is the whole point of asking: both are
/// `[1, heads, n, head_dim]`, so a decoder key fed to an encoder slot is not a
/// shape error, it is a translation that quietly falls apart.
fn is_self_slot(name: &str) -> bool {
    name.contains(SELF_SLOT)
}

/// Read one decoder session's cache contract, and refuse a graph that does not
/// have the one this decode needs.
///
/// The export must declare exactly as many slots as its own `config.json` says it
/// has layers (four per layer: a self- and a cross-attention key and value), with
/// no duplicates, all of them f32, and it must name both halves — a session
/// declaring only one would decode against a cache silently missing half its
/// attention.
///
/// Every one of those is worth a load-time error rather than a per-step failure,
/// because they are properties of the export rather than of the text. Nor is the
/// count a constant: a Marian pair declares 24 and NLLB 48, so it is compared
/// against the geometry read out of the model rather than a number written here.
pub(super) fn cache_slots(
    session: &Session,
    path: &std::path::Path,
    shape: &GraphShape,
) -> Result<CacheSlots> {
    let mut empty: Vec<(String, ArrayD<f32>)> = Vec::new();
    let mut self_attention: Vec<(usize, String)> = Vec::new();
    for input in session.inputs() {
        if !input.name().starts_with(PAST_PREFIX) {
            continue;
        }
        let name = input.name().to_string();
        let declared = match input.dtype() {
            ValueType::Tensor { shape, ty, .. }
                if *ty == ort::value::TensorElementType::Float32 =>
            {
                shape
                    .as_ref()
                    .iter()
                    .map(|d| if *d < 0 { None } else { Some(*d as usize) })
                    .collect::<Vec<_>>()
            }
            other => anyhow::bail!(
                "{} holds cache input {name} of a type this decode does not build: \
                 {other:?}",
                path.display()
            ),
        };
        // Which half a slot belongs to is read off the export's own name, so the
        // self slots are the ones a step replaces and the cross slots the ones it
        // leaves alone. A slot named neither is refused rather than guessed at:
        // guessing would put an encoder key where a decoder key belongs, and both
        // are `[1, heads, n, head_dim]`, so nothing would say so.
        if is_self_slot(&name) {
            self_attention.push((empty.len(), present_name(&name)));
        } else if !name.contains(CROSS_SLOT) {
            anyhow::bail!(
                "{} names cache input {name} as neither self- nor cross-attention; \
                 this decode cannot tell which half of the cache it belongs to",
                path.display()
            )
        }
        empty.push((
            name,
            ArrayD::from_shape_vec(empty_cache_shape(&declared, *shape), vec![])?,
        ));
    }
    let expected = shape.cache_slots();
    if empty.len() != expected {
        anyhow::bail!(
            "{} declares {} cache inputs, expected {expected} ({} layers)",
            path.display(),
            empty.len(),
            shape.layers,
        );
    }
    if self_attention.is_empty() {
        anyhow::bail!(
            "{} declares no self-attention cache input; a cached step would have \
             nothing to write its keys into",
            path.display()
        );
    }
    let mut names: Vec<&str> = empty.iter().map(|(name, _)| name.as_str()).collect();
    names.sort_unstable();
    names.dedup();
    if names.len() != expected {
        anyhow::bail!("{} declares duplicate cache input names", path.display());
    }
    Ok(CacheSlots {
        empty,
        self_attention,
    })
}

/// The shape of one empty cache tensor, built from the graph's declared axes and
/// the export's own config.
///
/// A cache slot is `[batch, heads, sequence, head_dim]`, so axis 2 is the one that
/// goes to zero. Every other axis keeps the value the metadata declares, and an
/// axis the metadata leaves dynamic falls back to `shape` — read from
/// `config.json`, never compiled in, because a zero-length tensor cannot carry a
/// dynamic axis and the two families disagree on both numbers that matter (8 heads
/// of 64 against 16 of 64).
fn empty_cache_shape(declared: &[Option<usize>], shape: GraphShape) -> Vec<usize> {
    let default = [1usize, shape.heads, 0, shape.head_dim()];
    if declared.len() != default.len() {
        return default.to_vec();
    }
    std::array::from_fn::<usize, 4, _>(|axis| {
        if axis == 2 {
            0
        } else {
            declared[axis].unwrap_or(default[axis])
        }
    })
    .to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator_store::Family;

    /// The two exports' own geometry, as their `config.json` files publish it. Read
    /// out of the ONNX protobuf for the cache slot count and confirmed against each
    /// config for the rest: a Marian merged decoder declares
    /// `past_key_values.0..5.{decoder,encoder}.{key,value}` and NLLB the same to
    /// `..11`.
    fn marian_shape() -> GraphShape {
        GraphShape::new(6, 8, 64)
    }
    fn nllb_shape() -> GraphShape {
        GraphShape::new(12, 16, 64)
    }

    #[test]
    fn an_empty_cache_shape_keeps_the_declared_axes_and_zeroes_the_sequence() {
        // The metadata of a Marian export: `[batch, heads, sequence, head_dim]`
        // with the sequence dynamic. `ort` reports a dynamic axis as -1.
        assert_eq!(
            empty_cache_shape(&[None, Some(8), None, Some(64)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        // A fully declared sequence axis still goes to zero: that is the whole
        // point of the empty past this decode sends every step.
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8), Some(20), Some(64)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        // A dynamic head count falls back to the export's own config, because a
        // zero-length tensor cannot carry a dynamic axis. This is the assertion
        // that the fallback is *per family*: the same dynamic metadata must yield
        // 8 heads for a pair and 16 for NLLB.
        assert_eq!(
            empty_cache_shape(&[None, None, None, None], marian_shape()),
            vec![1, 8, 0, 64]
        );
        assert_eq!(
            empty_cache_shape(&[None, None, None, None], nllb_shape()),
            vec![1, 16, 0, 64]
        );
        // A rank this decode does not know falls back to the whole shape.
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8)], marian_shape()),
            vec![1, 8, 0, 64]
        );
        assert_eq!(
            empty_cache_shape(&[Some(1), Some(8)], nllb_shape()),
            vec![1, 16, 0, 64]
        );
    }

    /// The prompt step is the only one that sends an empty cache, and a cached step
    /// is the only one that sends a filled one. Pinned as a shape assertion because
    /// the failure it prevents is a load-time shape error, not a wrong answer.
    #[test]
    fn a_past_slot_is_fed_empty_by_the_prompt_step() {
        for (shape, heads) in [(marian_shape(), 8), (nllb_shape(), 16)] {
            let slot = empty_cache_shape(&[None, Some(heads), None, Some(64)], shape);
            assert_eq!(slot, vec![1, heads, 0, 64]);
            assert_eq!(
                slot.iter().product::<usize>(),
                0,
                "an empty past holds no elements at all"
            );
            assert_eq!(slot[2], 0, "only the sequence axis is zeroed");
        }
    }

    /// Which slots a cached step replaces, and what it reads them out of.
    ///
    /// The one thing nothing else can check: `cache_slots` only ever runs against a
    /// loaded session, and getting the split wrong feeds each layer's
    /// cross-attention tensors to its self-attention slot — the same
    /// `[1, heads, n, head_dim]` either way, so nothing errors and only the
    /// transcript shows it. So the naming, the split and the output name are pinned
    /// here against the export's own convention, over both families' layer counts.
    #[test]
    fn a_cached_step_replaces_the_self_attention_slots_only() {
        for shape in [marian_shape(), nllb_shape()] {
            // What a real export calls its slots, in its own order: per layer, the
            // self key and value, then the cross key and value.
            let slots: Vec<String> = (0..shape.layers)
                .flat_map(|layer| {
                    [SELF_SLOT, CROSS_SLOT].into_iter().flat_map(move |half| {
                        ["key", "value"]
                            .into_iter()
                            .map(move |kv| format!("{PAST_PREFIX}{layer}{half}{kv}"))
                    })
                })
                .collect();
            assert_eq!(slots.len(), shape.cache_slots(), "four slots per layer");

            let self_slots: Vec<&String> = slots.iter().filter(|n| is_self_slot(n)).collect();
            assert_eq!(
                self_slots.len(),
                shape.layers * 2,
                "one self key and a value per layer"
            );
            for name in &self_slots {
                assert!(
                    !name.contains(CROSS_SLOT),
                    "{name} is one half or the other"
                );
                // And each is read back out under its own name with the prefix
                // swapped, which is the only derivation that keeps a name and the
                // tensor stored under it from drifting apart.
                let present = present_name(name);
                assert_eq!(present, name.replacen(PAST_PREFIX, "present.", 1));
                assert!(
                    present.starts_with("present.") && !present.contains(PAST_PREFIX),
                    "{present} is not a present name"
                );
            }
            // The cross half is captured once, at the prompt, and never rewritten:
            // a cached step that replaced it would copy the encoder's keys and
            // values out of the output and back on every token, unchanged.
            for name in slots.iter().filter(|n| n.contains(CROSS_SLOT)) {
                assert!(!is_self_slot(name), "{name} must stay out of the self half");
            }
        }
        // A name the export did not use is not a slot this decode can place, which
        // is why `cache_slots` refuses one rather than guessing which half it is.
        assert!(!is_self_slot("past_key_values.0.encoder.key"));
        assert!(!is_self_slot("past_key_values.0.something.key"));
    }

    /// The cache contract is read from the model, so what is pinned here is the
    /// *arithmetic* — four slots per layer — and the two shapes it produces. The
    /// slot count is not written down in either family: a Marian export declares
    /// 24 and NLLB 48, and `check_cache_inputs` compares the graph against whatever
    /// the config said rather than against a constant.
    #[test]
    fn the_cache_slot_count_follows_the_layer_count() {
        assert_eq!(
            marian_shape().cache_slots(),
            24,
            "6 x self+cross x key+value"
        );
        assert_eq!(
            nllb_shape().cache_slots(),
            48,
            "12 x self+cross x key+value"
        );
        // Both agree with `d_model / head_dim == heads`, which is where the head
        // width comes from.
        for (shape, d_model) in [(marian_shape(), 512), (nllb_shape(), 1024)] {
            assert_eq!(d_model / shape.head_dim(), shape.heads);
            assert_eq!(shape.cache_slots(), shape.layers * 4);
        }
        // The decode caps are per family for the same reason the shapes are: the
        // same O(n^2) decode over a decoder four times the width.
        assert!(Family::Nllb.max_new_tokens() < Family::Marian.max_new_tokens());
    }
}
