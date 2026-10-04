// ---------------------------------------------------------------------------
// whisper_core dyn_encoder: make the exported encoder accept any window.
// ---------------------------------------------------------------------------
//
// The `onnx-community` export of the encoder pins the 30 s window in two
// places, and nowhere else (it has no `Shape` nodes):
//
// - 128 `Reshape` shape constants of the form `[-1, 1500, 20, 64]` /
//   `[-1, 1500, 1280]` (4 per layer x 32 layers). Position 1 is the encoder
//   length. Setting it to `0` makes `Reshape` copy that size from its input.
// - the `embed_positions.weight` table (`[1500, 1280]`) that node `/Add_2` adds
//   after the convolutions. Slicing it to the real length (`Shape` ->
//   `Gather` -> `Slice`) lets any shorter sequence through.
//
// Plus the declared shapes of the graph input (`input_features`, axis 2 = 3000)
// and output (`last_hidden_state`, axis 1 = 1500), which ORT checks, and the
// `value_info` hints (stale 1500s), which are dropped.
//
// There is no ONNX crate in the dependency tree, and the file is 644 MB, so this
// is a small protobuf field walker: it copies every field it does not change
// byte for byte. The 128 constants are raw `int64` data, so they are edited in
// place; everything else is a few small splices. The result is patched in memory
// on every load and handed to `ort` with `commit_from_memory`, so no file is
// written next to the (possibly root-owned) model.
//
// The patcher refuses any graph that does not match what was measured on the
// shipped export (see `PatchError::Unexpected`); the caller then keeps the
// fixed 30 s window.
// ---------------------------------------------------------------------------

use std::collections::HashSet;
use std::ops::Range;

/// Reshape constants that pin the encoder length in the shipped export:
/// 4 attention reshapes x 32 encoder layers.
const EXPECTED_RESHAPES: usize = 128;
/// The pinned encoder length (3000 mel frames / 2).
const PINNED_POSITIONS: i64 = 1500;

const POSITIONS_NAME: &[u8] = b"embed_positions.weight";
const ADD_NODE_NAME: &[u8] = b"/Add_2";
const INPUT_NAME: &[u8] = b"input_features";
const OUTPUT_NAME: &[u8] = b"last_hidden_state";

// Field numbers from onnx.proto3.
const MODEL_GRAPH: u32 = 7;
const GRAPH_NODE: u32 = 1;
const GRAPH_INITIALIZER: u32 = 5;
const GRAPH_INPUT: u32 = 11;
const GRAPH_OUTPUT: u32 = 12;
const GRAPH_VALUE_INFO: u32 = 13;
const NODE_INPUT: u32 = 1;
const NODE_OUTPUT: u32 = 2;
const NODE_NAME: u32 = 3;
const NODE_OP_TYPE: u32 = 4;
const TENSOR_DIMS: u32 = 1;
const TENSOR_DATA_TYPE: u32 = 2;
const TENSOR_NAME: u32 = 8;
const TENSOR_RAW_DATA: u32 = 9;
const VALUE_INFO_NAME: u32 = 1;
const VALUE_INFO_TYPE: u32 = 2;
const TYPE_TENSOR: u32 = 1;
const TENSOR_TYPE_SHAPE: u32 = 2;
const SHAPE_DIM: u32 = 1;
const DIM_PARAM: u32 = 2;
/// `TensorProto.DataType.INT64`.
const DATA_TYPE_INT64: u64 = 7;

#[derive(thiserror::Error, Debug)]
pub enum PatchError {
    #[error("malformed protobuf: {0}")]
    Malformed(&'static str),
    #[error("encoder graph is not the export this patch was written for: {0}")]
    Unexpected(String),
}

/// One field of a protobuf message, with its position in the buffer.
struct Field {
    num: u32,
    /// Offset of the tag.
    start: usize,
    /// Offset just past the field.
    end: usize,
    /// Payload: the bytes of a length-delimited field, or of the varint/fixed value.
    body: Range<usize>,
    /// Value of a varint field (0 for the other wire types).
    varint: u64,
}

fn read_varint(buf: &[u8], pos: &mut usize) -> Result<u64, PatchError> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *buf
            .get(*pos)
            .ok_or(PatchError::Malformed("varint runs past the buffer"))?;
        *pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(PatchError::Malformed("varint longer than 10 bytes"))
}

/// The fields of the message occupying `range` of `buf`.
fn fields(buf: &[u8], range: Range<usize>) -> impl Iterator<Item = Result<Field, PatchError>> + '_ {
    let mut pos = range.start;
    let end = range.end;
    std::iter::from_fn(move || {
        if pos >= end {
            return None;
        }
        Some(next_field(buf, &mut pos, end))
    })
}

fn next_field(buf: &[u8], pos: &mut usize, limit: usize) -> Result<Field, PatchError> {
    let start = *pos;
    let tag = read_varint(buf, pos)?;
    let num = (tag >> 3) as u32;
    let (body, varint) = match tag & 7 {
        0 => {
            let from = *pos;
            let value = read_varint(buf, pos)?;
            (from..*pos, value)
        }
        1 => {
            *pos += 8;
            (*pos - 8..*pos, 0)
        }
        2 => {
            let len = read_varint(buf, pos)? as usize;
            let from = *pos;
            *pos = from
                .checked_add(len)
                .ok_or(PatchError::Malformed("length overflows"))?;
            (from..*pos, 0)
        }
        5 => {
            *pos += 4;
            (*pos - 4..*pos, 0)
        }
        _ => return Err(PatchError::Malformed("unsupported wire type")),
    };
    if *pos > limit {
        return Err(PatchError::Malformed("field runs past its message"));
    }
    Ok(Field {
        num,
        start,
        end: *pos,
        body,
        varint,
    })
}

fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// A length-delimited field: tag, length, body.
fn len_field(num: u32, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() + 6);
    put_varint(&mut out, u64::from(num) << 3 | 2);
    put_varint(&mut out, body.len() as u64);
    out.extend_from_slice(body);
    out
}

fn varint_field(num: u32, value: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    put_varint(&mut out, u64::from(num) << 3);
    put_varint(&mut out, value);
    out
}

/// Copy the message in `range`, replacing the fields `replace` returns bytes
/// for. `replace` gets each field in order and returns the full encoded
/// replacement, or `None` to keep the field as it is.
fn rebuild(
    buf: &[u8],
    range: Range<usize>,
    mut replace: impl FnMut(&Field) -> Result<Option<Vec<u8>>, PatchError>,
) -> Result<Vec<u8>, PatchError> {
    let mut out = Vec::with_capacity(range.len() + 16);
    for field in fields(buf, range) {
        let field = field?;
        match replace(&field)? {
            Some(bytes) => out.extend_from_slice(&bytes),
            None => out.extend_from_slice(&buf[field.start..field.end]),
        }
    }
    Ok(out)
}

/// The string fields of a node that the patch looks at.
struct NodeInfo<'a> {
    name: &'a [u8],
    op_type: &'a [u8],
    inputs: Vec<&'a [u8]>,
}

fn parse_node(buf: &[u8], range: Range<usize>) -> Result<NodeInfo<'_>, PatchError> {
    let mut info = NodeInfo {
        name: &[],
        op_type: &[],
        inputs: Vec::new(),
    };
    for field in fields(buf, range) {
        let field = field?;
        match field.num {
            NODE_INPUT => info.inputs.push(&buf[field.body]),
            NODE_NAME => info.name = &buf[field.body],
            NODE_OP_TYPE => info.op_type = &buf[field.body],
            _ => {}
        }
    }
    Ok(info)
}

/// Name, element type and raw data location of a small tensor; `None` for a
/// large one (weights are skipped without being parsed further) or one that
/// does not store its data raw.
struct TensorHead<'a> {
    name: &'a [u8],
    data_type: u64,
    raw: Option<Range<usize>>,
}

fn parse_small_tensor(buf: &[u8], range: Range<usize>) -> Result<TensorHead<'_>, PatchError> {
    let mut head = TensorHead {
        name: &[],
        data_type: 0,
        raw: None,
    };
    for field in fields(buf, range) {
        let field = field?;
        match field.num {
            TENSOR_NAME => head.name = &buf[field.body],
            TENSOR_DATA_TYPE => head.data_type = field.varint,
            TENSOR_RAW_DATA => head.raw = Some(field.body),
            _ => {}
        }
    }
    Ok(head)
}

/// A splice of the graph body: `range` is replaced by `with` (empty = removed).
struct Edit {
    range: Range<usize>,
    with: Vec<u8>,
}

/// Declared-shape rewrite: axis `axis` of the tensor `ValueInfoProto` in
/// `range` becomes the symbolic dimension `param`.
fn relax_axis(
    buf: &[u8],
    range: Range<usize>,
    axis: usize,
    param: &str,
) -> Result<Vec<u8>, PatchError> {
    let mut found = false;
    let out = rebuild(buf, range, |type_field| {
        if type_field.num != VALUE_INFO_TYPE {
            return Ok(None);
        }
        let type_proto = rebuild(buf, type_field.body.clone(), |tensor_field| {
            if tensor_field.num != TYPE_TENSOR {
                return Ok(None);
            }
            let tensor = rebuild(buf, tensor_field.body.clone(), |shape_field| {
                if shape_field.num != TENSOR_TYPE_SHAPE {
                    return Ok(None);
                }
                let mut index = 0usize;
                let shape = rebuild(buf, shape_field.body.clone(), |dim_field| {
                    if dim_field.num != SHAPE_DIM {
                        return Ok(None);
                    }
                    let this = index;
                    index += 1;
                    if this != axis {
                        return Ok(None);
                    }
                    found = true;
                    let dim = len_field(DIM_PARAM, param.as_bytes());
                    Ok(Some(len_field(SHAPE_DIM, &dim)))
                })?;
                Ok(Some(len_field(TENSOR_TYPE_SHAPE, &shape)))
            })?;
            Ok(Some(len_field(TYPE_TENSOR, &tensor)))
        })?;
        Ok(Some(len_field(VALUE_INFO_TYPE, &type_proto)))
    })?;
    if !found {
        return Err(PatchError::Unexpected(format!(
            "declared shape has no axis {axis}"
        )));
    }
    Ok(out)
}

fn node(op_type: &str, name: &str, inputs: &[&[u8]], output: &str) -> Vec<u8> {
    let mut body = Vec::new();
    for input in inputs {
        body.extend(len_field(NODE_INPUT, input));
    }
    body.extend(len_field(NODE_OUTPUT, output.as_bytes()));
    body.extend(len_field(NODE_NAME, name.as_bytes()));
    body.extend(len_field(NODE_OP_TYPE, op_type.as_bytes()));
    len_field(GRAPH_NODE, &body)
}

/// A one-element `int64` initializer of shape `[1]`.
fn int64_vector(name: &str, value: i64) -> Vec<u8> {
    let mut body = varint_field(TENSOR_DIMS, 1);
    body.extend(varint_field(TENSOR_DATA_TYPE, DATA_TYPE_INT64));
    body.extend(len_field(TENSOR_NAME, name.as_bytes()));
    body.extend(len_field(TENSOR_RAW_DATA, &value.to_le_bytes()));
    len_field(GRAPH_INITIALIZER, &body)
}

/// Patch the encoder so `input_features` may have any length (an even number of
/// frames) and `last_hidden_state` follows it. See the module comment.
pub fn patch_encoder(model: Vec<u8>) -> Result<Vec<u8>, PatchError> {
    patch_with(model, EXPECTED_RESHAPES)
}

fn patch_with(mut model: Vec<u8>, expected_reshapes: usize) -> Result<Vec<u8>, PatchError> {
    // Walk every top-level field, not just up to the graph: the tail is copied
    // unread, so a truncated file must be caught here.
    let mut graph = None;
    for field in fields(&model, 0..model.len()) {
        let field = field?;
        if field.num == MODEL_GRAPH {
            graph = Some(field);
        }
    }
    let graph = graph.ok_or_else(|| PatchError::Unexpected("no graph in the model".into()))?;

    // Pass 1: which constants feed a Reshape's shape input, and the position-add
    // node.
    let mut reshape_shapes: HashSet<Vec<u8>> = HashSet::new();
    let mut add_node: Option<(Range<usize>, Vec<u8>)> = None;
    for field in fields(&model, graph.body.clone()) {
        let field = field?;
        if field.num != GRAPH_NODE {
            continue;
        }
        let info = parse_node(&model, field.body.clone())?;
        if info.op_type == b"Reshape" {
            if let Some(shape) = info.inputs.get(1) {
                reshape_shapes.insert(shape.to_vec());
            }
        } else if info.op_type == b"Add"
            && info.name == ADD_NODE_NAME
            && info.inputs.len() == 2
            && info.inputs[1] == POSITIONS_NAME
        {
            if add_node.is_some() {
                return Err(PatchError::Unexpected("two position-add nodes".into()));
            }
            add_node = Some((field.start..field.end, info.inputs[0].to_vec()));
        }
    }
    let (add_range, add_source) = add_node.ok_or_else(|| {
        PatchError::Unexpected("no `/Add_2` node adding `embed_positions.weight`".into())
    })?;

    // Pass 2: the shape constants with the pinned length, and the splices.
    let mut pinned: Vec<usize> = Vec::new();
    let mut edits: Vec<Edit> = Vec::new();
    let mut saw_input = false;
    let mut saw_output = false;
    for field in fields(&model, graph.body.clone()) {
        let field = field?;
        match field.num {
            GRAPH_INITIALIZER if field.body.len() <= 128 => {
                let head = parse_small_tensor(&model, field.body.clone())?;
                let Some(raw) = head.raw else { continue };
                if head.data_type == DATA_TYPE_INT64
                    && reshape_shapes.contains(head.name)
                    && raw.len() >= 16
                    && raw.len() % 8 == 0
                {
                    let value = |i: usize| {
                        i64::from_le_bytes(
                            model[raw.start + 8 * i..raw.start + 8 * i + 8]
                                .try_into()
                                .expect("8 bytes"),
                        )
                    };
                    if value(0) == -1 && value(1) == PINNED_POSITIONS {
                        pinned.push(raw.start + 8);
                    }
                }
            }
            GRAPH_VALUE_INFO => edits.push(Edit {
                range: field.start..field.end,
                with: Vec::new(),
            }),
            GRAPH_INPUT | GRAPH_OUTPUT => {
                let (want, axis, param) = if field.num == GRAPH_INPUT {
                    (INPUT_NAME, 2, "n_frames")
                } else {
                    (OUTPUT_NAME, 1, "n_ctx")
                };
                let name = fields(&model, field.body.clone())
                    .find_map(|f| match f {
                        Ok(f) if f.num == VALUE_INFO_NAME => Some(Ok(f)),
                        Ok(_) => None,
                        Err(e) => Some(Err(e)),
                    })
                    .transpose()?;
                if name.map(|f| &model[f.body]) != Some(want) {
                    return Err(PatchError::Unexpected(format!(
                        "graph {} is not `{}`",
                        if field.num == GRAPH_INPUT {
                            "input"
                        } else {
                            "output"
                        },
                        String::from_utf8_lossy(want)
                    )));
                }
                let relaxed = relax_axis(&model, field.body.clone(), axis, param)?;
                edits.push(Edit {
                    range: field.start..field.end,
                    with: len_field(field.num, &relaxed),
                });
                if field.num == GRAPH_INPUT {
                    saw_input = true;
                } else {
                    saw_output = true;
                }
            }
            _ => {}
        }
    }
    if pinned.len() != expected_reshapes {
        return Err(PatchError::Unexpected(format!(
            "{} Reshape constants pin {PINNED_POSITIONS} positions, expected {expected_reshapes}",
            pinned.len()
        )));
    }
    if !(saw_input && saw_output) {
        return Err(PatchError::Unexpected(
            "graph input or output declaration missing".into(),
        ));
    }

    // The position-add node: slice the table to the real length first. The new
    // nodes go in front of it (ONNX wants topological order); the initializers
    // can sit anywhere, so they ride along.
    let add_body = fields(&model, add_range.clone())
        .next()
        .ok_or(PatchError::Malformed("empty node"))??;
    debug_assert_eq!(add_body.num, GRAPH_NODE);
    let mut input_index = 0usize;
    let new_add = rebuild(&model, add_body.body.clone(), |f| {
        if f.num != NODE_INPUT {
            return Ok(None);
        }
        input_index += 1;
        Ok((input_index == 2).then(|| len_field(NODE_INPUT, b"pos/table")))
    })?;
    let mut splice = Vec::new();
    splice.extend(int64_vector("pos/one", 1));
    splice.extend(int64_vector("pos/zero", 0));
    splice.extend(int64_vector("pos/axis0", 0));
    splice.extend(node("Shape", "pos/Shape", &[&add_source], "pos/shape"));
    splice.extend(node(
        "Gather",
        "pos/Gather",
        &[b"pos/shape", b"pos/one"],
        "pos/len",
    ));
    splice.extend(node(
        "Slice",
        "pos/Slice",
        &[POSITIONS_NAME, b"pos/zero", b"pos/len", b"pos/axis0"],
        "pos/table",
    ));
    splice.extend(len_field(GRAPH_NODE, &new_add));
    edits.push(Edit {
        range: add_range,
        with: splice,
    });
    edits.sort_by_key(|e| e.range.start);

    // Same-size edits first, then one pass that splices the rest.
    for at in pinned {
        model[at..at + 8].copy_from_slice(&0i64.to_le_bytes());
    }
    let removed: usize = edits.iter().map(|e| e.range.len()).sum();
    let added: usize = edits.iter().map(|e| e.with.len()).sum();
    let graph_len = graph.body.len() - removed + added;

    let mut out = Vec::with_capacity((model.len() + added).saturating_sub(removed) + 16);
    out.extend_from_slice(&model[..graph.start]);
    put_varint(&mut out, u64::from(MODEL_GRAPH) << 3 | 2);
    put_varint(&mut out, graph_len as u64);
    let mut cursor = graph.body.start;
    for edit in &edits {
        if edit.range.start < cursor {
            return Err(PatchError::Malformed("overlapping edits"));
        }
        out.extend_from_slice(&model[cursor..edit.range.start]);
        out.extend_from_slice(&edit.with);
        cursor = edit.range.end;
    }
    out.extend_from_slice(&model[cursor..graph.body.end]);
    out.extend_from_slice(&model[graph.end..]);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    fn tensor(name: &str, data_type: u64, raw: &[u8]) -> Vec<u8> {
        let mut body = varint_field(TENSOR_DIMS, (raw.len() / 8) as u64);
        body.extend(varint_field(TENSOR_DATA_TYPE, data_type));
        body.extend(len_field(TENSOR_NAME, name.as_bytes()));
        body.extend(len_field(TENSOR_RAW_DATA, raw));
        len_field(GRAPH_INITIALIZER, &body)
    }

    fn i64s(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn dim_value(v: u64) -> Vec<u8> {
        len_field(SHAPE_DIM, &varint_field(1, v))
    }

    fn dim_param(p: &str) -> Vec<u8> {
        len_field(SHAPE_DIM, &len_field(DIM_PARAM, p.as_bytes()))
    }

    fn value_info(field: u32, name: &str, dims: &[Vec<u8>]) -> Vec<u8> {
        let shape: Vec<u8> = dims.concat();
        let mut tensor_type = varint_field(1, 1);
        tensor_type.extend(len_field(TENSOR_TYPE_SHAPE, &shape));
        let type_proto = len_field(TYPE_TENSOR, &tensor_type);
        let mut body = len_field(VALUE_INFO_NAME, name.as_bytes());
        body.extend(len_field(VALUE_INFO_TYPE, &type_proto));
        len_field(field, &body)
    }

    /// A miniature of the shipped encoder: the same field kinds, 2 pinned
    /// reshapes, one decoy of every way a constant can fail to qualify, and a
    /// `/Add_2`. The extra top-level fields stand in for `ir_version` etc.
    fn synthetic(with_add: bool) -> Vec<u8> {
        let mut graph = Vec::new();
        graph.extend(node("Reshape", "r0", &[b"x", b"s0"], "y0"));
        graph.extend(node("Reshape", "r1", &[b"x", b"s1"], "y1"));
        graph.extend(node("Transpose", "t", &[b"y0"], "t_out"));
        if with_add {
            graph.extend(node("Add", "/Add_2", &[b"t_out", POSITIONS_NAME], "sum"));
        }
        graph.extend(node("Foo", "foo", &[b"sum", b"decoy_nonreshape"], "z"));
        graph.extend(len_field(2, b"graph-name"));
        graph.extend(tensor("s0", 7, &i64s(&[-1, 1500, 20, 64])));
        graph.extend(tensor("s1", 7, &i64s(&[-1, 1500, 1280])));
        graph.extend(tensor("decoy_nonreshape", 7, &i64s(&[-1, 1500])));
        graph.extend(tensor("decoy_other_value", 7, &i64s(&[-1, 7, 3])));
        graph.extend(tensor("decoy_f32", 1, &i64s(&[-1, 1500])));
        graph.extend(tensor(
            std::str::from_utf8(POSITIONS_NAME).unwrap(),
            1,
            &[0u8; 32],
        ));
        graph.extend(value_info(
            GRAPH_INPUT,
            "input_features",
            &[dim_param("batch_size"), dim_value(128), dim_value(3000)],
        ));
        graph.extend(value_info(
            GRAPH_OUTPUT,
            "last_hidden_state",
            &[dim_param("batch_size"), dim_value(1500), dim_value(1280)],
        ));
        for i in 0..3 {
            graph.extend(value_info(
                GRAPH_VALUE_INFO,
                &format!("vi{i}"),
                &[dim_value(1500)],
            ));
        }
        let mut model = varint_field(1, 10);
        model.extend(len_field(2, b"producer"));
        model.extend(len_field(MODEL_GRAPH, &graph));
        model.extend(len_field(8, b"\x0a\x00\x10\x0e")); // opset_import
        model
    }

    /// The graph's top-level fields, as `(field number, payload)`.
    fn graph_fields(model: &[u8]) -> Vec<(u32, Vec<u8>)> {
        let graph = fields(model, 0..model.len())
            .map(Result::unwrap)
            .find(|f| f.num == MODEL_GRAPH)
            .unwrap();
        fields(model, graph.body)
            .map(Result::unwrap)
            .map(|f| (f.num, model[f.body].to_vec()))
            .collect()
    }

    fn node_names(model: &[u8]) -> Vec<String> {
        graph_fields(model)
            .into_iter()
            .filter(|(n, _)| *n == GRAPH_NODE)
            .map(|(_, body)| {
                String::from_utf8(parse_node(&body, 0..body.len()).unwrap().name.to_vec()).unwrap()
            })
            .collect()
    }

    fn raw_of(model: &[u8], tensor_name: &str) -> Vec<i64> {
        for (num, body) in graph_fields(model) {
            if num != GRAPH_INITIALIZER {
                continue;
            }
            let head = parse_small_tensor(&body, 0..body.len()).unwrap();
            if head.name == tensor_name.as_bytes() {
                let raw = &body[head.raw.unwrap()];
                return raw
                    .chunks(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            }
        }
        panic!("no tensor {tensor_name}");
    }

    /// Dim `axis` of the value info `field` (`input`/`output`): `Ok(value)` for
    /// a fixed size, `Err(param)` for a symbolic one.
    fn declared_dim(model: &[u8], field: u32, axis: usize) -> Result<u64, String> {
        let (_, vi) = graph_fields(model)
            .into_iter()
            .find(|(n, _)| *n == field)
            .unwrap();
        let ty = fields(&vi, 0..vi.len())
            .map(Result::unwrap)
            .find(|f| f.num == VALUE_INFO_TYPE)
            .unwrap();
        let tt = fields(&vi, ty.body).map(Result::unwrap).next().unwrap();
        let shape = fields(&vi, tt.body)
            .map(Result::unwrap)
            .find(|f| f.num == TENSOR_TYPE_SHAPE)
            .unwrap();
        let dim = fields(&vi, shape.body)
            .map(Result::unwrap)
            .nth(axis)
            .unwrap();
        let inner = fields(&vi, dim.body).map(Result::unwrap).next().unwrap();
        if inner.num == DIM_PARAM {
            Err(String::from_utf8(vi[inner.body].to_vec()).unwrap())
        } else {
            Ok(inner.varint)
        }
    }

    #[test]
    fn varints_round_trip() {
        for v in [
            0u64,
            1,
            127,
            128,
            300,
            16_383,
            16_384,
            u32::MAX as u64,
            u64::MAX,
        ] {
            let mut buf = Vec::new();
            put_varint(&mut buf, v);
            let mut pos = 0;
            assert_eq!(read_varint(&buf, &mut pos).unwrap(), v);
            assert_eq!(pos, buf.len());
        }
        assert!(read_varint(&[0x80], &mut 0).is_err(), "truncated varint");
    }

    #[test]
    fn patch_unpins_the_reshapes_and_nothing_else() {
        let before = synthetic(true);
        let after = patch_with(before.clone(), 2).unwrap();

        // The pinned constants lose the length; every other element stays.
        assert_eq!(raw_of(&after, "s0"), [-1, 0, 20, 64]);
        assert_eq!(raw_of(&after, "s1"), [-1, 0, 1280]);
        // Decoys: not feeding a Reshape, a different value, not int64.
        assert_eq!(raw_of(&after, "decoy_nonreshape"), [-1, 1500]);
        assert_eq!(raw_of(&after, "decoy_other_value"), [-1, 7, 3]);
        assert_eq!(raw_of(&after, "decoy_f32"), [-1, 1500]);

        // Fields outside the graph survive byte for byte.
        let head = varint_field(1, 10).len() + len_field(2, b"producer").len();
        assert!(after.starts_with(&before[..head]));
        assert!(after.ends_with(b"\x42\x04\x0a\x00\x10\x0e"));

        // Dropped hints; relaxed declared shapes.
        let names: Vec<u32> = graph_fields(&after).iter().map(|(n, _)| *n).collect();
        assert!(!names.contains(&GRAPH_VALUE_INFO));
        assert_eq!(
            declared_dim(&after, GRAPH_INPUT, 0),
            Err("batch_size".into())
        );
        assert_eq!(declared_dim(&after, GRAPH_INPUT, 1), Ok(128));
        assert_eq!(declared_dim(&after, GRAPH_INPUT, 2), Err("n_frames".into()));
        assert_eq!(declared_dim(&after, GRAPH_OUTPUT, 1), Err("n_ctx".into()));
        assert_eq!(declared_dim(&after, GRAPH_OUTPUT, 2), Ok(1280));
    }

    #[test]
    fn patch_slices_the_position_table_before_the_add() {
        let after = patch_with(synthetic(true), 2).unwrap();
        assert_eq!(
            node_names(&after),
            [
                "r0",
                "r1",
                "t",
                "pos/Shape",
                "pos/Gather",
                "pos/Slice",
                "/Add_2",
                "foo"
            ]
        );
        for name in ["pos/one", "pos/zero", "pos/axis0"] {
            assert!(raw_of(&after, name).len() == 1, "{name}");
        }
        assert_eq!(raw_of(&after, "pos/one"), [1]);
        let (_, add) = graph_fields(&after)
            .into_iter()
            .filter(|(n, _)| *n == GRAPH_NODE)
            .find(|(_, b)| parse_node(b, 0..b.len()).unwrap().name == b"/Add_2")
            .unwrap();
        let info = parse_node(&add, 0..add.len()).unwrap();
        assert_eq!(info.inputs, [&b"t_out"[..], b"pos/table"]);
        // The shape node reads the same tensor the add did.
        let (_, shape) = graph_fields(&after)
            .into_iter()
            .filter(|(n, _)| *n == GRAPH_NODE)
            .find(|(_, b)| parse_node(b, 0..b.len()).unwrap().name == b"pos/Shape")
            .unwrap();
        assert_eq!(
            parse_node(&shape, 0..shape.len()).unwrap().inputs,
            [&b"t_out"[..]]
        );
    }

    #[test]
    fn patched_model_is_itself_walkable_end_to_end() {
        // Every length prefix must agree with the bytes: walking the whole
        // output must consume it exactly.
        let after = patch_with(synthetic(true), 2).unwrap();
        let mut count = 0;
        for f in fields(&after, 0..after.len()) {
            f.unwrap();
            count += 1;
        }
        assert_eq!(count, 4);
        // And a second patch finds nothing left to unpin.
        assert!(matches!(
            patch_with(after, 2),
            Err(PatchError::Unexpected(_))
        ));
    }

    #[test]
    fn patch_refuses_an_unexpected_graph() {
        // Wrong count of pinned reshapes.
        assert!(matches!(
            patch_with(synthetic(true), 3),
            Err(PatchError::Unexpected(m)) if m.contains("expected 3")
        ));
        // No position add.
        assert!(matches!(
            patch_with(synthetic(false), 2),
            Err(PatchError::Unexpected(m)) if m.contains("/Add_2")
        ));
        // Not a protobuf.
        assert!(patch_with(vec![0xff; 12], 2).is_err());
        assert!(patch_with(Vec::new(), 2).is_err());
        // Truncated message.
        let mut cut = synthetic(true);
        cut.truncate(cut.len() - 3);
        assert!(patch_with(cut, 2).is_err());
    }

    /// Deterministic pseudo-random features in the encoder's input range.
    fn features(frames: usize) -> ndarray17::Array3<f32> {
        let mut state = 0x2545_F491u32;
        ndarray17::Array3::from_shape_fn((1, 128, frames), |_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state as f32 / u32::MAX as f32) * 1.4 - 0.6
        })
    }

    /// The real export, when downloaded: the patched encoder must reproduce the
    /// original at the full window and accept shorter ones. Skipped without
    /// the weights, like the tokenizer tests.
    #[test]
    fn patched_real_encoder_matches_the_original_and_accepts_short_windows() {
        let path = crate::model_store::catalog_dir(crate::config::WHISPER_DEFAULT_SPEC)
            .join("encoder_model_int8.onnx");
        if !path.exists() {
            eprintln!("skipping: {} not downloaded", path.display());
            return;
        }
        let bytes = std::fs::read(&path).unwrap();
        let patched = patch_encoder(bytes.clone()).expect("shipped export must patch");
        let mut original = super::super::engine::session_builder(4, Device::Cpu)
            .unwrap()
            .commit_from_memory(&bytes)
            .unwrap();
        let mut dynamic = super::super::engine::session_builder(4, Device::Cpu)
            .unwrap()
            .commit_from_memory(&patched)
            .unwrap();
        let run = |s: &mut ort::session::Session, f: &ndarray17::Array3<f32>| {
            let out = s
                .run(ort::inputs![
                    "input_features" => ort::value::TensorRef::from_array_view(f.view()).unwrap()
                ])
                .unwrap();
            out["last_hidden_state"]
                .try_extract_array::<f32>()
                .unwrap()
                .to_owned()
        };
        let full = features(3000);
        let want = run(&mut original, &full);
        let got = run(&mut dynamic, &full);
        assert_eq!(got.shape(), [1, 1500, 1280]);
        let worst = want
            .iter()
            .zip(got.iter())
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(worst < 1e-3, "patched output differs by {worst}");

        let short = run(&mut dynamic, &features(1000));
        assert_eq!(short.shape(), [1, 500, 1280]);
        assert!(short.iter().all(|v| v.is_finite()));
    }
}
