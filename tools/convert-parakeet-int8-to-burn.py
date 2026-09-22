#!/usr/bin/env python3
"""Convert int8 Parakeet-TDT ONNX to Burn safetensors (int8-now path).

The encoder/decoder_joint int8 graphs use *dynamic* quantization
(DynamicQuantizeLinear + MatMulInteger + ConvInteger + DynamicQuantizeLSTM),
which `burn-import` cannot ingest. This tool extracts every static int8
weight + its scale (+ bias/fp extras) into safetensors that
`src/parakeet_burn/weights.rs` loads:

  models/parakeet-tdt-0.6b-v3-int8/encoder-burn.safetensors
  models/parakeet-tdt-0.6b-v3-int8/decoder_joint-burn.safetensors

int8 tensors are stored as I8 + `<name>.scale` (F32, per-row or scalar);
float tensors pass through as F32. No requantization, no calibration.

Usage:
  python3 tools/convert-parakeet-int8-to-burn.py --model-dir models/parakeet-tdt-0.6b-v3-int8
  python3 tools/convert-parakeet-int8-to-burn.py --model-dir models/parakeet-tdt-0.6b-v3-int8 --check-only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _import_onnx():
    try:
        import onnx
        import numpy as np
        from onnx import numpy_helper
    except ImportError as e:
        print(
            "missing dependency: pip install onnx numpy safetensors",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    try:
        from safetensors.numpy import save_file
    except ImportError as e:
        print(
            "missing dependency: pip install safetensors",
            file=sys.stderr,
        )
        raise SystemExit(2) from e
    return onnx, np, numpy_helper, save_file


def convert_one(onnx_path: Path, out_path: Path) -> dict:
    onnx, np, numpy_helper, save_file = _import_onnx()
    model = onnx.load(str(onnx_path), load_external_data=True)
    raw: dict[str, "np.ndarray"] = {}
    for init in model.graph.initializer:
        raw[init.name] = np.ascontiguousarray(numpy_helper.to_array(init))

    out: dict[str, "np.ndarray"] = {}
    n_u8 = n_i8 = n_fp = 0

    def keep_fp(name: str) -> None:
        out[name] = np.ascontiguousarray(raw[name].astype(np.float32))
        return None

    # Static-quantized weights come in triples:
    #   <base>_quantized (U8 or I8) + <base>_scale (F32) + <base>_zero_point
    # rewritten as <base> (+ .scale / .zero_point), which is what
    # `src/parakeet_burn/weights.rs` loads (see its doc comment).
    for name, arr in raw.items():
        if not name.endswith("_quantized"):
            continue
        base = name[: -len("_quantized")]
        scale = raw.get(base + "_scale")
        zp = raw.get(base + "_zero_point")
        if scale is None or zp is None:
            print(f"  skip {name}: no scale/zero_point pair")
            continue
        if arr.dtype == np.uint8:
            out[base] = np.ascontiguousarray(arr)
            out[base + ".scale"] = np.ascontiguousarray(scale.astype(np.float32))
            out[base + ".zero_point"] = np.ascontiguousarray(zp.astype(np.uint8))
            n_u8 += 1
        elif arr.dtype == np.int8:
            out[base] = np.ascontiguousarray(arr)
            out[base + ".scale"] = np.ascontiguousarray(scale.astype(np.float32))
            out[base + ".zero_point"] = np.ascontiguousarray(zp.astype(np.int8))
            n_i8 += 1
        else:
            print(f"  skip {name}: dtype {arr.dtype}")

    # Plain float tensors (biases, LSTM hidden states): keep as F32.
    for name, arr in raw.items():
        if name.endswith(("_quantized", "_scale", "_zero_point")):
            continue
        if arr.dtype in (np.float32, np.float16):
            keep_fp(name)
            n_fp += 1
        elif arr.dtype == np.int64 and arr.size <= 8:
            continue  # shape constants, not weights
        else:
            print(f"  skip {name}: dtype {arr.dtype} shape {arr.shape}")

    # Frozen large F32 constants baked into Constant nodes (not
    # initializers): the encoder's relative-position table
    # (Constant_1723, (1, 9999, 1024)). Without this the Rust side would
    # have to regenerate NeMo's sinusoid convention; the exact table
    # removes a whole class of parity risk. Threshold keeps tiny scalars
    # (scales-as-constants, pads) out.
    n_const = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            arr = np.ascontiguousarray(numpy_helper.to_array(attr.t))
            if arr.dtype != np.float32 or arr.size <= 10_000:
                continue
            if arr.shape == (1, 9999, 1024):
                out["pos_enc.table"] = arr.reshape(9999, 1024)
            else:
                out[f"const/{node.name}"] = arr
            n_const += 1
    if n_const:
        print(f"  kept {n_const} large F32 constant(s)")
    save_file(out, str(out_path))
    return {"u8": n_u8, "i8": n_i8, "fp": n_fp, "tensors": len(out)}


def _attr(node, name, default=None):
    for a in node.attribute:
        if a.name == name:
            if a.type == 7:  # INTS
                return list(a.ints)
            if a.type == 2:  # INT
                return a.i
    return default


def emit_encoder_manifest(onnx_path: Path, out_path: Path) -> dict:
    """Map every quantized encoder node to its (layer, role).

    Node names carry NeMo role paths (`/layers.N/self_attn/linear_q/...`,
    `/pre_encode/conv/conv.0/...`); weight initializers are anonymous
    (`onnx::..._quantized`). The manifest joins them so the Rust loader
    (`src/parakeet_burn/encoder.rs`) can fetch each role's int8 weight by
    name. Conv entries also carry kernel/stride/groups/pads.
    """
    onnx, np, numpy_helper, _ = _import_onnx()
    import json
    import re

    model = onnx.load(str(onnx_path), load_external_data=False)
    shapes = {}
    init_names = set()
    for init in model.graph.initializer:
        init_names.add(init.name)
        try:
            shapes[init.name] = list(numpy_helper.to_array(init, base_dir=str(onnx_path.parent)).shape)
        except Exception:
            shapes[init.name] = None

    entries = []
    counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type not in ("MatMulInteger", "ConvInteger"):
            continue
        m = re.match(r"^/(layers\.\d+/.+|pre_encode/.+)/[^/]+$", node.name)
        if not m:
            print(f"  manifest: unmapped quant node {node.name}")
            continue
        role = m.group(1)
        # The activation input is *also* `_quantized`-suffixed: disambiguate
        # by initializer membership (activations are graph edges, weights
        # are initializers).
        weight = next(
            (i for i in node.input if i.endswith("_quantized") and i in init_names),
            None,
        )
        base = weight[: -len("_quantized")] if weight else None
        entry: dict = {
            "role": role,
            "node": node.name,
            "kind": "matmul" if node.op_type == "MatMulInteger" else "conv",
            "weight": base,
            # shapes[] is keyed by full initializer name (with suffix).
            "weight_shape": shapes.get(weight) if weight else None,
        }
        if node.op_type == "ConvInteger":
            entry["conv"] = {
                "kernel": _attr(node, "kernel_shape"),
                "strides": _attr(node, "strides"),
                "pads": _attr(node, "pads"),
                "group": _attr(node, "group", 1),
                "dilations": _attr(node, "dilations"),
            }
        entries.append(entry)
        counts[role.split("/")[1] if role.startswith("layers.") else "pre_encode"] = (
            counts.get(role.split("/")[1] if role.startswith("layers.") else "pre_encode", 0) + 1
        )

    # Per-layer norm + pos-bias names (already named in safetensors).
    norms: dict[str, list[str]] = {}
    for init in model.graph.initializer:
        m = re.match(r"^(layers\.\d+\.norm_\w+)\.(weight|bias)$", init.name)
        if m:
            norms.setdefault(m.group(1), []).append(init.name)
    pos_bias = sorted(n for n in shapes if re.match(r"^layers\.\d+\.self_attn\.pos_bias_[uv]$", n))

    # Depthwise-conv biases are anonymous F32 initializers (onnx::Conv_*):
    # trace each layer's *bias_add through its Reshape to the initializer.
    init_names = set(shapes)
    out2node = {}
    for node in model.graph.node:
        for o in node.output:
            out2node[o] = node
    dw_bias: dict[str, str] = {}
    for node in model.graph.node:
        m = re.match(r"^/layers\.(\d+)/conv/depthwise_conv/.*bias_add$", node.name)
        if not m:
            continue
        layer = m.group(1)
        for inp in node.input:
            cur = inp
            seen = 0
            while cur in out2node and out2node[cur].op_type == "Reshape" and seen < 3:
                data_in = out2node[cur].input[0]
                cur = data_in
                seen += 1
            if cur in init_names:
                dw_bias[f"layers.{layer}/conv/depthwise_bias"] = cur
    if len(dw_bias) != len(set(re.match(r"^layers\.(\d+)", e["role"]).group(1) for e in entries if e["role"].startswith("layers."))):
        print(f"  manifest: dw_bias mapped for {len(dw_bias)} layers (expected 24)")

    manifest = {
        "model": onnx_path.name,
        "quant_ops": entries,
        "norms": norms,
        "pos_bias": pos_bias,
        "dw_bias": dw_bias,
    }
    out_path.write_text(json.dumps(manifest, indent=1) + "\n")
    return {"roles": len(entries), "norm_groups": len(norms), "pos_bias": len(pos_bias)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="e.g. models/parakeet-tdt-0.6b-v3-int8")
    ap.add_argument("--check-only", action="store_true", help="inspect, don't write")
    args = ap.parse_args()

    d = Path(args.model_dir)
    pairs = [
        (d / "encoder-model.int8.onnx", d / "encoder-burn.safetensors"),
        (d / "decoder_joint-model.int8.onnx", d / "decoder_joint-burn.safetensors"),
    ]
    for src, _ in pairs:
        if not src.is_file():
            print(f"missing {src}", file=sys.stderr)
            return 1

    onnx, np, numpy_helper, _ = _import_onnx()
    from collections import Counter

    for src, dst in pairs:
        model = onnx.load(str(src), load_external_data=False)
        ops = Counter(n.op_type for n in model.graph.node)
        print(f"== {src.name}: {len(model.graph.node)} nodes")
        for op, c in ops.most_common(12):
            print(f"   {op:28s} {c}")
        n_init = len(model.graph.initializer)
        print(f"   initializers (inline): {n_init}")
        if not args.check_only:
            stats = convert_one(src, dst)
            print(
                f"   wrote {dst} ({stats['tensors']} tensors: "
                f"{stats['u8']} u8, {stats['i8']} i8, {stats['fp']} fp)"
            )
            if "encoder" in src.name:
                manifest_path = dst.with_name("encoder-manifest.json")
                mstats = emit_encoder_manifest(src, manifest_path)
                print(
                    f"   wrote {manifest_path} ({mstats['roles']} roles, "
                    f"{mstats['norm_groups']} norm groups, {mstats['pos_bias']} pos-bias)"
                )
    if args.check_only:
        print("check-only: no files written")
    else:
        print("done. Retry with `--parakeet-engine burn`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
