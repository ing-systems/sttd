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
    save_file(out, str(out_path))
    return {"u8": n_u8, "i8": n_i8, "fp": n_fp, "tensors": len(out)}


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
    if args.check_only:
        print("check-only: no files written")
    else:
        print("done. Retry with `--parakeet-engine burn`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
