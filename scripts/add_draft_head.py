"""Copies an LFM3 GGUF that has output_sketch.weight and adds a speculative-decoding draft head.

The draft head comes from the MLX training code (safetensors): fc [n_embd, 2*n_embd], one SwiGLU block
(blocks.0.norm, gate, up, down) and a final RMSNorm. llama.cpp with apple5-lfm3.patch runs it in the same graph
as the verify step when the context outputs nextn embeddings. Matrices are stored as Q8_0 (or Q4_0), norms as F32.
Needs numpy and the gguf-py of a llama.cpp checkout ($LLAMA_CPP_DIR, default llama.cpp/).

Usage: python3 add_draft_head.py in.gguf draft.safetensors out.gguf [q8_0|q4_0]
"""

import json
import os
import struct
import sys
import types
from pathlib import Path

# gguf/__init__.py imports yaml for model card metadata, which this script does not use
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
# gguf-py of the llama.cpp checkout: $LLAMA_CPP_DIR, else llama.cpp/ in this repository (scripts/setup.sh)
LLAMA_CPP_DIR = Path(
    os.environ.get(
        "LLAMA_CPP_DIR", Path(__file__).resolve().parent.parent / "llama.cpp"
    )
)
sys.path.insert(0, str(LLAMA_CPP_DIR / "gguf-py"))

import gguf
import numpy as np

# GGUF tensor name: (safetensors name, quantize)
DRAFT = {
    "draft_fc.weight": ("fc.weight", True),
    "draft_ffn_norm.weight": ("blocks.0.norm.weight", False),
    "draft_ffn_gate.weight": ("blocks.0.gate.weight", True),
    "draft_ffn_up.weight": ("blocks.0.up.weight", True),
    "draft_ffn_down.weight": ("blocks.0.down.weight", True),
    "draft_norm.weight": ("norm.weight", False),
}


def read_safetensors(path: str) -> dict[str, np.ndarray]:
    """The float32 tensors of a safetensors file."""
    raw = Path(path).read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8 : 8 + n])
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if info["dtype"] != "F32":
            sys.exit(f"{name}: expected F32, got {info['dtype']}")
        a, b = info["data_offsets"]
        if not 0 <= a <= b <= len(raw) - 8 - n:
            sys.exit(f"{name}: data offsets outside the file")
        out[name] = np.frombuffer(raw[8 + n + a : 8 + n + b], np.float32).reshape(
            info["shape"]
        )
    return out


def main() -> None:
    if len(sys.argv) not in (4, 5):
        sys.exit(__doc__)
    src, weights, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    if Path(src).resolve() == Path(dst).resolve():
        sys.exit("out.gguf must differ from in.gguf")
    qname = sys.argv[4] if len(sys.argv) > 4 else "q8_0"
    qtype = {
        "q8_0": gguf.GGMLQuantizationType.Q8_0,
        "q4_0": gguf.GGMLQuantizationType.Q4_0,
    }.get(qname)
    if qtype is None:
        sys.exit(f"unknown type {qname}: use q8_0 or q4_0")
    reader = gguf.GGUFReader(src)
    if not any(t.name == "output_sketch.weight" for t in reader.tensors):
        sys.exit("the input needs output_sketch.weight (scripts/add_output_sketch.py)")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith(
            "GGUF."
        ):
            continue
        vt = field.types[0]
        sub = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=sub)

    st = read_safetensors(weights)
    draft = []
    for name, (key, quant) in DRAFT.items():
        w = np.ascontiguousarray(st[key])
        if quant:
            q = gguf.quants.quantize(w, qtype)
            draft.append((name, q, qtype))
        else:
            draft.append((name, w, gguf.GGMLQuantizationType.F32))
        print(f"{name}: {list(w.shape)} {qname if quant else 'f32'}")

    for t in reader.tensors:
        writer.add_tensor_info(
            t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type
        )
    for name, data, ttype in draft:
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ttype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for _, data, _ in draft:
        writer.write_tensor_data(data)
    writer.close()


if __name__ == "__main__":
    main()
