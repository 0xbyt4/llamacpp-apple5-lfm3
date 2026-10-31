"""Copies an LFM3 GGUF and adds output_sketch.weight, a q1_0 copy of the tied output matrix (token_embd).

llama.cpp with apple5-lfm3.patch uses the sketch when the file has it: for each of up to 8 output rows it ranks the
vocabulary with the sketch and computes exact logits only for the best candidates and the most common token ids.
q1_0 as in quantize_row_q1_0_ref: per 128 values, d = mean |w| (f16) and bit j set when w_j >= 0.
The source token_embd must be Q8_0.
Needs numpy and the gguf-py of a llama.cpp checkout ($LLAMA_CPP_DIR, default llama.cpp/).

Usage: python3 add_output_sketch.py in.gguf out.gguf
"""

import os
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


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    if Path(src).resolve() == Path(dst).resolve():
        sys.exit("out.gguf must differ from in.gguf")
    reader = gguf.GGUFReader(src)
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

    emb = next(t for t in reader.tensors if t.name == "token_embd.weight")
    if emb.tensor_type != gguf.GGMLQuantizationType.Q8_0:
        sys.exit("token_embd.weight must be Q8_0")
    n_embd = int(emb.shape[0])
    raw = np.asarray(emb.data).reshape(-1, 34)
    d8 = raw[:, :2].copy().view(np.float16).astype(np.float32)
    w = (raw[:, 2:].view(np.int8).astype(np.float32) * d8).reshape(-1, n_embd)

    blocks = w.reshape(-1, 128)
    d = np.abs(blocks).mean(axis=1).astype(np.float16)
    bits = np.packbits(blocks >= 0, axis=1, bitorder="little")
    sketch = np.concatenate([d.view(np.uint8).reshape(-1, 2), bits], axis=1).reshape(
        w.shape[0], -1
    )
    print(
        f"output_sketch.weight: {w.shape[0]} x {n_embd}, {sketch.nbytes / 1e6:.1f} MB"
    )

    for t in reader.tensors:
        writer.add_tensor_info(
            t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type
        )
    writer.add_tensor_info(
        "output_sketch.weight",
        sketch.shape,
        np.uint8,
        sketch.nbytes,
        gguf.GGMLQuantizationType.Q1_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.write_tensor_data(sketch)
    writer.close()


if __name__ == "__main__":
    main()
