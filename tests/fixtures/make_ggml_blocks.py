"""Write tests/fixtures/ggml_blocks.gguf: random blocks of every ggml type dlm
decodes, each next to llama.cpp's own f32 dequantization of it.

`tests/ggml_blocks.rs` checks dlm's decoders against that file weight for
weight. The reference is gguf-py, llama.cpp's Python port of its
`dequantize_row_*` functions, so rerun this only to add a type:

    python -m venv env && env/bin/pip install gguf
    env/bin/python tests/fixtures/make_ggml_blocks.py tests/fixtures/ggml_blocks.gguf
"""

import sys

import numpy as np
import gguf
from gguf import GGMLQuantizationType as T

# Each type with the byte offsets of its f16 scales. Every other byte is
# random; the scales are set to small finite values so no weight is inf or NaN.
TYPES = {
    T.Q4_0: [0], T.Q4_1: [0, 2], T.Q5_0: [0], T.Q5_1: [0, 2], T.Q8_0: [0],
    T.Q2_K: [80, 82], T.Q3_K: [108], T.Q4_K: [0, 2], T.Q5_K: [0, 2], T.Q6_K: [208],
    T.IQ2_XXS: [0], T.IQ2_XS: [0], T.IQ2_S: [0], T.IQ3_XXS: [0], T.IQ3_S: [0],
    T.IQ4_NL: [0], T.IQ4_XS: [0],
}
ROWS, BLOCKS_PER_ROW = 2, 2

rng = np.random.default_rng(7)
w = gguf.GGUFWriter(sys.argv[1], "dlm-fixture")
for qtype, halves in TYPES.items():
    block_elems, block_bytes = gguf.GGML_QUANT_SIZES[qtype]
    n_blocks = ROWS * BLOCKS_PER_ROW
    blocks = rng.integers(0, 256, size=(n_blocks, block_bytes), dtype=np.uint8)
    for at in halves:
        d = rng.uniform(-0.05, 0.05, size=n_blocks).astype(np.float16)
        blocks[:, at:at + 2] = d.view(np.uint8).reshape(n_blocks, 2)
    raw = blocks.reshape(ROWS, -1)
    want = gguf.quants.dequantize(raw, qtype).astype(np.float32)
    assert want.shape == (ROWS, BLOCKS_PER_ROW * block_elems) and np.isfinite(want).all()
    w.add_tensor(qtype.name, raw, raw_dtype=qtype)
    w.add_tensor(qtype.name + ".f32", want)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
