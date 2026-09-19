# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

"""Regenerate the dequant_q4k_tt / dequant_q4k_tt_asm golden fixture.

Golden = first 4 tiles (tilized-flat F32) of blk.1.attn_qkv.weight, dequantized
by gguf-py itself (dequantize_row_q4_K). Tilized layout: tile t covers row-block
t / tiles_per_row, col-block t % tiles_per_row; a tile is row-major 32x32.

Writes examples/data/qwen3_attn_qkv_q4k_golden_first4tiles.safetensors, key "x".
"""

import numpy as np
from safetensors.numpy import save_file
import gguf

GGUF = "/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf"
OUT = "/home/x/Dev/rust/zyx/examples/data/qwen3_attn_qkv_q4k_golden_first4tiles.safetensors"
NAME = "blk.1.attn_qkv.weight"
TILES = 4

r = gguf.GGUFReader(GGUF)
t = next(t for t in r.tensors if t.name == NAME)
w = gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
assert w.shape == (10240, 5120), w.shape

tiles_per_row = w.shape[1] // 32
out = np.empty(TILES * 1024, dtype=np.float32)
for tt in range(TILES):
    r0 = (tt // tiles_per_row) * 32
    c0 = (tt % tiles_per_row) * 32
    out[tt * 1024 : (tt + 1) * 1024] = w[r0 : r0 + 32, c0 : c0 + 32].reshape(-1)

save_file({"x": out}, OUT)
print("wrote", OUT, out.shape, "first 6:", out[:6])
