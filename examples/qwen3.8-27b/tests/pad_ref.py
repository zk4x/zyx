# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for pad_kernel.

`pad_kernel(s, m, d)`: in [s, d] F32, out [m, d] F16, zero-pad rows s..m.

Two uses in qwen3.8-27b: input (d=HIDDEN=5120) and normed (d=VAL_DIM=6144).
Generates random F32 inputs (matching kernel input dtype), pads with zeros,
saves input (F32) and output (F16) to safetensors.

Run from this directory: python3.12 pad_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
M_PAD = 16
HIDDEN = 5120
VAL_DIM = 6144


def make_padded(s: int, m: int, d: int):
    x = torch.randn(s, d)  # F32 to match kernel input
    out = torch.zeros(m, d, dtype=torch.float16)
    out[:s].copy_(x.half())
    return x, out


# pad_input
inp_i, out_i = make_padded(S, M_PAD, HIDDEN)
save_file(
    {"input": inp_i, "output": out_i},
    "../../data/qwen3_pad_input.safetensors",
)
print(f"wrote ../../data/qwen3_pad_input.safetensors shape {tuple(out_i.shape)}")

# pad_normed
inp_n, out_n = make_padded(S, M_PAD, VAL_DIM)
save_file(
    {"input": inp_n, "output": out_n},
    "../../data/qwen3_pad_normed.safetensors",
)
print(f"wrote ../../data/qwen3_pad_normed.safetensors shape {tuple(out_n.shape)}")
