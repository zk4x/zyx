# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for conv_silu_kernel.

Depthwise causal conv1d (kernel 4, left pad 3) + SiLU.
out [M_PAD, CONV_DIM] = silu( sum_k convw[c, k] * inp[max(0, t+k-3), c] )

Random F32 input (first S=6 rows are valid, rest is zero — kernel uses S=6
via in-kernel check but reference just uses the full [M_PAD, CONV_DIM] input).
Real convw from blk.0.ssm_conv1d.weight.

Run from this directory: python3.12 conv_silu_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
S = 6
CONV_DIM = 10240
CK = 4


def silu(x):
    return x / (1.0 + math.exp(-x))


# Real convw from GGUF: [4, 10240] in metadata, [10240, 4] dequantized
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.ssm_conv1d.weight":
        w_deq = gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        convw = torch.from_numpy(w_deq).contiguous()  # [CONV_DIM, CK]
        break
assert convw.shape == (CONV_DIM, CK)

# Random F32 input [M_PAD, CONV_DIM] (kernel uses S=6 implicitly via clamp,
# reference just uses the full padded input).
input = torch.zeros(M_PAD, CONV_DIM, dtype=torch.float32)
input[:S] = torch.randn(S, CONV_DIM, dtype=torch.float32)

# Reference: depthwise causal conv (kernel 4, left pad 3) + SiLU.
# out[t, c] = silu( sum_k convw[c, k] * inp[max(0, t+k-3), c] )
out = torch.zeros(M_PAD, CONV_DIM, dtype=torch.float32)
for t in range(M_PAD):
    for c in range(CONV_DIM):
        acc = 0.0
        for k in range(CK):
            src_t = t + k - (CK - 1)  # = t + k - 3
            if src_t >= 0:
                acc += convw[c, k] * input[src_t, c]
        out[t, c] = silu(acc)

save_file(
    {"input": input, "convw": convw, "output": out},
    "../../data/qwen3_conv_silu.safetensors",
)
print(f"wrote ../../data/qwen3_conv_silu.safetensors input {tuple(input.shape)} convw {tuple(convw.shape)} output {tuple(out.shape)}")
