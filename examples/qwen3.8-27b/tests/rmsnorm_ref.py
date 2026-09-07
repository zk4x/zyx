# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for rmsnorm_kernel.

Gated RMSNorm: out[t, h*VD + d] = silu(z[t, h*VD + d]) * nw[d] * c[h, t, d] / sqrt(mean(c^2) + eps)

Random F32 core [VH, S, VD] and zp [S, VAL_DIM]. Real nw from blk.0.ssm_norm.weight.

Run from this directory: python3.12 rmsnorm_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
VH = 48
VD = 128
VAL_DIM = 6144  # VH * VD
EPS = 1e-6


def silu(x):
    return x / (1.0 + math.exp(-x))


# Real nw from GGUF: blk.0.ssm_norm.weight shape [128] = VD
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.ssm_norm.weight":
        nw = torch.from_numpy(np.array(t.data).astype(np.float32).copy())  # [VD]
        break
assert nw.shape == (VD,)

# Random F32 inputs
core = torch.randn(VH, S, VD, dtype=torch.float32)  # [VH, S, VD]
zp = torch.randn(S, VAL_DIM, dtype=torch.float32)    # [S, VAL_DIM]

# Reference: for each (t, h, d), compute mean over d of c^2, then scale, gate, weight.
out = torch.zeros(S, VAL_DIM, dtype=torch.float32)
for t in range(S):
    for h in range(VH):
        # mean of c^2 over d
        c = core[h, t, :]
        mean = (c * c).mean()
        denom = mean + EPS
        scale = 1.0 / math.sqrt(denom)
        for d in range(VD):
            n = c[d] * scale
            w = nw[d]
            z = zp[t, h * VD + d]
            gz = silu(z)
            y = n * w * gz
            out[t, h * VD + d] = y

save_file(
    {"core": core, "zp": zp, "nw": nw, "output": out},
    "../../data/qwen3_rmsnorm.safetensors",
)
print(f"wrote ../../data/qwen3_rmsnorm.safetensors core {tuple(core.shape)} zp {tuple(zp.shape)} nw {tuple(nw.shape)} output {tuple(out.shape)}")
