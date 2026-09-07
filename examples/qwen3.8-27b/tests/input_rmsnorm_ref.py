# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for input_rmsnorm_kernel.

out [S, HIDDEN] = x * rsqrt(mean(x^2) + eps) * weight.
Random F32 inputs; real weight from blk.0.attn_norm.weight.

Run from this directory: python3.12 input_rmsnorm_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
HIDDEN = 5120
EPS = 1e-6

# Real weight from blk.0.attn_norm.weight (F32, shape [HIDDEN])
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.attn_norm.weight":
        w = torch.from_numpy(np.array(t.data).astype(np.float32).copy())
        break
assert w.shape == (HIDDEN,)

# Random F32 input
x = torch.randn(S, HIDDEN, dtype=torch.float32)

# Reference
ss = (x * x).sum(dim=-1, keepdim=True)  # [S, 1]
mean = ss / HIDDEN
denom = mean + EPS
scale = 1.0 / torch.sqrt(denom)
out = x * scale * w

save_file(
    {"x": x, "w": w, "output": out},
    "../../data/qwen3_input_rmsnorm.safetensors",
)
print(f"wrote ../../data/qwen3_input_rmsnorm.safetensors x {tuple(x.shape)} w {tuple(w.shape)} output {tuple(out.shape)}")
