# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for rope_kernel(6, 24, 256, 64).

Qwen3.5-27B full-attn rope: out [H*S, D] = rope(x [H*S, D], cos [S, rot_dim], sin [S, rot_dim])
where rope uses rotate_half on the first rot_dim and passes through the rest.

Run from this directory: python3.12 rope_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
HEADS = 24
HEAD_DIM = 256
ROT_DIM = 64


def rotate_half(x):
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:d]
    return torch.cat((-x2, x1), dim=-1)


x = torch.randn(HEADS * S, HEAD_DIM, dtype=torch.float32)
cos = torch.randn(S, ROT_DIM, dtype=torch.float32)
sin = torch.randn(S, ROT_DIM, dtype=torch.float32)

out = torch.zeros(HEADS * S, HEAD_DIM, dtype=torch.float32)
for hs in range(HEADS * S):
    s = hs % S
    x_rot = x[hs, :ROT_DIM]
    cos_s = cos[s, :]
    sin_s = sin[s, :]
    y_rot = x_rot * cos_s + rotate_half(x_rot.unsqueeze(0)).squeeze(0) * sin_s
    out[hs, :ROT_DIM] = y_rot
    out[hs, ROT_DIM:] = x[hs, ROT_DIM:]

save_file(
    {"x": x, "cos": cos, "sin": sin, "output": out},
    "../../data/qwen3_rope.safetensors",
)
print(f"wrote ../../data/qwen3_rope.safetensors x {tuple(x.shape)} cos {tuple(cos.shape)} sin {tuple(sin.shape)} output {tuple(out.shape)}")
