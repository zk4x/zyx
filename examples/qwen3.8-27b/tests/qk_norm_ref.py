# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for qk_norm_kernel.

Per-head norm on q [H, S, KD] and k [KV, S, KD]:
  q[h, t, :] = q[h, t, :] * rsqrt(mean(q[h, t, :]^2) + eps) * qw
  k[kv, t, :] = k[kv, t, :] * rsqrt(mean(k[kv, t, :]^2) + eps) * kw

Random F32 inputs.

Run from this directory: python3.12 qk_norm_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
H = 24
KV = 4
KD = 128  # head_dim for full attention
EPS = 1e-6

q = torch.randn(H, S, KD, dtype=torch.float32)
k = torch.randn(KV, S, KD, dtype=torch.float32)
qw = torch.randn(KD, dtype=torch.float32)
kw = torch.randn(KD, dtype=torch.float32)

ss_q = (q * q).sum(dim=-1, keepdim=True)
scale_q = 1.0 / torch.sqrt(ss_q / KD + EPS)
q_out = q * scale_q * qw

ss_k = (k * k).sum(dim=-1, keepdim=True)
scale_k = 1.0 / torch.sqrt(ss_k / KD + EPS)
k_out = k * scale_k * kw

save_file(
    {"q": q, "k": k, "qw": qw, "kw": kw, "q_out": q_out, "k_out": k_out},
    "../../data/qwen3_qk_norm.safetensors",
)
print(f"wrote ../../data/qwen3_qk_norm.safetensors q {tuple(q.shape)} k {tuple(k.shape)} q_out {tuple(q_out.shape)} k_out {tuple(k_out.shape)}")
