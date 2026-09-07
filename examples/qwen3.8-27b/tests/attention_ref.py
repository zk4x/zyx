# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for attention_kernel.

Causal GQA attention with sigmoid gate:
out[s, h*d] = sigmoid(gate[s, h*d]) * sum_{s2<=s} softmax(QK^T/sqrt(d))[s, s2] * V[s2]
With KV grouping: head h uses KV head h/(h/kv).

Random F32 q [H, S, D], k/v [KV, S, D], gate [S, H*D].

Run from this directory: python3.12 attention_ref.py
"""

import math
import torch
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
H = 24
KV = 4
D = 256

# Random F32 inputs
q = torch.randn(H, S, D, dtype=torch.float32)
k = torch.randn(KV, S, D, dtype=torch.float32)
v = torch.randn(KV, S, D, dtype=torch.float32)
gate = torch.randn(S, H * D, dtype=torch.float32)

scale = 1.0 / math.sqrt(D)
h_per_kv = H // KV

# Reference: for each head h, attend over s2<=s
out = torch.zeros(S, H * D, dtype=torch.float32)
for h in range(H):
    kv_h = h // h_per_kv
    for s in range(S):
        # scores [s2] for s2 in 0..s+1
        scores = []
        for s2 in range(s + 1):
            # dot = sum_d q[h, s, d] * k[kv_h, s2, d]
            dot = (q[h, s, :] * k[kv_h, s2, :]).sum().item()
            scores.append(dot * scale)
        # softmax
        max_s = max(scores)
        exps = [math.exp(s - max_s) for s in scores]
        sum_e = sum(exps)
        probs = [e / sum_e for e in exps]
        # out[s, h*d..h*d+D] = sum_{s2<=s} probs[s2] * v[kv_h, s2, :]
        for s2 in range(s + 1):
            out[s, h * D:(h + 1) * D] += probs[s2] * v[kv_h, s2, :]
        # apply gate
        for d in range(D):
            g = torch.sigmoid(gate[s, h * D + d])
            out[s, h * D + d] *= g

save_file(
    {"q": q, "k": k, "v": v, "gate": gate, "output": out},
    "../../data/qwen3_attention.safetensors",
)
print(f"wrote ../../data/qwen3_attention.safetensors output {tuple(out.shape)}")
