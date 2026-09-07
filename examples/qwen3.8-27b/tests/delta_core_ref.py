# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for delta_core_kernel.

Gated delta-rule core (llama.cpp gated_delta_net, KDA variant):
  For each (h, t):
    q, k = l2norm(q_raw, k_raw); q = q * 1/sqrt(KD)
    beta = sigmoid(b); g = exp(-ealog * softplus(a + dt_bias))
    kv = S^T k  (S is the state, init 0)
    v_new = v - g * kv
    delta = beta * v_new
    S = g * S + k * delta^T
    out[t, h, col] = S^T q (one col per warp)
  Final out: [VH, S, VD]

Inputs:
  mixed [M_PAD, CONV_DIM]  - conv+silu output (q/k/v packed at offsets 0, KEY_DIM, 2*KEY_DIM)
  bp    [M_PAD, DT_RANK]   - b per (token, head)
  ap    [M_PAD, DT_RANK]   - a per (token, head)
  ealog [VH]               - ssm_a (raw, not log)
  dtb   [VH]               - ssm_dt.bias
Output: [VH, S, VD]

Random F32 for mixed/bp/ap, real ealog from blk.0.ssm_a, real dtb from
blk.0.ssm_dt.bias.

Run from this directory: python3.12 delta_core_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
S = 6
VH = 48
VD = 128
KD = 128
DT_RANK = 48
KEY_DIM = 2048  # KH * KD
CONV_DIM = 10240
EPS = 1e-6

# Real ealog = ssm_a, dtb = ssm_dt.bias
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.ssm_a":
        ealog = torch.from_numpy(np.array(t.data).astype(np.float32).copy())  # [VH]
    if t.name == "blk.0.ssm_dt.bias":
        dtb = torch.from_numpy(np.array(t.data).astype(np.float32).copy())  # [VH]
assert ealog.shape == (VH,)
assert dtb.shape == (VH,)

# Random F32 inputs
mixed = torch.randn(M_PAD, CONV_DIM, dtype=torch.float32)
bp = torch.randn(M_PAD, DT_RANK, dtype=torch.float32)
ap = torch.randn(M_PAD, DT_RANK, dtype=torch.float32)

qscale = 1.0 / math.sqrt(KD)

# Reference: per (h, t), compute the gated delta rule.
# State S is [VD, KD] (state is between value-dim and key-dim). Initial S=0.
# Each warp processes 4 cols of S (one col = one value head's col).
# Output: out[h, t, col] is the dot product of S[:, col] with normalized q.
out = torch.zeros(VH, S, VD, dtype=torch.float32)

for h in range(VH):
    # state S[VD, KD] = 0
    S_state = torch.zeros(VD, KD, dtype=torch.float32)
    for t in range(S):
        # Extract q, k, v from mixed at this token, this head's group
        # Qwen3.5 mamba has 1 group (KH=16 key heads, but each delta rule uses 1 head, no GQA here)
        # Each head h uses its own k block: offset = h * KD (no KH/kv division in this kernel)
        # Actually kernel: kh = h / (VH/KH) = h / 3, so head h maps to key head h/3
        kh = h // (VH // 16)  # VH=48, KH=16, so kh = h/3
        q_row = mixed[t, kh * KD : (kh + 1) * KD]
        k_row = mixed[t, KEY_DIM + kh * KD : KEY_DIM + (kh + 1) * KD]
        v_row = mixed[t, 2 * KEY_DIM + h * VD : 2 * KEY_DIM + (h + 1) * VD]

        # l2norm
        q_ss = (q_row * q_row).sum()
        k_ss = (k_row * k_row).sum()
        qn = q_row / torch.sqrt(q_ss + EPS) * qscale
        kn = k_row / torch.sqrt(k_ss + EPS)

        # beta, g
        beta = torch.sigmoid(bp[t, h])
        adt = ap[t, h] + dtb[h]
        # softplus(x) = log(1 + exp(x)), with cap at 20
        sp = torch.where(adt > 20, adt, torch.log(1.0 + torch.exp(adt)))
        g = torch.exp(-ealog[h] * sp)

        # kv[col] = sum_k S[col, k] * kn[k]
        kv = S_state @ kn  # [VD]

        # v_new = v - g * kv
        v_new = v_row - g * kv

        # delta = beta * v_new
        delta = beta * v_new

        # S[col, k] = g * S[col, k] + kn[k] * delta[col]
        S_state = g * S_state + torch.outer(delta, kn)

        # out[h, t, col] = sum_k S[col, k] * qn[k]
        out[h, t, :] = S_state @ qn

save_file(
    {"mixed": mixed, "bp": bp, "ap": ap, "ealog": ealog, "dtb": dtb, "output": out},
    "../../data/qwen3_delta_core.safetensors",
)
print(f"wrote ../../data/qwen3_delta_core.safetensors output {tuple(out.shape)}")
