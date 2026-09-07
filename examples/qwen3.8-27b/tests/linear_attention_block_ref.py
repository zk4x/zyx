# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for linear_attention_block (Qwen3.5-27B layer 0).

End-to-end forward through blk.0 (linear_attention layer):
  pad -> [gemm_qkv, gemm_z, gemm_ba] -> conv_silu -> delta_core -> rmsnorm ->
  pad -> gemm_o -> out (first S rows).

Real weights from GGUF blk.0:
  attn_qkv.weight, attn_gate.weight (gemm_z), ssm_alpha.weight (b),
  ssm_beta.weight (a), ssm_conv1d.weight, ssm_a, ssm_dt.bias, ssm_norm.weight,
  ssm_out.weight.
Random F32 input.

Run from this directory: python3.12 linear_attention_block_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
HIDDEN = 5120
INTERMEDIATE = 17408
DT_RANK = 48
VH = 48
VD = 128
CONV_DIM = 10240  # KEY_DIM*2 + VAL_DIM = 2*2048 + 6144
KEY_DIM = 2048
VAL_DIM = 6144
M_PAD = 16
EPS = 1e-6


def silu(x):
    return x / (1.0 + torch.exp(-x))


# Load real weights
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
tensors = {}
for t in r.tensors:
    if t.name.startswith("blk.0.") and t.tensor_type == 0:
        tensors[t.name] = torch.from_numpy(np.array(t.data).astype(np.float32).copy())
    elif t.name.startswith("blk.0."):
        tensors[t.name] = torch.from_numpy(
            gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        )

w_qkv = tensors["blk.0.attn_qkv.weight"]            # [CONV_DIM, HIDDEN] dequantized
w_z   = tensors["blk.0.attn_gate.weight"]            # [VAL_DIM, HIDDEN]
w_b   = tensors["blk.0.ssm_beta.weight"]           # [DT_RANK, HIDDEN] dequantized
w_a   = tensors["blk.0.ssm_alpha.weight"]          # [DT_RANK, HIDDEN]
conv_w = tensors["blk.0.ssm_conv1d.weight"]           # [CONV_DIM, CK=4]
ss_a   = tensors["blk.0.ssm_a"]                       # [VH]
ss_dt  = tensors["blk.0.ssm_dt.bias"]                 # [VH]
ss_nw  = tensors["blk.0.ssm_norm.weight"]             # [VD]
w_o   = tensors["blk.0.ssm_out.weight"]               # [HIDDEN, VAL_DIM]

assert w_qkv.shape == (CONV_DIM, HIDDEN)
assert w_z.shape == (VAL_DIM, HIDDEN)
assert w_b.shape == (DT_RANK, HIDDEN)
assert w_a.shape == (DT_RANK, HIDDEN)
assert conv_w.shape == (CONV_DIM, 4)
assert ss_a.shape == (VH,)
assert ss_dt.shape == (VH,)
assert ss_nw.shape == (VD,)
assert w_o.shape == (HIDDEN, VAL_DIM)

# Random F32 input [S, HIDDEN]
input = torch.randn(S, HIDDEN, dtype=torch.float32)

# 1. Pad [S, HIDDEN] -> [M_PAD, HIDDEN] (cast to F16 like the kernel)
pinned = torch.zeros(M_PAD, HIDDEN, dtype=torch.float32)
pinned[:S] = input
pinned = pinned.half().float()  # simulate F16 conversion

# 2. Projections
qkv = pinned @ w_qkv.T          # [M_PAD, CONV_DIM] F32
z   = pinned @ w_z.T            # [M_PAD, VAL_DIM]
b   = pinned @ w_b.T            # [M_PAD, DT_RANK]
a   = pinned @ w_a.T            # [M_PAD, DT_RANK]

# 3. Conv + SiLU on qkv
mixed = torch.zeros(M_PAD, CONV_DIM, dtype=torch.float32)
for t in range(M_PAD):
    for c in range(CONV_DIM):
        acc = 0.0
        for k in range(4):
            src_t = t + k - 3
            if src_t >= 0:
                acc += conv_w[c, k] * qkv[src_t, c]
        mixed[t, c] = silu(acc)

# 4. Delta core (per h, t)
core = torch.zeros(VH, S, VD, dtype=torch.float32)
for h in range(VH):
    S_state = torch.zeros(VD, 128, dtype=torch.float32)  # KD=128
    for t in range(S):
        kh = h // (VH // 16)
        q_row = mixed[t, kh * 128 : (kh + 1) * 128]
        k_row = mixed[t, KEY_DIM + kh * 128 : KEY_DIM + (kh + 1) * 128]
        v_row = mixed[t, 2 * KEY_DIM + h * VD : 2 * KEY_DIM + (h + 1) * VD]
        q_ss = (q_row * q_row).sum()
        k_ss = (k_row * k_row).sum()
        qn = q_row / torch.sqrt(q_ss + EPS) / math.sqrt(128)
        kn = k_row / torch.sqrt(k_ss + EPS)
        beta = torch.sigmoid(b[t, h])
        adt = a[t, h] + ss_dt[h]
        sp = torch.where(adt > 20, adt, torch.log(1.0 + torch.exp(adt)))
        g = torch.exp(-ss_a[h] * sp)
        kv = S_state @ kn
        v_new = v_row - g * kv
        delta = beta * v_new
        S_state = g * S_state + torch.outer(delta, kn)
        core[h, t, :] = S_state @ qn

# 5. Gated RMSNorm
mean = (core * core).sum(dim=-1, keepdim=True) / 128  # [VH, S, 1]
scale = 1.0 / torch.sqrt(mean + EPS)
normed_core = core * scale * ss_nw.view(1, 1, VD)  # [VH, S, VD]
gz = silu(z[:S]).reshape(S, VH, VD)  # [S, VH, VD]  (z is [M_PAD, VAL_DIM]=[16, 6144])
normed = (normed_core * gz.permute(1, 0, 2)).reshape(S, VAL_DIM)  # [S, VH*VD]=[S, 6144]

# 6. Pad [S, VAL_DIM] -> [M_PAD, VAL_DIM]
normed_p = torch.zeros(M_PAD, VAL_DIM, dtype=torch.float32)
normed_p[:S] = normed
normed_p = normed_p.half().float()

# 7. Output projection
out_padded = normed_p @ w_o.T  # [M_PAD, HIDDEN]

# 8. Final output = first S rows
out = out_padded[:S]

save_file({
    "input": input,
    "w_qkv": w_qkv.half().float(),  # store as F32 (simulates F16->F32 ref)
    "w_z": w_z.half().float(),
    "w_b": w_b.half().float(),
    "w_a": w_a.half().float(),
    "conv_w": conv_w,
    "ss_a": ss_a,
    "ss_dt_bias": ss_dt,
    "ss_norm_w": ss_nw,
    "w_o": w_o.half().float(),
    "output": out,
}, "../../data/qwen3_linear_attention_block.safetensors")
print(f"wrote ../../data/qwen3_linear_attention_block.safetensors output {tuple(out.shape)}")
