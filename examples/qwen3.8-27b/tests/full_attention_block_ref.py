# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for full_attention_block (Qwen3.5-27B layer 3).

End-to-end forward through blk.3 (full_attention layer) per vLLM's
Qwen3NextAttention.forward:
  input_rmsnorm -> qkv_proj (q+gate fused) -> q,k split -> qk_norm (with
  per-head weight) -> rope -> attention (with output gate) -> o_proj ->
  residual_add -> out.

Real weights from GGUF blk.3.

Run from this directory: python3.12 full_attention_block_ref.py
"""

import math
import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

S = 6
HIDDEN = 5120
HEADS = 24
KV_HEADS = 4
HEAD_DIM = 256
ROT_DIM = 64
H_TOTAL = HEADS * HEAD_DIM        # 6144
KV_TOTAL = KV_HEADS * HEAD_DIM    # 1024
QKV_TOTAL = 2 * H_TOTAL           # 12288 (q + gate)
EPS = 1e-6


def silu(x):
    return x / (1.0 + torch.exp(-x))


def rotate_half(x):
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:d]
    return torch.cat((-x2, x1), dim=-1)


# Load real weights
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
tensors = {}
for t in r.tensors:
    if not t.name.startswith("blk.3."):
        continue
    if t.tensor_type == 0:
        tensors[t.name] = torch.from_numpy(np.array(t.data).astype(np.float32).copy())
    else:
        tensors[t.name] = torch.from_numpy(
            gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        )

# w_q is [2*H*HEAD_DIM, HIDDEN] dequantized (q+gate fused, [q, gate] stacked)
w_q = tensors["blk.3.attn_q.weight"]            # [12288, 5120]
q_w_only = w_q[:H_TOTAL, :]                    # [H*D, HIDDEN] = [6144, 5120]
gate_w_only = w_q[H_TOTAL:, :]                 # [H*D, HIDDEN] = [6144, 5120]
w_k = tensors["blk.3.attn_k.weight"]            # [1024, 5120]
w_v = tensors["blk.3.attn_v.weight"]            # [1024, 5120]
w_o = tensors["blk.3.attn_output.weight"]       # [5120, 6144]
attn_norm = tensors["blk.3.attn_norm.weight"]   # [5120]
q_norm_w = tensors["blk.3.attn_q_norm.weight"]  # [256]
k_norm_w = tensors["blk.3.attn_k_norm.weight"]  # [256]

assert q_w_only.shape == (H_TOTAL, HIDDEN), f"q_w {q_w_only.shape}"
assert gate_w_only.shape == (H_TOTAL, HIDDEN), f"gate_w {gate_w_only.shape}"
assert w_k.shape == (KV_TOTAL, HIDDEN), f"w_k {w_k.shape}"
assert w_v.shape == (KV_TOTAL, HIDDEN), f"w_v {w_v.shape}"
assert w_o.shape == (HIDDEN, H_TOTAL), f"w_o {w_o.shape}"

# Random F32 input
input = torch.randn(S, HIDDEN, dtype=torch.float32)
cos = torch.randn(S, ROT_DIM, dtype=torch.float32)
sin = torch.randn(S, ROT_DIM, dtype=torch.float32)

# 1. Input RMSNorm
ss = (input * input).sum(dim=-1, keepdim=True)
mean = ss / HIDDEN
scale = 1.0 / torch.sqrt(mean + EPS)
x = input * scale * attn_norm

# 2. QKV projections (q and gate projected separately)
qkv_flat = x @ w_q.T  # [S, QKV_TOTAL=12288]
q_flat = x @ q_w_only.T  # [S, H*D=6144]
gate_flat = x @ gate_w_only.T  # [S, H*D=6144]
k_flat = x @ w_k.T    # [S, KV_TOTAL=1024]
v_flat = x @ w_v.T    # [S, KV_TOTAL=1024]

# 3. Split qkv into q and gate
q_flat_split = qkv_flat[:, :H_TOTAL]  # [S, H*D=6144]
gate_flat = qkv_flat[:, H_TOTAL:]      # [S, H*D=6144]
# Reshape to [H, S, D]: need to transpose since [S, H*D] is S-major.
q = q_flat_split.reshape(S, HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()  # [H, S, D]
gate = gate_flat.reshape(S, HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()  # [H, S, D]
k = k_flat.reshape(S, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()  # [KV, S, D]
v = v_flat.reshape(S, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()  # [KV, S, D]

# 4. Q/K norm (per-head, with weight)
ss_q = (q * q).sum(dim=-1, keepdim=True)
scale_q = 1.0 / torch.sqrt(ss_q / HEAD_DIM + EPS)
q_n = q * scale_q * q_norm_w
ss_k = (k * k).sum(dim=-1, keepdim=True)
scale_k = 1.0 / torch.sqrt(ss_k / HEAD_DIM + EPS)
k_n = k * scale_k * k_norm_w

# 5. RoPE on Q and K (partial RoPE, rot_dim=64)
def apply_rope_partial(x_flat, cos, sin, n_heads):
    out = x_flat.clone()
    for hs in range(n_heads * S):
        s = hs % S
        x_row = x_flat[hs, :]
        x_rot = x_row[:ROT_DIM]
        y_rot = x_rot * cos[s, :] + rotate_half(x_rot.unsqueeze(0)).squeeze(0) * sin[s, :]
        out[hs, :ROT_DIM] = y_rot
        out[hs, ROT_DIM:] = x_row[ROT_DIM:]
    return out

q_n_flat = q_n.reshape(HEADS * S, HEAD_DIM)
k_n_flat = k_n.reshape(KV_HEADS * S, HEAD_DIM)
q_roped_flat = apply_rope_partial(q_n_flat, cos, sin, HEADS)
k_roped_flat = apply_rope_partial(k_n_flat, cos, sin, KV_HEADS)
q_roped = q_roped_flat.reshape(HEADS, S, HEAD_DIM)
k_roped = k_roped_flat.reshape(KV_HEADS, S, HEAD_DIM)

# 6. Attention (causal, GQA, with sigmoid gate)
out = torch.zeros(S, H_TOTAL, dtype=torch.float32)
scale_att = 1.0 / math.sqrt(HEAD_DIM)
h_per_kv = HEADS // KV_HEADS  # 6
for h in range(HEADS):
    kv_h = h // h_per_kv
    for s in range(S):
        scores = []
        for s2 in range(s + 1):
            dot = (q_roped[h, s, :] * k_roped[kv_h, s2, :]).sum().item()
            scores.append(dot * scale_att)
        max_s = max(scores)
        exps = [math.exp(sc - max_s) for sc in scores]
        sum_e = sum(exps)
        probs = [e / sum_e for e in exps]
        for s2 in range(s + 1):
            out[s, h * HEAD_DIM:(h + 1) * HEAD_DIM] += probs[s2] * v[kv_h, s2, :]
        # Apply output gate
        for d in range(HEAD_DIM):
            g = torch.sigmoid(gate[h, s, d])
            out[s, h * HEAD_DIM + d] *= g

# 7. Output projection
out_proj = out @ w_o.T  # [S, HIDDEN]

# 8. Residual add
final_out = input + out_proj

save_file({
    "input": input,
    "w_q": w_q.clone(),
    "q_w_only": q_w_only.clone(),
    "gate_w_only": gate_w_only.clone(),
    "w_k": w_k,
    "w_v": w_v,
    "w_o": w_o,
    "attn_norm": attn_norm,
    "q_norm_w": q_norm_w,
    "k_norm_w": k_norm_w,
    "cos": cos,
    "sin": sin,
    "output": final_out,
}, "../../data/qwen3_full_attention_block.safetensors")
print(f"wrote ../../data/qwen3_full_attention_block.safetensors output {tuple(final_out.shape)}")
