# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for the linear-attention (GatedDeltaNet) block.

Uses the real Qwen3_5GatedDeltaNet class with the real Qwen3.8-27B linear
attention geometry (hidden 5120, 16 k-heads, 48 v-heads, k/v dim 128,
conv kernel 4, seq 6 — a single chunk of 64) with the torch fallback kernel
(no cache, no mask). Dumps all nine weights + input/output (float32).

Run from this directory: python3.12 linear_attention_ref.py
"""

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.manual_seed(5)
torch.backends.cuda.matmul.allow_tf32 = False

config = Qwen3_5TextConfig(
    hidden_size=5120,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_conv_kernel_dim=4,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)
net = Qwen3_5GatedDeltaNet(config, layer_idx=1)
h = torch.randn(1, 6, 5120)
with torch.no_grad():
    output = net(h, None, None)

sd = net.state_dict()
out_ref = output

# Per-kernel intermediates, recomputed in plain torch from the dumped
# weights. Shapes match what each custom kernel in src/lib.rs consumes:
# pad rows 6 -> 16, f32 linears, depthwise causal conv1d (k=4, left pad 3)
# + SiLU, the token-recurrent gated delta rule, gated RMSNorm.
with torch.no_grad():
    dev = h.device
    h1 = h[0]  # [6, 5120]
    pin = torch.cat([h1, torch.zeros(10, 5120, device=dev)], 0)  # [16, 5120]
    mixed = pin @ sd["in_proj_qkv.weight"].T  # [16, 10240]
    z = pin @ sd["in_proj_z.weight"].T  # [16, 6144]
    b = pin @ sd["in_proj_b.weight"].T  # [16, 48]
    a = pin @ sd["in_proj_a.weight"].T  # [16, 48]
    # GEMM kernels take f16: reference from half inputs (tf32 off above).
    mixed_href = (pin.half() @ sd["in_proj_qkv.weight"].half().T).float()
    xp = F.pad(mixed.T.unsqueeze(0), (3, 0))  # left pad 3
    mixed_s = F.silu(F.conv1d(xp, sd["conv1d.weight"], groups=10240)).squeeze(0).T.contiguous()
    # Recurrent gated delta rule (llama.cpp gated_delta_net convention).
    ea = sd["A_log"].exp()
    dtb = sd["dt_bias"]
    q = mixed_s[:6, :2048].view(6, 16, 128).repeat_interleave(3, dim=1)
    k = mixed_s[:6, 2048:4096].view(6, 16, 128).repeat_interleave(3, dim=1)
    v = mixed_s[:6, 4096:].view(6, 48, 128)
    b6, a6 = b[:6], a[:6]
    z6 = z[:6].view(6, 48, 128)

    def l2norm(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    qn = l2norm(q) * (128**-0.5)
    kn = l2norm(k)
    beta = torch.sigmoid(b6)
    g = torch.exp(-ea * F.softplus(a6 + dtb, beta=1.0, threshold=20.0))
    state = torch.zeros(48, 128, 128, device=dev)
    outs = []
    for t in range(6):
        kv = torch.einsum("hij,hi->hj", state, kn[t])
        delta = (v[t] - g[t, :, None] * kv) * beta[t, :, None]
        state = g[t, :, None, None] * state + torch.einsum("hi,hj->hij", kn[t], delta)
        outs.append(torch.einsum("hij,hi->hj", state, qn[t]))
    core = torch.stack(outs, 1)  # [48, 6, 128]
    mean = (core * core).mean(-1, keepdim=True)
    normed = (core.permute(1, 0, 2) * torch.rsqrt(mean.permute(1, 0, 2) + 1e-6)
              * sd["norm.weight"] * F.silu(z6)).reshape(6, 6144).contiguous()
    out_href = (normed.half() @ sd["out_proj.weight"].half().T).float()
    # Prove the intermediates are faithful: out-proj must match hf.
    check = normed @ sd["out_proj.weight"].T
    assert torch.allclose(check, out_ref[0], atol=1e-4), (
        f"intermediates drift from hf: max err {(check - out_ref[0]).abs().max()}")
    print("intermediates faithful to hf output")
    print(f"out_href vs hf max err {(out_href - out_ref[0]).abs().max():.6f}")
save_file(
    {
        "in_proj_qkv": sd["in_proj_qkv.weight"],
        "in_proj_z": sd["in_proj_z.weight"],
        "in_proj_b": sd["in_proj_b.weight"],
        "in_proj_a": sd["in_proj_a.weight"],
        "conv": sd["conv1d.weight"],
        "dt_bias": sd["dt_bias"],
        "a_log": sd["A_log"],
        "norm_weight": sd["norm.weight"],
        "out_proj": sd["out_proj.weight"],
        "input": h,
        "output": output,
        "pin": pin,
        "mixed": mixed,
        "z": z,
        "b": b,
        "a": a,
        "mixed_href": mixed_href,
        "mixed_s": mixed_s,
        "core": core,
        "normed": normed,
        "out_href": out_href,
    },
    "../../data/qwen3_8b_linear_attention.safetensors",
)
print("wrote ../../data/qwen3_8b_linear_attention.safetensors, output shape:", tuple(output.shape))
