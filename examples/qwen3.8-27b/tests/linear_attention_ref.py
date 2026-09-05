# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for the linear-attention (GatedDeltaNet) block.

Uses the real Qwen3_5GatedDeltaNet class (hidden 32, 2 k-heads, 2 v-heads,
k/v dim 8, conv kernel 4, seq 6) with the torch fallback kernel (no cache,
no mask). Dumps all nine weights + input/output (float32).

Run from this directory: python3.12 linear_attention_ref.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.manual_seed(5)

config = Qwen3_5TextConfig(
    hidden_size=32,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_num_key_heads=2,
    linear_num_value_heads=2,
    linear_conv_kernel_dim=4,
    hidden_act="silu",
    rms_norm_eps=1e-6,
)
net = Qwen3_5GatedDeltaNet(config, layer_idx=1)
h = torch.randn(1, 6, 32)
with torch.no_grad():
    output = net(h, None, None)

sd = net.state_dict()
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
    },
    "../../data/qwen3_8b_linear_attention.safetensors",
)
print("wrote ../../data/qwen3_8b_linear_attention.safetensors, output shape:", tuple(output.shape))
