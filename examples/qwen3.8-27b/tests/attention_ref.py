"""Golden reference for the full (GQA) attention block.

Uses the real Qwen3_5Attention class (hidden 64, 4 q-heads, 2 kv-heads,
head_dim 8, no bias, causal, no cache) with real rotary cos/sin. Dumps all
six weights + input/output (float32). q_norm/k_norm weights are
zero-centered (scale = 1 + weight); the dump stores scale directly.

Run from this directory: python3.12 attention_ref.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextRotaryEmbedding,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.manual_seed(4)

config = Qwen3_5TextConfig(
    hidden_size=64,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    attention_bias=False,
    rms_norm_eps=1e-6,
    attention_dropout=0.0,
)
attn = Qwen3_5Attention(config, layer_idx=0)
rot = Qwen3_5TextRotaryEmbedding(config)

h = torch.randn(1, 4, 64)
cos, sin = rot(torch.randn(1, 2, 4, 8), torch.arange(4)[None, :])
# Explicit causal mask: eager mode applies NO mask when mask is None.
causal_mask = torch.full((4, 4), float("-inf")).triu(1)
with torch.no_grad():
    output, _ = attn(h, (cos, sin), causal_mask, None)

sd = attn.state_dict()
save_file(
    {
        "q_proj": sd["q_proj.weight"],
        "k_proj": sd["k_proj.weight"],
        "v_proj": sd["v_proj.weight"],
        "o_proj": sd["o_proj.weight"],
        "q_scale": (1.0 + sd["q_norm.weight"]).squeeze(),
        "k_scale": (1.0 + sd["k_norm.weight"]).squeeze(),
        "cos": cos,
        "sin": sin,
        "mask": causal_mask,
        "input": h,
        "output": output,
    },
    "../../data/qwen3_8b_attention.safetensors",
)
print("wrote ../../data/qwen3_8b_attention.safetensors, output shape:", tuple(output.shape))
