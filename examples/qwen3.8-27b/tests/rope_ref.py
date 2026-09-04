"""Golden reference for the partial-RoPE application.

Uses the real Qwen3_5TextRotaryEmbedding (head_dim 16, partial factor 0.25
-> rot_dim 4) plus the modeling apply_rotary_pos_emb. Dumps q, k, cos, sin
and the rotated outputs (float32). cos/sin are kernel inputs (precomputed
host-side); the kernel only applies the rotation.

Run from this directory: python3.12 rope_ref.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.manual_seed(3)

config = Qwen3_5TextConfig(head_dim=16)
rot = Qwen3_5TextRotaryEmbedding(config)
x = torch.randn(1, 2, 4, 16)
cos, sin = rot(x, torch.arange(4)[None, :])

q = torch.randn(1, 2, 4, 16)
k = torch.randn(1, 2, 4, 16)
with torch.no_grad():
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

save_file(
    {"q": q, "k": k, "cos": cos, "sin": sin, "q_rot": q_rot, "k_rot": k_rot},
    "../../data/qwen3_8b_rope.safetensors",
)
print("wrote ../../data/qwen3_8b_rope.safetensors, cos shape:", tuple(cos.shape))
