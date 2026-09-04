"""Golden reference for the SwiGLU MLP block.

Uses the real Qwen3_5MLP class (hidden 64, intermediate 128, no bias):
down(silu(gate(x)) * up(x)). Dumps weights + input/output (float32).

Run from this directory: python3.12 mlp_ref.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

torch.manual_seed(2)

config = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, hidden_act="silu")
mlp = Qwen3_5MLP(config, 128)
x = torch.randn(2, 4, 64)
with torch.no_grad():
    output = mlp(x)

sd = mlp.state_dict()
save_file(
    {
        "gate": sd["gate_proj.weight"],
        "up": sd["up_proj.weight"],
        "down": sd["down_proj.weight"],
        "input": x,
        "output": output,
    },
    "../../data/qwen3_8b_mlp.safetensors",
)
print("wrote ../../data/qwen3_8b_mlp.safetensors, output shape:", tuple(output.shape))
