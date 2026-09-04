"""Golden reference for the untied LM head (plain matmul).

vocab 256, hidden 64: logits = x @ W.T (no bias). Dumps weight + input/output
(float32).

Run from this directory: python3.12 lm_head_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

VOCAB = 256
HIDDEN = 64

weight = torch.randn(VOCAB, HIDDEN)
x = torch.randn(2, 4, HIDDEN)
with torch.no_grad():
    output = x @ weight.T

save_file(
    {"weight": weight, "input": x, "output": output},
    "../../data/qwen3_8b_lm_head.safetensors",
)
print("wrote ../../data/qwen3_8b_lm_head.safetensors, output shape:", tuple(output.shape))
