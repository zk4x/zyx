# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for the untied LM head (plain matmul).

vocab 256, hidden 64: logits = x @ W.T (no bias). Dumps weight + input as
float16, output as float32 accumulated from the f16-rounded values (matches
mma.sync f32.f16.f16.f32 and zyx Tensor::dot_dtype).

Run from this directory: python3.12 lm_head_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

VOCAB = 256
HIDDEN = 64

weight = torch.randn(VOCAB, HIDDEN).half()
x = torch.randn(2, 4, HIDDEN).half()
with torch.no_grad():
    output = x.float() @ weight.float().T

save_file(
    {"weight": weight, "input": x, "output": output},
    "../../data/qwen3_8b_lm_head.safetensors",
)
print("wrote ../../data/qwen3_8b_lm_head.safetensors, output shape:", tuple(output.shape))
