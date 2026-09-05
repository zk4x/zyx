# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for the RMSNorm op.

Uses the real Qwen3_5RMSNorm class (tiny dim 64). Note its zero-centered
weight convention: output = norm(x) * (1 + weight). The dumped `scale` is
already (1 + weight), i.e. what the kernel consumes.

Run from this directory: python3.12 rmsnorm_ref.py
"""

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm

torch.manual_seed(1)

DIM = 64

norm = Qwen3_5RMSNorm(DIM)
x = torch.randn(2, 8, DIM)
with torch.no_grad():
    output = norm(x)

save_file(
    {"scale": (1.0 + norm.weight).detach(), "input": x, "output": output},
    "../../data/qwen3_8b_rmsnorm.safetensors",
)
print("wrote ../../data/qwen3_8b_rmsnorm.safetensors, output shape:", tuple(output.shape))
