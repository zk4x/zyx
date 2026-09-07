# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for mlp_kernel.

out [m, inter] = silu(gate) * up

Random F32 gate and up [M_PAD, INTERMEDIATE].

Run from this directory: python3.12 mlp_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
INTERMEDIATE = 17408


def silu(x):
    return x / (1.0 + torch.exp(-x))


gate = torch.randn(M_PAD, INTERMEDIATE, dtype=torch.float32)
up = torch.randn(M_PAD, INTERMEDIATE, dtype=torch.float32)

out = silu(gate) * up

save_file(
    {"gate": gate, "up": up, "output": out},
    "../../data/qwen3_mlp.safetensors",
)
print(f"wrote ../../data/qwen3_mlp.safetensors gate {tuple(gate.shape)} up {tuple(up.shape)} output {tuple(out.shape)}")
