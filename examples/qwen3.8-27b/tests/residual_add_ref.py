# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for residual_add_kernel.

out [M_PAD, HIDDEN] = a + b (F32).
Random F32 inputs.

Run from this directory: python3.12 residual_add_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
HIDDEN = 5120

a = torch.randn(M_PAD, HIDDEN, dtype=torch.float32)
b = torch.randn(M_PAD, HIDDEN, dtype=torch.float32)
out = a + b

save_file(
    {"a": a, "b": b, "output": out},
    "../../data/qwen3_residual_add.safetensors",
)
print(f"wrote ../../data/qwen3_residual_add.safetensors a {tuple(a.shape)} b {tuple(b.shape)} output {tuple(out.shape)}")
