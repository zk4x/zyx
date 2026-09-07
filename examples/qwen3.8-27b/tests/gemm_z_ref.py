# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for gemm_kernel(M_PAD, HIDDEN, VAL_DIM).

out [M_PAD, VAL_DIM] = A [M_PAD, HIDDEN] @ W [VAL_DIM, HIDDEN].T (F16 inputs, F32 output).

Dequantizes real Q4_K weight from blk.0.attn_gate.weight to F16, generates random
F16 input, computes reference with torch (FP32 accum), saves input/weight/output.

Run from this directory: python3.12 gemm_z_ref.py
"""

import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
HIDDEN = 5120
VAL_DIM = 6144

r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.attn_gate.weight":
        w_deq = gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        w = torch.from_numpy(w_deq).half()  # [VAL_DIM, HIDDEN]
        break
assert w.shape == (VAL_DIM, HIDDEN)

a = torch.randn(M_PAD, HIDDEN).half()  # [M_PAD, HIDDEN]

out = (a.float() @ w.float().T).contiguous()

save_file(
    {"input": a, "weight": w, "output": out},
    "../../data/qwen3_gemm_z.safetensors",
)
print(f"wrote ../../data/qwen3_gemm_z.safetensors input {tuple(a.shape)} weight {tuple(w.shape)} output {tuple(out.shape)}")
