# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for gemm_kernel(M_PAD, HIDDEN, CONV_DIM).

out [M_PAD, CONV_DIM] = A [M_PAD, HIDDEN] @ W [HIDDEN, CONV_DIM] (F16 inputs, F32 output).

Dequantizes real Q4_K weight from blk.0.attn_qkv.weight to F16, generates random
F16 input, computes reference with torch (FP32 accum), saves input/weight/output.

Run from this directory: python3.12 gemm_qkv_ref.py
"""

import torch
import gguf
import numpy as np
from safetensors.torch import save_file

torch.manual_seed(6)

M_PAD = 16
HIDDEN = 5120
CONV_DIM = 10240

# Load blk.0.attn_qkv.weight (Q4_K), dequantize to F32, cast to F16.
r = gguf.GGUFReader("/home/x/Dev/rust/zyx/examples/models/Qwen3.8-27B-UD-Q4_K_XL.gguf")
for t in r.tensors:
    if t.name == "blk.0.attn_qkv.weight":
        w_deq = gguf.dequantize(np.array(t.data), t.tensor_type).astype(np.float32)
        w = torch.from_numpy(w_deq).half()  # [CONV_DIM, HIDDEN]
        break
assert w.shape == (CONV_DIM, HIDDEN)

# Random F16 input
a = torch.randn(M_PAD, HIDDEN).half()  # [M_PAD, HIDDEN]

# Reference: F16 inputs, F32 accum (matches mma.f32.f16.f16.f32).
out = (a.float() @ w.float().T).contiguous()

save_file(
    {"input": a, "weight": w, "output": out},
    "../../data/qwen3_gemm_qkv.safetensors",
)
print(f"wrote ../../data/qwen3_gemm_qkv.safetensors input {tuple(a.shape)} weight {tuple(w.shape)} output {tuple(out.shape)}")
