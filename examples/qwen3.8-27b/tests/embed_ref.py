# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for embed_kernel.

out [seq, dim] = w[ids[s], d]

Random F32 weight [VOCAB, DIM] and I64 ids [SEQ]. (Real model uses quantized
embeddings — out of scope for this kernel; the kernel takes F32 inputs.)

Run from this directory: python3.12 embed_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(6)

VOCAB = 248320
DIM = 5120
SEQ = 6

# Random F32 weight (real model uses Q8_0; out of scope here)
w = torch.randn(VOCAB, DIM, dtype=torch.float32)

# Random token ids in [0, VOCAB)
ids = torch.randint(0, VOCAB, (SEQ,), dtype=torch.int64)

# Reference
out = w[ids.long()].contiguous()  # [SEQ, DIM]

save_file(
    {"w": w, "ids": ids, "output": out},
    "../../data/qwen3_embed.safetensors",
)
print(f"wrote ../../data/qwen3_embed.safetensors w {tuple(w.shape)} ids {tuple(ids.shape)} output {tuple(out.shape)}")
