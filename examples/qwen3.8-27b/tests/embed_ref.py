# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Golden reference for the embedding gather op.

Builds a tiny nn.Embedding (vocab 256, dim 64), gathers 8 ids, and dumps
{weight, input_ids, output} to ../../data/qwen3_8b_embed.safetensors (float32).

Run from this directory: python3.12 embed_ref.py
"""

import torch
from safetensors.torch import save_file

torch.manual_seed(0)

VOCAB = 256
DIM = 64

weight = torch.randn(VOCAB, DIM)
input_ids = torch.tensor([3, 17, 42, 100, 128, 200, 231, 255])
output = torch.nn.functional.embedding(input_ids, weight)

save_file(
    {"weight": weight, "input_ids": input_ids, "output": output},
    "../../data/qwen3_8b_embed.safetensors",
)
print("wrote ../../data/qwen3_8b_embed.safetensors, output shape:", tuple(output.shape))
