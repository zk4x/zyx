# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

"""llama-3.2-3B GEMMs in tinygrad, per projection and per phase.

Mirrors `zyx-bench llama prefill_gemm` and `zyx-bench llama decode_gemm`: the
same seven projections at the same lengths (`x [m, k] @ w [n, k].T`, both
operands f16), plus `lm_head` at decode length. A single GEMM is already one
kernel, so the comparison is tinygrad's kernel against zyx's; `TinyJit` (the
`Tape::freeze` equivalent, as in `examples/mnist`) only removes tinygrad's
per-`realize` Python scheduling, which at ~2ms would otherwise dominate a
0.1ms kernel and measure nothing.

Timing is a batch of `REPS` jitted calls plus one device sync, best of
`TRIALS`. Each phase ends with its aggregate over the projections.

    python3.12 examples/llama/gemm_tinygrad.py
"""

import time

from tinygrad import Device, Tensor, TinyJit, dtypes

# `(name, k, n)` in the order a decoder block runs them; matches
# `zyx-bench/src/bench/llama/matmul.rs` and `shapes.rs`.
HIDDEN = 3072
INTERMEDIATE = 8192
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 128_256
SEQ = 128

PROJS = [
    ("q_proj", HIDDEN, HEADS * HEAD_DIM),
    ("k_proj", HIDDEN, KV_HEADS * HEAD_DIM),
    ("v_proj", HIDDEN, KV_HEADS * HEAD_DIM),
    ("o_proj", HEADS * HEAD_DIM, HIDDEN),
    ("gate_proj", HIDDEN, INTERMEDIATE),
    ("up_proj", HIDDEN, INTERMEDIATE),
    ("down_proj", INTERMEDIATE, HIDDEN),
]

# `(phase, sequence length, extra projections)`; decode adds `lm_head`, exactly
# like `zyx-bench llama decode_gemm` (`matmul.rs::lm_head`).
PHASES = [
    ("prefill", SEQ, []),
    ("decode", 1, [("lm_head", HIDDEN, VOCAB)]),
]

TRIALS = 5
REPS = 50
WARMUP = 20


def measure(dev, fn):
    for _ in range(WARMUP):
        fn().realize()
    dev.synchronize()
    best = float("inf")
    for _ in range(TRIALS):
        start = time.perf_counter()
        for _ in range(REPS):
            fn()
        dev.synchronize()
        best = min(best, (time.perf_counter() - start) / REPS)
    return best


def main():
    dev = Device[Device.DEFAULT]
    print(f"tinygrad, {Device.DEFAULT}")
    for phase, seq, extra in PHASES:
        total_ms = 0.0
        total_flops = 0.0
        for name, k, n in PROJS + extra:
            x = Tensor.rand(seq, k, dtype=dtypes.float16).realize()
            w = Tensor.rand(n, k, dtype=dtypes.float16).realize()

            @TinyJit
            def gemm(x, w):
                return (x @ w.transpose()).realize()

            ms = measure(dev, lambda: gemm(x, w)) * 1e3
            flops = 2.0 * seq * k * n
            tflops = flops / (ms * 1e-3) / 1e12
            total_ms += ms
            total_flops += flops
            print(f"{phase:8s} {name:10s} [{seq}x{k}]x[{k}x{n}]  {ms:8.4f} ms  {tflops:6.2f} TFLOP/s")
        print(f"{phase:8s} {'total':10s} {total_ms:8.4f} ms  {total_flops / (total_ms * 1e-3) / 1e12:6.2f} TFLOP/s")


if __name__ == "__main__":
    main()
