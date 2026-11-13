# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

"""llama-3.2-3B GEMMs in torch/cuBLAS, per projection and per phase.

Mirrors `zyx-bench llama prefill_gemm` and `zyx-bench llama decode_gemm`: the
same seven projections at the same lengths (`x [m, k] @ w [n, k].T`, both
operands f16), plus the `lm_head` projection at decode length, which is what
`decode_gemm` adds. The only thing that differs between the two runs is the
matmul implementation. Plain eager torch, no `compile`: the reference is
cuBLAS itself, not a fused step.

Timing is a batch of `REPS` back-to-back launches followed by one sync, so CPU
submit cost is amortized the way it is inside a real layer; the best of
`TRIALS` batches is reported (min, not mean: a stray sync in one batch should
not inflate the kernel's cost). Each phase ends with its aggregate over the
projections, which is the number to compare against zyx's phase total.

`--cold` measures the way zyx's `LaunchTimed` does, for a like-for-like
comparison: one launch per trial, bracketed by CUDA events, after the device
has been left idle long enough for the clocks to fall back. Each trial records
the SM clock it ran at (`nvidia-smi`), because "cold" is a clock state and not
something the wall clock can assert; the reported number carries its clock so
the two sides can be compared at equal clocks. A batch protocol warms the
device to boost clocks, so the warm number is not the one to compare against a
single-launch kernel.

    python3.12 examples/llama/gemm_torch.py [--cold]
"""

import subprocess
import sys
import time

import torch

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

# `--cold`: one launch per trial after an idle gap. The gap must outlast the
# clock ramp-down, and the clock is sampled per trial so the reported number
# carries the state it ran at.
COLD_TRIALS = 5
COLD_IDLE_S = 2.0


def measure(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(TRIALS):
        start = time.perf_counter()
        for _ in range(REPS):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - start) / REPS)
    return best


def sm_clock_mhz(device_index):
    """The SM clock `nvidia-smi` reports for the device, or None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=clocks.sm",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def measure_cold(fn, device_index):
    """One event-bracketed launch per trial, after the device sits idle.

    Mirrors zyx's `LaunchTimed` (one launch, CUDA events, no queue) instead of
    the batched warm measurement above, so the two numbers are measured the
    same way. The first call is initialization only (cuBLAS/cuDNN plan
    selection), which must not be timed; the idle gap afterwards is what puts
    the clocks back where a single-launch kernel finds them.
    """
    fn()
    torch.cuda.synchronize()
    best = (float("inf"), None)
    for _ in range(COLD_TRIALS):
        time.sleep(COLD_IDLE_S)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        mhz = sm_clock_mhz(device_index)
        start.record()
        fn()
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end)
        if ms < best[0]:
            best = (ms, mhz)
    return best


def main():
    cold = "--cold" in sys.argv[1:]
    device = torch.device("cuda")
    device_index = torch.cuda.current_device()
    print(f"torch {torch.__version__} (cuda {torch.version.cuda}), {torch.cuda.get_device_name(device)}")
    if cold:
        print(f"cold: {COLD_TRIALS} trials, {COLD_IDLE_S:g} s idle before each, one launch per trial")
    for phase, seq, extra in PHASES:
        total_ms = 0.0
        total_flops = 0.0
        total_bytes = 0.0
        for name, k, n in PROJS + extra:
            x = torch.randn(seq, k, device=device, dtype=torch.float16)
            w = torch.randn(n, k, device=device, dtype=torch.float16)
            with torch.no_grad():
                if cold:
                    out = torch.empty(seq, n, device=device, dtype=torch.float16)
                    # `elapsed_time` is already milliseconds; the warm path
                    # returns seconds from the wall clock.
                    ms, mhz = measure_cold(lambda: torch.matmul(x, w.t(), out=out), device_index)
                else:
                    ms = measure(lambda: x @ w.t()) * 1e3
                    mhz = None
            flops = 2.0 * seq * k * n
            tflops = flops / (ms * 1e-3) / 1e12
            gbs = (seq * k + n * k + seq * n) * 2 / (ms * 1e-3) / 1e9
            total_ms += ms
            total_flops += flops
            total_bytes += (seq * k + n * k + seq * n) * 2
            clock = "" if mhz is None else f"  @ {mhz} MHz"
            print(f"{phase:8s} {name:10s} [{seq}x{k}]x[{k}x{n}]  {ms:8.4f} ms  {tflops:6.2f} TFLOP/s  {gbs:6.0f} GB/s{clock}")
        print(
            f"{phase:8s} {'total':10s} {total_ms:8.4f} ms  "
            f"{total_flops / (total_ms * 1e-3) / 1e12:6.2f} TFLOP/s  {total_bytes / (total_ms * 1e-3) / 1e9:6.0f} GB/s"
        )


if __name__ == "__main__":
    main()
