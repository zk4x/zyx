# tt_gemm — multi-core BF16 GEMM for Tenstorrent Blackhole

Hand-tiled GEMM via the `zyx` custom-`Kernel` API (bypasses egraph/autotune).
Each core owns a `4x4` grid of 32x32 output tiles; row-stationary reuse shares
one A tile across the row's matmuls. Correctness is checked against a
CUDA (or CPU) reference matmul.

## Run

```bash
./run.sh                        # needs TT_METAL_ROOT for the TT runtime build
ZYX_DRY_RUN=1 cargo run -p tt_gemm   # compile only, no board launch (value check fails by design)
```

## Reading perf (use the device lines, not the summary line)

Run with the dev debug bit to get per-launch device-synced numbers:

```bash
ZYX_DEBUG=1 cargo run -p tt_gemm
```

Each `forward` then prints a `get_perf` line, e.g.:

```text
1.2 ms ~ 2.91 TFLOP/s, 113.73 GB/s r, 2.84 GB/s w
```

That line times the synchronous device launch only (flops from the
`MatmulTile` count, DRAM bytes read/written) — **this is the device number.**

The `tt_gemm: best ... TFLOPS` summary line is host-inclusive: every timed
iteration does `forward` + `to(Dev::C)` + `cast` + `untilize` + `to_vec`,
so it understates device throughput (~0.08 TFLOPS end-to-end) and must not
be quoted as kernel performance.

## Current state

- Correctness: `bad 0 / 1802240` vs CUDA reference, max err ~0.05 (BF16).
- Device: ~2.4–3.1 TFLOP/s at M=1280, N=1408, K=1024 on a 10x11 core grid.
- Bottleneck: private per-core DRAM reads (~110 GB/s); multicast/L1 sharing
  not yet implemented — see the 300 TFLOPS gap analysis in session notes.
