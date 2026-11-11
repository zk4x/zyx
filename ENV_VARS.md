# Environment Variables

## `ZYX_DEBUG`

Set `ZYX_DEBUG` to a bitmask to enable tracing. Values combine by summing:

| Value | Flag | Description |
|-------|------|-------------|
| 1 | dev | Devices and configuration |
| 2 | egraph | Graph/e-graph dump |
| 4 | sched | Kernel IR before linearization (the scheduler's view) |
| 8 | ir | Winning kernel IR after the autotune search |
| 16 | asm | Generated assembly/source |
| 32 | launch | Kernel launches (program id, grid, args) |
| 64 | memory | Memory allocation and deallocation |
| 128 | compile | Kernel compilation |
| 256 | no_search | Skip the beam search: compile a cleaned seed once (debug bisect) |

Examples:

```bash
ZYX_DEBUG=8 cargo run      # winning kernel IR
ZYX_DEBUG=24 cargo run     # IR (8) + generated code (16)
ZYX_DEBUG=256 cargo run    # NoSearch: one seed, no optimization search
```

## `ZYX_DRY_RUN`

Set `ZYX_DRY_RUN=1` to skip all device launches across every backend.
`Dev::launch` returns `Ok(())` immediately and `Dev::launch_timed` returns a
fixed 1s placeholder (`1_000_000_000` nanos, never a measurement). Compilation
still runs (codegen + JIT + compile-time checks), so combine it with
`ZYX_DEBUG=8`/`16` to capture IR and generated code without executing:

```bash
ZYX_DRY_RUN=1 ZYX_DEBUG=24 cargo run   # IR + code, no launch
```

Output buffers hold uninitialized contents under dry run — callers must not
read them. Autotune winner-picking under dry run is arbitrary (the first seed
wins on the fixed placeholder).

## Other variables

| Variable | Effect |
|----------|--------|
| `ZYX_DEV` | Pin one device (`cpu`, `cuda:0`, `vulkan`, `tt`, ...); see [CONFIG.md](CONFIG.md) |
| `ZYX_PTX` | CUDA codegen emits PTX directly instead of CUDA C + NVRTC |
| `ZYX_BENCH_OUT` | Append one JSON line per measured autotune candidate to this path |
| `ZYX_BENCH_TAG` | Tag for the benchmark rows (`untagged` by default) |
| `ZYX_TT_IR` | Dump the Tenstorrent IR before/after `tt_tile`, `tt_reindex`, `tt_dataflow_select` |
| `ZYX_TT_DUMP` | Dump each Tenstorrent role program (bytes, entries, length) |
| `TT_TTSIM_LIB` | Path to the Tenstorrent simulator library (default `vendor/libttsim_bh.so`) |
| `TT_TTSIM_LIB_X2` / `TT_TTSIM_LIB_X4` | Simulator library for dual-/quad-chip builds |
| `TT_MATH` | Tenstorrent SFPU math mode: `precise` (default) or `approx` |
| `ZYX_KERN_TRACE` | Kernelizer split/store tracing on stderr |
| `AGENT=1` | Colorless debug output (also auto-detected when stdout is not a terminal) |

The `ZYX_TT_*`, `TT_TTSIM_*`, and `TT_MATH` variables apply only to builds with
the `tenstorrent` feature.

**First debug step**: run with `ZYX_DEBUG=1` to see which backends initialized
and how many devices they found. If no devices appear, check whether a
[config file](CONFIG.md) is disabling them.
