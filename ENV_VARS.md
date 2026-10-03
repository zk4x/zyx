Set `ZYX_DEBUG` environment variable to enable debugging. It is a bitmask with the following options:

| Value | Flag | Description |
|-------|------|-------------|
| 1     | dev  | Print hardware devices and configuration |
| 2     | egraph | Print the egraph after realize (graph/extraction) |
| 4     | sched | Print kernels created by scheduler |
| 8     | ir   | Print kernels in intermediate representation |
| 16    | asm  | Print kernels in native assembly/code (OpenCL, WGSL, etc.) |
| 32    | launch  | Print kernel launches (program id, grid and args) |
| 64    | memory | Print memory allocation and deallocation |
| 128   | compile | Print kernel compilation |
| 256   | no_search | Skip beam search: compile seeds with linearize + epilogue only (debug bisect) |

Combine flags by summing values (e.g., `ZYX_DEBUG=24` enables ir + asm).

## ZYX_DRY_RUN

Set `ZYX_DRY_RUN=1` to skip all device launches across every backend.
`Dev::launch` returns `Ok(())` immediately and `Dev::launch_timed`
returns a fixed 1s placeholder (1_000_000_000 nanos, never a measurement).
Compile still runs (codegen + JIT + compile-time checks), so combine with
`ZYX_DEBUG=8`/`16` to capture IR and generated code without executing:

```bash
ZYX_DRY_RUN=1 ZYX_DEBUG=24 cargo run   # IR + code, no launch
```

Output buffers hold uninitialized contents under dry run — callers must not
read them. Autotune winner-picking under dry run is arbitrary (first seed
wins on the fixed placeholder).

**First debug step**: run with `ZYX_DEBUG=1` to see which backends initialized and how many devices.
If no devices appear, check whether a [config file](CONFIG.md) is disabling them.
