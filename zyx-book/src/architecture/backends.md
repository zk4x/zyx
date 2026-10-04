# Backend System

Zyx supports multiple hardware backends. Backends are enum-dispatched, compiled into the library (feature gates aside), and selected at runtime through a lightweight `Copy` handle.

## `Dev` — Device Handle and Selector

There is no `dyn Backend` and no device ids separate from the devices: `Dev` is both the selector and the handle — a `Copy` enum naming the backend plus the hardware ordinal.

```rust,ignore
pub enum Dev {
    /// Auto-select: resolves to the first available device (Dev::all).
    Auto,
    /// CPU backend (runs on Pool::Host).
    C,
    /// CBLAS backend for AOT matmuls (runs on Pool::Host).
    Cblas,
    /// CUDA GPU with the given driver ordinal.
    Cuda(u16),
    /// Tenstorrent chip with the given id.
    TT(u16),          // feature = "tenstorrent"
    /// Vulkan physical device with the given index.
    Vulkan(u16),
    /// OpenCL device with the given index.
    OpenCL(u16),
    /// WGPU device with the given index.
    WGPU(u16),        // feature = "wgpu"
}
```

Trait objects would require downcasting to reach backend-specific functionality; the enum makes every method an exhaustive match instead. `Dev::Auto` is the scheduling placeholder — it is resolved by the scheduler before placement, so the device API never sees it. The `panic!` arms for `Auto` in methods like `info()` and `free_compute()` are guards on that invariant, not behavior; and the device's memory pool is always derived from the device via `Dev::pool()` — never the reverse.

## `Pool` — Memory

Memory belongs to pools, and pools are process-wide: they outlive any `Runtime` and are never deinitialized.

```rust,ignore
pub enum Pool {
    /// Host RAM. Shared by the C and CBLAS devices, which own no pool.
    Host,
    /// Disk-backed tensors (paths, not bytes).
    Disk,
    Cuda(u16),
    OpenCL(u16),
    Vulkan(u16),
    TT(u16),          // feature = "tenstorrent"
    WGPU(u16),        // feature = "wgpu"
}
```

Each variant owns its globals — one `Mutex<pool>` per ordinal (`Host` is a `OnceLock<Mutex<_>>`, `Disk` is a const `Mutex` with no `OnceLock` at all, the rest are `OnceLock<Vec<Mutex<_>>>` with one `Mutex<()>` per backend serializing first construction where it matters — cheap backends have no init lock). All globals live at the top of each `backend/*.rs` file, so the process-wide state is visible before any logic. The only lock takers are the device-API entry points (alloc/free/copy/compile/launch); the per-op tensor path never touches them.

## Lazy Initialization

There is no upfront backend-initialization phase. `Dev::all()` triggers lazy init of every backend; backends that are configured out, whose driver is missing, or whose hardware is absent contribute nothing to the returned list. This keeps startup free and makes device discovery idempotent:

```rust,ignore
impl Dev {
    pub fn all() -> Vec<Dev> {
        // C, CBLAS: single devices, included if init succeeds.
        // CUDA / TT / Vulkan / OpenCL / WGPU: one entry per detected device.
    }
}
```

Device selection happens at schedule time: `Auto` resolves to the first available device; a kernel is placed on a device whose pool has enough free bytes, skipping AOT-only devices (see below) for generic kernels.

## Device API

Every backend implements the same surface, dispatched by the enum:

```rust,ignore
pub fn compile(self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError>;
pub fn launch(self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<(), BackendError>;
pub fn launch_timed(self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError>;
```

plus pool operations (`alloc`, `free`, `retain`/`release`, `pool_to_host`, `pool_to_pool`) and `info()`. One backend-specific concept lives in the shared API: **`GwsDim`** — the per-axis global work size. Each gws dim is a `Group` index length, either a constant or a `Param`-backed dynamic length resolved from launch args; how it maps to a launch grid (CUDA grid, OpenCL global size, ...) is each backend's own business, derived from its `Op::Range` ops at compile time.

## Codegen: Mostly Trivial, by Construction

For most backends, codegen is a straight line: deSSA, then one linear pass over the kernel IR emitting target code. This is not an accident — the kernel IR is optimized until nothing searchable remains (see [the search test](./codegen.md#the-search-test-which-ir-holds-an-op)): everything the autotuner could enumerate over has already been decided, measured, and frozen before codegen runs.

What is left for a backend is exactly the **non-searchable, non-SSA residue**: registers, sync and placement concerns, launch-argument conventions, format configs — machine-shaped facts the value-level IR cannot and should not express. For simple targets that residue is small enough to absorb into the single lowering pass. For hardware whose compute model needs explicit physical state (Tenstorrent: three RISC-V threads, CB FIFOs, DST locks, SFPU LREGs), the residue becomes its own typed physical IR — see [Codegen and the Physical IR](./codegen.md).

## Current Backends

| Backend | Source | Target | Runtime |
|---------|--------|--------|---------|
| C | `c.rs` | C99 (compiled to .so) | Clang/GCC |
| CBLAS | `cblas.rs` | AOT matmul calls | Host BLAS (shares `Pool::Host`) |
| CUDA | `cuda.rs` | CUDA C → SASS, cuDNN for AOT matmul | CUDA driver via `libloading` |
| HIP | `hip.rs` | HIP | ROCm via `libloading` |
| OpenCL | `opencl.rs` | OpenCL C | OpenCL runtime via `libloading` |
| Vulkan | `vulkan.rs` | SPIR-V | Vulkan via `ash` crate |
| WGPU | `wgpu.rs` | SPIR-V | WGPU (feature: `wgpu`) |
| Tenstorrent | `tenstorrent.rs` | C++ RISC-V kernels | TT-Metalium (feature: `tenstorrent`) |

All backends except WGPU and Tenstorrent are compiled in by default; those two require `--features wgpu` / `--features tenstorrent`.

### CBLAS

The CBLAS device runs **only AOT (precompiled) matmul kernels** and cannot compile generic zyx kernels. `Dev::aot_only()` marks it, and generic kernel autotuning skips it — it competes only where a precompiled BLAS call is an option.

### CUDA: micro-batched submission

Each CUDA device is owned by a worker thread behind a command channel. Launches accumulate in a pending window (`MICRO_BATCH_WINDOW = 100`); the **batched-submission algorithm** then distributes the whole window over per-device hardware queues (CUDA streams, default 12, config `cuda.queues`) with stream-wait dependencies computed between them, and submits the window at once. Events exist only inside this worker — for queue dependencies and timing (`launch_timed`) — never as user-facing or runtime-level objects. Buffers carry a host-side refcount and are freed behind all their in-flight work.

## Configuration

Backends are configured through the process-wide config file — see [Configuration](./config.md). Each backend has its own section; missing sections mean that backend's defaults.
