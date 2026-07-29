# Adding a new backend to zyx

This document describes what a backend needs to provide and how the compiler
adapts to your hardware.

## Philosophy

Zyx decomposes ML computation into simple kernel-level operations,
then rewrites and autotunes based on your hardware's capabilities.
A vendor provides:

1. A `DeviceInfo` describing what the hardware supports (capability matrix)
2. ~500 lines of codegen mapping kernel ops to target code

Everything else — optimization, op decomposition, autotuning — is the compiler's job.

## Initialization

Backends return any number of memory pools and devices.

## Memory pools

Each memory pool must be able to:
- Load data from CPU (`&[u8]`)
- Store data to CPU (`&mut [u8]`)
- Copy to another memory pool

For transfers between different backends (e.g., CUDA GPU to Intel GPU),
zyx routes through CPU memory unless special override is provided.

## Devices

Each device has a set of queues. Devices compile Kernel IR into machine code.

## Programs (kernels)

Programs are compiled from Kernel IR. After lowering, the IR contains a small set
of operations that backends must handle:

| Op | Required | Notes |
|----|----------|-------|
| `Define` | Yes | Declares a global/local/register variable |
| `Const` | Yes | Constant value |
| `Cast` | Yes | Type conversion |
| `Index` | Yes | Thread/block index (gidx, gidy, lidx, etc.) |
| `Load` | Yes | Load from memory with pointer + offset |
| `Store` | Yes | Store to memory with pointer + offset |
| `Unary` | Yes | Element-wise (neg, exp, sqrt, sin, etc.) |
| `Binary` | Yes | Element-wise (add, mul, sub, div, pow, etc.) |
| `Loop` / `EndLoop` | Yes | Bounded loop with loop variable |
| `Barrier` | No | Synchronization barrier (optional) |
| `If` / `EndIf` | No | Conditional branch (optional) |
| `Mad` | No | Fused multiply-add (optional optimization) |
| `Wmma` | No | Warp matrix multiply-accumulate (optional) |
| `Vectorize` / `Devectorize` | No | Explicit vectorization (optional) |

## The DeviceInfo capability matrix

Backends report what the hardware supports through `DeviceInfo`:

```rust
pub struct DeviceInfo {
    pub compute: u128,                      // Device FLOPs estimate
    pub max_global_work_dims: Vec<Dim>,     // Max grid dimensions
    pub max_local_threads: Dim,             // Max threads per block
    pub max_local_work_dims: Vec<Dim>,      // Max block dimensions
    pub preferred_vector_size: u8,          // Preferred vector width (bytes)
    pub local_mem_size: Dim,                // SRAM / shared memory size
    pub max_register_bytes: Dim,            // Register file per thread
    pub tensor_cores: bool,                 // Has tensor core hardware
    pub warp_size: u16,                     // Warp / wavefront size
    pub dtype_capability: [DTypeCapability; N_DTYPES],  // Per-dtype op support
    pub has_native_exp2: bool,              // Whether backend has exp2 natively
    pub supported_vec_lens: Vec<u8>,        // Supported vector load/store lengths
}
```

`DTypeCapability` is a bitfield of ~30 operations per dtype (add, sub, mul, div,
exp, sin, etc.). A backend says "F16 supports ADD and MUL but not SIN" and the
compiler adapts.

Examples from real backends:

- **CUDA**: starts from `all()` then removes ops per dtype — no BF16 before
  compute capability 8, no native LOG2/SIN/SQRT for F16, no LOG2/SIN/SQRT for F64
- **C backend**: reports `all()` for every dtype; it can emulate everything in software
- **OpenCL**: disables BF16 and optionally F16 entirely

## How the compiler uses DeviceInfo

The autotuner and optimizer passes are all `DeviceInfo`-aware:

- **Vectorizer** checks `supported_vec_lens` and `preferred_vector_size`
- **Local reduce** checks `local_mem_size` and `max_local_threads`
- **Split loops** checks local memory constraints
- **Cost model** uses `compute`, `warp_size`, memory sizes
- **exp/ln conversion** (in `autotune.rs`): if `has_native_exp2` is false,
  `exp` → `exp2`, `ln` → `log2`; if true, the reverse. This lets backends
  choose which variant they have hardware support for.

Optimization passes run identically for every backend but tune to the
hardware's reported constraints. The autotuner searches combinations of
9 optimizations (reassociation, split global-to-local, thread coarsening,
register blocking, local reduce, split loop, pad index, vectorize, merge loops)
and selects the best configuration by cost model and timing.

## Datatypes

Devices can support any subset of datatypes. If a kernel uses an unsupported
dtype, the backend should return a compilation error and zyx falls back to
a different device or reports the error.

## Custom kernels via fused ops

If a backend has a fused op (e.g., WMMA on tensor cores), zyx supports it as
an optional kernel-level op. The `fuse_mma` pass in `kernel/mma.rs` patterns
multiply+add into WMMA ops. These are declarative — backends declare what fused
ops they support in the capability matrix; the compiler discovers when to use
them.

## Verifying correctness

**No optimization is needed for tests to pass.** All tests must pass with all
optimizations disabled, and all tests must pass no matter which sequence of
optimizations (including empty) is applied. If any sequence breaks correctness,
the optimization that produced invalid IR is buggy.

## Proven backends

| Backend | Approximate time | Notes |
|---------|-----------------|-------|
| C | 2 days | Software CPU, Clang/GCC codegen |
| SPIR-V | 2 days | Cross-API GPU (Vulkan) |
| PTX | 4 days | Direct NVIDIA codegen |
| Tenstorrent | — | Non-SIMT NPU, 32x32 tile constraints |

Tenstorrent is the most exotic — 32x32 tile constraints,
no real branching. Since zyx works there, it will likely work on your hardware.

## Comparison

| System | Real backends | Time to add | Vendor work |
|--------|--------------|-------------|-------------|
| XLA | CPU, GPU, TPU | Months | StreamExecutor + Compiler + Executable |
| TVM | CPU, GPU, FPGA | Months | Schedule primitives per backend |
| Triton | NVIDIA, AMD, Intel GPU | Months | MLIR dialect lowering pipeline |
| IREE | CPU, CUDA, Vulkan, Metal | Months | HAL driver + compiler target |
| Mojo/MAX | NVIDIA, AMD, CPU | Quarters | LLVM backend + kernel library |
| **Zyx** | **C, CUDA, PTX, SPIR-V, Vulkan, OpenCL, Tenstorrent** | **Days** | **DeviceInfo + ~500 LOC codegen** |

## Final notes

This is pretty much all that is needed to add new backends to zyx. If you have
any problems adding support for your device, please do not hesitate to create
an issue on [github.com/zk4x/zyx](https://github.com/zk4x/zyx), we're happy to
assist you. Hardware support is the second primary goal of zyx (first one is
correctness, obviously).
