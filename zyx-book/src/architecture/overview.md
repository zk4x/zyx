# Architecture Overview

The zyx pipeline transforms high-level tensor operations into device-specific machine code.

```text
Tensor API ──► Eager-ish: append to kernel ──► compile + execute
            │
            └── Tape: Graph (lazy) ──► Kernelizer ──► Kernel IR ──► Opt Passes ──► Backend Codegen
                                                                                    │
                                       Autotune (clone + evaluate)  └── deSSA + linear pass
```

## Pipeline Stages

### 1. Tensor API

The user creates tensors and applies operations:

```rust
# extern crate zyx;
# use zyx::{DType, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let x = Tensor::randn([1024, 1024], DType::F32)?;
let y = x.relu();
let z = y * 2.0;
# Ok(())
# }
```

Outside a `Tape`, each op is appended directly to the kernel that produced its inputs. When fusion is not possible, the kernel compiles and executes — no graph, no separate realize step. Inside a `Tape`, ops build graph ops lazily. Either way, each operation returns a lightweight `Tensor` handle.

### 2. The Graph

The graph opset was taken from tinygrad, with changes to make it even smaller. This is the minimal set of operations that can express ALL linear algebra operations and ALL PyTorch ops — by stacking these ops:

| Variant | Meaning |
|---------|---------|
| `Const` | A constant value baked into kernels |
| `Param` | A kernel parameter — one of the arguments passed to the compiled GPU kernel at launch |
| `Expand` | Broadcast a dimension |
| `Permute` | Transpose axes |
| `Reshape` | Change shape without changing data |
| `Pad` | Pad one axis with zeros |
| `Flip` | Reverse one or more axes |
| `Narrow` | Slice one axis |
| `Stack` | Build a vector from operands |
| `Reduce` | Sum, max, etc. |
| `Cast` | Change dtype |
| `Bitcast` | Reinterpret raw bits |
| `Unary` | Element-wise: relu, exp, sin, etc. |
| `Binary` | Element-wise: add, mul, etc. |
| `After` | Ordering for side-effecting ops |
| `ToDevice` | Move data between devices |
| `Contiguous` | Materialize a layout |
| `Kernel` | A compiled kernel boundary |
| `Custom` | An opaque custom kernel |

Tensor handles are `u32` (4 bytes) into the slab. Still small enough that 10,000 handles cost ~40 kB.

The graph is stored in a `Slab` — a dense array with free-list tracking. `TensorId` is a `u32` index into this slab, making tensor handles 4 bytes.

### 3. The Kernelizer

The kernelizer fuses compatible graph ops into kernels. Outside a tape it runs incrementally as ops are added; inside a tape it runs when `Tape::realize()` is called (dropping only cleans up graph state). The kernelizer uses heuristics to decide where kernel boundaries go — it's not a simple rule. A reduce op used by multiple downstream ops does not necessarily force a split. If two downstream ops are both expand ops, that may force fusion. Element-wise chains will almost always fuse into one kernel.

View operations (reshape, expand, permute, pad) are unfolded into index arithmetic in the kernel, becoming "free" — they don't create separate operations.

### 4. The Kernel IR

After unfolding, the DAG is converted into a **linear structure** — a doubly-linked list of ops stored in an arena (the `Slab<OpId, OpNode>`). Each `OpNode` is **32 bytes** stored inline in the arena — no `Box`, no vtables, no indirection.

OpId is a `u32` index into the slab — random access is O(1). The IR is SSA, except for loops and `Define` ops (which can be mutable).

The design goal of the small opset: optimizations are easy to write. If you understand the IR, you can add a new pass in an afternoon.

### 5. Optimization Passes

Optimization passes work on the linear IR. Kernels are cloned and each variant is evaluated separately — no egraphs. The cost function evaluates **thousands of variants per second**.

Optimization passes emit IR that is deliberately simple to lower to backend instructions. A backend just does deSSA + a single linear pass over the IR. No complex backend-specific lowering.

The autotune system explores the search space by:
1. Starting with the initial kernel
2. Applying one optimization variant
3. Hashing the kernel to detect duplicates
4. Evaluating with the cost function (or launching and timing)
5. Repeating by combining optimization sequences
6. Selecting the best variant

There is no fixed number of optimization passes zyx aims to have. The design goal was a small opset in the graph and IR so that passes are easy to write. More passes will be added over time.

### 6. Backend Codegen

Each backend converts the stabilized kernel IR into target code. Since the IR is designed for this, codegen is a straight line: deSSA, then one pass over the ops emitting instructions. No further optimizations, no complex lowering. Hardware whose compute model can't be expressed this way (Tenstorrent) gets an explicit **physical IR** — see [Codegen and the Physical IR](./codegen.md).

Backends are dispatch via enums (no `dyn Backend` — that would require downcasting, which is ugly in Rust):

```rust,ignore
pub enum Device {
    C(CDevice),
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
    Vulkan(VulkanDevice),
    WGPU(WGPUDevice),
    HIP(HIPDevice),
    Dummy(DummyDevice),
    // Tenstorrent, etc.
}
```

All backends are compiled into the library and selected at runtime.

### 7. Runtime and Scheduler (Current)

The scheduler picks a device based on free memory and compute capacity. Cross-device data transfers are scheduled as kernels; asynchronous execution is managed per backend (e.g. CUDA's micro-batched submission over hardware queues — see [Backend System](./backends.md)).

## Debugging the Pipeline

| `ZYX_DEBUG` | Output |
|-------------|--------|
| 1 | Devices + configuration, kernel launches |
| 2 | Egraph print (after realize) |
| 4 | Scheduler kernels — IR before linearization |
| 8 | Kernel IR after linearization + optimization |
| 16 | Generated assembly/code |
| 32 | Launch + memory movement |
| 64 | Alloc/dealloc |
| 128 | Kernel compilation |
| 256 | Autotune exploration |

## Key Design Decisions

- **One graph for everything** — autograd and computation share the same graph. No need to specify which tensors require gradients.
- **Symbolic dims everywhere** — the eager path and the graph path BOTH work with symbolic dimensions, always. Every symbolic dim bottoms out in a `Param { Variable }` scalar (`IDX_T`) whose value lives in a backend pool's variable slot, so any dim expression can be fully evaluated to a concrete constant at any time: walk the tree, fold `Const` leaves via `Constant::unary` / `Constant::binary`, and read variable slots at `Param { Variable }` leaves. Consumers must evaluate dims this way instead of fabricating placeholder values (`0`, `-1`, ...) where evaluation can produce the real value.
- **Inline ops** — all ops live in the arena as flat 32-byte entries. No `Box`, no vtables, no indirection. Passes allocate their own working data (hash maps, vecs) as needed.
- **Linear IR** — linked list of fixed-size ops. Optimizations traverse front-to-back or back-to-back.
- **Backend codegen is trivial** — the hard work is in the IR-level optimization passes.
- **Tape-scoped lazy graph** — ops inside a tape build graph ops instead of executing eagerly. Enables egraph fusion, device allocation search, and plan caching across iterations.
