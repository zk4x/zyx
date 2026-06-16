# Architecture Overview

The zyx pipeline transforms high-level tensor operations into device-specific machine code.

```text
Tensor API ──► Graph ──► Kernelizer ──► Kernel IR ──► Opt Passes ──► Backend Codegen
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

These calls do not compute anything. Each operation appends a node to the graph and returns a lightweight `Tensor` handle.

### 2. The Graph

The graph opset was taken from tinygrad, with changes to make it even smaller. This is the minimal set of operations that can express ALL linear algebra operations and ALL PyTorch ops — by stacking these nodes:

| Variant | Meaning |
|---------|---------|
| `Const` | A constant value baked into kernels |
| `Leaf` | A tensor stored on device |
| `Expand` | Broadcast a dimension |
| `Permute` | Transpose axes |
| `Reshape` | Change shape without changing data |
| `Pad` | Pad with zeros |
| `Reduce` | Sum, max, etc. |
| `Cast` | Change dtype |
| `Unary` | Element-wise: relu, exp, sin, etc. |
| `Binary` | Element-wise: add, mul, etc. |

Each node is ~16 bytes, plus a 4-byte reference count and slab metadata overhead. Still small enough that 10,000 nodes cost ~200 KB.

The graph is stored in a `Slab` — a dense array with free-list tracking. `TensorId` is a `u32` index into this slab, making tensor handles 4 bytes.

### 3. The Kernelizer

When `realize()` is called, the kernelizer traverses the graph bottom-up and fuses compatible nodes into kernels. The kernelizer uses heuristics to decide where kernel boundaries go — it's not a simple rule. A reduce node used by multiple downstream nodes does not necessarily force a split. If two downstream nodes are both expand ops, that may force fusion. Element-wise chains will almost always fuse into one kernel.

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

Each backend converts the stabilized kernel IR into target code. Since the IR is designed for this, codegen is a straight line: deSSA, then one pass over the ops emitting instructions. No further optimizations, no complex lowering.

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

The scheduler picks a device based on free memory and compute capacity. It handles cross-device data transfers and tracks async execution via events.

**Note**: A new scheduler is under development in `search.rs` and `search2.rs`. It will use an e-graph-like budget-guided exhaustive fusion enumeration, including costs for memory movement. The current `schedule.rs` will be replaced with something much more powerful.

## Debugging the Pipeline

| `ZYX_DEBUG` | Output |
|-------------|--------|
| 1 | Backend selection and hardware info |
| 2 | Performance info during realize |
| 4 | Kernels generated by the kernelizer |
| 8 | Kernel IR (before GPU execution) |
| 16 | Generated backend code (PTX, OpenCL C, etc.) |
| 32 | Autotune exploration |

## Key Design Decisions

- **One graph for everything** — autograd and computation share the same graph. No need to specify which tensors require gradients.
- **Inline ops** — all ops live in the arena as flat 32-byte entries. No `Box`, no vtables, no indirection. Passes allocate their own working data (hash maps, vecs) as needed.
- **Linear IR** — linked list of fixed-size nodes. Optimizations traverse front-to-back or back-to-back.
- **Backend codegen is trivial** — the hard work is in the IR-level optimization passes.
- **Explicit GradientTape** — prevents graph node deletion instead of building a separate graph.
