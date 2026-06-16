# Introduction

This book documents the internals of zyx — a machine learning library and compiler.

Unlike traditional ML frameworks that separate the eager execution graph from the autograd graph, zyx uses a **single unified graph** for everything. This design eliminates duplication, enables seamless kernel fusion, and keeps the implementation lean — tensors are only 4 bytes and the graph uses ~10 node types.

## Who This is For

This book is for developers who want to understand how zyx works under the hood: the architecture decisions, the optimization passes, the backend system, and how pieces fit together.

## The Architecture at a Glance

```text
User Code (Tensor API)
       │
       ▼
   Tensor Graph ─── Autograd (reverse-mode on same graph)
       │
       ▼
   Kernelizer (greedy fusion of graph nodes)
       │
       ▼
   Kernel IR (linked-list of ops)
       │
       ▼
   Optimization Passes (always-on + autotune)
       │
       ▼
   Backend Codegen (C, CUDA, OpenCL, Vulkan, WGPU, HIP)
```

Every tensor operation builds a graph node. When you call `.realize()` or `.item()`, the graph is traversed bottom-up, compatible nodes are fused into kernels, the kernels are optimized, and finally compiled to native code for the target device.

## Why This Design

Most deep learning libraries use two separate graphs:
1. A compute graph for eager execution
2. An autograd graph for backpropagation

Zyx uses **one graph** for both. This means:
- The autograd system doesn't need its own graph infrastructure — it reuses the same nodes
- Kernel fusion works across operation boundaries without special handling
- The implementation is debuggable (one graph to inspect, not two)
- Memory overhead is minimal: each graph node is ~16 bytes

The trade-off is that evaluation is lazy — you must call `realize()` to trigger computation. But this laziness enables optimizations that eager execution cannot: kernel fusion, dead code elimination, and cross-operation constant folding.
