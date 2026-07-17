# Introduction

This book documents the internals of zyx — a machine learning library and compiler.

Unlike traditional ML frameworks that separate the eager execution graph from the autograd graph, zyx uses a **single unified graph** — but only inside a `Tape` scope. Outside a tape, there is no graph: ops go directly to the kernelizer. Inside a tape, the graph is shared between computation and autograd, eliminating duplication and keeping the implementation lean — tensors are only 4 bytes and the graph uses ~10 node types.

## Who This is For

This book is for developers who want to understand how zyx works under the hood: the architecture decisions, the optimization passes, the backend system, and how pieces fit together.

## Two Execution Modes

Zyx operates in two modes:

- **Eager-ish (default)**: Tensor operations fuse into kernels as you write them. When no more fusion is possible, the kernel executes immediately. No separate realize step. Ideal for one-off work like data preprocessing and initialization.

- **Tape**: Wrap loops in a `Tape` to enable lazy graph building, autograd, and complex optimization (egraph-based fusion comparison, device allocation search, plan caching across structurally identical iterations). Think of it as `torch.compile`, but less strict. Kernel caching (compiled program reuse) is shared across both modes.

## The Architecture at a Glance

Every tensor operation creates a graph node. What happens next depends on the mode:

**Eager-ish (default) — direct fusion, no graph:**

```text
Tensor op ──► append to kernel ──► compile + execute (if fusion not possible)
```

Each op is appended directly to the kernel that produced its inputs. When a new op can't fuse (different device, incompatible data flow), the pending kernel compiles and executes. No graph, no separate realize step.

**Tape — lazy graph, egraph exploration:**

```text
Tensor op ──► graph node (accumulates) ──► Autograd (reverse-mode on same graph)
                                  │
                                  ▼
                          Tape::realize() / drop
                                  │
                                  ▼
                    Kernelizer (batch, egraph explores
                     fusion variants + device allocations)
                                  │
                                  ▼
                    Kernel IR ──► Opt ──► Codegen ──► Execute
```

Inside a tape, graph nodes accumulate lazily. At realize/drop time, the kernelizer processes the full graph while egraph exploration tries different fusion schemes and device allocations, selecting the fastest. Autograd reuses the same graph nodes — the tape prevents their deletion so the backward pass can traverse them.

## Why This Design

Most deep learning libraries use two separate graphs:
1. A compute graph for eager execution
2. An autograd graph for backpropagation

Zyx uses **one graph** for both. This means:
- The autograd system doesn't need its own graph infrastructure — it reuses the same nodes
- Kernel fusion works across operation boundaries without special handling
- The implementation is debuggable (one graph to inspect, not two)
- Memory overhead is minimal: each graph node is ~16 bytes

The trade-off in tape mode is that evaluation is lazy — you must call `realize()` or drop the tape to trigger computation. But this laziness enables optimizations that eager execution cannot: egraph exploration of fusion variants, device allocation search, and plan caching across structurally identical iterations. Outside a tape, optimization is lighter-weight (greedy fusion only), which is appropriate for one-off ops. Kernel caching (compiled program reuse) is shared across both modes.
