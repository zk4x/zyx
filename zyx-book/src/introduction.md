# Introduction

This book documents the internals of zyx — a machine learning library and compiler.

## Who This is For

This book is for developers who want to understand how zyx works under the hood, namely the architecture decisions.

## Two Execution Modes

Zyx operates in two modes:

- **Eager (default)**: Tensor operations fuse into kernels as you write them. When no more fusion is beneficial (based on a heuristic), the kernel executes immediately. No separate realize step. Ideal for one-off work like data preprocessing and initialization.

- **Tape**: Wrap loops in a `Tape` to enable lazy graph building, autograd, and complex optimization (egraph-based fusion comparison, device allocation search, plan caching across structurally identical iterations). Think of it as `torch.compile`, but less strict. Kernels are shared across both modes.

## The Architecture at a Glance

Every tensor operation creates a graph op. What happens next depends on the mode:

**Eager — direct fusion, no graph:**

```text
Tensor op ──► append to kernel ──► compile + execute
```

Each op is appended directly to the kernel that produced its inputs. When a new op isn't fused, the pending kernel compiles and executes. No graph, no separate realize step. Everything is automated and runs in the background, user can't really observe it and shouldn't care about it.

In eager mode, the fusion heuristic is conservative, won't overfuse as to not break the premise of the eager mode.

**Tape — lazy graph, egraph exploration:**

Tape is used for repeated computation. The opset and kernels are the same, it's just when you use tape, you tell zyx that you are fine with being explicit (manual realize) and deferred (computation-wise) while you want to reap benefits of more optimizations (which comes from better fusion due to deferred computtation) as well as autograd. Additionally zyx takes this as a signal that it can take more time optimizing the DAG (directed acyclic graph) of tensor ops.

```text
Tensor op ──► graph op (accumulates) ──► Autograd (reverse-mode on same graph)
                                  │
                                  ▼
                           Tape::realize()
                                  │
                                  ▼
                    Kernelizer (batch, egraph explores
                     fusion variants + device allocations)
                                  │
                                  ▼
                    Kernel IR ──► Opt ──► Codegen ──► Execute
```

Inside a tape, graph ops accumulate lazily. At realize time, the kernelizer processes the full graph while egraph exploration tries different fusion schemes and device allocations, selecting the fastest. Autograd reuses the same graph ops — the tape prevents their deletion so the backward pass can traverse them.

The trade-off in tape mode is that evaluation is lazy — you must call `realize()` to trigger computation. But this laziness enables optimizations that eager execution cannot do: autograd without allocation of unnecessary intermediaries (there is no "save for backward"), egraph exploration of fusion variants, device allocation search, and plan caching across structurally identical iterations. Outside a tape, optimization is lighter-weight (greedy fusion only), which is basically the "numpy" mode. But you still get to run on all the devices and you get minimalistic fusion too, so for example matmul + relu will be just one kernel, as it should be.
