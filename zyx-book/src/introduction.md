# Introduction

This book documents the internals of zyx — a machine learning library and compiler.

## Who This is For

This book is for developers who want to understand how zyx works under the hood, namely the architecture decisions.

## Two Execution Modes

Zyx operates in two modes:

- **Eager (default)**: Tensor operations fuse into kernels as you write them. When fusion is no longer possible, the pending kernel is compiled and executed when its value is needed. No separate realize step. Ideal for one-off work like data preprocessing and initialization.

- **Tape**: Wrap loops in a `Tape` to enable lazy graph building, autograd, and complex optimization (fusion variants compared by measured time, device placement, plan caching across structurally identical iterations). Think of it as `torch.compile`, but less strict. Kernels are shared across both modes.

## The Architecture at a Glance

Every tensor operation is appended somewhere; what happens next depends on the mode:

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
                    Kernelizer (batch fusion variants
                     + device placement)
                                  │
                                  ▼
                    Kernel IR ──► Opt ──► Codegen ──► Execute
```

Inside a tape, graph ops accumulate lazily. At realize time, the kernelizer builds fusion variants for the full graph, each candidate kernel is autotuned and measured, and extraction keeps the fastest path (including device placement). Autograd reuses the same graph ops — the tape prevents their deletion so the backward pass can traverse them.

The trade-off in tape mode is that evaluation is lazy — you must call `realize()` to trigger computation. But this laziness enables optimizations that eager execution cannot do: autograd without allocation of unnecessary intermediaries (there is no "save for backward"), fusion variants chosen by measured time, device placement, and plan caching across structurally identical iterations. Outside a tape, optimization is lighter-weight (greedy fusion only), which is basically the "numpy" mode. But you still get to run on all the devices and you get minimalistic fusion too, so for example matmul + relu will be just one kernel, as it should be.
