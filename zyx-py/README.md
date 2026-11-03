# zyx — Python bindings

[![PyPI version](https://badge.fury.io/py/zyx-py.svg)](https://pypi.org/project/zyx-py/)
[![License: LGPL 3.0](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

**zyx** is a machine learning library that runs on your hardware: a lazy tensor
library with automatic kernel fusion and autotuning, written in Rust, with
backends for CUDA, Vulkan, OpenCL, CPU (C and CBLAS), WGPU, and Tenstorrent.

The whole install is **2 MB and has no dependencies** — there is no bundled
CUDA runtime, no BLAS to link, no toolchain to install. Backends are loaded
from the drivers already on the machine.

## Install

```bash
pip install zyx-py
```

> **The package is `zyx-py`, not `zyx`.** `pip install zyx` installs an
> unrelated project from PyPI. The import name is still `zyx`.

```bash
pip install zyx-py
```

```python
import zyx
```

Requirements: **Python 3.8+**. Wheels are published for Linux `x86_64` and
macOS `arm64` for CPython 3.8–3.14. Other platforms (Windows, Linux `aarch64`,
macOS Intel, PyPy) need a source build, which requires a Rust toolchain.

## Quick start

Tensors are lazy. Building an expression records a graph; the kernel is
compiled and run when a value is actually needed — `.numpy()`, `.item()`, or a
`Tape` realization.

```python
import zyx

x = zyx.Tensor([1.0, -2.0, 3.0])
y = zyx.Tensor([4.0, 5.0, 6.0])

z = x.relu() + y

print(z.is_realized())   # False — only the graph exists so far
print(z.numpy())         # [5. 5. 9.] — compile + execute happens here
print(z.is_realized())   # True
```

Nothing is computed until the value is read, so a chain of ops becomes one
fused kernel rather than one launch per op.

```python
a = zyx.Tensor.randn(512, 512)
b = zyx.Tensor.randn(512, 512)

c = ((a @ b) + 1.0).relu()   # one fused kernel, not three
print(c.numpy().shape)       # (512, 512)
```

### Eager mode vs `Tape`

Outside a `Tape`, ops fuse with heuristics and execute when read. That is the
right mode for one-off computations.

Inside a `Tape`, the graph is kept for the duration of the scope, which enables
autograd, autotuning over the whole block, and reuse of the compiled plan
across iterations. That is the right mode for a training or inference loop.

## Automatic differentiation

`Tape` records operations on the tensors you add to it. Gradients are
themselves tensors and are computed by the same graph machinery, so
higher-order derivatives are just another `gradient` call.

```python
import zyx

x = zyx.Tensor([1.0, 2.0, 3.0])
w = zyx.Tensor([0.5, 0.5, 0.5])

tape = zyx.Tape()
tape.add(w)                  # tensors to differentiate with respect to

loss = (x * w).sum()
(w_grad,) = tape.gradient(loss, [w])

# Gradients are lazy like everything else: realize them to read the values.
tape.realize([loss, w_grad])
print(w_grad.numpy())        # [1. 2. 3.]
```

`tape.realize([...])` executes exactly the tensors you pass. Reading a tensor
that was not realized raises `GraphTensorNotRealized`, so pass every value you
intend to read.

## Tensor operations

```python
import zyx

# Creation
a = zyx.Tensor([1, 2, 3])                    # from a list or numpy array
b = zyx.Tensor.randn(4, 8)                   # standard normal
c = zyx.Tensor.zeros(4, 8)
d = zyx.Tensor.ones(4, 8)
e = zyx.Tensor.eye(4)
f = zyx.Tensor.full(4, 8, fill_value=2.5)    # shape args, then fill_value
g = zyx.Tensor.arange(0, 10)

# Elementwise
h = b.relu()
h = b.sigmoid()
h = b.tanh()
h = b.gelu()
h = b.swish()                                # SiLU is spelled swish
h = b.softmax(dim=-1)

# Binary
h = b + c
h = b * c
h = b @ c.transpose(0, 1)                    # same as b.matmul(c.transpose(0, 1))

# Reductions
h = b.sum(dim=0)
h = b.mean(dim=1)
h = b.max(dim=-1)
h = b.argmax(dim=-1)

# Shape
h = b.reshape(8, 4)
h = b.transpose(0, 1)
h = b.unsqueeze(0)
h = b.squeeze()

# Dtype
h = zyx.Tensor([1, 2, 3]).cast(zyx.DType.F32)
```

`Tensor.softmax` takes `dim`; `Tensor.argmax` takes `dim=None` (no argument
returns the flat index of the maximum).

The full method list is on `zyx.Tensor`; 150+ ops are exposed.

## Neural network modules

```python
from zyx import nn

linear = nn.Linear(in_features=128, out_features=64)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
ln = nn.LayerNorm(normalized_shape=(128,))
bn = nn.BatchNorm(num_features=64)
gn = nn.GroupNorm(num_groups=8, num_channels=64)
rn = nn.RMSNorm(dim=128)
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
block = nn.TransformerEncoderLayer(d_model=512, nhead=8)
```

Layers expose their parameters directly (`linear.weight`, `linear.bias`) and
are callable with `forward(x)`.

Both import forms work:

```python
from zyx import nn          # or:
import zyx.nn as nn
```

## Device selection

zyx is lazy about devices too. `Tensor.device()` reports `Dev::Auto` — **the
device is not chosen until the graph is executed**, at which point zyx picks the
first available backend and can also split a graph across several of them. This
is deliberate: it is what lets the same code run on a CUDA box, a Vulkan box,
and a CPU-only machine without changes, and what lets the autotuner compare
backends for a block instead of inheriting a choice made at tensor construction.

```python
import zyx

t = zyx.Tensor.randn(4, 4)
print(t.device())                        # Dev::Auto — undecided

pinned = t.to(zyx.Dev.cuda(0))           # pin to a concrete device
print(pinned.device())                   # Dev::Cuda(0)

pinned.numpy()                           # [4, 4] float array
```

Available selectors: `Dev.cpu()`, `Dev.cblas()`, `Dev.cuda(id)`,
`Dev.opencl(id)`, `Dev.vulkan(id)`, `Dev.dummy()`. `Dev.auto()` resolves the
pending choice.

### Configuring backends

Backends can be disabled, enabled, or restricted through
`~/.config/zyx/config.json` (or `$XDG_CONFIG_HOME/zyx/config.json`):

```json
{
  "cuda":        {"device_ids": [0], "cudnn": true},
  "opencl":      {"platform_ids": []},
  "vulkan":      {"device_ids": []},
  "tenstorrent": {"device_ids": []}
}
```

- `null` (or absent) = use every available device
- `[]` = disable that backend
- `[0]` = use device 0 only

CUDA, HIP, OpenCL, Vulkan, and Tenstorrent are always compiled in and look for
their driver libraries at runtime. WGPU is a cargo feature.

## Data types

```python
import zyx

zyx.DType.F64    # 64-bit float
zyx.DType.F32    # 32-bit float
zyx.DType.F16    # 16-bit float
zyx.DType.BF16   # bfloat16
zyx.DType.F8E4M3 # fp8, 4-bit exponent / 3-bit mantissa
zyx.DType.F8E5M2 # fp8, 5-bit exponent / 2-bit mantissa
zyx.DType.I8     # signed integers
zyx.DType.I16
zyx.DType.I32
zyx.DType.I64
zyx.DType.U8     # unsigned integers
zyx.DType.U16
zyx.DType.U32
zyx.DType.U64
zyx.DType.Bool

t = zyx.Tensor([1, 2, 3]).cast(zyx.DType.F32)
```

A backend that cannot represent a dtype fails the operation explicitly rather
than silently downcasting.

## Debugging

`ZYX_DEBUG` is a bitmask:

```bash
ZYX_DEBUG=1    # print hardware devices and configuration
ZYX_DEBUG=2    # print the e-graph after realize
ZYX_DEBUG=4    # print kernels created by the scheduler
ZYX_DEBUG=8    # print kernels in intermediate representation
ZYX_DEBUG=16   # print kernels in native assembly/code
ZYX_DEBUG=32   # print kernel launches (program id, grid, args)
ZYX_DEBUG=64   # print memory allocation and deallocation
ZYX_DEBUG=128  # print kernel compilation
ZYX_DEBUG=256  # skip the beam search: compile seeds without autotuning
```

Combine by summing (e.g. `ZYX_DEBUG=24` = IR + assembly).

`ZYX_DRY_RUN=1` skips all device launches while still compiling, which is useful
for inspecting generated code without hardware.

## Optimizers — not usable from Python yet

`zyx.optim` exposes `SGD`, `Adam`, `AdamW`, and `RMSprop`, and
`optimizer.update(model, gradients)` exists, but it requires the model object to
implement `get_params()` and `set_params(list)`, which the built-in `nn` layers
do not. Until that is wired up, calling `update()` raises:

```
AttributeError: 'builtins.Linear' object has no attribute 'get_params'
```

Gradient computation itself works (see above); only the optimizer step is
missing. Rust users have the full path via the `Module` trait.

## Rust API

This package is bindings over the `zyx` crate. The Rust API is the complete one:
[docs.rs/zyx](https://docs.rs/zyx) and the [book](https://zk4x.github.io/zyx/).

## Status

Experimental. The API is stabilizing and performance is under active
optimization. Reported numbers should be taken as in-progress; see
`BENCHMARKS.md` in the repository.

## License

LGPL-3.0-only WITH Classpath-exception-2.0.

## Links

- **PyPI**: https://pypi.org/project/zyx-py/
- **GitHub**: https://github.com/zk4x/zyx
- **Rust docs**: https://docs.rs/zyx
- **Book**: https://zk4x.github.io/zyx/
