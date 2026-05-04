# zyx - Python Bindings

[![PyPI version](https://badge.fury.io/py/zyx.svg)](https://pypi.org/project/zyx/)
[![License: LGPL 3.0](https://img.shields.io/badge/License-LGPL%203.0-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

**zyx** is a high-performance machine learning library for Python, powered by Rust. It features **lazy evaluation**, **automatic kernel fusion**, and **multi-backend support** (CPU, GPU via OpenCL/CUDA/WebGPU).

## Installation

```bash
pip install zyx
```

Requirements:
- Python 3.5+
- Linux x86_64 (more platforms coming soon)

## Quick Start

```python
import zyx

# Create tensors
x = zyx.Tensor([1.0, -2.0, 3.0])
y = zyx.Tensor([4.0, 5.0, 6.0])

# Lazy evaluation - operations build a graph
z = x.relu() + y

# Realize to compute the result
z.realize()
print(z.numpy())  # [5.0, 5.0, 9.0]

# Random tensors
a = zyx.Tensor.randn(3, 3)
b = zyx.Tensor.randn(3, 3)
c = a @ b  # Matrix multiplication
c.realize()
```

## Key Features

- **Lazy Evaluation** - Operations accumulate until `realize()`, reducing temporary allocations
- **Kernel Fusion** - Multiple operations compile into single optimized GPU kernels
- **Immutable Tensors** - No in-place modification errors common in other frameworks
- **Explicit Gradient Tape** - Control what's recorded, no `no_grad()` semantics needed
- **Arbitrary-Order Gradients** - Native support for 2nd, 3rd, and higher-order derivatives
- **Cross-Platform** - OpenCL (CPU/GPU), WebGPU, CUDA/ROCm backends

## Tensor Operations

### Creation

```python
import zyx

# From Python lists
t1 = zyx.Tensor([1, 2, 3])

# Random initialization
t2 = zyx.Tensor.randn(100, 100)
t3 = zyx.Tensor.rand((50, 50), dtype=zyx.DType.F32)

# Constant tensors
zeros = zyx.Tensor.zeros(10, 10)
ones = zyx.Tensor.ones(10, 10)
eye = zyx.Tensor.eye(5)

# Full tensor with specific value
filled = zyx.Tensor.full((3, 3), 2.5)

# With numpy arrays
import numpy as np
arr = np.array([[1, 2], [3, 4]])
t = zyx.Tensor(arr)
```

### Math Operations

```python
x = zyx.Tensor.randn(100, 100)

# Unary operations
y = x.relu()
y = x.sigmoid()
y = x.tanh()
y = x.gelu()
y = x.softmax(dim=-1)
y = x.log_softmax(dim=-1)

# Binary operations
a = zyx.Tensor.randn(100, 100)
b = zyx.Tensor.randn(100, 100)
c = a + b
c = a - b
c = a * b
c = a / b
c = a.matmul(b)  # Matrix multiplication

# Reduction operations
mean = x.mean(dim=0)
sum_val = x.sum(dim=1)
max_val = x.max(dim=-1)
min_val = x.min(dim=-1)
argmax = x.argmax(dim=-1)
argmin = x.argmin(dim=-1)

# Shape operations
y = x.reshape(10, 10)
y = x.transpose(0, 1)
y = x.permute(1, 0, 2)
y = x.squeeze()
y = x.unsqueeze(0)

# Realize to get numpy array
result = y.numpy()
```

## Automatic Differentiation

```python
import zyx

# Create a gradient tape
tape = zyx.GradientTape()

# Forward pass
x = zyx.Tensor([1.0, 2.0, 3.0])
w = zyx.Tensor([0.5, 0.5, 0.5])
y = (x * w).sum()

# Compute gradients
grads = tape.gradient(y, [w])
print(grads[0].numpy())  # [1.0, 2.0, 3.0]

# Higher-order gradients
tape2 = zyx.GradientTape()
y = x.relu().sum()
grads = tape2.gradient(y, [x])
```

## Neural Network Modules

```python
import zyx
import zyx.nn as nn

# Linear layer
linear = nn.Linear(in_features=128, out_features=64)

# Convolutional layer
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))

# Normalization layers
ln = nn.LayerNorm(normalized_shape=(128,))
bn = nn.BatchNorm(num_features=64)
gn = nn.GroupNorm(num_groups=8, num_channels=64)
rn = nn.RMSNorm(dim=128)

# Activation functions
x = zyx.Tensor.randn(10, 128)
y = x.relu()
y = x.gelu()
y = x.silu()

# Multihead attention
attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
query = zyx.Tensor.randn(32, 128, 512)  # batch, seq, embed
key = zyx.Tensor.randn(32, 128, 512)
value = zyx.Tensor.randn(32, 128, 512)
output, _ = attn(query, key, value)

# Transformer layers
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
```

## Optimizers

```python
import zyx
import zyx.nn as nn
import zyx.optim as optim

# Create model
model = nn.Linear(10, 5)

# Create optimizer
optimizer = optim.Adam(learning_rate=0.001)
# or: optim.SGD(learning_rate=0.01, momentum=0.9)
# or: optim.AdamW(learning_rate=0.001, weight_decay=0.01)
# or: optim.RMSprop(learning_rate=0.001)

# Training loop
tape = zyx.GradientTape()
x = zyx.Tensor.randn(32, 10)
target = zyx.Tensor.randn(32, 5)

# Forward
pred = model(x)
loss = ((pred - target) ** 2).mean()

# Backward
grads = tape.gradient(loss, model.get_params())
optimizer.update(model, grads)

# Realize all pending computations
zyx.Tensor.realize_all()
```

## Training Example

```python
import zyx
import zyx.nn as nn
import zyx.optim as optim

# Set random seed
zyx.manual_seed(42)

# Create model
class SimpleNet:
    def __init__(self):
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

    def get_params(self):
        return self.fc1.get_params() + self.fc2.get_params()

    def set_params(self, params):
        self.fc1.set_params(params[:2])
        self.fc2.set_params(params[2:])

model = SimpleNet()
optimizer = optim.Adam(learning_rate=0.001)

# Training loop
for epoch in range(10):
    tape = zyx.GradientTape()

    # Dummy data
    x = zyx.Tensor.randn(64, 784)
    y = zyx.Tensor.randint(0, 10, (64,))

    # Forward pass
    logits = model.forward(x)
    loss = logits.cross_entropy(y)

    # Backward pass
    grads = tape.gradient(loss, model.get_params())
    optimizer.update(model, grads)

    # Compute accuracy
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean()

    zyx.Tensor.realize_all()
    print(f"Epoch {epoch}, Loss: {loss.numpy():.4f}, Acc: {acc.numpy():.4f}")
```

## Data Types

```python
import zyx

# Supported dtypes
zyx.DType.F32    # 32-bit float
zyx.DType.F64    # 64-bit float
zyx.DType.F16    # 16-bit float
zyx.DType.BF16   # BFloat16
zyx.DType.I8     # 8-bit int
zyx.DType.I16    # 16-bit int
zyx.DType.I32    # 32-bit int
zyx.DType.I64    # 64-bit int
zyx.DType.U8     # 8-bit uint
zyx.DType.U16    # 16-bit uint
zyx.DType.U32    # 32-bit uint
zyx.DType.U64    # 64-bit uint
zyx.DType.Bool   # Boolean

# Create tensor with specific dtype
t = zyx.Tensor([1, 2, 3], dtype=zyx.DType.F32)
```

## Device Support

zyx automatically selects the best available backend:

- **OpenCL** - CPU and GPU support via POCL or vendor drivers
- **WebGPU** - Modern cross-platform GPU API
- **CUDA/ROCm** - NVIDIA and AMD GPU support

To check or set the runtime backend, use environment variables:

```bash
export ZYX_BACKEND=opencl    # Use OpenCL backend
export ZYX_BACKEND=wgpu      # Use WebGPU backend
export ZYX_BACKEND=cuda      # Use CUDA backend
```

## Debug Options

Enable debug output with the `ZYX_DEBUG` environment variable:

```bash
ZYX_DEBUG=1    # Print hardware devices
ZYX_DEBUG=2    # Print performance info
ZYX_DEBUG=4    # Print kernel scheduling
ZYX_DEBUG=8    # Print kernel IR
ZYX_DEBUG=16   # Print native assembly
```

Combine flags: `ZYX_DEBUG=18` (2 + 16) for perf + assembly output.

## Why zyx?

| Feature | zyx | PyTorch |
|---------|-----|---------|
| Gradient recording | Explicit `GradientTape` | Implicit, requires `no_grad()` |
| Tensor mutability | Immutable (no in-place errors) | Mutable (can cause back-prop failures) |
| Higher-order gradients | Arbitrary order natively | Supported but more complex |
| Kernel fusion | Automatic via lazy graph | Manual or via torch.compile |
| Disk I/O | Lazy loading parallel to compute | Typically blocking |

## License

LGPL-3.0-only - see LICENSE file for details.

## Links

- **GitHub**: https://github.com/zk4x/zyx
- **Documentation**: https://docs.rs/zyx (Rust API)
- **Book**: https://zk4x.github.io/zyx/

## Status

**Experimental** - API is stabilizing, performance under active optimization. Not production-ready yet.
