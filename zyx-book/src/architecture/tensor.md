# The Tensor

The `Tensor` is the primary user-facing type in zyx. It is designed to be a lightweight handle — only **4 bytes**.

```rust,ignore
pub struct Tensor {
    pub(super) id: TensorId,  // u32 index into the graph slab
}
```

## Design Choices

### Why 4 Bytes?

Most ML frameworks have heavyweight tensor objects. PyTorch's `Tensor` is a `TensorImpl*` with shape, stride, dtype, device, storage, and autograd metadata — easily 100+ bytes. In zyx, all metadata lives in the **graph**, not the tensor handle. The tensor is just an index.

### Reference Counting

Tensors are reference-counted via the global `RT` (a `Mutex<Runtime>`):

```rust,ignore
impl Clone for Tensor {
    fn clone(&self) -> Self {
        RT.lock().retain(self.id);
        Tensor { id: self.id }
    }
}
```

If we used `Arc` instead, we would still need `Mutex` for the `Runtime` — `Tensor(id, Arc<Mutex<Runtime>>)`. The current approach avoids the `Arc` overhead and keeps `Tensor` at 4 bytes. Since every tensor operation already locks the runtime to append a graph node, there's no additional lock contention from reference counting.

### Execution

Outside a tape, each op is appended directly to the kernel that produced its inputs. When fusion is not possible, the kernel compiles and executes:

```rust
# extern crate zyx;
# use zyx::{DType, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let x = Tensor::randn([1024, 1024], DType::F32)?;
let y = x.relu();     // appended to x's kernel
let z = y.tanh();     // appended to same kernel
// at some point the kernel compiles and executes
# Ok(())
# }
```

Inside a tape, operations build graph nodes lazily and execute when the tape is realized (dropping only cleans up graph state — no computation is performed). The key insight: repeated graph patterns are automatically recognized and cached across structurally identical iterations.

### Construction Methods

Tensors can be created from:

```rust
# extern crate zyx;
# use zyx::{DType, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let t = Tensor::from([1.0f32, 2.0, 3.0]);
let t = Tensor::randn([1024, 1024], DType::F32)?;
let t = Tensor::uniform([1024, 1024], -1.0f32..1.0)?;
let ones = Tensor::ones([3, 3], DType::F32);
let zeros = Tensor::zeros([3, 3], DType::F32);
# Ok(())
# }
```

This also works from files on disk (lazy loading).

### The Immutability Rule

Tensors are **immutable** — there is no in-place mutation:

```rust
# extern crate zyx;
# use zyx::{DType, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let x = Tensor::randn([3, 3], DType::F32)?;
let x_plus_one = &x + 1.0;  // new tensor, no mutation
# Ok(())
# }
```

This makes autograd simpler (no mutation to track) and eliminates backpropagation errors from in-place modifications.
