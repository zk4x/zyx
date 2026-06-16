# The Graph

The graph is the heart of zyx. Every operation a user performs builds a node in this graph. The graph is shared between computation and autograd — there is only one.

## Data Structure

The graph is stored in the `Runtime`:

```rust,ignore
pub struct Runtime {
    pub graph: Graph,
    // ...
}

pub struct Graph {
    pub nodes: Slab<TensorId, (u32, Node)>,
    pub gradient_tape: Option<Set<TensorId>>,
    pub shapes: Map<TensorId, Box<[Dim]>>,
    // ...
}
```

### Slab Allocator

The `Slab<TensorId, (u32, Node)>` is a dense array with free-list tracking. Insertion is O(1) amortized, and iteration is cache-friendly. Each node is a `(reference_count, Node)` pair stored inline.

A `TensorId` is just a `u32` index into this slab. This is why `Tensor` is 4 bytes — it's an index, not a pointer.

### Node Types

The graph opset was derived from tinygrad, with changes to make it even smaller. By stacking these types, zyx can express ALL linear algebra operations and ALL PyTorch ops:

```rust,ignore
pub enum Node {
    Const { value: Constant },
    Leaf { dtype: DType },
    Expand { x: TensorId },
    Permute { x: TensorId },
    Reshape { x: TensorId },
    Pad { x: TensorId },
    Reduce { x: TensorId, rop: BOp },
    Cast { x: TensorId, dtype: DType },
    Unary { x: TensorId, uop: UOp },
    Binary { x: TensorId, y: TensorId, bop: BOp },
    ToDevice { x: TensorId, device: DeviceId },
    Custom(Box<CustomKernel>),
}
```

## Lifecycle with GradientTape

Without a gradient tape, realized graph nodes are immediately released:

```text
realize(x) → compute x → replace x's node with Leaf → release x's inputs
```

With a gradient tape alive, realized nodes that the tape references are preserved:

```text
realize(x) with tape → compute x → if tape.contains(x), keep node → else replace with Leaf
```

This is how autograd works on the same graph: the gradient tape prevents node deletion, so when you later call `tape.gradient()`, it can traverse the graph backward.

## Graph Size

The graph is designed to stay small. With ~16 bytes per node + 4 bytes reference count, a training iteration with 10,000 operations costs ~200 KB. When the tape is dropped at the end of each iteration, the graph shrinks back to baseline.

## Debugging the Graph

Inspect the graph at runtime:

```rust,ignore
Tensor::plot_graph(&[&output], "graph")?;
```

Or with environment variables:

```bash
ZYX_DEBUG=8 cargo run  # prints kernel IR during compilation
```
