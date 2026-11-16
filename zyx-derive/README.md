# zyx-derive

This crate contains procedural macros for zyx.

Macro Module automatically implements IntoIterator<Item = &Tensor>
for your module, so that you can use it in backpropagation and save it to disk.
```rust
use zyx::Tensor;
use zyx_derive::Module;

#[derive(Module)]
struct MyNet {
    b: Tensor,
    w: Tensor,
}

impl MyNet {
    fn forward(&self, x: Tensor) -> Tensor {
        x.dot(&self.w).unwrap() + &self.b
    }
}
```

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

For more details, there is a [book](https://zk4x.github.io/zyx).

Macro gguf reads a model's hyperparameters from a GGUF checkpoint, so the
file is the single source of truth and no `config.json` sits beside it. Every
field is read from `<general.architecture>.<key>`; a field without an attribute
uses its own name, `#[gguf(skip)]` fields come from the trailing constructor
parameters, and `#[gguf(group = "rope.scaling", gate = "rope.scaling.type")]`
builds an `Option<Nested>` from a key group behind a gate key. `default =
<value>` sets a field's value in the `impl Default` the derive also emits
(fields without one use `Default::default()`; a skipped field must declare
one).
```rust
use zyx::GGUFMetadata;
use zyx_derive::gguf;

#[derive(gguf)]
struct Config {
    #[gguf("embedding_length", default = 32)]
    n_embd: i64,
    #[gguf("attention.head_count_kv", default = Some(2))]
    n_kv_head: Option<i64>,
    #[gguf("rope.freq_base", default = 10000.0)]
    theta: f64,
    #[gguf(skip, default = 4096)]
    max_context: usize,
}

fn config(metadata: &GGUFMetadata, max_context: usize) -> Result<Config, zyx::ZyxError> {
    Config::from_gguf(metadata, max_context)
}
```
