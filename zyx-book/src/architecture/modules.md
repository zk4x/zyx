# Module System

The module system provides a way to group tensors (parameters) into neural network layers. It's defined by the `Module` trait and powered by `#[derive(Module)]`.

## The Module Trait

```rust,ignore
pub trait Module {
    fn iter(&self) -> impl Iterator<Item = &Tensor>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tensor>;
    fn iter_tensors(&self) -> impl Iterator<Item = (String, &Tensor)>;
    fn iter_tensors_mut(&mut self) -> impl Iterator<Item = (String, &mut Tensor)>;
    fn set_params(&mut self, params: &mut HashMap<String, Tensor>);

    fn save(&self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
    fn save_numpy(&self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
}
```

## #[derive(Module)]

The `#[derive(Module)]` macro (from `zyx-derive`) generates the trait implementation, collecting all tensor fields recursively. This works with nested modules:

```rust
# extern crate zyx;
# extern crate zyx_nn;
# use zyx::Tensor;
# use zyx_nn::Module;
#[derive(Module)]
struct Linear {
    weight: Tensor,
    bias: Tensor,
}

#[derive(Module)]
struct MLP {
    layer1: Linear,
    layer2: Linear,
    layer3: Linear,
}
# fn main() {}
```

## Using Modules

```rust
# extern crate zyx;
# extern crate zyx_nn;
# extern crate zyx_optim;
# use zyx::{DType, Module, Tape, Tensor, ZyxError};
# use zyx_nn::Linear;
# use zyx_nn::Module;
# use zyx_optim::SGD;
#[derive(Module)]
struct SimpleNet {
    linear1: Linear,
    linear2: Linear,
}
# impl SimpleNet {
#     fn new() -> Result<Self, ZyxError> {
#         Ok(Self {
#             linear1: Linear::new(4, 8, true, DType::F32)?,
#             linear2: Linear::new(8, 2, true, DType::F32)?,
#         })
#     }
#     fn forward(&self, x: &Tensor) -> Tensor {
#         let x = self.linear1.forward(x).unwrap().relu();
#         self.linear2.forward(&x).unwrap()
#     }
# }

fn train_step(model: &mut SimpleNet, optim: &mut SGD, x: &Tensor, target: &Tensor) -> Result<f32, ZyxError> {
    let tape = Tape::new(&*model)?;
    let output = model.forward(x);
    let loss = output.mse_loss(target)?;
    let grads = tape.gradient(&loss, &*model);
    optim.update(&mut *model, grads);
    tape.realize(model.iter().chain([&loss]))?;
    // tape drop cleans up graph state; next iteration reuses cached plan
    Ok(loss.item())
}
# fn main() -> Result<(), ZyxError> {
#     let mut model = SimpleNet::new()?;
#     let mut optim = SGD::default();
#     let x = Tensor::randn([2, 4], DType::F32)?;
#     let target = Tensor::randn([2, 2], DType::F32)?;
#     let _ = train_step(&mut model, &mut optim, &x, &target)?;
#     Ok(())
# }
```

The `tape.gradient(&loss, &model)` call passes the model itself as the sources. The autograd system iterates over `model.iter()` to get all parameters.

## Serialization

Modules can save and load parameters in safetensors format:

```rust,no_run
# extern crate zyx;
# extern crate zyx_nn;
# use zyx::{DType, Module, Tensor, ZyxError};
# use zyx_nn::Linear;
#
# #[derive(zyx_nn::Module)]
# struct SimpleNet {
#     linear1: Linear,
# }
#
# fn main() -> Result<(), ZyxError> {
# let mut model = SimpleNet { linear1: Linear::new(4, 2, true, DType::F32)? };
model.save("model.safetensors")?;
let mut params = Tensor::load_safetensors("model.safetensors")?;
model.set_params(&mut params);
# Ok(())
# }
```
