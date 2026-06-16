# Module System

The module system provides a way to group tensors (parameters) into neural network layers. It's defined by the `Module` trait and powered by `#[derive(Module)]`.

## The Module Trait

```rust,ignore
pub trait Module {
    fn iter(&self) -> impl Iterator<Item = &Tensor>;
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Tensor>;
    fn iter_tensors(&self) -> impl Iterator<Item = (String, &Tensor)>;
    fn iter_tensors_mut(&mut self) -> impl Iterator<Item = (String, &mut Tensor)>;
    fn realize(&self) -> Result<(), ZyxError>;
    fn save(&self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
    fn set_params(&mut self, params: &mut HashMap<String, Tensor>);
}
```

## #[derive(Module)]

The `#[derive(Module)]` macro (from `zyx-derive`) generates the trait implementation, collecting all tensor fields recursively. This works with nested modules:

```rust,ignore
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
```

## Using Modules

```rust,ignore
#[derive(Module)]
struct SimpleNet {
    linear1: Linear,
    linear2: Linear,
}

fn train_step(model: &mut SimpleNet, optim: &mut SGD, x: &Tensor, target: &Tensor) -> f32 {
    let tape = GradientTape::new();
    let output = model.forward(x);
    let loss = output.mse_loss(target)?;
    let grads = tape.gradient(&loss, &model);
    optim.update(model, grads);
    Tensor::realize_all()?;
    loss.item()
}
```

The `tape.gradient(&loss, &model)` call passes the model itself as the sources. The autograd system iterates over `model.iter()` to get all parameters.

## Serialization

Modules can save and load parameters in safetensors format:

```rust,ignore
model.save("model.safetensors")?;
let params = Tensor::load_safetensors("model.safetensors")?;
model.set_params(&mut params);
```
