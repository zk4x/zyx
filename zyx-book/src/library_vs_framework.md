# Library vs. Framework: The Zyx Philosophy

## Core Philosophy

Zyx is designed as a **library**, not a framework. This distinction is critical for developers who value flexibility and control over their machine learning workflows. While frameworks impose rigid structures and design patterns, libraries like Zyx adapt to your needs, enabling seamless integration into any project architecture.

This philosophy manifests in several key ways:
- **No scaffolding requirements**: Zyx doesn't force you to follow specific project templates or directory structures
- **No global state**: Unlike frameworks that maintain persistent runtime state, Zyx operations are localized to individual tensors
- **No enforced abstractions**: You're free to build your own abstractions without fighting framework conventions

## Technical Advantages

### 1. Zero Design Enforcement

Zyx avoids dictating how you structure your code. This means:

- **No mandatory base classes**: You're not required to implement predefined interfaces.
- **No opinionated training loops**: Unlike many ML frameworks that provide a single `fit()` or `train()` method, Zyx allows you to build custom training loops tailored to your application's requirements.
- **No static graph requirements**: Zyx's tape-based autograd system (see [autograd documentation](autograd.md)) records gradients only when needed, without requiring upfront graph definitions.

### 2. Minimal Compilation Footprint

Zyx is engineered to be **tiny when compiled**, making it ideal for users who want to keep their disk free. It's just a few MB.

This lightweight nature stems from:
- **No code generation**: Zyx avoids macros that expand into large codebases
- **No feature flags bloat**: Most functionality and hardware works without enabling any features
- **No unnecessary abstractions**: Extensive ops support, without traits or generics

The benefits are tangible:
- **Embedded systems**: Deploy ML capabilities in resource-constrained environments
- **Edge computing**: Run inference on devices with limited storage capacity
- **Fast iteration**: Quick recompilation during development cycles
- **No generic/lifetime headaches**: Spend time writing your code, not refactoring to fit into the framework

### 3. Dependency Management

Zyx maintains a **minimal dependency profile**:

- **Core dependencies**: Only essential libraries like `nanoserde` for config parsing and `libloading` for dynamic backend loading.
- **Optional features**: Big dependencies (e.g., WGPU backend) are available as features that can be enabled when needed.
- **No dependency bloat**: Zyx avoids unnecessary dependencies that could increase binary size or complexity.

This approach offers several advantages:
- **Reduced security surface**: Fewer dependencies mean fewer potential vulnerabilities
- **Simpler builds**: Minimal dependency chain reduces compilation issues
- **Smaller binaries**: No transitive dependencies bloating your final executable

### 4. Tensor Design Without Generics

Zyx's tensor implementation avoids pervasive generics, which offers several benefits:

- **No type parameter propagation**: Unlike systems that require generic parameters throughout the codebase, Zyx's tensors have fixed types.
- **Simpler API**: Developers don't need to manage complex generic type signatures across operations.
- **Easier integration**: Tensors can be used in more contexts without requiring generic type alignment.

This design choice has practical implications:
```rust
// Zyx tensor operation
impl CustomModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.layer1.forward(x).unwrap().relu();
        self.layer2.forward(&x).unwrap()
    }
}

// Hypothetical generic-based approach
impl<T: DType, B: Backend> CustomModel<T, B> {
    fn forward<S: Shape>(&self, x: &Tensor<S, T, B>) -> Tensor<S, T, B> {
        let x = self.layer1.forward(x).unwrap().relu();
        self.layer2.forward(&x).unwrap()
    }
}
```

## Practical Usage Scenarios

### General Linear Algebra Library

Zyx's design allows it to function as a general-purpose linear algebra library:

```rust
use zyx::{Tensor, DType};

// Basic tensor operations
let a = Tensor::from([1.0, 2.0, 3.0]);
let b = Tensor::from([4.0, 5.0, 6.0]);
let c = a + b;  // Element-wise addition
let d = c * 2.0;  // Scalar multiplication
```

This example demonstrates how Zyx can be used for standard tensor operations without any framework-specific boilerplate.

### Custom Training Loop Integration

The library approach makes custom modules and training the default:

```rust
use zyx::{Tensor, DType, GradientTape};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct CustomModel {
    layer1: Linear,
    layer2: Linear,
    learning_rate: f32,
}

impl CustomModel {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.layer1.forward(x).unwrap().relu();
        self.layer2.forward(&x).unwrap()
    }
}

fn train_step(model: &mut CustomModel, optimizer: &mut SGD, inputs: &Tensor, targets: &Tensor) -> f32 {
    let tape = GradientTape::new();
    let outputs = model.forward(inputs);
    let loss = outputs.mse_loss(targets).unwrap();
    
    let gradients = tape.gradient(&loss, model);
    optimizer.update(model, gradients);
    
    loss.item()
}
```

This demonstrates how Zyx components can be integrated into a custom training workflow without framework constraints.

### Memory-Efficient Deployment

Zyx's minimal footprint makes it suitable for deployment scenarios:

```rust
// In a deployment context
use zyx::{Tensor, DType};

fn process_input(input: &[f32]) -> Vec<f32> {
    let tensor = Tensor::from_slice(input, [1, input.len()]);
    let processed = tensor.sigmoid();  // Example activation
    processed.try_into().unwrap()
}
```

This example shows how Zyx can be used in deployment without carrying framework-specific runtime overhead.

## Comparative Analysis

| Feature                | Traditional Frameworks       | Zyx Library               |
|------------------------|-----------------------------|---------------------------|
| Graph definition       | Static upfront              | Dynamic, on-demand        |
| Training loop          | Predefined `fit()` method   | Custom implementation     |
| Memory usage           | High (stores intermediates) | Optimized (tape-based)    |
| Compilation size       | Large binaries              | Minimal footprint         |
| Dependency chain       | Complex                     | Minimal, focused          |
| Generic type usage     | Pervasive                   | Avoided                   |
| Debugging flexibility  | Limited                     | Full control              |
| Hardware utilization   | Framework-bound             | User-controlled           |

## Technical Implementation Details

### Lazy Execution Model

Zyx's lazy execution model contributes to its lightweight nature:

- Tensors aren't realized until explicitly requested
- Computation graphs are built only when needed
- Memory allocation is optimized through deferred execution

This approach enables efficient memory usage while maintaining flexibility:
```rust
let tape = GradientTape::new();
let x = Tensor::randn([1024, 1024], DType::F32);
let y = x.relu();  // Not computed yet
let z = y * 2.0;    // Still just building the graph

// Computation happens here
Tensor::realize([&z]).unwrap();
```

### Backend Agnosticism

Zyx supports multiple backends without framework-level constraints:

- CUDA (PTX)
- OpenCL
- WGPU (WGSL)

This allows users to choose hardware acceleration without being tied to framework-specific configuration. Developers don't need to be concerned where models will be deployed. Hardware is abstracted away as much as possible. But we are aware of leaky abstractions. Zyx works similarly to standard language compilers - code written in Rust can be deployed to many hardware targets, but knowledge of hardware quirks is still necessary when optimizing performance.

### Error Handling Philosophy

Zyx's error handling aligns with its library approach:

- Returns `Result` types for recoverable errors
- Uses `panic!` only for unrecoverable hardware issues or internal bugs
- Allows integration with both simple and complex error handling systems

This approach provides flexibility:
```rust
// Simple error handling
let tape = GradientTape::new();
let x = Tensor::randn([1024, 1024], DType::F32);
let y = x.relu();  // No Result type needed for non-fallible operations

// Complex error handling
fn safe_operation() -> Result<(), ZyxError> {
    let tape = GradientTape::new();
    let x = Tensor::randn([1024, 1024], DType::F32)?; // Result will be returned on allocation failure
    let y = x.relu();
    let z = tape.gradient(&y, &[&x]);
    Ok(())
}
```

## Why This Matters

### For Researchers

The library model enables:
- **Rapid experimentation**: Modify code without framework constraints
- **Custom optimization strategies**: Implement domain-specific optimizations
- **Fine-grained control**: Debug and inspect every computation step

### For Engineers

When integrating ML into existing systems:
- **No architecture changes**: Zyx adapts to your codebase, not vice versa
- **Minimal binary impact**: Add ML capabilities without bloating your executable
- **Simplified dependency management**: Avoid complex framework dependency trees

### For Deployments

Zyx's design shines in production:
- **Small footprint**: Ideal for edge devices with limited storage
- **Flexible execution**: Choose hardware acceleration based on deployment environment
- **Simple error handling**: Integrate with your existing error management system

## Conclusion

Zyx's library-first design avoids framework-level constraints while offering:
1. **Flexibility** in code organization
2. **Minimal compilation footprint** for efficient deployment
3. **Simplified tensor operations** without generic type propagation
4. **Custom workflow support** for both research and production environments
5. **Backend agnosticism** with hardware flexibility

This approach makes Zyx suitable for:
- Researchers needing custom training dynamics
- Engineers integrating ML or linear algebra routines into existing systems
- Developers working in resource-constrained environments

The library model enables Zyx to be both powerful and lightweight, providing the best of both worlds for modern machine learning and linear algebra needs.
