# Mathematical Foundations

This chapter provides the essential mathematical background needed for understanding machine learning with Zyx. We'll cover the key mathematical concepts that form the foundation of neural networks and deep learning.

## Overview

Machine learning is fundamentally applied mathematics. To understand how neural networks work and how to effectively use Zyx, you need a solid grasp of:

- Linear algebra: The language of neural networks
- Calculus: The mathematics of optimization
- Probability and statistics: The mathematics of uncertainty
- Optimization theory: The mathematics of learning

## Linear Algebra Review

Linear algebra is the mathematical foundation of machine learning. Neural networks are essentially complex functions composed of linear transformations and nonlinear activations.

### Vectors and Matrices

Vectors are one-dimensional arrays of numbers, while matrices are two-dimensional arrays. In Zyx, these are represented as tensors.

```rust
use zyx::{Tensor, DType};

// Vector (1D tensor)
let vector = Tensor::from([1.0, 2.0, 3.0]);
assert_eq!(vector.shape(), [3]);

// Matrix (2D tensor)
let matrix = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
assert_eq!(matrix.shape(), [2, 2]);
```

### Key Operations

#### Matrix Multiplication

Matrix multiplication is the fundamental operation in neural networks. In Zyx, this is done using the `dot` method.

```rust
let a = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
let b = Tensor::from([[5.0, 6.0], [7.0, 8.0]]);

// Matrix multiplication
let c = a.dot(&b);
// Result: [[19.0, 22.0], [43.0, 50.0]]
```

#### Transposition

Transposing a matrix swaps its rows and columns.

```rust
let matrix = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);
let transposed = matrix.t();
// Result: [[1.0, 3.0], [2.0, 4.0]]
```

#### Element-wise Operations

Element-wise operations apply the same operation to each element independently.

```rust
let a = Tensor::from([1.0, 2.0, 3.0]);
let b = Tensor::from([4.0, 5.0, 6.0]);

// Element-wise addition
let c = &a + &b;  // [5.0, 7.0, 9.0]

// Element-wise multiplication
let d = &a * &b;  // [4.0, 10.0, 18.0]
```

### Tensor Properties

#### Rank and Shape

The rank of a tensor is the number of dimensions, and the shape describes the size of each dimension.

```rust
let tensor = Tensor::randn([2, 3, 4], DType::F32);

assert_eq!(tensor.rank(), 3);  // 3D tensor
assert_eq!(tensor.shape(), [2, 3, 4]);  // 2x3x4 tensor
```

#### Broadcasting

Broadcasting allows operations between tensors of different shapes by automatically expanding dimensions.

```rust
let matrix = Tensor::randn([3, 3], DType::F32);
let vector = Tensor::from([1.0, 2.0, 3.0]);

// Vector will be broadcasted to match matrix shape
let result = matrix + &vector;
```

## Calculus for Machine Learning

Calculus provides the tools for optimization and understanding how neural networks learn.

### Derivatives and Gradients

The derivative measures how a function changes as its input changes. In neural networks, we use gradients (multidimensional derivatives) to update parameters.

```rust
use zyx::{Tensor, GradientTape};

// Forward pass
let tape = GradientTape::new();
let x = Tensor::from([2.0, 3.0]);
let y = x.pow(2);  // y = x^2

// Backward pass to compute gradients
let grads = tape.gradient(&y, &[&x]);
// dy/dx = 2*x = [4.0, 6.0]
```

### Chain Rule

The chain rule allows us to compute gradients through composed functions, which is essential for backpropagation.

```rust
// f(g(x)) where f(y) = y^2 and g(x) = x + 1
let tape = GradientTape::new();
let x = Tensor::from([2.0]);
let g = x + 1.0;  // g(x) = x + 1
let f = g.pow(2);  // f(g) = g^2

// df/dx = df/dg * dg/dx = 2g * 1 = 2(x + 1)
let grads = tape.gradient(&f, &[&x]);
// Result: [6.0] (since 2*(2 + 1) = 6)
```

### Partial Derivatives

Partial derivatives measure how a function changes with respect to one variable while holding others constant.

```rust
// f(x, y) = x^2 + y^2
let tape = GradientTape::new();
let x = Tensor::from([2.0]);
let y = Tensor::from([3.0]);
let f = x.pow(2) + y.pow(2);

// df/dx = 2x = 4.0
// df/dy = 2y = 6.0
let grads = tape.gradient(&f, &[&x, &y]);
```

## Probability and Statistics

Probability and statistics provide the framework for understanding uncertainty and making predictions.

### Probability Distributions

```rust
// Normal distribution
let normal = Tensor::randn([1000], DType::F32);

// Uniform distribution
let uniform = Tensor::rand([1000], DType::F32);
```

### Statistical Measures

```rust
// Mean
let mean = normal.mean();

// Variance and standard deviation
let variance = normal.var();
let std_dev = normal.std();

// Percentiles
let p50 = normal.quantile(0.5);  // Median
let p95 = normal.quantile(0.95);
```

### Cross-Entropy

Cross-entropy is commonly used as a loss function for classification tasks.

```rust
// Softmax and cross-entropy
let logits = Tensor::randn([10], DType::F32);  // 10 classes
let targets = Tensor::from([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);  # One-hot

// Softmax probabilities
let probs = logits.softmax();

// Cross-entropy loss
let loss = -(&targets * probs.ln()).sum();
```

## Optimization Theory

Optimization theory provides the mathematical foundation for training neural networks.

### Gradient Descent

Gradient descent is the fundamental optimization algorithm used in neural networks.

```rust
// Simple gradient descent implementation
fn gradient_descent(
    params: &mut Tensor,
    gradients: &Tensor,
    learning_rate: f32
) {
    // Update parameters: params = params - learning_rate * gradients
    *params = params - &gradients * learning_rate;
}
```

### Learning Rate Scheduling

```rust
// Exponential decay learning rate
fn exponential_decay(epoch: i32, initial_lr: f32, decay_rate: f32) -> f32 {
    initial_lr * decay_rate.powi(epoch)
}

// Step decay learning rate
fn step_decay(epoch: i32, initial_lr: f32, step_size: i32, decay_rate: f32) -> f32 {
    initial_lr * decay_rate.powi(epoch / step_size)
}
```

### Momentum

Momentum helps accelerate gradient descent in relevant directions and dampens oscillations.

```rust
// Momentum-based parameter update
struct Momentum {
    velocity: Tensor,
    momentum: f32,
}

impl Momentum {
    fn new(shape: &[usize], momentum: f32) -> Self {
        Self {
            velocity: Tensor::zeros(shape, DType::F32),
            momentum,
        }
    }

    fn update(&mut self, params: &mut Tensor, gradients: &Tensor, learning_rate: f32) {
        // Update velocity: v = momentum * v + learning_rate * gradients
        self.velocity = &self.velocity * self.momentum + gradients * learning_rate;
        
        // Update parameters: params = params - velocity
        *params = params - &self.velocity;
    }
}
```

### Adaptive Learning Rates

Adaptive optimizers like Adam adjust learning rates for each parameter individually.

```rust
// Simplified Adam optimizer
struct Adam {
    m: Tensor,  // First moment (momentum)
    v: Tensor,  // Second moment (RMSProp)
    t: i32,     // Time step
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Adam {
    fn new(shape: &[usize]) -> Self {
        Self {
            m: Tensor::zeros(shape, DType::F32),
            v: Tensor::zeros(shape, DType::F32),
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }

    fn update(&mut self, params: &mut Tensor, gradients: &Tensor, learning_rate: f32) {
        self.t += 1;
        
        // Update biased first moment estimate
        self.m = &self.m * self.beta1 + gradients * (1.0 - self.beta1);
        
        // Update biased second raw moment estimate
        self.v = &self.v * self.beta2 + gradients.pow(2) * (1.0 - self.beta2);
        
        // Compute bias-corrected first moment estimate
        let m_hat = &self.m / (1.0 - self.beta1.powi(self.t));
        
        // Compute bias-corrected second raw moment estimate
        let v_hat = &self.v / (1.0 - self.beta2.powi(self.t));
        
        // Update parameters
        let update = m_hat / (&v_hat.sqrt() + self.eps);
        *params = params - &update * learning_rate;
    }
}
```

## Mathematical Concepts in Zyx

### Vectorization

Zyx automatically handles vectorization of operations, allowing you to work with batches of data efficiently.

```rust
// Batch processing
let batch = Tensor::randn([32, 784], DType::F32);  // 32 samples, 784 features each
let weights = Tensor::randn([784, 10], DType::F32);  // 784 input features, 10 output classes

// Process entire batch at once
let logits = batch.dot(&weights);  // [32, 10]
```

### Automatic Differentiation

Zyx's automatic differentiation system handles the complex mathematics of backpropagation automatically.

```rust
// Complex computation graph
let tape = GradientTape::new();
let x = Tensor::randn([100, 100], DType::F32);
let y = x.relu().mm(&x.t()).softmax();
let loss = y.sum();

// Automatic gradient computation
let gradients = tape.gradient(&loss, &[&x]);
```

## Practical Examples

### Linear Regression

```rust
// Linear regression: y = Wx + b
fn linear_regression(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    x.dot(w) + b
}

// Mean squared loss
fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).pow(2).mean()
}

// Training loop
fn train_linear_regression() {
    let x_train = Tensor::randn([100, 10], DType::F32);
    let y_train = Tensor::randn([100, 1], DType::F32);
    
    let mut w = Tensor::randn([10, 1], DType::F32);
    let mut b = Tensor::zeros([1], DType::F32);
    
    let learning_rate = 0.01;
    let epochs = 1000;
    
    for epoch in 0..epochs {
        let tape = GradientTape::new();
        let predictions = linear_regression(&x_train, &w, &b);
        let loss = mse_loss(&predictions, &y_train);
        
        let gradients = tape.gradient(&loss, [&w, &b]);
        
        // Update parameters
        w = w - &gradients[0] * learning_rate;
        b = b - &gradients[1] * learning_rate;
        
        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {}", epoch, loss.item());
        }
    }
}
```

### Logistic Regression

```rust
// Logistic regression with sigmoid activation
fn logistic_regression(x: &Tensor, w: &Tensor, b: &Tensor) -> Tensor {
    let logits = x.dot(w) + b;
    logits.sigmoid()
}

// Binary cross-entropy loss
fn binary_cross_entropy(predictions: &Tensor, targets: &Tensor) -> Tensor {
    -(&targets * predictions.ln() + &(1.0 - targets) * (1.0 - predictions).ln()).mean()
}

// Training loop
fn train_logistic_regression() {
    let x_train = Tensor::randn([1000, 20], DType::F32);
    let y_train = Tensor::randn([1000, 1], DType::F32);  # Binary labels
    
    let mut w = Tensor::randn([20, 1], DType::F32);
    let mut b = Tensor::zeros([1], DType::F32);
    
    let learning_rate = 0.1;
    let epochs = 100;
    
    for epoch in 0..epochs {
        let tape = GradientTape::new();
        let predictions = logistic_regression(&x_train, &w, &b);
        let loss = binary_cross_entropy(&predictions, &y_train);
        
        let gradients = tape.gradient(&loss, [&w, &b]);
        
        // Update parameters
        w = w - &gradients[0] * learning_rate;
        b = b - &gradients[1] * learning_rate;
        
        if epoch % 10 == 0 {
            let accuracy = (&predictions > 0.5).eq(&y_train).mean();
            println!("Epoch {}: Loss = {:.4}, Accuracy = {:.4}", 
                    epoch, loss.item(), accuracy.item());
        }
    }
}
```

## Summary

The mathematical foundations covered in this chapter are essential for understanding and working with machine learning in Zyx:

- **Linear algebra** provides the language for representing and manipulating data
- **Calculus** gives us the tools for optimization and learning
- **Probability and statistics** help us handle uncertainty and measure performance
- **Optimization theory** provides the algorithms for training neural networks

With this mathematical foundation, you're ready to dive deeper into neural networks and start building sophisticated machine learning models with Zyx.