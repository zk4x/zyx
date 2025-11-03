# Data Preprocessing

Data preprocessing is a critical step in machine learning that transforms raw data into a clean, usable format for model training. Proper preprocessing can significantly impact model performance and is often the most time-consuming part of the ML pipeline.

## Why Data Preprocessing Matters

Raw data is rarely ready for machine learning models. It often contains:
- Missing values
- Outliers and anomalies
- Inconsistent scales and distributions
- Categorical variables that need encoding
- Noise and irrelevant features

Effective preprocessing addresses these issues, ensuring that models can learn meaningful patterns from the data.

## Common Preprocessing Techniques

### Normalization and Standardization

Normalization scales features to a range [0, 1], while standardization transforms data to have zero mean and unit variance.

```rust
use zyx::{Tensor, DType};

// Normalize data to [0, 1] range
fn normalize_data(data: &Tensor) -> Tensor {
    let min = data.min();
    let max = data.max();
    (data - min) / (max - min)
}

// Standardize data (zero mean, unit variance)
fn standardize_data(data: &Tensor) -> Tensor {
    let mean = data.mean();
    let std = data.std();
    (data - mean) / std
}
```

### Handling Missing Values

Missing values can be handled through imputation or removal. Common strategies include:
- Mean/median/mode imputation
- Forward/backward filling
- Predictive imputation

```rust
// Fill missing values with mean
fn fill_missing_with_mean(data: &Tensor) -> Tensor {
    let mean = data.mean();
    // Create mask for missing values (assuming NaN represents missing)
    let mask = data.is_nan();
    data * !mask.clone() + mean * mask
}
```

### Categorical Encoding

Categorical variables need to be converted to numerical format. Common approaches include:
- One-hot encoding
- Label encoding
- Target encoding

```rust
// One-hot encode categorical data
fn one_hot_encode(data: &Tensor, num_classes: i32) -> Tensor {
    let shape = [data.shape()[0], num_classes as usize];
    let mut encoded = Tensor::zeros(shape, DType::F32);
    
    for i in 0..data.shape()[0] {
        let class = data[[i]].item::<i32>().unwrap() as usize;
        encoded[[i, class]] = 1.0;
    }
    encoded
}
```

### Feature Scaling

Different features may have different scales, which can bias models. Feature scaling ensures all features contribute equally.

```rust
// Min-max scaling to specific range
fn min_max_scale(data: &Tensor, new_min: f32, new_max: f32) -> Tensor {
    let current_min = data.min();
    let current_max = data.max();
    new_min + (data - current_min) * (new_max - new_min) / (current_max - current_min)
}
```

## Data Quality Assessment

Before preprocessing, assess data quality by examining:
- Distribution of each feature
- Presence of outliers
- Missing value patterns
- Correlation between features

```rust
// Basic data quality assessment
fn assess_data_quality(data: &Tensor) {
    println!("Data shape: {:?}", data.shape());
    println!("Missing values: {}", data.is_nan().sum().item::<i32>());
    println!("Mean: {:.4}", data.mean().item());
    println!("Std: {:.4}", data.std().item());
    println!("Min: {:.4}", data.min().item());
    println!("Max: {:.4}", data.max().item());
}
```

## Advanced Preprocessing Techniques

### Feature Engineering

Create new features that may be more predictive than the original ones:
- Polynomial features
- Interaction terms
- Domain-specific transformations

### Dimensionality Reduction

Reduce the number of features while preserving important information:
- Principal Component Analysis (PCA)
- t-SNE
- Autoencoders

### Data Augmentation

 artificially expand training data by creating modified versions of existing samples.

```rust
// Simple data augmentation for images
fn augment_image(image: &Tensor) -> Tensor {
    // Random horizontal flip
    let flipped = image.flip(1);
    
    // Random rotation (simplified example)
    // In practice, you'd use more sophisticated rotation
    flipped
}
```

## Best Practices

1. **Fit on training data only**: Preprocessing parameters should be learned from training data and applied to validation/test data
2. **Handle data leakage**: Ensure information from validation/test sets doesn't leak into training
3. **Document preprocessing**: Keep track of all preprocessing steps for reproducibility
4. **Validate preprocessing**: Check that preprocessing doesn't introduce artifacts or distort data

## Summary

Data preprocessing is essential for building effective machine learning models. By properly normalizing, cleaning, and transforming your data, you create the foundation for successful model training and deployment.
