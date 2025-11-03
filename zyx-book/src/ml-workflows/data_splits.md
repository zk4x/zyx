# Train/Validation/Test Splits

Proper data splitting is crucial for building robust machine learning models. The way you split your data determines how well your model will generalize to unseen data and helps prevent overfitting.

## Why Data Splitting Matters

Data splitting allows you to:
- **Train models** on a portion of your data
- **Validate** model performance during development
- **Test** final model performance on unseen data

Without proper splitting, you risk overfitting to your training data and getting an overly optimistic view of your model's performance.

## Common Splitting Strategies

### Simple Random Split

The most basic approach where data is randomly divided into sets.

```rust
use zyx::{Tensor, DType};

// Simple random split
fn random_split(data: &Tensor, labels: &Tensor, train_ratio: f32) -> (Tensor, Tensor, Tensor, Tensor) {
    let n_samples = data.shape()[0];
    let n_train = (n_samples as f32 * train_ratio) as usize;
    
    // Shuffle indices
    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rand::thread_rng());
    
    // Split indices
    let train_indices = &indices[..n_train];
    let test_indices = &indices[n_train..];
    
    // Create masks
    let mut train_mask = Tensor::zeros([n_samples], DType::F32);
    let mut test_mask = Tensor::zeros([n_samples], DType::F32);
    
    for &idx in train_indices {
        train_mask[[idx]] = 1.0;
    }
    for &idx in test_indices {
        test_mask[[idx]] = 1.0;
    }
    
    // Apply masks
    let train_data = data * &train_mask.unsqueeze(1);
    let train_labels = labels * &train_mask.unsqueeze(1);
    let test_data = data * &test_mask.unsqueeze(1);
    let test_labels = labels * &test_mask.unsqueeze(1);
    
    (train_data, train_labels, test_data, test_labels)
}
```

### Stratified Split

For classification tasks, stratified splitting ensures class distribution is preserved in each split.

```rust
// Stratified split for classification
fn stratified_split(
    data: &Tensor, 
    labels: &Tensor, 
    train_ratio: f32
) -> (Tensor, Tensor, Tensor, Tensor) {
    let n_samples = data.shape()[0];
    let n_classes = labels.max().item::<i32>() + 1;
    
    let mut class_indices: Vec<Vec<usize>> = (0..n_classes).map(|_| Vec::new()).collect();
    
    // Group indices by class
    for i in 0..n_samples {
        let class = labels[[i]].item::<i32>().unwrap() as usize;
        class_indices[class].push(i);
    }
    
    // Split each class
    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();
    
    for class_indices in &class_indices {
        let n_class = class_indices.len();
        let n_train = (n_class as f32 * train_ratio) as usize;
        
        let train_part = &class_indices[..n_train];
        let test_part = &class_indices[n_train..];
        
        train_indices.extend_from_slice(train_part);
        test_indices.extend_from_slice(test_part);
    }
    
    // Create final splits
    let train_data = select_rows(data, &train_indices);
    let train_labels = select_rows(labels, &train_indices);
    let test_data = select_rows(data, &test_indices);
    let test_labels = select_rows(labels, &test_indices);
    
    (train_data, train_labels, test_data, test_labels)
}

// Helper function to select rows by indices
fn select_rows(data: &Tensor, indices: &[usize]) -> Tensor {
    let mut result = Tensor::zeros([indices.len(), data.shape()[1]], data.dtype());
    for (i, &idx) in indices.iter().enumerate() {
        result.slice(i..=i, 0..=data.shape()[1] - 1).copy_(data.slice(idx..=idx, 0..=data.shape()[1] - 1));
    }
    result
}
```

### Time Series Split

For time-dependent data, use temporal splits to prevent data leakage.

```rust
// Time series split
fn time_series_split(data: &Tensor, labels: &Tensor, train_ratio: f32) -> (Tensor, Tensor, Tensor, Tensor) {
    let n_samples = data.shape()[0];
    let split_point = (n_samples as f32 * train_ratio) as usize;
    
    let train_data = data.slice(0..=split_point - 1, 0..=data.shape()[1] - 1);
    let train_labels = labels.slice(0..=split_point - 1, 0..=labels.shape()[1] - 1);
    let test_data = data.slice(split_point..=n_samples - 1, 0..=data.shape()[1] - 1);
    let test_labels = labels.slice(split_point..=n_samples - 1, 0..=labels.shape()[1] - 1);
    
    (train_data, train_labels, test_data, test_labels)
}
```

## Cross-Validation

Cross-validation provides more robust performance estimates by using multiple train/test splits.

### K-Fold Cross-Validation

```rust
// K-fold cross validation
fn k_fold_cross_validation(
    data: &Tensor,
    labels: &Tensor,
    k: i32,
    model_factory: fn() -> Box<dyn Module>,
    epochs: i32
) -> f32 {
    let n_samples = data.shape()[0];
    let fold_size = n_samples / k as usize;
    let mut scores = Vec::new();
    
    for fold in 0..k {
        let start = fold as usize * fold_size;
        let end = if fold == k - 1 { n_samples } else { start + fold_size };
        
        // Create validation fold
        let val_data = data.slice(start..=end - 1, 0..=data.shape()[1] - 1);
        let val_labels = labels.slice(start..=end - 1, 0..=labels.shape()[1] - 1);
        
        // Create training folds (all other folds)
        let mut train_data = Tensor::zeros([n_samples - fold_size, data.shape()[1]], data.dtype());
        let mut train_labels = Tensor::zeros([n_samples - fold_size, labels.shape()[1]], labels.dtype());
        let mut train_idx = 0;
        
        for other_fold in 0..k {
            if other_fold == fold { continue; }
            
            let other_start = other_fold as usize * fold_size;
            let other_end = if other_fold == k - 1 { n_samples } else { other_start + fold_size };
            
            let other_data = data.slice(other_start..=other_end - 1, 0..=data.shape()[1] - 1);
            let other_labels = labels.slice(other_start..=other_end - 1, 0..=labels.shape()[1] - 1);
            
            train_data.slice(train_idx..=train_idx + other_data.shape()[0] - 1, 0..=data.shape()[1] - 1).copy_(&other_data);
            train_labels.slice(train_idx..=train_idx + other_labels.shape()[0] - 1, 0..=labels.shape()[1] - 1).copy_(&other_labels);
            train_idx += other_data.shape()[0];
        }
        
        // Train and evaluate model
        let mut model = model_factory();
        train_model(&mut model, &train_data, &train_labels, epochs);
        let score = evaluate_model(&model, &val_data, &val_labels);
        scores.push(score);
    }
    
    // Return average score
    scores.iter().sum::<f32>() / scores.len() as f32
}
```

## Split Ratios

Common split ratios include:
- **70/15/15**: Training/Validation/Test
- **80/10/10**: Training/Validation/Test
- **60/20/20**: Training/Validation/Test
- **80/20**: Training/Test (when validation isn't needed)

The choice depends on:
- Dataset size (larger datasets can afford smaller validation/test sets)
- Model complexity (complex models need more training data)
- Task requirements (critical applications may need larger test sets)

## Best Practices

1. **Always split before preprocessing**: Fit preprocessing parameters on training data only
2. **Use stratified splits** for classification tasks
3. **Shuffle data** before splitting (except for time series)
4. **Document your splits** for reproducibility
5. **Use consistent splits** across experiments

## Summary

Proper data splitting is essential for building reliable machine learning models. By choosing the right splitting strategy and ratios, you ensure your model generalizes well to unseen data and provides accurate performance estimates.
