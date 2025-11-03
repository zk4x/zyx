# Cross-Validation Techniques

Cross-validation is a robust technique for estimating model performance and selecting the best model. It provides more reliable performance estimates than a single train/test split by using multiple different splits of the data.

## Why Cross-Validation Matters

Single train/test splits can be misleading due to:
- Random variation in the split
- Unrepresentative splits
- Overfitting to the specific test set

Cross-validation addresses these issues by:
- Using multiple different train/test splits
- Averaging performance across all splits
- Providing more stable performance estimates

## K-Fold Cross-Validation

The most common cross-validation method, where data is divided into K equal parts. Each fold serves as the test set once, while the remaining K-1 folds form the training set.

```rust
use zyx::{Tensor, DType};

// Simple K-fold cross-validation
fn k_fold_cross_validation(data: &Tensor, k: i32) -> f32 {
    let n_samples = data.shape()[0];
    let fold_size = n_samples / k as usize;
    let mut scores = Vec::new();
    
    for fold in 0..k {
        let start = fold as usize * fold_size;
        let end = if fold == k - 1 { n_samples } else { start + fold_size };
        
        // Create validation fold
        let val_data = data.slice(start..=end - 1, 0..=data.shape()[1] - 1);
        
        // Train on remaining folds and evaluate
        let score = train_and_evaluate(val_data);
        scores.push(score);
    }
    
    // Return average score
    scores.iter().sum::<f32>() / scores.len() as f32
}
```

## Common Cross-Validation Methods

### Stratified K-Fold

For classification tasks, stratified K-fold ensures each fold has the same class distribution as the original dataset. This prevents situations where some folds might lack certain classes.

### Leave-One-Out Cross-Validation (LOOCV)

A special case where each sample is used once as test data. Provides unbiased estimates but is computationally expensive.

### Time Series Cross-Validation

For time-dependent data, uses specialized splits that respect temporal order to prevent data leakage.

## Choosing the Right Method

- **K-Fold (K=5 or 10)**: Good balance for most problems
- **Stratified K-Fold**: Best for classification
- **LOOCV**: Small datasets where you need unbiased estimates
- **Time Series CV**: Temporal data where order matters

## Best Practices

1. **Use stratified K-fold** for classification tasks
2. **Consider computational cost** - LOOCV is unbiased but expensive
3. **Use time series CV** for temporal data
4. **Document your CV strategy** for reproducibility

## Summary

Cross-validation provides robust performance estimates and helps select the best model. By choosing the appropriate method for your data type and problem, you can build more reliable and generalizable machine learning models.
