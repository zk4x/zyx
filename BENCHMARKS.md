# Benchmarks

## MNIST Training Step Time

| Hardware | Backend | Batch Size | Step Time |
|----------|---------|------------|-----------|
| RTX 3090 / A100 / T4 | zyx (CUDA PTX) | 128 | 0.7ms |
| RTX 3090 / A100 / T4 | torch (CUDA) | 129 | 0.7ms |

### Notes

- Both implementations use random sampling with replacement, momentum 0.9, lr 0.01
- zyx: 200 steps, torch: 5 epochs (2325 batches)
- Both hit GPU-bound regime (kernel launch overhead dominates for small batches)

---
