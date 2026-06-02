# Regression Results Log

## Data: 47,557 entries, 11 sections

## Current Best (Feature Count: 171-177)
| Date | Model | Features | Min R² | MatMul R² | Reduce Axis R² | Notes |
|------|-------|----------|--------|-----------|----------------|-------|
| - | Ridge(alpha=0.00001) | 206 | 0.7502 | 0.9331 | 0.7502 | Original baseline |
| - | ElasticNet(alpha=0.001,l1=0.5) | 206→82 | 0.7543 | 0.9342 | 0.7543 | ElasticNet best |
| - | Ridge(alpha=1e-10) | 171 | 0.7333 | 0.9358 | 0.7333 | Near-OLS, all feats |
| - | Ridge(alpha=0.01) | 171 | 0.7340 | 0.9361 | 0.7340 | Ridge higher alpha |
| - | Ridge(alpha=0.0001) | 171 | 0.7344 | 0.9358 | 0.7344 | Ridge mid alpha |

## Targets
- MatMul > 0.95
- Worst section > 0.80

## Tried without success
- Adding targeted features (matmul_*, layer_norm_*, etc.)
- Adding high-order interaction features (ops_barr_ci, threads_barr_gmem, etc.)
- ElasticNet with various alpha/l1_ratio
- Ridge with various alphas (1e-10 to 0.01)
- Feature selection at 50th/60th percentile
- Grouped barrier regimes (barr_low, barr_med, barr_high)
- Barrier group interactions (barr_med_ng, barr_nonzero_lops, etc.)

## What's next to try
- Different target transform (sqrt, inverse, level)
- Per-section weighting bias towards worst sections
- Interaction features specifically targeting reduction patterns
