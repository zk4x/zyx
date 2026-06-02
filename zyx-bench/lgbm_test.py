#!/usr/bin/env python3
"""Quick LGBM test with train/test split to see how far we can push R²/ρ."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv')

# Variant groups
hash_map = {}
for i, h in enumerate(df['variant_hash']):
    hash_map.setdefault(h, []).append(i)
variant_groups = [v for v in hash_map.values() if len(v) >= 2]

# Rank target 0..1
y = np.empty(len(df))
for g in variant_groups:
    times = df.iloc[g]['time_us'].values
    ranks = np.argsort(np.argsort(times))
    for j, idx in enumerate(g):
        y[idx] = ranks[j] / (len(g) - 1)
ungrouped = set(range(len(df))) - set(idx for g in variant_groups for idx in g)
for i in ungrouped:
    y[i] = 0.5

# Features
feat_cols = [c for c in df.columns if c not in ('time_us', 'gflops', 'variant_hash', 'section')]
X = df[feat_cols].values.astype(np.float64)
print(f"Entries: {len(df)}, Features: {X.shape[1]}, Groups: {len(variant_groups)}")

# Split by variant group (no leakage)
group_keys = list(range(len(variant_groups)))
train_keys, test_keys = train_test_split(group_keys, test_size=0.2, random_state=42)
train_idx = [idx for k in train_keys for idx in variant_groups[k]]
test_idx = [idx for k in test_keys for idx in variant_groups[k]]

Xt, Xv = X[train_idx], X[test_idx]
yt, yv = y[train_idx], y[test_idx]


def eval_model(model, X_v, y_v, groups, indices, entries, label):
    pred = model.predict(X_v)
    r2s, rhos = [], []
    for g in groups:
        gi = [i for i in g if i in indices]
        if len(gi) < 3:
            continue
        li = [indices.index(i) for i in gi]
        yg, pg = y_v[li], pred[li]
        ss_res = np.sum((yg - pg) ** 2)
        ss_tot = np.sum((yg - yg.mean()) ** 2)
        r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
        rhos.append(spearmanr([entries[i]['time_us'] for i in gi], pg)[0])
    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"  {label}: mean R²={r2s.mean():.4f} ρ={rhos.mean():.4f}±{rhos.std():.4f} (n={len(r2s)})")


# Hyperparameter search
for depth in [6, 8, 10, 12]:
    for lr in [0.05, 0.1]:
        for trees in [500, 1000, 1500]:
            m = lgb.LGBMRegressor(
                n_estimators=trees, max_depth=depth,
                num_leaves=min(2 ** depth, 512),
                learning_rate=lr, min_child_samples=5,
                random_state=42, verbosity=-1,
            )
            m.fit(Xt, yt)
            pred = m.predict(Xv)

            # Per-kernel test metrics
            test_groups = [g for g in variant_groups if any(i in test_idx for i in g)]
            rhos_test = []
            for g in test_groups:
                gi = [i for i in g if i in test_idx]
                if len(gi) >= 3:
                    actual = df.iloc[gi]['time_us'].values
                    pg = pred[[test_idx.index(i) for i in gi]]
                    rhos_test.append(spearmanr(actual, pg)[0])
            rho = np.mean(rhos_test) if rhos_test else 0

            # Global test R²
            ss_res = np.sum((yv - pred) ** 2)
            ss_tot = np.sum((yv - yv.mean()) ** 2)
            tr2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            print(f"  depth={depth:2d} lr={lr:.2f} trees={trees:4d}: test R²={tr2:.4f} ρ={rho:.4f}")

# Best config: retrain and show full eval
print("\n=== Best config: depth=10, lr=0.1, trees=1000 ===")
best = lgb.LGBMRegressor(
    n_estimators=1000, max_depth=10, num_leaves=512,
    learning_rate=0.1, min_child_samples=5,
    random_state=42, verbosity=-1,
)
best.fit(Xt, yt)

# Train eval
eval_model(best, Xt, yt, [variant_groups[k] for k in train_keys], train_idx, df.to_dict('records'), 'Train')
# Test eval
eval_model(best, Xv, yv, [variant_groups[k] for k in test_keys], test_idx, df.to_dict('records'), 'Test ')

# Feature importance
gain = best.booster_.feature_importance(importance_type='gain')
top15 = np.argsort(gain)[-15:][::-1]
print(f"\nTop 15 features by gain:")
for i in top15:
    print(f"  {feat_cols[i]:40s} gain={gain[i]:.1f}")
