#!/usr/bin/env python3
import numpy as np, pandas as pd
from sklearn.linear_model import HuberRegressor
from scipy.stats import spearmanr

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'

def main():
    df = pd.read_csv(BENCH_CSV).tail(9000).reset_index(drop=True)
    print(f"Read {len(df)} entries")

    y = df.groupby('variant_hash')['time_us'].rank(pct=True).fillna(0.5)

    # --- features ---
    df['f1'] = df['wi_global_load_bits'] * df['num_groups'] * df['wi_per_group']
    df['f2'] = df['wi_compute_ops'] / (df['wi_global_load_bits'] + df['wi_global_store_bits'])
    df['f3'] = df['max_register_bytes']

    X = df[['f1', 'f2', 'f3']].values

    groups = [g.index for _, g in df.groupby('variant_hash') if len(g) >= 2]
    print(f"Features: {X.shape[1]}, Groups: {len(groups)}")

    model = HuberRegressor(fit_intercept=False, max_iter=200)
    model.fit(X, y)
    pred = model.predict(X)

    # Debug: print a random group sorted by actual time
    rng = np.random.default_rng()
    h = rng.choice(df['variant_hash'].unique())
    g = df[df['variant_hash'] == h]
    order = g['time_us'].argsort()
    print(f"\nvariant_hash: {h}, {len(g)} variants")
    cols = '  '.join(f'{c:<12s}' for c in ['time_us', 'pred', 'f1', 'f2', 'f3'])
    print(cols)
    pred_vals = pred[g.index]
    for rank, i in enumerate(order[:40]):
        idx = g.index[i]
        idx = g.index[i]
        pred_rank = (pred_vals < pred_vals[i]).sum()
        vals = [f"{g['time_us'].iloc[i]:<12.0f}", f'{pred[idx]:<12.4f}']
        for j in range(1, X.shape[1]+1):
            v = df.loc[idx, f'f{j}']
            vals += [f'{v:<12.4f}' if v < 1e6 else f'{v:<12.0f}']
        print('  '.join(vals))

    rhos, topk = [], []
    for g in groups:
        if len(g) >= 3:
            rhos.append(spearmanr(df.loc[g, 'time_us'], pred[g])[0])
        if len(g) >= 20:
            a = df.loc[g, 'time_us'].argsort()[:10]
            p = pred[g].argsort()[:20]
            topk.append(len(set(a) & set(p)) / 10.0)
    print(f"Spearman ρ:       {np.nanmean(rhos):.4f} ± {np.nanstd(rhos):.4f}")
    print(f"Top10-in-top20:   {np.mean(topk):.3f} (over {len(topk)} groups)")

if __name__ == '__main__':
    main()
