#!/usr/bin/env python3
import numpy as np, pandas as pd
from sklearn.linear_model import HuberRegressor
from scipy.stats import spearmanr

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'

def main():
    df = pd.read_csv(BENCH_CSV)
    print(f"Read {len(df)} entries")

    y = df.groupby('variant_hash')['time_us'].rank(pct=True).fillna(0.5)

    # --- features ---



    X = df.values

    groups = [g.index for _, g in df.groupby('variant_hash') if len(g) >= 2]
    print(f"Features: {X.shape[1]}, Groups: {len(groups)}")

    model = HuberRegressor(max_iter=200)
    model.fit(X, y)
    pred = model.predict(X)

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
