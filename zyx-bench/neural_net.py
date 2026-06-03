#!/usr/bin/env python3
"""Train a tiny sparse MLP on bench_data.csv to predict kernel cost rank."""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'
RAW_FEATURES = [
    'num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops', 'wi_barriers',
    'wi_global_load_bits', 'wi_global_store_bits', 'wi_local_load_bits',
    'wi_local_store_bits', 'wi_peak_reg_bytes', 'wi_branches',
    'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
    'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
    'warp_size', 'max_local_threads', 'max_register_bytes',
    'wi_register_load_bits', 'wi_register_store_bits',
    'gws0', 'gws1', 'gws2', 'lws0', 'lws1', 'lws2', 'max_loop_depth',
    'preferred_vector_size', 'local_mem_size',
]
N_FEAT = len(RAW_FEATURES)


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def l1_reg(self):
        return sum(p.abs().sum() for n, p in self.named_parameters() if 'weight' in n and 'norm' not in n)


def main():
    import pandas as pd
    df = pd.read_csv(BENCH_CSV)
    entries = df.to_dict('records')
    print(f"Read {len(entries)} entries")

    hash_map = defaultdict(list)
    for i, e in enumerate(entries):
        hash_map[e['variant_hash']].append(i)
    variant_groups = list(hash_map.values())
    print(f"Variant groups: {len(variant_groups)}")

    # Target: rank 0..1 within each group
    y = np.empty(len(entries), dtype=np.float32)
    for g in variant_groups:
        times = np.array([entries[i]['time_us'] for i in g])
        if len(g) >= 2:
            ranks = np.argsort(np.argsort(times))
            for j, idx in enumerate(g):
                y[idx] = ranks[j] / (len(g) - 1)
        else:
            for idx in g:
                y[idx] = 0.5

    # Features: log-transform select columns
    raw = np.array([[e[f] for f in RAW_FEATURES] for e in entries], dtype=np.float32)
    for c in [0, 1, 2, 3, 5, 6, 7, 8, 10, 18, 19, 27]:
        raw[:, c] = np.log(np.maximum(raw[:, c], 1.0))
    X = raw

    # Split by variant_hash
    hashes = list(hash_map.keys())
    rng = np.random.RandomState(42)
    rng.shuffle(hashes)
    n_train = int(len(hashes) * 0.8)
    train_hashes = set(hashes[:n_train])
    test_hashes = set(hashes[n_train:])
    train_idx = [i for h in train_hashes for i in hash_map[h]]
    test_idx = [i for h in test_hashes for i in hash_map[h]]
    test_idx_set = set(test_idx)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    device = torch.device('cuda')
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    model = MLP([N_FEAT, 256, 128, 64, 1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    opt = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    l1_lambda_start = 0.0
    l1_lambda_end = 5e-4
    batch_size = 4096
    n_epochs = 2000
    warmup = 500

    for epoch in range(n_epochs):
        # Anneal L1 from 0 to target over warmup epochs
        if epoch < warmup:
            l1_lambda = l1_lambda_start + (l1_lambda_end - l1_lambda_start) * epoch / warmup
        else:
            l1_lambda = l1_lambda_end

        perm = torch.randperm(len(X_train_t), device=device)
        for start in range(0, len(X_train_t), batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X_train_t[idx])
            loss = (pred - y_train_t[idx]).pow(2).mean()
            loss = loss + l1_lambda * model.l1_reg()
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                train_mse = (model(X_train_t) - y_train_t).pow(2).mean().item()
                test_mse = (model(X_test_t) - y_test_t).pow(2).mean().item()
            model.train()
            print(f"epoch {epoch:4d} | train MSE {train_mse:.4f} | test MSE {test_mse:.4f} | "
                  f"lr {sched.get_last_lr()[0]:.2e} | L1 {l1_lambda:.2e}")
        sched.step()

    # Final evaluation on CPU
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).cpu().numpy()
        pred_test = model(X_test_t).cpu().numpy()

    train_rho = spearmanr(y_train, pred_train)[0]
    test_rho = spearmanr(y_test, pred_test)[0]
    print(f"\nOverall Spearman ρ — train: {train_rho:.4f}, test: {test_rho:.4f}")

    # Within-group metrics
    r2s, rhos = [], []
    for g in variant_groups:
        idx_in_test = [i for i in g if i in test_idx_set]
        if len(idx_in_test) >= 3:
            yg = y[idx_in_test]
            pg = pred_test[[test_idx.index(i) for i in idx_in_test]]
            ss_res = np.sum((yg - pg) ** 2)
            ss_tot = np.sum((yg - yg.mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            rhos.append(spearmanr([entries[i]['time_us'] for i in idx_in_test], pg)[0])
    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"Within-group — R²: {r2s.mean():.4f}, ρ: {rhos.mean():.4f} ± {rhos.std():.4f}")

    # Sparsity
    for thresh in [1e-6, 1e-4, 1e-3, 1e-2]:
        total = sum(p.numel() for n, p in model.named_parameters() if 'weight' in n)
        nz = sum((p.abs() < thresh).sum().item() for n, p in model.named_parameters() if 'weight' in n)
        print(f"Sparsity @ {thresh:.0e}: {nz / total * 100:.0f}% ({nz}/{total} weights)")
    all_weights = torch.cat([p.flatten() for n, p in model.named_parameters() if 'weight' in n])
    print(f"Weight stats: min={all_weights.min().item():.2e} max={all_weights.max().item():.2e} "
          f"mean={all_weights.mean().item():.2e} median={all_weights.median().item():.2e}")


if __name__ == '__main__':
    main()
