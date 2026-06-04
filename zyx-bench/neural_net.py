#!/usr/bin/env python3
"""Train a sparse MLP on bench_data.csv to predict kernel cost rank."""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tqdm import tqdm

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'

# ---- Engineered features (from regression.py) ----
FEATURE_DEFS = []
FEATURE_NAMES = []

def register_feature(name, fn):
    FEATURE_DEFS.append((name, fn))
    FEATURE_NAMES.append(name)

register_feature('lng', lambda e: np.log(e['num_groups']))
register_feature('lwpg', lambda e: np.log(e['wi_per_group'] + 1))
register_feature('lops', lambda e: np.log(e['wi_ops']))
register_feature('lcop', lambda e: np.log(e['wi_compute_ops']))
register_feature('lgmem', lambda e: np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('barr', lambda e: e['wi_barriers'])
register_feature('wr', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
register_feature('rr', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1))
register_feature('total_threads', lambda e: e['num_groups'] * e['wi_per_group'])
register_feature('overhead', lambda e: e['wi_ops'] / max(e['wi_compute_ops'], 1))
register_feature('ci', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
register_feature('log1p_ci', lambda e: np.log1p(e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
register_feature('log1p_overhead', lambda e: np.log1p(e['wi_ops'] / max(e['wi_compute_ops'], 1)))
register_feature('log1p_mp', lambda e: np.log1p((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1)))
register_feature('log1p_lm', lambda e: np.log1p((e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1, 1)))
register_feature('log_ops_100', lambda e: np.log(e['wi_ops'] + 100))
register_feature('log_ops_10', lambda e: np.log(e['wi_ops'] + 10))
register_feature('log_cops_100', lambda e: np.log(e['wi_compute_ops'] + 100))
register_feature('log1p_1000_div_ops', lambda e: np.log1p(1000 / max(e['wi_ops'], 1)))
register_feature('log1p_100_div_ops', lambda e: np.log1p(100 / max(e['wi_ops'], 1)))
register_feature('log1p_1000_div_cops', lambda e: np.log1p(1000 / max(e['wi_compute_ops'], 1)))
register_feature('ops_per_thread', lambda e: e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
register_feature('cops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
register_feature('ops_per_group', lambda e: e['wi_ops'] / max(e['num_groups'], 1))
register_feature('log_lwpg', lambda e: np.log(max(np.log(e['wi_per_group'] + 1), 1e-8)))
register_feature('raw_ng', lambda e: e['num_groups'])
register_feature('raw_ng_wpg', lambda e: e['num_groups'] * e['wi_per_group'])
register_feature('lbranch', lambda e: np.log1p(e.get('wi_branches', 0)))
register_feature('barr*lbranch', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_branches', 0)))
register_feature('lwpg*lbranch', lambda e: np.log(e['wi_per_group'] + 1) * np.log1p(e.get('wi_branches', 0)))
register_feature('lng_lwpg', lambda e: np.log(e['num_groups']) * np.log(e['wi_per_group'] + 1))
register_feature('lwpg_lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops']))
register_feature('log_ops_per_group', lambda e: np.log(e['wi_ops'] / max(e['num_groups'], 1) + 1))
register_feature('log_opt', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))
register_feature('inv_threads', lambda e: 1.0 / max(e['num_groups'] * e['wi_per_group'], 1))
register_feature('log_inv_threads', lambda e: np.log1p(1.0 / max(e['num_groups'] * e['wi_per_group'], 1)))
register_feature('log_opt_barr', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'] * max(e['wi_barriers'], 1), 1)))
register_feature('log_wpg_barr', lambda e: np.log1p(e['wi_per_group'] * e['wi_barriers']))
for bval in [0, 3, 4, 5, 6, 7, 8]:
    register_feature(f'b{bval}', lambda e, b=bval: 1.0 if e['wi_barriers'] == b else 0.0)
register_feature('barr_low', lambda e: 1.0 if e['wi_barriers'] == 0 else 0.0)
register_feature('barr_med', lambda e: 1.0 if e['wi_barriers'] in [3, 4, 5] else 0.0)
register_feature('barr_high', lambda e: 1.0 if e['wi_barriers'] >= 6 else 0.0)
register_feature('barr_nonzero', lambda e: 1.0 if e['wi_barriers'] > 0 else 0.0)
for bval in [0, 3, 4, 5, 6, 7, 8]:
    register_feature(f'b{bval}*lng', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['num_groups']))
    register_feature(f'b{bval}*lwpg', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['wi_per_group'] + 1))
    register_feature(f'b{bval}*lops', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['wi_ops']))
register_feature('barr*lops', lambda e: e['wi_barriers'] * np.log(e['wi_ops']))
register_feature('barr*lcop', lambda e: e['wi_barriers'] * np.log(e['wi_compute_ops']))
register_feature('barr*lgmem', lambda e: e['wi_barriers'] * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('barr*lwpg', lambda e: e['wi_barriers'] * np.log(e['wi_per_group'] + 1))
register_feature('barr*lng', lambda e: e['wi_barriers'] * np.log(e['num_groups']))
register_feature('barr*wr', lambda e: e['wi_barriers'] * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
register_feature('lld_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)))
register_feature('lst_st', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)))
register_feature('lld_st*lops', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_ops']))
register_feature('lld_st*lcop', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_compute_ops']))
register_feature('lld_st*lgmem', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lops*lgmem', lambda e: np.log(e['wi_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lcop*lgmem', lambda e: np.log(e['wi_compute_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lng*lgmem', lambda e: np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lwpg*lgmem', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lwpg*lops_w', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops']))
register_feature('lng*lcop', lambda e: np.log(e['num_groups']) * np.log(e['wi_compute_ops']))
register_feature('lng*lgmem_w', lambda e: np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('warp_util', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
register_feature('log_warp_waste', lambda e: np.log(32 / max(e['wi_per_group'], 1)))
register_feature('lng_warp_waste', lambda e: np.log(e['num_groups']) * np.log(32 / max(e['wi_per_group'], 1)))
register_feature('lwpg_warp_waste', lambda e: np.log(e['wi_per_group'] + 1) * np.log(32 / max(e['wi_per_group'], 1)))
register_feature('coalescing_eff', lambda e: 1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0))
register_feature('coalescing_eff_st', lambda e: 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0))
register_feature('reduce_kind', lambda e: e['wi_barriers'] / max(np.log2(e['wi_per_group'] + 1), 1.0))
register_feature('element_ops', lambda e: e['wi_compute_ops'] / max(e['num_groups'], 1))
register_feature('warp_div', lambda e: np.log(max(32 / max(e['wi_per_group'], 1), 1e-8)))
register_feature('sm_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0))
register_feature('tree_height', lambda e: np.log2(e['wi_barriers'] + 1) if e['wi_barriers'] > 0 else 0)
register_feature('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)))
register_feature('element_ops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
register_feature('shared_mem_pressure', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0)
register_feature('warp_waste', lambda e: (32.0 - e['wi_per_group']) / 32.0)
register_feature('compute_per_barrier', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * max(e['wi_barriers'], 1), 1))
register_feature('barrier_overhead_cost', lambda e: e['wi_barriers'] * np.log1p(e['wi_barriers']) / max(np.log1p(e['wi_ops']), 1))
register_feature('group_work_density', lambda e: e['wi_ops'] / max(e['num_groups'] * np.log1p(e['wi_per_group']), 1))
register_feature('ops_per_barrier', lambda e: e['wi_ops'] / max(e['wi_barriers'], 1))
register_feature('barrier_factor', lambda e: e['wi_barriers'] / max(np.log1p(e['wi_ops']), 1))
register_feature('barrier_efficiency', lambda e: e['wi_barriers'] / max(np.log1p(e['wi_ops']), 1))
register_feature('mem_compute_ratio_log', lambda e: np.log((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1) + 1))
register_feature('log1p_lng', lambda e: np.log1p(np.abs(np.log(e['num_groups']))))
register_feature('log1p_lwpg_raw', lambda e: np.log1p(np.abs(np.log(e['wi_per_group'] + 1))))
register_feature('log1p_lops_raw', lambda e: np.log1p(np.abs(np.log(e['wi_ops']))))
register_feature('log_ng_per_ops', lambda e: np.log(e['num_groups'] / max(e['wi_ops'], 1) + 1))
register_feature('log_threads_per_ops', lambda e: np.log((e['num_groups'] * e['wi_per_group']) / max(e['wi_ops'], 1) + 1))
register_feature('barr_med_lops', lambda e: (1.0 if e['wi_barriers'] in [3, 4, 5] else 0.0) * np.log(e['wi_ops']))
register_feature('barr_nonzero_lops', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_ops']))
register_feature('barr_nonzero_lmem', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('compute_density', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
register_feature('thread_compute', lambda e: np.log(e['wi_compute_ops'] + 1) * np.log(e['num_groups'] * e['wi_per_group'] + 1))
register_feature('bgt_lng', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['num_groups']))
register_feature('bgt_lwpg', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_per_group'] + 1))
register_feature('bgt_lops', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_ops']))
register_feature('bgt_lcop', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_compute_ops']))
register_feature('bgt_lgmem', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
register_feature('lreg', lambda e: np.log1p(e.get('wi_register_load_bits', 0)))
register_feature('lreg_st', lambda e: np.log1p(e.get('wi_register_store_bits', 0)))
register_feature('reg_pressure', lambda e: e.get('wi_register_load_bits', 0) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
register_feature('lgws0', lambda e: np.log1p(e.get('gws0', 1)))
register_feature('lgws1', lambda e: np.log1p(e.get('gws1', 1)))
register_feature('lgws2', lambda e: np.log1p(e.get('gws2', 1)))
register_feature('llws0', lambda e: np.log1p(e.get('lws0', 1)))
register_feature('llws1', lambda e: np.log1p(e.get('lws1', 1)))
register_feature('llws2', lambda e: np.log1p(e.get('lws2', 1)))
register_feature('lws0_ratio', lambda e: e.get('lws0', 1) / max(e['wi_per_group'], 1))
register_feature('lws1_ratio', lambda e: e.get('lws1', 1) / max(e['wi_per_group'], 1))
register_feature('lws2_ratio', lambda e: e.get('lws2', 1) / max(e['wi_per_group'], 1))
register_feature('lws0xlws1', lambda e: e.get('lws0', 1) * e.get('lws1', 1))
register_feature('gws0_ratio', lambda e: e.get('gws0', 1) / max(e['num_groups'], 1))
register_feature('gws1_ratio', lambda e: e.get('gws1', 1) / max(e['num_groups'], 1))
register_feature('loop_depth', lambda e: e.get('max_loop_depth', 0))
register_feature('loop_depth*lops', lambda e: e.get('max_loop_depth', 0) * np.log(max(e['wi_ops'], 1)))
register_feature('loop_depth*barr', lambda e: e.get('max_loop_depth', 0) * e['wi_barriers'])
register_feature('log_local_mem', lambda e: np.log1p(e.get('local_mem_size', 1)))
register_feature('pref_vec', lambda e: e.get('preferred_vector_size', 0))
register_feature('total_global_load', lambda e: e['wi_global_load_bits'] * e['num_groups'] * e['wi_per_group'])
register_feature('total_global_store', lambda e: e['wi_global_store_bits'] * e['num_groups'] * e['wi_per_group'])
register_feature('total_global_mem', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) * e['num_groups'] * e['wi_per_group'])
register_feature('log_total_global_load', lambda e: np.log(max(e['wi_global_load_bits'] * e['num_groups'] * e['wi_per_group'], 1)))
register_feature('log_total_global_mem', lambda e: np.log(max((e['wi_global_load_bits'] + e['wi_global_store_bits']) * e['num_groups'] * e['wi_per_group'], 1)))
register_feature('tile_ratio', lambda e: (e['wi_local_load_bits'] + e['wi_local_store_bits']) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
register_feature('mem_per_compute', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) * e['num_groups'] * e['wi_per_group'] / max(e['wi_compute_ops'] * e['num_groups'] * e['wi_per_group'], 1))
register_feature('log_mem_per_compute', lambda e: np.log(max((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1), 1e-8)))
register_feature('occ_regs', lambda e: e['max_register_bytes'] / max(e['wi_peak_reg_bytes'], 1))
register_feature('occ_threads', lambda e: e['max_local_threads'] / max(e['wi_per_group'], 1))
register_feature('lmem_global_ratio', lambda e: e['wi_local_load_bits'] / max(e['wi_global_load_bits'], 1))
register_feature('total_local_traffic', lambda e: (e['wi_local_load_bits'] + e['wi_local_store_bits']) * e['num_groups'] * e['wi_per_group'])
register_feature('log_total_local', lambda e: np.log(max((e['wi_local_load_bits'] + e['wi_local_store_bits']) * e['num_groups'] * e['wi_per_group'], 1)))
register_feature('barrier_intensity', lambda e: e['wi_barriers'] / max(e['wi_compute_ops'], 1))
register_feature('occ_combined', lambda e: min(e['max_register_bytes'] / max(e['wi_peak_reg_bytes'], 1), e['max_local_threads'] / max(e['wi_per_group'], 1)))
register_feature('log_occ_combined', lambda e: np.log(max(min(e['max_register_bytes'] / max(e['wi_peak_reg_bytes'], 1), e['max_local_threads'] / max(e['wi_per_group'], 1)), 1e-8)))
register_feature('mem_per_warp', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) * 32 / max(e['wi_per_group'], 1))
register_feature('compute_per_warp', lambda e: e['wi_compute_ops'] * 32 / max(e['wi_per_group'], 1))
register_feature('log_global_per_local', lambda e: np.log(max((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_local_load_bits'] + e['wi_local_store_bits'], 1), 1e-8)))

RAW_NAMES = [
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
N_RAW = len(RAW_NAMES)
N_ENG = len(FEATURE_NAMES)
N_INT = N_RAW * (N_RAW - 1) // 2


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=True))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(nn.SELU())
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

    # Engineered features
    X_eng = np.array([[fn(e) for _, fn in FEATURE_DEFS] for e in entries], dtype=np.float32)

    # Raw features + log transform
    raw = np.array([[e[f] for f in RAW_NAMES] for e in entries], dtype=np.float32)
    for c in [0, 1, 2, 3, 5, 6, 7, 8, 10, 18, 19, 27]:
        raw[:, c] = np.log(np.maximum(raw[:, c], 1.0))

    # All pairwise interactions of raw features
    interactions = []
    for i in range(N_RAW):
        for j in range(i + 1, N_RAW):
            interactions.append(raw[:, i] * raw[:, j])
    X_int = np.column_stack(interactions)

    X = np.column_stack([X_eng, X_int])
    print(f"Features: eng={N_ENG} + interactions={N_INT} = {X.shape[1]}")

    # Fixed 20k test split by variant_hash
    rng = np.random.RandomState(42)
    hashes = list(hash_map.keys())
    rng.shuffle(hashes)
    test_idx = np.array([], dtype=int)
    for h in hashes:
        if len(test_idx) + len(hash_map[h]) > 20000:
            break
        test_idx = np.concatenate([test_idx, hash_map[h]])
    test_mask = np.zeros(len(entries), dtype=bool)
    test_mask[test_idx] = True
    train_mask = ~test_mask

    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X[train_mask])
    X_test = scaler.transform(X[test_mask])
    y_train_all, y_test = y[train_mask], y[test_mask]

    device = torch.device('cuda')
    X_test_t = torch.tensor(X_test, device=device)

    # Build group index: local indices into training subset
    train_global_to_local = {int(i): int(idx) for idx, i in enumerate(np.where(train_mask)[0])}
    train_hash_map = defaultdict(list)
    for i in np.where(train_mask)[0]:
        train_hash_map[entries[i]['variant_hash']].append(i)
    train_groups = [torch.tensor([train_global_to_local[int(i)] for i in v], device=device)
                    for v in train_hash_map.values() if len(v) >= 4]
    print(f"Training groups with >=4 variants: {len(train_groups)}")

    X_train_t = torch.tensor(X_train_all, device=device)
    y_train_t = torch.tensor(y_train_all, device=device)

    model = MLP([X.shape[1], 500, 200, 1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    l1_lambda = 5e-5
    margin = 0.1

    n_steps = 30000
    n_groups = len(train_groups)
    pbar = tqdm(total=n_steps, desc="Training")
    step = 0
    perm = np.arange(n_groups)
    while step < n_steps:
        rng.shuffle(perm)
        for gi in perm:
            if step >= n_steps:
                break
            local_idx = train_groups[gi]
            pred = model(X_train_t[local_idx])
            targets = y_train_t[local_idx]

            diff = pred.unsqueeze(0) - pred.unsqueeze(1)  # pred_i - pred_j
            # S_ij = 1 if target_i < target_j (i faster), -1 if target_i > target_j (i slower)
            S = torch.sign(targets.unsqueeze(1) - targets.unsqueeze(0))
            loss = torch.relu(S * diff + margin).mean()
            loss = loss + l1_lambda * model.l1_reg()
            opt.zero_grad()
            loss.backward()
            opt.step()

            step += 1
            if step % 100 == 0:
                pbar.update(100)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
    if step % 100 != 0:
        pbar.update(step % 100)
    pbar.close()

    model.eval()
    with torch.no_grad():
        pred_train = model(X_train_t).cpu().numpy()
        pred_test = model(X_test_t).cpu().numpy()
    train_rho = spearmanr(y_train_all, pred_train)[0]
    test_rho = spearmanr(y_test, pred_test)[0]
    print(f"\nOverall Spearman ρ — train: {train_rho:.4f}, test: {test_rho:.4f}")

    # Within-group on train set
    train_group_map = defaultdict(list)
    for i in np.where(train_mask)[0]:
        train_group_map[entries[i]['variant_hash']].append(i)
    train_rhos = []
    for g in train_group_map.values():
        if len(g) >= 3:
            local_idx = [train_global_to_local[int(i)] for i in g]
            train_rhos.append(spearmanr([entries[i]['time_us'] for i in g], pred_train[local_idx])[0])
    train_rhos = np.array(train_rhos)
    print(f"Train within-group ρ: {np.nanmean(train_rhos):.4f} ± {np.nanstd(train_rhos):.4f}")

    test_global_to_local = {int(i): int(idx) for idx, i in enumerate(test_idx)}
    test_group_map = defaultdict(list)
    for i in test_idx:
        h = entries[i]['variant_hash']
        test_group_map[h].append(i)
    r2s, rhos = [], []
    n_debug = 0
    for g in test_group_map.values():
        if len(g) >= 3:
            local_idx = [test_global_to_local[int(i)] for i in g]
            yg = y[g]
            pg = pred_test[local_idx]
            ss_res = np.sum((yg - pg) ** 2)
            ss_tot = np.sum((yg - yg.mean()) ** 2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0)
            rho = spearmanr([entries[i]['time_us'] for i in g], pg)[0]
            rhos.append(rho)
            if n_debug < 3 and not np.isnan(rho):
                print(f"\n  Test group {g[0]:6d}: {len(g)} variants, ρ={rho:.3f}")
                times = [entries[i]['time_us'] for i in g]
                order = np.argsort(times)
                print(f"    actual rank: {np.argsort(np.argsort(times))}")
                print(f"    pred rank:   {np.argsort(np.argsort(pg))}")
                print(f"    pred vals:   {np.array([f'{v:.3f}' for v in pg])}")
                n_debug += 1
    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"\nTest within-group — R²: {r2s.mean():.4f}, ρ: {rhos.mean():.4f} ± {rhos.std():.4f}")
    print(f"  Worst 5% ρ: {np.quantile(rhos, 0.05):.4f}, Best 5% ρ: {np.quantile(rhos, 0.95):.4f}")


if __name__ == '__main__':
    main()
