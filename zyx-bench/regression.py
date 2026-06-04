#!/usr/bin/env python3
"""Train cost model: Ridge on ~167 engineered features, no DT in Rust code."""

import os
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr as _spearmanr
from collections import defaultdict

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'

# ---- Feature registration ----
FEATURE_DEFS = []
FEATURE_NAMES = []

def register_feature(name, fn):
    FEATURE_DEFS.append((name, fn))
    FEATURE_NAMES.append(name)

def register_all_features():
    if FEATURE_DEFS:
        return
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
    register_feature('lops*ci', lambda e: np.log(e['wi_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    register_feature('lcop*ci', lambda e: np.log(e['wi_compute_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    register_feature('lng*ci', lambda e: np.log(e['num_groups']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    register_feature('lng*wr', lambda e: np.log(e['num_groups']) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    register_feature('lwpg*wr', lambda e: np.log(e['wi_per_group'] + 1) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    register_feature('lng*rr', lambda e: np.log(e['num_groups']) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)))
    register_feature('lwpg*rr', lambda e: np.log(e['wi_per_group'] + 1) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)))
    register_feature('lng*lcop', lambda e: np.log(e['num_groups']) * np.log(e['wi_compute_ops']))
    register_feature('lwpg*lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops']))
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
    register_feature('warp_efficiency', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
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
    # New IR features
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
    # Compute/memory bound
    register_feature('comp_intensity', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    register_feature('is_comp_bound', lambda e: 1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0)
    register_feature('is_mem_bound', lambda e: 1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0)
    register_feature('cb*lgmem', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    register_feature('cb*lops', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_ops']))
    register_feature('cb*lcop', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_compute_ops']))
    register_feature('cb*lng', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['num_groups']))
    register_feature('cb*lwpg', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_per_group'] + 1))
    register_feature('cb*barr', lambda e: (1.0 if e['wi_compute_ops'] > 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * e['wi_barriers'])
    register_feature('mb*lgmem', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    register_feature('mb*lops', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_ops']))
    register_feature('mb*lcop', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_compute_ops']))
    register_feature('mb*lng', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['num_groups']))
    register_feature('mb*lwpg', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * np.log(e['wi_per_group'] + 1))
    register_feature('mb*barr', lambda e: (1.0 if e['wi_compute_ops'] <= 0.2 * max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1) else 0.0) * e['wi_barriers'])


def build_feature_matrix(entries):
    X = None
    for _, fn in FEATURE_DEFS:
        feat = np.array([fn(e) for e in entries]).reshape(-1, 1)
        X = feat if X is None else np.column_stack([X, feat])
    return X


# ---- Rust codegen ----
def _feature_to_rust(name):
    d = {
        'lng': 'lng', 'lwpg': 'lwpg', 'lops': 'lops', 'lcop': 'lcop', 'lgmem': 'lgmem',
        'barr': 'barr', 'wr': 'wr', 'rr': 'rr', 'ci': 'ci',
        'log_inv_threads': '(1.0 / (num_groups * wi_per_group).max(1.0)).ln_1p()',
        'overhead': 'wi_ops / wi_compute_ops.max(1.0)',
        'log1p_ci': '(wi_compute_ops / (wi_global_load_bits + wi_global_store_bits).max(1.0)).ln_1p()',
        'log1p_overhead': '(wi_ops / wi_compute_ops.max(1.0)).ln_1p()',
        'total_threads': 'num_groups * wi_per_group',
        'raw_ng': 'num_groups',
        'raw_ng_wpg': 'num_groups * wi_per_group',
        'warp_util': 'wr',
        'log_warp_waste': '(32.0 / wi_per_group.max(1.0)).ln()',
        'lng_warp_waste': 'lng * (32.0 / wi_per_group.max(1.0)).ln()',
        'lwpg_warp_waste': 'lwpg * (32.0 / wi_per_group.max(1.0)).ln()',
        'llm': '(wi_local_load_bits + wi_local_store_bits).ln_1p()',
        'barr_wpg': 'barr * (wi_per_group + 1.0).ln_1p()',
        'reduce_kind': 'barr / (wi_per_group + 1.0).log2().max(1.0)',
        'element_ops': 'wi_compute_ops / num_groups.max(1.0)',
        'warp_div': '(32.0 / wi_per_group.max(1.0)).max(1e-8).ln()',
        'sm_occupancy': '((num_groups * wi_per_group) / 2048.0).min(1.0)',
        'tree_height': 'if wi_barriers > 0.0 { (wi_barriers + 1.0).log2() } else { 0.0 }',
        'tree_reduce_cost': 'barr * (wi_local_load_bits + wi_local_store_bits).ln_1p()',
        'element_ops_per_thread': 'wi_compute_ops / (num_groups * wi_per_group).max(1.0)',
        'coalescing_eff': '1.0 - (wi_global_load_lidx_stride / 32.0).min(1.0)',
        'coalescing_eff_st': '1.0 - (wi_global_store_lidx_stride / 32.0).min(1.0)',
        'lld_st': 'wi_global_load_lidx_stride.ln_1p()',
        'lst_st': 'wi_global_store_lidx_stride.ln_1p()',
        'log_ops_100': '(wi_ops + 100.0).ln()',
        'log_ops_10': '(wi_ops + 10.0).ln()',
        'log_cops_100': '(wi_compute_ops + 100.0).ln()',
        'log1p_1000_div_ops': '(1000.0 / wi_ops.max(1.0)).ln_1p()',
        'log1p_100_div_ops': '(100.0 / wi_ops.max(1.0)).ln_1p()',
        'log1p_1000_div_cops': '(1000.0 / wi_compute_ops.max(1.0)).ln_1p()',
        'ops_per_thread': 'wi_ops / (num_groups * wi_per_group).max(1.0)',
        'cops_per_thread': 'wi_compute_ops / (num_groups * wi_per_group).max(1.0)',
        'ops_per_group': 'wi_ops / num_groups.max(1.0)',
        'log_lwpg': 'lwpg.max(1e-8).ln()',
        'log_opt': '(wi_ops / (num_groups * wi_per_group).max(1.0)).ln_1p()',
        'inv_threads': '1.0 / (num_groups * wi_per_group).max(1.0)',
        'lbranch': 'wi_branches.ln_1p()',
        'lng_lwpg': 'lng * lwpg',
        'lwpg_lops': 'lwpg * lops',
        'lng*lgmem': 'lng * lgmem',
        'lcop*lgmem': 'lcop * lgmem',
        'lops*lgmem': 'lops * lgmem',
        'lwpg*lgmem': 'lwpg * lgmem',
        'lng*lcop': 'lng * lcop',
        'lwpg*lops': 'lwpg * lops',
        'lng_sq': 'lng * lng',
        'lcop_sq': 'lcop * lcop',
        'lgmem_sq': 'lgmem * lgmem',
        'lops_sq': 'lops * lops',
        'lwpg_sq': 'lwpg * lwpg',
        'log1p_mp': '((wi_global_load_bits + wi_global_store_bits) / (num_groups * wi_per_group).max(1.0)).ln_1p()',
        'log1p_lm': '((wi_local_load_bits + wi_local_store_bits + 1.0) / (wi_global_load_bits + wi_global_store_bits + 1.0).max(1.0)).ln_1p()',
        'log_ops_per_group': '(wi_ops / num_groups.max(1.0) + 1.0).ln()',
        'log_opt_barr': '(wi_ops / (num_groups * wi_per_group * wi_barriers.max(1.0)).max(1.0)).ln_1p()',
        'log_wpg_barr': '(wi_per_group * wi_barriers).ln_1p()',
        'barr_low': 'if wi_barriers == 0.0 { 1.0 } else { 0.0 }',
        'barr_med': 'if wi_barriers >= 3.0 && wi_barriers <= 5.0 { 1.0 } else { 0.0 }',
        'barr_high': 'if wi_barriers >= 6.0 { 1.0 } else { 0.0 }',
        'barr_nonzero': 'if wi_barriers > 0.0 { 1.0 } else { 0.0 }',
        'barr*lops': 'barr * lops',
        'barr*lcop': 'barr * lcop',
        'barr*lgmem': 'barr * lgmem',
        'barr*lwpg': 'barr * lwpg',
        'barr*lng': 'barr * lng', 'barr*wr': 'barr * wr',
        'shared_mem_pressure': '(wi_local_load_bits + wi_local_store_bits) / 65536.0',
        'warp_efficiency': 'wr',
        'warp_waste': '(32.0 - wi_per_group) / 32.0',
        'compute_per_barrier': 'wi_compute_ops / (num_groups * wi_barriers.max(1.0)).max(1.0)',
        'barrier_overhead_cost': 'barr * (barr).ln_1p() / (wi_ops).ln_1p().max(1.0)',
        'group_work_density': 'wi_ops / (num_groups * (wi_per_group).ln_1p()).max(1.0)',
        'ops_per_barrier': 'wi_ops / wi_barriers.max(1.0)',
        'barrier_factor': 'barr / (wi_ops).ln_1p().max(1.0)',
        'barrier_efficiency': 'barr / (wi_ops).ln_1p().max(1.0)',
        'mem_compute_ratio_log': '((wi_global_load_bits + wi_global_store_bits) / wi_compute_ops.max(1.0) + 1.0).ln()',
        'log1p_lng': '(lng.abs()).ln_1p()',
        'log1p_lwpg_raw': '(lwpg.abs()).ln_1p()',
        'log1p_lops_raw': '(lops.abs()).ln_1p()',
        'log_ng_per_ops': '(num_groups / wi_ops.max(1.0) + 1.0).ln()',
        'log_threads_per_ops': '((num_groups * wi_per_group) / wi_ops.max(1.0) + 1.0).ln()',
        'barr_med_lops': '(if wi_barriers >= 3.0 && wi_barriers <= 5.0 { 1.0 } else { 0.0 }) * lops',
        'barr_nonzero_lops': '(if wi_barriers > 0.0 { 1.0 } else { 0.0 }) * lops',
        'barr_nonzero_lmem': '(if wi_barriers > 0.0 { 1.0 } else { 0.0 }) * lgmem',
        'compute_density': 'wi_compute_ops / (num_groups * wi_per_group).max(1.0)',
        'thread_compute': '(wi_compute_ops + 1.0).ln() * ((num_groups * wi_per_group) + 1.0).ln()',
        'barr*lbranch': 'barr * wi_branches.ln_1p()',
        'lwpg*lbranch': 'lwpg * wi_branches.ln_1p()',
        'lld_st*lops': 'wi_global_load_lidx_stride.ln_1p() * lops',
        'lld_st*lcop': 'wi_global_load_lidx_stride.ln_1p() * lcop',
        'lld_st*lgmem': 'wi_global_load_lidx_stride.ln_1p() * lgmem',
        'lng_lgmem': 'lng * lgmem',
        'lng_lcop': 'lng * lcop',
        'lreg': 'wi_register_load_bits.ln_1p()',
        'lreg_st': 'wi_register_store_bits.ln_1p()',
        'reg_pressure': 'wi_register_load_bits / (wi_global_load_bits + wi_global_store_bits).max(1.0)',
        'lgws0': 'gws0.ln_1p()',
        'lgws1': 'gws1.ln_1p()',
        'lgws2': 'gws2.ln_1p()',
        'llws0': 'lws0.ln_1p()',
        'llws1': 'lws1.ln_1p()',
        'llws2': 'lws2.ln_1p()',
        'lws0_ratio': 'lws0 / wi_per_group.max(1.0)',
        'lws1_ratio': 'lws1 / wi_per_group.max(1.0)',
        'lws2_ratio': 'lws2 / wi_per_group.max(1.0)',
        'lws0xlws1': 'lws0 * lws1',
        'gws0_ratio': 'gws0 / num_groups.max(1.0)',
        'gws1_ratio': 'gws1 / num_groups.max(1.0)',
        'loop_depth': 'max_loop_depth',
        'loop_depth*lops': 'max_loop_depth * lops',
        'loop_depth*barr': 'max_loop_depth * barr',
        'log_local_mem': 'local_mem_size.ln_1p()',
        'pref_vec': 'preferred_vector_size',
        'comp_intensity': 'ci',
        'is_comp_bound': 'is_comp_bound',
        'is_mem_bound': 'is_mem_bound',
        'cb*lgmem': 'is_comp_bound * lgmem',
        'cb*lops': 'is_comp_bound * lops',
        'cb*lcop': 'is_comp_bound * lcop',
        'cb*lng': 'is_comp_bound * lng',
        'cb*lwpg': 'is_comp_bound * lwpg',
        'cb*barr': 'is_comp_bound * barr',
        'mb*lgmem': 'is_mem_bound * lgmem',
        'mb*lops': 'is_mem_bound * lops',
        'mb*lcop': 'is_mem_bound * lcop',
        'mb*lng': 'is_mem_bound * lng',
        'mb*lwpg': 'is_mem_bound * lwpg',
        'mb*barr': 'is_mem_bound * barr',
    }
    if name in d:
        return d[name]
    if '*' in name:
        parts = name.split('*')
        return ' * '.join(_feature_to_rust(p) for p in parts)
    if name.startswith('bgt_'):
        rest = name[4:]
        return f'(if wi_barriers > 0.0 {{ 1.0 }} else {{ 0.0 }}) * {_feature_to_rust(rest)}'
    if name.startswith('b') and len(name) > 1 and name[1:].isdigit():
        return f'(if wi_barriers == {name[1:]}.0 {{ 1.0 }} else {{ 0.0 }})'
    return f'/* UNKNOWN: {name} */ 0.0'


# ---- MAIN ----
def main():
    import sys
    import pandas as pd
    matmul_only = '--matmul' in sys.argv
    df = pd.read_csv(BENCH_CSV)
    if matmul_only:
        df = df.tail(9000)
        print(f"Filtered to last {len(df)} matmul entries")
    entries = df.to_dict('records')
    print(f"Read {len(entries)} entries from {BENCH_CSV}")

    hash_map = defaultdict(list)
    for i, e in enumerate(entries):
        h = e.get('variant_hash', '')
        hash_map[h].append(i)
    variant_groups = [v for v in hash_map.values() if len(v) >= 2]

    y = np.empty(len(entries))
    for g in variant_groups:
        times = np.array([entries[i]['time_us'] for i in g])
        ranks = np.argsort(np.argsort(times))
        for j, idx in enumerate(g):
            y[idx] = ranks[j] / (len(g) - 1)
    ungrouped = set(range(len(entries))) - set(idx for g in variant_groups for idx in g)
    for i in ungrouped:
        y[i] = 0.5

    register_all_features()
    X = build_feature_matrix(entries)
    n_feat = X.shape[1]
    print(f"Features: {n_feat}, Groups: {len(variant_groups)}")

    barr_vals = np.array([e['wi_barriers'] for e in entries])
    n_b0 = np.sum(barr_vals == 0)
    n_bg = np.sum(barr_vals > 0)
    sw = np.where(barr_vals > 0, 0.5 / n_bg, 0.5 / n_b0) * len(entries)
    sw /= np.mean(sw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    ridge_cv.fit(X_scaled, y, sample_weight=sw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=ridge_cv.alpha_)
    ridge.fit(X_scaled, y, sample_weight=sw)
    pred = ridge.predict(X_scaled)

    r2s, rhos = [], []
    for g in variant_groups:
        if len(g) >= 3:
            yg, pg = y[g], pred[g]
            ss_res = np.sum((yg - pg)**2)
            ss_tot = np.sum((yg - yg.mean())**2)
            r2s.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)
            rhos.append(_spearmanr([entries[i]['time_us'] for i in g], pg)[0])
    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"\n=== Ridge ({n_feat} features, no DT) ===")
    print(f"  Mean R²:          {r2s.mean():.4f}  (over {len(r2s)} kernels)")
    print(f"  Median R²:        {np.median(r2s):.4f}")
    print(f"  Worst 5% quantile:{np.quantile(r2s, 0.05):.4f}")
    print(f"  Worst R²:         {r2s.min():.4f}")
    print(f"  Best R²:          {r2s.max():.4f}")
    print(f"  Spearman ρ:       {rhos.mean():.4f} ± {rhos.std():.4f}")

    # Top-k metric: how many of actual fastest 10 are in model's top 20
    top10_in_top20 = []
    for g in variant_groups:
        n = len(g)
        if n < 20:
            continue
        times = np.array([entries[i]['time_us'] for i in g])
        actual_top10 = np.argsort(times)[:10]
        pred_top20 = np.argsort(pred[g])[:20]
        hits = len(set(actual_top10) & set(pred_top20))
        top10_in_top20.append(hits / 10.0)
    if top10_in_top20:
        print(f"  Top10-in-top20:   {np.mean(top10_in_top20):.3f} (over {len(top10_in_top20)} groups)")

    # Generate predict_cost.rs
    rust_path = os.path.join(os.path.dirname(__file__), '..', 'zyx', 'src', 'kernel', 'predict_cost.rs')
    with open(rust_path, 'w') as f:
        f.write("// Copyright (C) 2025 zk4x\n")
        f.write("// SPDX-License-Identifier: LGPL-3.0-only\n//\n")
        f.write(f"// Auto-generated by regression.py. {n_feat} features, Ridge.\n")
        f.write("#![allow(unused)]\nuse super::cost::Cost;\n")
        f.write("impl Cost {\n")
        f.write("    pub fn predict_time_us(\n")
        f.write("        num_groups: u32, wi_per_group: u32, wi_ops: u32,\n")
        f.write("        wi_compute_ops: u32, wi_barriers: u32,\n")
        f.write("        wi_global_load_bits: u32, wi_global_store_bits: u32,\n")
        f.write("        wi_local_load_bits: u32, wi_local_store_bits: u32,\n")
        f.write("        wi_peak_reg_bytes: u32, wi_branches: u32,\n")
        f.write("        wi_global_load_lidx_stride: u32, wi_global_store_lidx_stride: u32,\n")
        f.write("        wi_local_load_lidx_stride: u32, wi_local_store_lidx_stride: u32,\n")
        f.write("        warp_size: u32, max_local_threads: u32, max_register_bytes: u32,\n")
        f.write("        wi_register_load_bits: u32, wi_register_store_bits: u32,\n")
        f.write("        gws0: u32, gws1: u32, gws2: u32,\n")
        f.write("        lws0: u32, lws1: u32, lws2: u32,\n")
        f.write("        max_loop_depth: u32,\n")
        f.write("        preferred_vector_size: u32, local_mem_size: u32,\n")
        f.write("    ) -> f64 {\n")
        PARAMS = ['num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops', 'wi_barriers',
                  'wi_global_load_bits', 'wi_global_store_bits', 'wi_local_load_bits',
                  'wi_local_store_bits', 'wi_peak_reg_bytes', 'wi_branches',
                  'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
                  'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
                  'warp_size', 'max_local_threads', 'max_register_bytes',
                  'wi_register_load_bits', 'wi_register_store_bits',
                  'gws0', 'gws1', 'gws2', 'lws0', 'lws1', 'lws2', 'max_loop_depth',
                  'preferred_vector_size', 'local_mem_size']
        for var in PARAMS:
            f.write(f"        let {var} = {var} as f64;\n")
        f.write("        let lng = num_groups.ln();\n")
        f.write("        let lwpg = (wi_per_group + 1.0).ln();\n")
        f.write("        let lops = wi_ops.ln();\n")
        f.write("        let lcop = wi_compute_ops.ln();\n")
        f.write("        let lgmem = (wi_global_load_bits + wi_global_store_bits + 1.0).ln();\n")
        f.write("        let ci = wi_compute_ops / (wi_global_load_bits + wi_global_store_bits).max(1.0);\n")
        f.write("        let barr = wi_barriers;\n")
        f.write("        let wr = wi_per_group / warp_size.max(1.0);\n")
        f.write("        let rr = wi_peak_reg_bytes / max_register_bytes.max(1.0);\n")
        f.write("        let is_comp_bound = if ci > 0.2 { 1.0 } else { 0.0 };\n")
        f.write("        let is_mem_bound = if ci <= 0.2 { 1.0 } else { 0.0 };\n\n")
        f.write(f"        let features: [f64; {n_feat}] = [\n")
        for i in range(n_feat):
            expr = _feature_to_rust(FEATURE_NAMES[i])
            f.write(f"            {expr},\n")
        f.write("        ];\n\n")
        def write_arr(name, vals, n_col=8):
            f.write(f"        const {name}: &[f64] = &[\n")
            for i in range(0, len(vals), n_col):
                line = ', '.join(f'{v:.8e}' for v in vals[i:i+n_col])
                f.write(f"            {line},\n")
            f.write("        ];\n")
        write_arr('SM', scaler.mean_)
        write_arr('SC', scaler.scale_)
        write_arr('RC', ridge.coef_)
        f.write(f"        let ri: f64 = {ridge.intercept_:.8e};\n\n")
        f.write(f"        let mut pred = ri;\n")
        f.write(f"        for i in 0..{n_feat} {{\n")
        f.write("            pred += RC[i] * (features[i] - SM[i]) / SC[i];\n")
        f.write("        }\n")
        f.write("        pred * 1_000_000.0\n")
        f.write("    }\n}\n")
    print(f"Wrote {rust_path}")


if __name__ == '__main__':
    main()
