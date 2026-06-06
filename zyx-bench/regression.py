#!/usr/bin/env python3
"""Train cost model: Ridge on ~167 engineered features, no DT in Rust code."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import spearmanr as _spearmanr
warnings.filterwarnings('ignore')

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
    register_feature('lng', lambda df: np.log(df['num_groups']))
    register_feature('lwpg', lambda df: np.log(df['wi_per_group'] + 1))
    register_feature('lops', lambda df: np.log(df['wi_ops']))
    register_feature('lcop', lambda df: np.log(df['wi_compute_ops']))
    register_feature('lgmem', lambda df: np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('barr', lambda df: df['wi_barriers'])
    register_feature('wr', lambda df: df['wi_per_group'] / np.maximum(df['warp_size'], 1))
    register_feature('rr', lambda df: df['wi_peak_reg_bytes'].fillna(0) / np.maximum(df['max_register_bytes'], 1))
    register_feature('total_threads', lambda df: df['num_groups'] * df['wi_per_group'])
    register_feature('overhead', lambda df: df['wi_ops'] / np.maximum(df['wi_compute_ops'], 1))
    register_feature('ci', lambda df: df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1))
    register_feature('log1p_ci', lambda df: np.log1p(df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)))
    register_feature('log1p_overhead', lambda df: np.log1p(df['wi_ops'] / np.maximum(df['wi_compute_ops'], 1)))
    register_feature('log1p_mp', lambda df: np.log1p((df['wi_global_load_bits'] + df['wi_global_store_bits']) / np.maximum(df['num_groups'] * df['wi_per_group'], 1)))
    register_feature('log1p_lm', lambda df: np.log1p((df['wi_local_load_bits'].fillna(0) + df['wi_local_store_bits'].fillna(0) + 1) / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1, 1)))
    register_feature('log_ops_100', lambda df: np.log(df['wi_ops'] + 100))
    register_feature('log_ops_10', lambda df: np.log(df['wi_ops'] + 10))
    register_feature('log_cops_100', lambda df: np.log(df['wi_compute_ops'] + 100))
    register_feature('log1p_1000_div_ops', lambda df: np.log1p(1000 / np.maximum(df['wi_ops'], 1)))
    register_feature('log1p_100_div_ops', lambda df: np.log1p(100 / np.maximum(df['wi_ops'], 1)))
    register_feature('log1p_1000_div_cops', lambda df: np.log1p(1000 / np.maximum(df['wi_compute_ops'], 1)))
    register_feature('ops_per_thread', lambda df: df['wi_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'], 1))
    register_feature('cops_per_thread', lambda df: df['wi_compute_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'], 1))
    register_feature('ops_per_group', lambda df: df['wi_ops'] / np.maximum(df['num_groups'], 1))
    register_feature('log_lwpg', lambda df: np.log(np.maximum(np.log(df['wi_per_group'] + 1), 1e-8)))
    register_feature('raw_ng', lambda df: df['num_groups'])
    register_feature('raw_ng_wpg', lambda df: df['num_groups'] * df['wi_per_group'])
    register_feature('lbranch', lambda df: np.log1p(df['wi_branches'].fillna(0)))
    register_feature('barr*lbranch', lambda df: df['wi_barriers'] * np.log1p(df['wi_branches'].fillna(0)))
    register_feature('lwpg*lbranch', lambda df: np.log(df['wi_per_group'] + 1) * np.log1p(df['wi_branches'].fillna(0)))
    register_feature('lng_lwpg', lambda df: np.log(df['num_groups']) * np.log(df['wi_per_group'] + 1))
    register_feature('lwpg_lops', lambda df: np.log(df['wi_per_group'] + 1) * np.log(df['wi_ops']))
    register_feature('log_ops_per_group', lambda df: np.log(df['wi_ops'] / np.maximum(df['num_groups'], 1) + 1))
    register_feature('log_opt', lambda df: np.log1p(df['wi_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'], 1)))
    register_feature('inv_threads', lambda df: 1.0 / np.maximum(df['num_groups'] * df['wi_per_group'], 1))
    register_feature('log_inv_threads', lambda df: np.log1p(1.0 / np.maximum(df['num_groups'] * df['wi_per_group'], 1)))
    register_feature('log_opt_barr', lambda df: np.log1p(df['wi_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'] * np.maximum(df['wi_barriers'], 1), 1)))
    register_feature('log_wpg_barr', lambda df: np.log1p(df['wi_per_group'] * df['wi_barriers']))
    for bval in [0, 3, 4, 5, 6, 7, 8]:
        register_feature(f'b{bval}', lambda df, b=bval: (df['wi_barriers'] == b).astype(float))
    register_feature('barr_low', lambda df: (df['wi_barriers'] == 0).astype(float))
    register_feature('barr_med', lambda df: df['wi_barriers'].isin([3, 4, 5]).astype(float))
    register_feature('barr_high', lambda df: (df['wi_barriers'] >= 6).astype(float))
    register_feature('barr_nonzero', lambda df: (df['wi_barriers'] > 0).astype(float))
    for bval in [0, 3, 4, 5, 6, 7, 8]:
        register_feature(f'b{bval}*lng', lambda df, b=bval: (df['wi_barriers'] == b).astype(float) * np.log(df['num_groups']))
        register_feature(f'b{bval}*lwpg', lambda df, b=bval: (df['wi_barriers'] == b).astype(float) * np.log(df['wi_per_group'] + 1))
        register_feature(f'b{bval}*lops', lambda df, b=bval: (df['wi_barriers'] == b).astype(float) * np.log(df['wi_ops']))
    register_feature('barr*lops', lambda df: df['wi_barriers'] * np.log(df['wi_ops']))
    register_feature('barr*lcop', lambda df: df['wi_barriers'] * np.log(df['wi_compute_ops']))
    register_feature('barr*lgmem', lambda df: df['wi_barriers'] * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('barr*lwpg', lambda df: df['wi_barriers'] * np.log(df['wi_per_group'] + 1))
    register_feature('barr*lng', lambda df: df['wi_barriers'] * np.log(df['num_groups']))
    register_feature('barr*wr', lambda df: df['wi_barriers'] * (df['wi_per_group'] / np.maximum(df['warp_size'], 1)))
    register_feature('lld_st', lambda df: np.log1p(df['wi_global_load_lidx_stride'].fillna(0)))
    register_feature('lst_st', lambda df: np.log1p(df['wi_global_store_lidx_stride'].fillna(0)))
    register_feature('lld_st*lops', lambda df: np.log1p(df['wi_global_load_lidx_stride'].fillna(0)) * np.log(df['wi_ops']))
    register_feature('lld_st*lcop', lambda df: np.log1p(df['wi_global_load_lidx_stride'].fillna(0)) * np.log(df['wi_compute_ops']))
    register_feature('lld_st*lgmem', lambda df: np.log1p(df['wi_global_load_lidx_stride'].fillna(0)) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('lops*lgmem', lambda df: np.log(df['wi_ops']) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('lcop*lgmem', lambda df: np.log(df['wi_compute_ops']) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('lng*lgmem', lambda df: np.log(df['num_groups']) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('lwpg*lgmem', lambda df: np.log(df['wi_per_group'] + 1) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('lops*ci', lambda df: np.log(df['wi_ops']) * (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)))
    register_feature('lcop*ci', lambda df: np.log(df['wi_compute_ops']) * (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)))
    register_feature('lng*ci', lambda df: np.log(df['num_groups']) * (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)))
    register_feature('lng*wr', lambda df: np.log(df['num_groups']) * (df['wi_per_group'] / np.maximum(df['warp_size'], 1)))
    register_feature('lwpg*wr', lambda df: np.log(df['wi_per_group'] + 1) * (df['wi_per_group'] / np.maximum(df['warp_size'], 1)))
    register_feature('lng*rr', lambda df: np.log(df['num_groups']) * (df['wi_peak_reg_bytes'].fillna(0) / np.maximum(df['max_register_bytes'], 1)))
    register_feature('lwpg*rr', lambda df: np.log(df['wi_per_group'] + 1) * (df['wi_peak_reg_bytes'].fillna(0) / np.maximum(df['max_register_bytes'], 1)))
    register_feature('lng*lcop', lambda df: np.log(df['num_groups']) * np.log(df['wi_compute_ops']))
    register_feature('lwpg*lops', lambda df: np.log(df['wi_per_group'] + 1) * np.log(df['wi_ops']))
    register_feature('warp_util', lambda df: df['wi_per_group'] / np.maximum(df['warp_size'], 1))
    register_feature('log_warp_waste', lambda df: np.log(df['warp_size'] / np.maximum(df['wi_per_group'], 1)))
    register_feature('lng_warp_waste', lambda df: np.log(df['num_groups']) * np.log(df['warp_size'] / np.maximum(df['wi_per_group'], 1)))
    register_feature('lwpg_warp_waste', lambda df: np.log(df['wi_per_group'] + 1) * np.log(df['warp_size'] / np.maximum(df['wi_per_group'], 1)))
    register_feature('coalescing_eff', lambda df: 1.0 - np.minimum(df['wi_global_load_lidx_stride'].fillna(0) / df['warp_size'], 1.0))
    register_feature('coalescing_eff_st', lambda df: 1.0 - np.minimum(df['wi_global_store_lidx_stride'].fillna(0) / df['warp_size'], 1.0))
    register_feature('reduce_kind', lambda df: df['wi_barriers'] / np.maximum(np.log2(df['wi_per_group'] + 1), 1.0))
    register_feature('element_ops', lambda df: df['wi_compute_ops'] / np.maximum(df['num_groups'], 1))
    register_feature('warp_div', lambda df: np.log(np.maximum(df['warp_size'] / np.maximum(df['wi_per_group'], 1), 1e-8)))
    register_feature('sm_occupancy', lambda df: np.minimum((df['num_groups'] * df['wi_per_group']) / 2048.0, 1.0))
    register_feature('tree_height', lambda df: np.where(df['wi_barriers'] > 0, np.log2(df['wi_barriers'] + 1), 0))
    register_feature('tree_reduce_cost', lambda df: df['wi_barriers'] * np.log1p(df['wi_local_load_bits'].fillna(0) + df['wi_local_store_bits'].fillna(0)))
    register_feature('element_ops_per_thread', lambda df: df['wi_compute_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'], 1))
    register_feature('shared_mem_pressure', lambda df: (df['wi_local_load_bits'].fillna(0) + df['wi_local_store_bits'].fillna(0)) / 65536.0)
    register_feature('warp_efficiency', lambda df: df['wi_per_group'] / np.maximum(df['warp_size'], 1))
    register_feature('warp_waste', lambda df: (df['warp_size'] - df['wi_per_group']) / df['warp_size'])
    register_feature('compute_per_barrier', lambda df: df['wi_compute_ops'] / np.maximum(df['num_groups'] * np.maximum(df['wi_barriers'], 1), 1))
    register_feature('barrier_overhead_cost', lambda df: df['wi_barriers'] * np.log1p(df['wi_barriers']) / np.maximum(np.log1p(df['wi_ops']), 1))
    register_feature('group_work_density', lambda df: df['wi_ops'] / np.maximum(df['num_groups'] * np.log1p(df['wi_per_group']), 1))
    register_feature('ops_per_barrier', lambda df: df['wi_ops'] / np.maximum(df['wi_barriers'], 1))
    register_feature('barrier_factor', lambda df: df['wi_barriers'] / np.maximum(np.log1p(df['wi_ops']), 1))
    register_feature('barrier_efficiency', lambda df: df['wi_barriers'] / np.maximum(np.log1p(df['wi_ops']), 1))
    register_feature('mem_compute_ratio_log', lambda df: np.log((df['wi_global_load_bits'] + df['wi_global_store_bits']) / np.maximum(df['wi_compute_ops'], 1) + 1))
    register_feature('log1p_lng', lambda df: np.log1p(np.abs(np.log(df['num_groups']))))
    register_feature('log1p_lwpg_raw', lambda df: np.log1p(np.abs(np.log(df['wi_per_group'] + 1))))
    register_feature('log1p_lops_raw', lambda df: np.log1p(np.abs(np.log(df['wi_ops']))))
    register_feature('log_ng_per_ops', lambda df: np.log(df['num_groups'] / np.maximum(df['wi_ops'], 1) + 1))
    register_feature('log_threads_per_ops', lambda df: np.log((df['num_groups'] * df['wi_per_group']) / np.maximum(df['wi_ops'], 1) + 1))
    register_feature('barr_med_lops', lambda df: df['wi_barriers'].isin([3, 4, 5]).astype(float) * np.log(df['wi_ops']))
    register_feature('barr_nonzero_lops', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_ops']))
    register_feature('barr_nonzero_lmem', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('compute_density', lambda df: df['wi_compute_ops'] / np.maximum(df['num_groups'] * df['wi_per_group'], 1))
    register_feature('thread_compute', lambda df: np.log(df['wi_compute_ops'] + 1) * np.log((df['num_groups'] * df['wi_per_group']) + 1))
    register_feature('bgt_lng', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['num_groups']))
    register_feature('bgt_lwpg', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_per_group'] + 1))
    register_feature('bgt_lops', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_ops']))
    register_feature('bgt_lcop', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_compute_ops']))
    register_feature('bgt_lgmem', lambda df: (df['wi_barriers'] > 0).astype(float) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    # New IR features
    register_feature('lreg', lambda df: np.log1p(df['wi_register_load_bits'].fillna(0)))
    register_feature('lreg_st', lambda df: np.log1p(df['wi_register_store_bits'].fillna(0)))
    register_feature('reg_pressure', lambda df: df['wi_register_load_bits'].fillna(0) / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1))
    register_feature('lgws0', lambda df: np.log1p(df['gws0'].fillna(1)))
    register_feature('lgws1', lambda df: np.log1p(df['gws1'].fillna(1)))
    register_feature('lgws2', lambda df: np.log1p(df['gws2'].fillna(1)))
    register_feature('llws0', lambda df: np.log1p(df['lws0'].fillna(1)))
    register_feature('llws1', lambda df: np.log1p(df['lws1'].fillna(1)))
    register_feature('llws2', lambda df: np.log1p(df['lws2'].fillna(1)))
    register_feature('lws0_ratio', lambda df: df['lws0'].fillna(1) / np.maximum(df['wi_per_group'], 1))
    register_feature('lws1_ratio', lambda df: df['lws1'].fillna(1) / np.maximum(df['wi_per_group'], 1))
    register_feature('lws2_ratio', lambda df: df['lws2'].fillna(1) / np.maximum(df['wi_per_group'], 1))
    register_feature('lws0xlws1', lambda df: df['lws0'].fillna(1) * df['lws1'].fillna(1))
    register_feature('gws0_ratio', lambda df: df['gws0'].fillna(1) / np.maximum(df['num_groups'], 1))
    register_feature('gws1_ratio', lambda df: df['gws1'].fillna(1) / np.maximum(df['num_groups'], 1))
    register_feature('loop_depth', lambda df: df['max_loop_depth'].fillna(0))
    register_feature('loop_depth*lops', lambda df: df['max_loop_depth'].fillna(0) * np.log(np.maximum(df['wi_ops'], 1)))
    register_feature('loop_depth*barr', lambda df: df['max_loop_depth'].fillna(0) * df['wi_barriers'])
    register_feature('log_local_mem', lambda df: np.log1p(df['local_mem_size'].fillna(1)))
    register_feature('pref_vec', lambda df: df['preferred_vector_size'].fillna(0))
    # Compute/memory interactions (continuous)
    register_feature('ci*lgmem', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * np.log(df['wi_global_load_bits'] + df['wi_global_store_bits'] + 1))
    register_feature('ci*lops', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * np.log(df['wi_ops']))
    register_feature('ci*lcop', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * np.log(df['wi_compute_ops']))
    register_feature('ci*lng', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * np.log(df['num_groups']))
    register_feature('ci*lwpg', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * np.log(df['wi_per_group'] + 1))
    register_feature('ci*barr', lambda df: (df['wi_compute_ops'] / np.maximum(df['wi_global_load_bits'] + df['wi_global_store_bits'], 1)) * df['wi_barriers'])


def build_feature_matrix(df):
    X = pd.DataFrame(index=df.index)
    for name, fn in FEATURE_DEFS:
        X[name] = fn(df)
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
        'log_warp_waste': '(warp_size / wi_per_group.max(1.0)).ln()',
        'lng_warp_waste': 'lng * (warp_size / wi_per_group.max(1.0)).ln()',
        'lwpg_warp_waste': 'lwpg * (warp_size / wi_per_group.max(1.0)).ln()',
        'llm': '(wi_local_load_bits + wi_local_store_bits).ln_1p()',
        'barr_wpg': 'barr * (wi_per_group + 1.0).ln_1p()',
        'reduce_kind': 'barr / (wi_per_group + 1.0).log2().max(1.0)',
        'element_ops': 'wi_compute_ops / num_groups.max(1.0)',
        'warp_div': '(warp_size / wi_per_group.max(1.0)).max(1e-8).ln()',
        'sm_occupancy': '((num_groups * wi_per_group) / 2048.0).min(1.0)',
        'tree_height': 'if wi_barriers > 0.0 { (wi_barriers + 1.0).log2() } else { 0.0 }',
        'tree_reduce_cost': 'barr * (wi_local_load_bits + wi_local_store_bits).ln_1p()',
        'element_ops_per_thread': 'wi_compute_ops / (num_groups * wi_per_group).max(1.0)',
        'coalescing_eff': '1.0 - (wi_global_load_lidx_stride / warp_size).min(1.0)',
        'coalescing_eff_st': '1.0 - (wi_global_store_lidx_stride / warp_size).min(1.0)',
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
        'warp_waste': '(warp_size - wi_per_group) / warp_size',
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
        'ci*lgmem': 'ci * lgmem',
        'ci*lops': 'ci * lops',
        'ci*lcop': 'ci * lcop',
        'ci*lng': 'ci * lng',
        'ci*lwpg': 'ci * lwpg',
        'ci*barr': 'ci * barr',
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
    matmul_only = '--matmul' in sys.argv
    df = pd.read_csv(BENCH_CSV)
    if matmul_only:
        df = df.tail(9000).reset_index(drop=True)
        print(f"Filtered to last {len(df)} matmul entries")
    print(f"Read {len(df)} entries from {BENCH_CSV}")

    y = df.groupby('variant_hash')['time_us'].rank(pct=True).fillna(0.5).values
    variant_groups = [g.index.values for _, g in df.groupby('variant_hash') if len(g) >= 2]

    register_all_features()
    X = build_feature_matrix(df)
    n_feat = X.shape[1]
    print(f"Features: {n_feat}, Groups: {len(variant_groups)}")

    model = HuberRegressor(alpha=1.0, max_iter=50)
    sfs = SFS(model, n_features_to_select=10, direction='forward',
              scoring='neg_mean_absolute_error', cv=2, n_jobs=-1)
    sfs.fit(X, y)
    selected = list(sfs.get_feature_names_out())
    print(f"Selected {len(selected)} features: {selected}")

    X_sel = X[selected]
    model.fit(X_sel, y)
    pred = model.predict(X_sel)

    r2s, rhos = [], []
    for g in variant_groups:
        if len(g) >= 3:
            yg, pg = y[g], pred[g]
            ss_res = np.sum((yg - pg)**2)
            ss_tot = np.sum((yg - yg.mean())**2)
            r2s.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)
            rhos.append(_spearmanr(df.loc[g, 'time_us'], pg)[0])
    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"\n=== Huber ({n_feat} features, no DT) ===")
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
        times = df.loc[g, 'time_us'].values
        actual_top10 = np.argsort(times)[:10]
        pred_top20 = np.argsort(pred[g])[:20]
        hits = len(set(actual_top10) & set(pred_top20))
        top10_in_top20.append(hits / 10.0)
    if top10_in_top20:
        print(f"  Top10-in-top20:   {np.mean(top10_in_top20):.3f} (over {len(top10_in_top20)} groups)")

    # Generate predict_cost.rs — only selected features
    rust_path = os.path.join(os.path.dirname(__file__), '..', 'zyx', 'src', 'kernel', 'predict_cost.rs')
    with open(rust_path, 'w') as f:
        f.write("// Copyright (C) 2025 zk4x\n")
        f.write("// SPDX-License-Identifier: LGPL-3.0-only\n//\n")
        f.write(f"// Auto-generated by regression.py. {len(selected)} features, Huber.\n")
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
        f.write("        let rr = wi_peak_reg_bytes / max_register_bytes.max(1.0);\n\n")
        n_sel = len(selected)
        f.write(f"        let features: [f64; {n_sel}] = [\n")
        for name in selected:
            expr = _feature_to_rust(name)
            f.write(f"            {expr},\n")
        f.write("        ];\n\n")
        def write_arr(name, vals, n_col=8):
            f.write(f"        const {name}: &[f64] = &[\n")
            for i in range(0, len(vals), n_col):
                line = ', '.join(f'{v:.8e}' for v in vals[i:i+n_col])
                f.write(f"            {line},\n")
            f.write("        ];\n")
        write_arr('RC', model.coef_)
        f.write(f"        let ri: f64 = {model.intercept_:.8e};\n\n")
        f.write(f"        let mut pred = ri;\n")
        f.write(f"        for i in 0..{n_sel} {{\n")
        f.write("            pred += RC[i] * features[i];\n")
        f.write("        }\n")
        f.write("        pred * 1_000_000.0\n")
        f.write("    }\n}\n")
    print(f"Wrote {rust_path}")


if __name__ == '__main__':
    main()
