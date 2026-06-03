#!/usr/bin/env python3
"""Train cost model from bench_data.csv -> predict_cost.rs"""

import io
import os
import re
import sys
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import spearmanr as _spearmanr
from collections import defaultdict

BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'

# Feature registration
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
    register_feature('lng_div_lops', lambda e: np.log(max(e['num_groups'] / max(e['wi_ops'], 1), 1e-8)))
    register_feature('llm', lambda e: np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)))
    register_feature('barr_wpg', lambda e: e['wi_barriers'] * np.log1p(e['wi_per_group']))
    register_feature('b0*lcop', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_compute_ops']))
    register_feature('b0*lgmem', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    register_feature('log_cops_100*b0', lambda e: np.log(e['wi_compute_ops'] + 100) * (1.0 if e['wi_barriers'] == 0 else 0.0))
    register_feature('lwpg*lops*b0', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log1p(e['wi_per_group']) * np.log(e['wi_ops']))
    register_feature('coalescing_eff', lambda e: 1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0))
    register_feature('coalescing_eff_st', lambda e: 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0))
    register_feature('reduce_kind', lambda e: e['wi_barriers'] / max(np.log2(e['wi_per_group'] + 1), 1.0))
    register_feature('element_ops', lambda e: e['wi_compute_ops'] / max(e['num_groups'], 1))
    register_feature('warp_div', lambda e: np.log(max(32 / max(e['wi_per_group'], 1), 1e-8)))
    register_feature('sm_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0))
    register_feature('tree_height', lambda e: np.log2(e['wi_barriers'] + 1) if e['wi_barriers'] > 0 else 0)
    register_feature('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)))
    register_feature('element_ops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
    register_feature('compute_intensity_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'] * 32, 1))
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
    register_feature('bgt_lng_lgmem', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    register_feature('bgt_lng_lcop', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['num_groups']) * np.log(e['wi_compute_ops']))

    # New IR features: register-scoped memory
    register_feature('lreg', lambda e: np.log1p(e.get('wi_register_load_bits', 0)))
    register_feature('lreg_st', lambda e: np.log1p(e.get('wi_register_store_bits', 0)))
    register_feature('reg_pressure', lambda e: e.get('wi_register_load_bits', 0) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))

    # Per-axis work sizes
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
    register_feature('gws_div_lws0', lambda e: e.get('gws0', 1) / max(e.get('lws0', 1), 1))
    register_feature('gws1_div_lws1', lambda e: e.get('gws1', 1) / max(e.get('lws1', 1), 1))

    # Loop depth
    register_feature('loop_depth', lambda e: e.get('max_loop_depth', 0))
    register_feature('loop_depth*lops', lambda e: e.get('max_loop_depth', 0) * np.log(max(e['wi_ops'], 1)))
    register_feature('loop_depth*lcop', lambda e: e.get('max_loop_depth', 0) * np.log(max(e['wi_compute_ops'], 1)))
    register_feature('loop_depth*lwpg', lambda e: e.get('max_loop_depth', 0) * np.log(e['wi_per_group'] + 1))
    register_feature('loop_depth*ci', lambda e: e.get('max_loop_depth', 0) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    register_feature('loop_depth*barr', lambda e: e.get('max_loop_depth', 0) * e['wi_barriers'])

    # Device properties
    register_feature('log_local_mem', lambda e: np.log1p(e.get('local_mem_size', 1)))
    register_feature('pref_vec', lambda e: e.get('preferred_vector_size', 0))

def build_feature_matrix(entries):
    X = None
    for _, fn in FEATURE_DEFS:
        feat = np.array([fn(e) for e in entries]).reshape(-1, 1)
        X = feat if X is None else np.column_stack([X, feat])
    return X


# Rust code generation helpers
def _fix_f64(expr):
    expr = expr.replace(' as f32', '')
    expr = re.sub(r'\.max\((\d+)\)', lambda m: f'.max({m.group(1)}.0)', expr)
    expr = re.sub(r'(wi_barriers\s*[=!><]+\s*)(\d+)\b(?!\.)', lambda m: m.group(1) + m.group(2) + '.0', expr)
    expr = re.sub(r'(\+ )(\d+)\b(?!\.)(?!\d)', lambda m: m.group(1) + m.group(2) + '.0', expr)
    return expr

def _feature_to_rust(name):
    direct = {
        'lng': 'lng', 'lwpg': 'lwpg', 'lops': 'lops', 'lcop': 'lcop', 'lgmem': 'lgmem',
        'barr': 'barr', 'wr': 'wr', 'rr': 'rr', 'log_inv_threads': 'log_inv_threads', 'ci': 'ci',
        'overhead': 'wi_ops as f32 / wi_compute_ops.max(1) as f32',
        'log1p_ci': '(ci as f32).ln_1p()',
        'log1p_overhead': '((wi_ops as f32 / wi_compute_ops.max(1) as f32)).ln_1p()',
    }
    if name in direct:
        return direct[name]
    special = {
        'total_threads': '(num_groups as f32) * (wi_per_group as f32)',
        'raw_ng': 'num_groups as f32', 'raw_ng_wpg': '(num_groups as f32) * (wi_per_group as f32)',
        'warp_util': 'wr', 'log_warp_waste': '(32.0 / wi_per_group.max(1) as f32).ln()',
        'lng_warp_waste': 'lng * (32.0 / wi_per_group.max(1) as f32).ln()',
        'lwpg_warp_waste': 'lwpg * (32.0 / wi_per_group.max(1) as f32).ln()',
        'lng_div_lops': '(num_groups as f32 / wi_ops.max(1) as f32).max(1e-8).ln()',
        'llm': '((wi_local_load_bits + wi_local_store_bits) as f32).ln_1p()',
        'barr_wpg': 'barr * (wi_per_group as f32).ln_1p()',
        'reduce_kind': 'barr / (wi_per_group as f32 + 1.0).log2().max(1.0)',
        'element_ops': 'wi_compute_ops as f32 / num_groups.max(1) as f32',
        'warp_div': '(32.0 / wi_per_group.max(1) as f32).max(1e-8).ln()',
        'sm_occupancy': '((num_groups as f32) * (wi_per_group as f32) / 2048.0).min(1.0)',
        'launch_overhead': 'if wi_ops > 0 { (-(wi_ops as f32) / 1000.0).exp() } else { 1.0 }',
        'tree_height': 'if wi_barriers > 0 { (wi_barriers as f32 + 1.0).log2() } else { 0.0 }',
        'tree_reduce_cost': 'barr * ((wi_local_load_bits + wi_local_store_bits) as f32).ln_1p()',
        'element_ops_per_thread': 'wi_compute_ops as f32 / ((num_groups as f32) * (wi_per_group as f32)).max(1.0)',
        'compute_intensity_per_thread': 'wi_compute_ops as f32 / ((num_groups as f32) * (wi_per_group as f32) * 32.0).max(1.0)',
        'coalescing_eff': '1.0 - (wi_global_load_lidx_stride as f32 / 32.0).min(1.0)',
        'coalescing_eff_st': '1.0 - (wi_global_store_lidx_stride as f32 / 32.0).min(1.0)',
        'lld_st': '(wi_global_load_lidx_stride as f32).ln_1p()',
        'lst_st': '(wi_global_store_lidx_stride as f32).ln_1p()',
        'log_ops_100': '(wi_ops as f32 + 100.0).ln()',
        'log_ops_10': '(wi_ops as f32 + 10.0).ln()',
        'log_cops_100': '(wi_compute_ops as f32 + 100.0).ln()',
        'log1p_1000_div_ops': '(1000.0 / wi_ops.max(1) as f32).ln_1p()',
        'log1p_100_div_ops': '(100.0 / wi_ops.max(1) as f32).ln_1p()',
        'log1p_1000_div_cops': '(1000.0 / wi_compute_ops.max(1) as f32).ln_1p()',
        'ops_per_thread': 'wi_ops as f32 / ((num_groups as f32) * (wi_per_group as f32)).max(1.0)',
        'cops_per_thread': 'wi_compute_ops as f32 / ((num_groups as f32) * (wi_per_group as f32)).max(1.0)',
        'ops_per_group': 'wi_ops as f32 / num_groups.max(1) as f32',
        'log_lwpg': 'lwpg.max(1e-8).ln()',
        'log_opt': '(wi_ops as f32 / (num_groups * wi_per_group).max(1) as f32).ln_1p()',
        'inv_threads': '1.0 / (num_groups * wi_per_group).max(1) as f32',
        'lbranch': '(wi_branches as f32).ln_1p()',
        'lng_lwpg': 'lng * lwpg', 'lwpg_lops': 'lwpg * lops',
        'lng*lgmem': 'lng * lgmem', 'lcop*lgmem': 'lcop * lgmem',
        'lops*lgmem': 'lops * lgmem', 'lwpg*lgmem': 'lwpg * lgmem',
        'lng*lcop': 'lng * lcop', 'lwpg*lops': 'lwpg * lops',
        'lng_sq': 'lng * lng', 'lcop_sq': 'lcop * lcop',
        'lgmem_sq': 'lgmem * lgmem', 'lops_sq': 'lops * lops', 'lwpg_sq': 'lwpg * lwpg',
        'log1p_mp': '((wi_global_load_bits + wi_global_store_bits) as f32 / (num_groups * wi_per_group).max(1) as f32).ln_1p()',
        'log1p_lm': '((wi_local_load_bits + wi_local_store_bits + 1) as f32 / (wi_global_load_bits + wi_global_store_bits + 1).max(1) as f32).ln_1p()',
        'log_ops_per_group': '(wi_ops as f32 / num_groups.max(1) as f32 + 1.0).ln()',
        'log_opt_barr': '(wi_ops as f32 / (num_groups * wi_per_group * wi_barriers.max(1)).max(1) as f32).ln_1p()',
        'log_wpg_barr': '(wi_per_group as f32 * wi_barriers as f32).ln_1p()',
        'barr_low': 'if wi_barriers == 0 { 1.0 } else { 0.0 }',
        'barr_med': 'if wi_barriers >= 3 && wi_barriers <= 5 { 1.0 } else { 0.0 }',
        'barr_high': 'if wi_barriers >= 6 { 1.0 } else { 0.0 }',
        'barr_nonzero': 'if wi_barriers > 0 { 1.0 } else { 0.0 }',
        'barr*lops': 'barr * lops', 'barr*lcop': 'barr * lcop',
        'barr*lgmem': 'barr * lgmem', 'barr*lwpg': 'barr * lwpg',
        'barr*lng': 'barr * lng', 'barr*wr': 'barr * wr',
        'shared_mem_pressure': '(wi_local_load_bits + wi_local_store_bits) as f32 / 65536.0',
        'warp_efficiency': 'wr', 'warp_waste': '(32.0 - wi_per_group as f32) / 32.0',
        'compute_per_barrier': 'wi_compute_ops as f32 / (num_groups * wi_barriers.max(1)).max(1) as f32',
        'barrier_overhead_cost': 'barr * (barr as f32).ln_1p() / (wi_ops as f32).ln_1p().max(1.0)',
        'group_work_density': 'wi_ops as f32 / (num_groups as f32 * (wi_per_group as f32).ln_1p()).max(1.0)',
        'ops_per_barrier': 'wi_ops as f32 / wi_barriers.max(1) as f32',
        'mem_compute_ratio_log': '((wi_global_load_bits + wi_global_store_bits) as f32 / wi_compute_ops.max(1) as f32 + 1.0).ln()',
        'barrier_factor': 'barr / (wi_ops as f32).ln_1p().max(1.0)',
        'barrier_efficiency': 'barr / (wi_ops as f32).ln_1p().max(1.0)',
        'log1p_lng': '(lng.abs()).ln_1p()',
        'log1p_lwpg_raw': '(lwpg.abs()).ln_1p()',
        'log1p_lops_raw': '(lops.abs()).ln_1p()',
        'log_ng_per_ops': '(num_groups as f32 / wi_ops.max(1) as f32 + 1.0).ln()',
        'log_threads_per_ops': '((num_groups * wi_per_group) as f32 / wi_ops.max(1) as f32 + 1.0).ln()',
        'barr_med_lops': '(if wi_barriers >= 3 && wi_barriers <= 5 { 1.0 } else { 0.0 }) * lops',
        'barr_nonzero_lops': '(if wi_barriers > 0 { 1.0 } else { 0.0 }) * lops',
        'barr_nonzero_lmem': '(if wi_barriers > 0 { 1.0 } else { 0.0 }) * lgmem',
        'compute_density': 'wi_compute_ops as f32 / (num_groups * wi_per_group).max(1) as f32',
        'thread_compute': '(wi_compute_ops as f32 + 1.0).ln() * ((num_groups * wi_per_group) as f32 + 1.0).ln()',
        'barr*lbranch': 'barr * (wi_branches as f32).ln_1p()',
        'lwpg*lbranch': 'lwpg * (wi_branches as f32).ln_1p()',
        'lld_st*lops': '(wi_global_load_lidx_stride as f32).ln_1p() * lops',
        'lld_st*lcop': '(wi_global_load_lidx_stride as f32).ln_1p() * lcop',
        'lld_st*lgmem': '(wi_global_load_lidx_stride as f32).ln_1p() * lgmem',
        'lng_lgmem': 'lng * lgmem', 'lng_lcop': 'lng * lcop',

        # New IR features: register memory
        'lreg': '(wi_register_load_bits as f64).ln_1p()',
        'lreg_st': '(wi_register_store_bits as f64).ln_1p()',
        'reg_pressure': 'wi_register_load_bits as f64 / (wi_global_load_bits + wi_global_store_bits).max(1.0)',

        # Per-axis work sizes
        'lgws0': '(gws0 as f64).ln_1p()',
        'lgws1': '(gws1 as f64).ln_1p()',
        'lgws2': '(gws2 as f64).ln_1p()',
        'llws0': '(lws0 as f64).ln_1p()',
        'llws1': '(lws1 as f64).ln_1p()',
        'llws2': '(lws2 as f64).ln_1p()',
        'lws0_ratio': 'lws0 as f64 / wi_per_group.max(1.0)',
        'lws1_ratio': 'lws1 as f64 / wi_per_group.max(1.0)',
        'lws2_ratio': 'lws2 as f64 / wi_per_group.max(1.0)',
        'lws0xlws1': '(lws0 as f64) * (lws1 as f64)',
        'gws0_ratio': 'gws0 as f64 / num_groups.max(1.0)',
        'gws1_ratio': 'gws1 as f64 / num_groups.max(1.0)',
        'gws_div_lws0': 'gws0 as f64 / lws0.max(1.0)',
        'gws1_div_lws1': 'gws1 as f64 / lws1.max(1.0)',

        # Per-axis interactions with memory
        'lws1*reg_pressure': 'lws1 as f64 * wi_register_load_bits as f64 / (wi_global_load_bits + wi_global_store_bits).max(1.0)',
        'lws0*reg_pressure': 'lws0 as f64 * wi_register_load_bits as f64 / (wi_global_load_bits + wi_global_store_bits).max(1.0)',
        'lgws0*lreg': '(gws0 as f64).ln_1p() * (wi_register_load_bits as f64).ln_1p()',
        'lgws1*lreg': '(gws1 as f64).ln_1p() * (wi_register_load_bits as f64).ln_1p()',
        'llws1*lreg': '(lws1 as f64).ln_1p() * (wi_register_load_bits as f64).ln_1p()',
        'llws0*lreg': '(lws0 as f64).ln_1p() * (wi_register_load_bits as f64).ln_1p()',
        'llws1*lgmem': '(lws1 as f64).ln_1p() * lgmem',
        'lws1_ratio*lgmem': '(lws1 as f64 / wi_per_group.max(1.0)) * lgmem',

        # Loop depth
        'loop_depth': 'max_loop_depth',
        'loop_depth*lops': 'max_loop_depth * lops',
        'loop_depth*lcop': 'max_loop_depth * lcop',
        'loop_depth*lwpg': 'max_loop_depth * lwpg',
        'loop_depth*ci': 'max_loop_depth * ci',
        'loop_depth*barr': 'max_loop_depth * barr',

        # Device properties
        'log_local_mem': '(local_mem_size as f64).ln_1p()',
        'pref_vec': 'preferred_vector_size',
    }
    if name in special:
        return special[name]
    if '*' in name:
        parts = name.split('*')
        return ' * '.join(_feature_to_rust(p) for p in parts)
    if name.startswith('bgt_'):
        rest = name[4:]
        return f'(if wi_barriers > 0 {{ 1.0 }} else {{ 0.0 }}) * {_feature_to_rust(rest)}'
    if name.startswith('b') and len(name) > 1 and name[1:].isdigit():
        return f'(if wi_barriers == {name[1:]} {{ 1.0 }} else {{ 0.0 }})'
    return f'/* UNKNOWN: {name} */ 0.0'

def _print_dt_rust(dt, feature_names, leaf_biases):
    tree = dt.tree_
    def print_node(node_id, depth=0):
        prefix = "    " * depth
        if tree.feature[node_id] == -2:
            leaf_idx = len([n for n in range(node_id) if tree.feature[n] == -2])
            print(f"{prefix}{leaf_biases[leaf_idx]:.6f}")
        else:
            fname = feature_names[tree.feature[node_id]]
            threshold = tree.threshold[node_id]
            expr = _fix_f64(_feature_to_rust(fname))
            left, right = tree.children_left[node_id], tree.children_right[node_id]
            print(f"{prefix}if ({expr}) < {threshold}f64 {{")
            print_node(left, depth + 1)
            print(f"{prefix}}} else {{")
            print_node(right, depth + 1)
            print(f"{prefix}}}")
    print_node(0)

def predict_entry(entry, dt, ridge, scaler):
    feat_vec = np.array([fn(entry) for _, fn in FEATURE_DEFS]).reshape(1, -1)
    dt_pred = dt.predict(feat_vec)
    X_stacked = np.column_stack([feat_vec, dt_pred.reshape(-1, 1)])
    X_scaled = scaler.transform(X_stacked)
    return ridge.predict(X_scaled)[0]


# ---- MAIN ----
def main():
    import pandas as pd
    df = pd.read_csv(BENCH_CSV)
    entries = df.to_dict('records')
    print(f"Read {len(entries)} entries from {BENCH_CSV}")

    # Variant groups from variant_hash
    hash_map = defaultdict(list)
    for i, e in enumerate(entries):
        h = e.get('variant_hash', '')
        hash_map[h].append(i)
    variant_groups = [v for v in hash_map.values() if len(v) >= 2]

    # Rank target 0..1 within each variant group
    y = np.empty(len(entries))
    for g in variant_groups:
        times = np.array([entries[i]['time_us'] for i in g])
        ranks = np.argsort(np.argsort(times))
        for j, idx in enumerate(g):
            y[idx] = ranks[j] / (len(g) - 1)
    ungrouped = set(range(len(entries))) - set(idx for g in variant_groups for idx in g)
    for i in ungrouped:
        y[i] = 0.5

    # Features
    register_all_features()
    X = build_feature_matrix(entries)
    print(f"Features: {X.shape[1]}, Groups: {len(variant_groups)}")

    # Sample weights
    barr_vals = np.array([e['wi_barriers'] for e in entries])
    n_b0 = np.sum(barr_vals == 0)
    n_bg = np.sum(barr_vals > 0)
    sw = np.where(barr_vals > 0, 0.5 / n_bg, 0.5 / n_b0) * len(entries)
    sw /= np.mean(sw)

    # DT + Ridge
    max_dt_leaves = 2000
    dt = DecisionTreeRegressor(max_leaf_nodes=max_dt_leaves, random_state=42, min_samples_leaf=5)
    dt.fit(X, y, sample_weight=sw)
    leaf_ids = dt.apply(X)
    tree = dt.tree_
    n_leaves = len([i for i in range(tree.node_count) if tree.feature[i] == -2])

    leaf_to_idx = {}
    for node_id in range(tree.node_count):
        if tree.feature[node_id] == -2:
            leaf_to_idx[node_id] = len(leaf_to_idx)
    leaf_ohe = np.zeros((len(entries), n_leaves))
    for i, leaf in enumerate(leaf_ids):
        leaf_ohe[i, leaf_to_idx[leaf]] = 1.0

    X_with_leaves = np.column_stack([X, leaf_ohe])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_leaves)
    ridge_cv = RidgeCV(alphas=[0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    ridge_cv.fit(X_scaled, y, sample_weight=sw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_leaves)
    ridge = Ridge(alpha=ridge_cv.alpha_)
    ridge.fit(X_scaled, y, sample_weight=sw)
    stacked_pred = ridge.predict(X_scaled)

    n_ridge_sel = X.shape[1]
    final_dt_coefs = ridge.coef_[n_ridge_sel:]

    # Evaluation
    r2s, rhos = [], []
    for g in variant_groups:
        if len(g) >= 3:
            yg, pg = y[g], stacked_pred[g]
            ss_res = np.sum((yg - pg)**2)
            ss_tot = np.sum((yg - yg.mean())**2)
            r2s.append(1 - ss_res/ss_tot if ss_tot > 0 else 0)
            rhos.append(_spearmanr([entries[i]['time_us'] for i in g], pg)[0])

    r2s, rhos = np.array(r2s), np.array(rhos)
    print(f"\n=== Ridge + DT leaves ===")
    print(f"  Mean R²:          {r2s.mean():.4f}  (over {len(r2s)} kernels)")
    print(f"  Median R²:        {np.median(r2s):.4f}")
    print(f"  Worst 5% quantile:{np.quantile(r2s, 0.05):.4f}")
    print(f"  Worst R²:         {r2s.min():.4f}")
    print(f"  Best R²:          {r2s.max():.4f}")
    print(f"  Spearman ρ:       {rhos.mean():.4f} ± {rhos.std():.4f}")

    # Generate predict_cost.rs
    rust_path = os.path.join(os.path.dirname(__file__), '..', 'zyx', 'src', 'kernel', 'predict_cost.rs')
    with open(rust_path, 'w') as f:
        f.write("// Copyright (C) 2025 zk4x\n// SPDX-License-Identifier: LGPL-3.0-only\n//\n")
        f.write("// Auto-generated by regression.py. Do not edit manually.\n")
        f.write(f"// Ridge+DT leaves: {n_ridge_sel} Ridge features, {n_leaves} DT leaves\n")
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
        for var in ['num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops', 'wi_barriers',
                     'wi_global_load_bits', 'wi_global_store_bits', 'wi_local_load_bits',
                     'wi_local_store_bits', 'wi_peak_reg_bytes', 'wi_branches',
                     'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
                     'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
                     'warp_size', 'max_local_threads', 'max_register_bytes',
                     'wi_register_load_bits', 'wi_register_store_bits',
                     'gws0', 'gws1', 'gws2',
                     'lws0', 'lws1', 'lws2',
                     'max_loop_depth', 'preferred_vector_size', 'local_mem_size']:
            f.write(f"        let {var} = {var} as f64;\n")
        f.write("        let lng = num_groups.ln();\n")
        f.write("        let lwpg = (wi_per_group + 1.0).ln();\n")
        f.write("        let lops = wi_ops.ln();\n")
        f.write("        let lcop = wi_compute_ops.ln();\n")
        f.write("        let lgmem = (wi_global_load_bits + wi_global_store_bits + 1.0).ln();\n")
        f.write("        let log_inv_threads = (1.0 / (num_groups * wi_per_group)).ln_1p();\n")
        f.write("        let ci = wi_compute_ops / (wi_global_load_bits + wi_global_store_bits).max(1.0);\n")
        f.write("        let barr = wi_barriers;\n")
        f.write("        let wr = wi_per_group / warp_size;\n")
        f.write("        let rr = wi_peak_reg_bytes / max_register_bytes.max(1.0);\n\n")
        f.write(f"        let features: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel):
            expr = _fix_f64(_feature_to_rust(FEATURE_NAMES[i]))
            f.write(f"            {expr},  // {FEATURE_NAMES[i]}\n")
        f.write("        ];\n\n")
        f.write("        let leaf_bias: f64 = {\n")
        buf = io.StringIO()
        old_out = sys.stdout
        sys.stdout = buf
        _print_dt_rust(dt, FEATURE_NAMES, final_dt_coefs)
        sys.stdout = old_out
        f.write(buf.getvalue())
        f.write("        };\n\n")
        sm = scaler.mean_
        ss = scaler.scale_
        rc = ridge.coef_
        f.write(f"        let scaler_mean: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel): f.write(f"            {sm[i]:.8e},\n")
        f.write("        ];\n")
        f.write(f"        let scaler_scale: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel): f.write(f"            {ss[i]:.8e},\n")
        f.write("        ];\n")
        f.write(f"        let ridge_coef: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel): f.write(f"            {rc[i]:.8e},\n")
        f.write("        ];\n")
        f.write(f"        let ridge_intercept: f64 = {ridge.intercept_:.8e};\n\n")
        f.write(f"        let mut pred = ridge_intercept;\n")
        f.write(f"        for i in 0..{n_ridge_sel} {{\n")
        f.write("            pred += ridge_coef[i] * (features[i] - scaler_mean[i]) / scaler_scale[i];\n")
        f.write("        }\n")
        f.write("        pred += leaf_bias;\n")
        f.write("        pred * 1_000_000.0\n")
        f.write("    }\n}\n")
    print(f"Wrote {rust_path}")


if __name__ == '__main__':
    main()
