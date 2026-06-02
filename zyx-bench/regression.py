#!/usr/bin/env python3
"""Parse bench_output.txt -> ridge regression for cost model."""

import re
import csv
import os
import sys
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import SelectFromModel
from collections import Counter

BENCH_OUTPUT = '/home/x/Dev/rust/zyx/zyx-bench/bench_output.txt'
BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'


def parse_bench_output(filename):
    entries = []
    current_section = None

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('===') and line.endswith('==='):
            current_section = line[3:-3].strip()
            i += 1
            continue

        if line.startswith('num_groups=') or line.startswith('const=') or line.startswith('cost='):
            m = re.match(
                r'(?:cost=(\d+), )?num_groups=(\d+), wi_per_group=(\d+), wi_ops=(\d+), wi_compute_ops=(\d+), '
                r'wi_barriers=(\d+), wi_global_load_bits=(\d+), wi_global_store_bits=(\d+)',
                line
            )
            if m:
                entry = {
                    'section': current_section,
                    'predicted_cost_us': int(m.group(1)) if m.group(1) and int(m.group(1)) != 18446744073709551615 else None,
                    'num_groups': int(m.group(2)),
                    'wi_per_group': int(m.group(3)),
                    'wi_ops': int(m.group(4)),
                    'wi_compute_ops': int(m.group(5)),
                    'wi_barriers': int(m.group(6)),
                    'wi_global_load_bits': int(m.group(7)),
                    'wi_global_store_bits': int(m.group(8)),
                }
                i += 1
                if i < len(lines):
                    m2 = re.match(
                        r'wi_local_load_bits=(\d+), wi_local_store_bits=(\d+), '
                        r'wi_peak_reg_bytes=(\d+), wi_branches=(\d+)(?:, '
                        r'wi_global_load_lidx_stride=(\d+))?, '
                        r'(?:wi_global_store_lidx_stride=(\d+))?, '
                        r'(?:wi_local_load_lidx_stride=(\d+))?, '
                        r'(?:wi_local_store_lidx_stride=(\d+))?, '
                        r'warp_size=(\d+), max_local_threads=(\d+), max_register_bytes=(\d+)',
                        lines[i].strip()
                    )
                    if m2:
                        entry['wi_local_load_bits'] = int(m2.group(1))
                        entry['wi_local_store_bits'] = int(m2.group(2))
                        entry['wi_peak_reg_bytes'] = int(m2.group(3))
                        entry['wi_branches'] = int(m2.group(4))
                        entry['wi_global_load_lidx_stride'] = int(m2.group(5)) / 10.0 if m2.group(5) else 0.0
                        entry['wi_global_store_lidx_stride'] = int(m2.group(6)) / 10.0 if m2.group(6) else 0.0
                        entry['wi_local_load_lidx_stride'] = int(m2.group(7)) / 10.0 if m2.group(7) else 0.0
                        entry['wi_local_store_lidx_stride'] = int(m2.group(8)) / 10.0 if m2.group(8) else 0.0
                        entry['warp_size'] = int(m2.group(9))
                        entry['max_local_threads'] = int(m2.group(10))
                        entry['max_register_bytes'] = int(m2.group(11))

                i += 1
                if i < len(lines):
                    time_match = re.match(r'([\d.]+)\s*(s|ms|μs)\s*~\s*([\d.]+)\s*[MGT]FLOP/s', lines[i].strip())
                    if time_match:
                        time_val = float(time_match.group(1))
                        unit = time_match.group(2)
                        entry['gflops'] = float(time_match.group(3))
                        if unit == 'ms':
                            time_val *= 1000.0
                        elif unit == 's':
                            time_val *= 1_000_000.0
                        entry['time_us'] = time_val
                        entries.append(entry)
        i += 1

    return entries


def write_csv(entries):
    with open(BENCH_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'section', 'num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops',
            'wi_barriers', 'wi_global_load_bits', 'wi_global_store_bits',
            'wi_local_load_bits', 'wi_local_store_bits', 'wi_peak_reg_bytes',
            'wi_branches', 'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
            'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
            'warp_size', 'max_local_threads', 'max_register_bytes',
            'time_us', 'gflops'
        ])
        for e in entries:
            writer.writerow([
                e['section'], e['num_groups'], e['wi_per_group'], e['wi_ops'],
                e['wi_compute_ops'], e['wi_barriers'], e['wi_global_load_bits'],
                e['wi_global_store_bits'], e.get('wi_local_load_bits', 0),
                e.get('wi_local_store_bits', 0), e.get('wi_peak_reg_bytes', 0),
                e.get('wi_branches', 0),
                e.get('wi_global_load_lidx_stride', 0),
                e.get('wi_global_store_lidx_stride', 0),
                e.get('wi_local_load_lidx_stride', 0),
                e.get('wi_local_store_lidx_stride', 0),
                e.get('warp_size', 32),
                e.get('max_local_threads', 1024), e.get('max_register_bytes', 256),
                e['time_us'], e['gflops']
            ])


def add_feature(entries, FEATURE_NAMES, X, name, fn):
    FEATURE_NAMES.append(name)
    new_feature = [fn(e) for e in entries]
    X = np.column_stack([X, new_feature])
    return X


def evaluate_model(model, Xs, y, section_groups):
    section_r2s = {}
    for section in section_groups.keys():
        mask = section_groups[section]
        if len(mask) > 1:
            y_sec = y[mask]
            y_pred_sec = model.predict(Xs[mask])
            ss_res_sec = np.sum((y_sec - y_pred_sec) ** 2)
            ss_tot_sec = np.sum((y_sec - np.mean(y_sec)) ** 2)
            r2_sec = 1 - ss_res_sec / ss_tot_sec if ss_tot_sec > 0 else 0
            section_r2s[section] = r2_sec
    return section_r2s


def _fix_f64(expr):
    """Convert a Rust f32 expression to f64 (no casts, float literals)."""
    expr = expr.replace(' as f32', '')
    expr = re.sub(r'\.max\((\d+)\)', lambda m: f'.max({m.group(1)}.0)', expr)
    # Fix integer comparisons with f64 variables
    expr = re.sub(r'(wi_barriers\s*[=!><]+\s*)(\d+)\b(?!\.)', lambda m: m.group(1) + m.group(2) + '.0', expr)
    # Fix integer arithmetic with f64 variables (e.g., + 1, - 1)
    expr = re.sub(r'(\+ )(\d+)\b(?!\.)(?!\d)', lambda m: m.group(1) + m.group(2) + '.0', expr)
    return expr


def _feature_to_rust(name):
    """Convert a feature name to a Rust expression using raw inputs."""
    # Simple features that map directly to a single log expression
    direct = {
        'lng': 'lng',
        'lwpg': 'lwpg',
        'lops': 'lops',
        'lcop': 'lcop',
        'lgmem': 'lgmem',
        'barr': 'barr',
        'wr': 'wr',
        'rr': 'rr',
        'log_inv_threads': 'log_inv_threads',
        'ci': 'ci',
        'overhead': 'wi_ops as f32 / wi_compute_ops.max(1) as f32',
        'log1p_ci': '(ci as f32).ln_1p()',
        'log1p_overhead': '((wi_ops as f32 / wi_compute_ops.max(1) as f32)).ln_1p()',
    }
    if name in direct:
        return direct[name]

    # Special cases (check before pattern matching to avoid greedy prefix handlers)
    special = {
        'total_threads': '(num_groups as f32) * (wi_per_group as f32)',
        'raw_ng': 'num_groups as f32',
        'raw_ng_wpg': '(num_groups as f32) * (wi_per_group as f32)',
        'warp_util': 'wr',
        'log_warp_waste': '(32.0 / wi_per_group.max(1) as f32).ln()',
        'lng_warp_waste': 'lng * (32.0 / wi_per_group.max(1) as f32).ln()',
        'lwpg_warp_waste': 'lwpg * (32.0 / wi_per_group.max(1) as f32).ln()',
        'lng_div_lops': '(num_groups as f32 / wi_ops.max(1) as f32).max(1e-8).ln()',
        'llm': '((wi_local_load_bits + wi_local_store_bits) as f32).ln_1p()',
        'barr_wpg': 'barr * (wi_per_group as f32).ln_1p()',
        'reduce_kind': 'barr / (wi_per_group as f32 + 1.0).log2().max(1.0)',
        'element_ops': 'wi_compute_ops as f32 / num_groups.max(1) as f32',
        'warp_div': '(32.0 / wi_per_group.max(1) as f32).max(1e-8).ln()',
        'sm_occupancy': '((num_groups as f32) * (wi_per_group as f32) / 2048.0).min(1.0)',
        'local_occupancy': '((wi_local_load_bits + wi_local_store_bits) as f32).ln_1p() / 65536.0',
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
        # === Missing mappings ===
        'overhead': 'wi_ops as f32 / wi_compute_ops.max(1) as f32',
        'log1p_ci': '(wi_compute_ops as f32 / (wi_global_load_bits + wi_global_store_bits).max(1) as f32).ln_1p()',
        'log1p_mp': '((wi_global_load_bits + wi_global_store_bits) as f32 / (num_groups * wi_per_group).max(1) as f32).ln_1p()',
        'log1p_lm': '((wi_local_load_bits + wi_local_store_bits + 1) as f32 / (wi_global_load_bits + wi_global_store_bits + 1).max(1) as f32).ln_1p()',
        'log_ops_per_group': '(wi_ops as f32 / num_groups.max(1) as f32 + 1.0).ln()',
        'log_inv_threads': '(1.0 / (num_groups * wi_per_group).max(1) as f32).ln_1p()',
        'log_opt_barr': '(wi_ops as f32 / (num_groups * wi_per_group * wi_barriers.max(1)).max(1) as f32).ln_1p()',
        'log_wpg_barr': '(wi_per_group as f32 * wi_barriers as f32).ln_1p()',
        'barr_low': 'if wi_barriers == 0 { 1.0 } else { 0.0 }',
        'barr_med': 'if wi_barriers >= 3 && wi_barriers <= 5 { 1.0 } else { 0.0 }',
        'barr_high': 'if wi_barriers >= 6 { 1.0 } else { 0.0 }',
        'barr_nonzero': 'if wi_barriers > 0 { 1.0 } else { 0.0 }',
        'barr*lops': 'barr * lops',
        'barr*lcop': 'barr * lcop',
        'barr*lgmem': 'barr * lgmem',
        'barr*lwpg': 'barr * lwpg',
        'barr*lng': 'barr * lng',
        'barr*wr': 'barr * wr',
        'shared_mem_pressure': '(wi_local_load_bits + wi_local_store_bits) as f32 / 65536.0',
        'warp_efficiency': 'wr',
        'warp_waste': '(32.0 - wi_per_group as f32) / 32.0',
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
        'coalescing_eff_st': '1.0 - (wi_global_store_lidx_stride as f32 / 32.0).min(1.0)',
        'lng_lgmem': 'lng * lgmem',
        'lng_lcop': 'lng * lcop',
    }
    if name in special:
        return special[name]
    
    # Handle interaction features with '*'
    if '*' in name:
        parts = name.split('*')
        sub_exprs = [_feature_to_rust(p) for p in parts]
        return ' * '.join(sub_exprs)
    
    # Handle bgt_ features (barrier>0 gated)
    if name.startswith('bgt_'):
        rest = name[4:]  # Remove 'bgt_'
        sub = _feature_to_rust(rest)
        return f'(if wi_barriers > 0 {{ 1.0 }} else {{ 0.0 }}) * {sub}'
    
    # Handle barr_ features (barrier interaction)
    if name.startswith('barr_'):
        rest = name[5:]  # Remove 'barr_'
        rest = name[5:]  # Remove 'barr_'
        sub = _feature_to_rust(rest)
        return f'barr * {sub}'
    
    # Handle b0, b3, b4 etc (barrier one-hot)
    if name.startswith('b') and len(name) > 1 and name[1:].isdigit():
        bval = name[1:]
        return f'(if wi_barriers == {bval} {{ 1.0 }} else {{ 0.0 }})'
    
    # Fallback
    return f'/* UNKNOWN: {name} */ 0.0'


def _print_dt_rust(dt, feature_names, leaf_biases, indent=0):
    """Print a sklearn DecisionTreeRegressor as Rust if-else returning leaf bias directly."""
    tree = dt.tree_
    
    def print_node(node_id, depth=0):
        prefix = "    " * depth
        if tree.feature[node_id] == -2:  # leaf
            # Find leaf index
            leaf_idx = len([n for n in range(node_id) if tree.feature[n] == -2])
            bias = leaf_biases[leaf_idx]
            print(f"{prefix}{bias:.6f}")
        else:
            feat_idx = tree.feature[node_id]
            fname = feature_names[feat_idx] if feat_idx < len(feature_names) else f'feat_{feat_idx}'
            threshold = tree.threshold[node_id]
            expr = _fix_f64(_feature_to_rust(fname))
            left = tree.children_left[node_id]
            right = tree.children_right[node_id]
            print(f"{prefix}if ({expr}) < {threshold}f64 {{")
            print_node(left, depth + 1)
            print(f"{prefix}}} else {{")
            print_node(right, depth + 1)
            print(f"{prefix}}}")
    
    return print_node(0)


# Global feature definitions (name, function) for reuse in prediction
FEATURE_DEFS = []
FEATURE_NAMES = []

def register_feature(name, fn):
    """Register a feature for both training and prediction."""
    FEATURE_DEFS.append((name, fn))
    FEATURE_NAMES.append(name)


def build_feature_matrix(entries):
    """Build feature matrix for a list of entries."""
    X = None
    for name, fn in FEATURE_DEFS:
        feat = np.array([fn(e) for e in entries]).reshape(-1, 1)
        if X is None:
            X = feat
        else:
            X = np.column_stack([X, feat])
    return X


def predict_entry(entry, dt, ridge, scaler):
    """Predict time_us for a single entry using trained Ridge+DT model."""
    feat_vec = np.array([fn(entry) for _, fn in FEATURE_DEFS]).reshape(1, -1)
    dt_pred = dt.predict(feat_vec)
    X_stacked = np.column_stack([feat_vec, dt_pred.reshape(-1, 1)])
    X_scaled = scaler.transform(X_stacked)
    log_pred = ridge.predict(X_scaled)[0]
    return np.exp(log_pred)


def predict_raw(lng, lwpg, lops, lcop, lgmem, barr, wr, rr,
                num_groups, wi_per_group, wi_ops, wi_compute_ops,
                wi_global_load_bits, wi_global_store_bits, wi_barriers,
                wi_local_load_bits, wi_local_store_bits, wi_peak_reg_bytes,
                wi_branches, wi_global_load_lidx_stride, wi_global_store_lidx_stride,
                wi_local_load_lidx_stride, wi_local_store_lidx_stride,
                warp_size, max_register_bytes):
    """Standalone predict using raw params (no entry dict needed)."""
    entry = {
        'num_groups': num_groups, 'wi_per_group': wi_per_group,
        'wi_ops': wi_ops, 'wi_compute_ops': wi_compute_ops,
        'wi_barriers': wi_barriers,
        'wi_global_load_bits': wi_global_load_bits,
        'wi_global_store_bits': wi_global_store_bits,
        'wi_local_load_bits': wi_local_load_bits,
        'wi_local_store_bits': wi_local_store_bits,
        'wi_peak_reg_bytes': wi_peak_reg_bytes,
        'wi_branches': wi_branches,
        'wi_global_load_lidx_stride': wi_global_load_lidx_stride,
        'wi_global_store_lidx_stride': wi_global_store_lidx_stride,
        'wi_local_load_lidx_stride': wi_local_load_lidx_stride,
        'wi_local_store_lidx_stride': wi_local_store_lidx_stride,
        'warp_size': warp_size, 'max_register_bytes': max_register_bytes,
    }
    from sklearn.tree import DecisionTreeRegressor
    return predict_entry(entry, dt, ridge, scaler)


def main():
    print("Parsing bench_output.txt...")
    entries = parse_bench_output(BENCH_OUTPUT)
    print(f"Parsed {len(entries)} entries")

    section_counts = Counter(e['section'] for e in entries)
    for section, count in sorted(section_counts.items()):
        print(f"  {section}: {count}")

    write_csv(entries)
    print(f"\nWrote {len(entries)} entries to bench_data.csv")

    barrier_counts = Counter(e['wi_barriers'] for e in entries)
    print(f"Barrier distribution: {dict(sorted(barrier_counts.items()))}")

    y = np.log(np.array([e['time_us'] for e in entries]))

    # Group entries by section
    section_groups = {}
    for i, entry in enumerate(entries):
        section = entry['section']
        if section not in section_groups:
            section_groups[section] = []
        section_groups[section].append(i)

    # === FEATURES: all hand-engineered features (registered for reuse in predict_entry) ===
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
    register_feature('mem_compute_ratio_log', lambda e: np.log((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1) + 1))
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

    # Build feature matrix after all features are registered
    X = build_feature_matrix(entries)

    print(f"Total features: {X.shape[1]}")

    # Create sample weights: 50% to barrier>0, 50% to barrier=0
    # Within each barrier group, each section gets equal weight
    barr_vals = np.array([e['wi_barriers'] for e in entries])
    barrier_mask = barr_vals > 0
    
    # Group sections by barrier regime
    barrier0_sections = set()
    barrier_gt0_sections = set()
    for i, e in enumerate(entries):
        if e['wi_barriers'] > 0:
            barrier_gt0_sections.add(e['section'])
        else:
            barrier0_sections.add(e['section'])
    
    # Within each regime, give each section equal weight
    barrier_weights = np.zeros(len(entries))
    
    for section_list, total_weight in [(barrier0_sections, 0.5), (barrier_gt0_sections, 0.5)]:
        for section in section_list:
            indices = section_groups[section]
            weight_per_sample = total_weight / len(section_list) / len(indices)
            for idx in indices:
                barrier_weights[idx] = weight_per_sample
    
    sample_weights = barrier_weights * len(entries)
    sample_weights = sample_weights / np.mean(sample_weights)

    # === Ridge+DT Stacking ===
    # DT carves the feature space into regions where relationships are roughly linear.
    # Ridge on [X, DT_pred] learns regional intercepts via DT_pred and linear slopes via X.
    max_dt_leaves = 2000
    print(f"\nTraining DT with max {max_dt_leaves} leaves...")
    dt = DecisionTreeRegressor(max_leaf_nodes=max_dt_leaves, random_state=42, min_samples_leaf=5)
    dt.fit(X, y, sample_weight=sample_weights)
    dt_pred = dt.predict(X)
    tree = dt.tree_
    leaf_node_ids = [i for i in range(tree.node_count) if tree.feature[i] == -2]
    dt_leaf_values = tree.value[leaf_node_ids, 0, 0]
    print(f"DT has {len(leaf_node_ids)} leaves, R²={dt.score(X, y):.4f}")

    # Stack Ridge on [X, dt_pred]
    X_stacked = np.column_stack([X, dt_pred])
    alphas = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stacked)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X_scaled, y, sample_weight=sample_weights)
    best_alpha = ridge_cv.alpha_
    # Re-fit with best alpha
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stacked)
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y, sample_weight=sample_weights)
    ridge_coef = ridge.coef_
    ridge_intercept = ridge.intercept_
    stacked_pred = ridge.predict(X_scaled)
    print(f"Ridge alpha={best_alpha:.4f}, stacked R²={ridge.score(X_scaled, y, sample_weight=sample_weights):.4f}")

    # === Evaluate DT results ===
    def evaluate_model_impl(model_fn, y, section_groups):
        y_pred = model_fn()
        section_r2s = {}
        for section in section_groups.keys():
            mask = section_groups[section]
            if len(mask) > 1:
                y_sec = y[mask]
                y_pred_sec = y_pred[mask]
                ss_res = np.sum((y_sec - y_pred_sec) ** 2)
                ss_tot = np.sum((y_sec - np.mean(y_sec)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                section_r2s[section] = r2
        return section_r2s

    dt_r2_sections = evaluate_model_impl(lambda: dt.predict(X), y, section_groups)
    dt_min_r2 = min(dt_r2_sections.values()) if dt_r2_sections else 0
    print(f"\n=== DT-Only Results ===")
    print(f"DT leaves: {len(leaf_node_ids)}")
    print(f"Total parameters: {len(leaf_node_ids)}")
    print(f"Minimum per-section R²: {dt_min_r2:.4f}")
    print()
    for section in sorted(section_groups.keys()):
        if section in dt_r2_sections:
            r2 = dt_r2_sections[section]
            flag = " *** ABOVE 0.95 ***" if r2 >= 0.95 else ""
            print(f"  {section}: R²={r2:.4f}{flag}")

    # === Evaluate stacked Ridge+DT results ===
    stacked_r2_sections = evaluate_model_impl(lambda: stacked_pred, y, section_groups)
    stacked_min_r2 = min(stacked_r2_sections.values()) if stacked_r2_sections else 0
    print(f"\n=== Ridge+DT Stacked Results ===")
    print(f"DT leaves: {len(leaf_node_ids)}")
    print(f"Ridge features: {X_stacked.shape[1]} (143 original + 1 DT pred)")
    print(f"Total parameters: {len(ridge_coef)} + 1 intercept = {len(ridge_coef) + 1}")
    print(f"Minimum per-section R²: {stacked_min_r2:.4f}")
    print()
    for section in sorted(section_groups.keys()):
        if section in stacked_r2_sections:
            r2 = stacked_r2_sections[section]
            flag = " *** ABOVE 0.95 ***" if r2 >= 0.95 else ""
            print(f"  {section}: R²={r2:.4f}{flag}")

    if stacked_min_r2 >= 0.95:
        print("\n*** All targets MET! ***")
    else:
        print(f"\nWorst section R²={stacked_min_r2:.4f} (target: 0.95)")
        for section in sorted(section_groups.keys()):
            if section in stacked_r2_sections and stacked_r2_sections[section] < 0.95:
                print(f"  {section}: R²={stacked_r2_sections[section]:.4f}")

    # === Cost.rs code generation ===
    # Model: log_time = ridge_intercept + sum(ridge_coef[i] * scaled_feature[i])
    # where scaled_feature[i] = (feature[i] - scaler_mean[i]) / scaler_scale[i]
    # Feature 144 (0-indexed) is the DT leaf_bias.
    rust_path = os.path.join(os.path.dirname(__file__), '..', 'zyx', 'src', 'kernel', 'predict_cost.rs')
    with open(rust_path, 'w') as f:
        f.write("// Copyright (C) 2025 zk4x\n")
        f.write("// SPDX-License-Identifier: LGPL-3.0-only\n")
        f.write("//\n")
        f.write("// Auto-generated by regression.py. Do not edit manually.\n")
        f.write(f"// Ridge+DT stacked model: {len(leaf_node_ids)} DT leaves, {len(ridge_coef)} Ridge features\n")
        f.write(f"// Total parameters: {len(ridge_coef)} Ridge + 1 intercept + DT tree\n")
        f.write("\n")
        f.write("#![allow(unused)]\n")
        f.write("\n")
        f.write("use super::cost::Cost;\n")
        f.write("\n")
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
        f.write("    ) -> f64 {\n")
        f.write("        // Convert all inputs to f64 immediately (no overflow risk)\n")
        f.write("        let num_groups = num_groups as f64;\n")
        f.write("        let wi_per_group = wi_per_group as f64;\n")
        f.write("        let wi_ops = wi_ops as f64;\n")
        f.write("        let wi_compute_ops = wi_compute_ops as f64;\n")
        f.write("        let wi_barriers = wi_barriers as f64;\n")
        f.write("        let wi_global_load_bits = wi_global_load_bits as f64;\n")
        f.write("        let wi_global_store_bits = wi_global_store_bits as f64;\n")
        f.write("        let wi_local_load_bits = wi_local_load_bits as f64;\n")
        f.write("        let wi_local_store_bits = wi_local_store_bits as f64;\n")
        f.write("        let wi_peak_reg_bytes = wi_peak_reg_bytes as f64;\n")
        f.write("        let wi_branches = wi_branches as f64;\n")
        f.write("        let wi_global_load_lidx_stride = wi_global_load_lidx_stride as f64;\n")
        f.write("        let wi_global_store_lidx_stride = wi_global_store_lidx_stride as f64;\n")
        f.write("        let wi_local_load_lidx_stride = wi_local_load_lidx_stride as f64;\n")
        f.write("        let wi_local_store_lidx_stride = wi_local_store_lidx_stride as f64;\n")
        f.write("        let warp_size = warp_size as f64;\n");
        f.write("        let max_register_bytes = max_register_bytes as f64;\n");
        f.write("\n")
        f.write("        // Compute raw features (all f64, no overflow)\n")
        f.write("        let lng = num_groups.ln();\n")
        f.write("        let lwpg = (wi_per_group + 1.0).ln();\n")
        f.write("        let lops = wi_ops.ln();\n")
        f.write("        let lcop = wi_compute_ops.ln();\n")
        f.write("        let lgmem = (wi_global_load_bits + wi_global_store_bits + 1.0).ln();\n")
        f.write("        let log_inv_threads = (1.0 / (num_groups * wi_per_group)).ln_1p();\n")
        f.write("        let ci = wi_compute_ops / (wi_global_load_bits + wi_global_store_bits).max(1.0);\n")
        f.write("        let barr = wi_barriers;\n")
        f.write("        let wr = wi_per_group / warp_size;\n")
        f.write("        let rr = wi_peak_reg_bytes / max_register_bytes.max(1.0);\n")
        f.write("\n")
        # Compute scaled features from raw features
        f.write("        // Raw features array\n")
        f.write("        let raw: [f64; 143] = [\n")
        for i in range(X.shape[1]):
            expr = _fix_f64(_feature_to_rust(FEATURE_NAMES[i]))
            f.write(f"            {expr},\n")
        f.write("        ];\n")
        f.write("\n")
        f.write("        // DT leaf bias\n")
        f.write("        let leaf_bias: f64 = {\n")
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        _print_dt_rust(dt, FEATURE_NAMES, dt_leaf_values)
        sys.stdout = old_stdout
        f.write(buf.getvalue())
        f.write("        };\n")
        f.write("\n")
        # Output scaler means and scales
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        f.write("        // StandardScaler parameters (mean, scale)\n")
        f.write("        let scaler_mean: [f64; 144] = [\n")
        for i in range(144):
            f.write(f"            {scaler_mean[i]:.8e},\n")
        f.write("        ];\n")
        f.write("        let scaler_scale: [f64; 144] = [\n")
        for i in range(144):
            f.write(f"            {scaler_scale[i]:.8e},\n")
        f.write("        ];\n")
        f.write("\n")
        f.write("        // Ridge coefficients (post-scaling)\n")
        f.write("        let ridge_coef: [f64; 144] = [\n")
        for i in range(144):
            f.write(f"            {ridge_coef[i]:.8e},\n")
        f.write("        ];\n")
        f.write("\n")
        f.write(f"        let ridge_intercept: f64 = {ridge_intercept:.8e};\n")
        f.write("\n")
        f.write("        // Build X_stacked = [raw_features(143), leaf_bias]\n")
        f.write("        let mut scaled: [f64; 144] = [0.0; 144];\n")
        f.write("        for i in 0..143 {\n")
        f.write("            scaled[i] = (raw[i] - scaler_mean[i]) / scaler_scale[i];\n")
        f.write("        }\n")
        f.write("        scaled[143] = (leaf_bias - scaler_mean[143]) / scaler_scale[143];\n")
        f.write("\n")
        f.write("        let mut pred = ridge_intercept;\n")
        f.write("        for i in 0..144 {\n")
        f.write("            pred += ridge_coef[i] * scaled[i];\n")
        f.write("        }\n")
        f.write("\n")
        f.write("        pred.exp()\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"\nWrote {rust_path}")

    # === Diagnostic ===
    full_pred = stacked_pred
    matmul_mask = np.array([e['section'] == 'MatMul' for e in entries])
    matmul_idx = np.where(matmul_mask)[0]
    mm_actual_us = np.array([entries[i]['time_us'] for i in matmul_idx])
    mm_pred_us = np.exp(full_pred[matmul_mask])

    print("\n=== MatMul Within-Section Ranking ===")
    sort_order = np.argsort(mm_actual_us)
    print(f"{'#':>4} {'ng':>10} {'wipg':>5} {'ops':>6} {'actual_us':>10} {'pred_us':>10}")
    print("-" * 55)
    for rank, idx in enumerate(sort_order[:15]):
        e = entries[matmul_idx[idx]]
        a = mm_actual_us[idx]; p = mm_pred_us[idx]
        print(f"{rank:4d} {e['num_groups']:10} {e['wi_per_group']:5} {e['wi_ops']:6} {a:10.1f} {p:10.1f}")
    print("  ...")
    for rank, idx in enumerate(sort_order[-5:]):
        e = entries[matmul_idx[idx]]
        a = mm_actual_us[idx]; p = mm_pred_us[idx]
        print(f"{len(sort_order)-5+rank:4d} {e['num_groups']:10} {e['wi_per_group']:5} {e['wi_ops']:6} {a:10.1f} {p:10.1f}")

    from scipy.stats import spearmanr as _spearmanr
    rho, _ = _spearmanr(mm_actual_us, mm_pred_us)
    print(f"\nMatMul Spearman ρ = {rho:.4f}")

    pred_order = np.argsort(mm_pred_us)
    print(f"\n{'Pred rank':>9} {'ng':>10} {'wipg':>5} {'ops':>6} {'actual_us':>10} {'pred_us':>10}")
    print("-" * 55)
    for rank, si in enumerate(pred_order[:10]):
        e = entries[matmul_idx[si]]
        a = mm_actual_us[si]; p = mm_pred_us[si]
        print(f"{rank:9d} {e['num_groups']:10} {e['wi_per_group']:5} {e['wi_ops']:6} {a:10.1f} {p:10.1f}")

    print(f"\n--- High-ops MatMul entries (ops > 50000) ---")
    high_mask = np.array([entries[i]['wi_ops'] > 50000 for i in matmul_idx])
    high_si = np.where(high_mask)[0]
    high_actual = mm_actual_us[high_mask]
    high_pred = mm_pred_us[high_mask]
    high_pred_order = np.argsort(high_pred)
    print(f"{'Pred rank':>9} {'ng':>10} {'wipg':>5} {'ops':>8} {'us_time':>8} {'actual_us':>10} {'pred_us':>10}")
    print("-" * 70)
    for rank, si in enumerate(high_pred_order[:15]):
        actual_si = high_si[si]
        e = entries[matmul_idx[actual_si]]
        a = high_actual[si]; p = high_pred[si]
        print(f"{rank:9d} {e['num_groups']:10} {e['wi_per_group']:5} {e['wi_ops']:8} {e['time_us']:8.0f} {a:10.1f} {p:10.1f}")
    rho_high, _ = _spearmanr(high_actual, high_pred)
    print(f"High-ops MatMul Spearman ρ = {rho_high:.4f}")

    print("\n=== Large MatMul Within-Section Ranking ===")
    lm_mask = np.array([e['section'] == 'Large MatMul' for e in entries])
    lm_idx = np.where(lm_mask)[0]
    lm_actual_us = np.array([entries[i]['time_us'] for i in lm_idx])
    lm_pred_us = np.exp(full_pred[lm_mask])
    lm_pred_order = np.argsort(lm_pred_us)
    print(f"{'Pred rank':>9} {'ng':>10} {'wipg':>5} {'ops':>6} {'actual_us':>10} {'pred_us':>10}")
    print("-" * 55)
    for rank, si in enumerate(lm_pred_order[:10]):
        e = entries[lm_idx[si]]
        a = lm_actual_us[si]; p = lm_pred_us[si]
        print(f"{rank:9d} {e['num_groups']:10} {e['wi_per_group']:5} {e['wi_ops']:6} {a:10.1f} {p:10.1f}")
    rho_lm, _ = _spearmanr(lm_actual_us, lm_pred_us)
    print(f"Large MatMul Spearman ρ = {rho_lm:.4f}")

    return dt, entries, ridge, scaler

if __name__ == '__main__':
    main()
