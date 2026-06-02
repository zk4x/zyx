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
from scipy.stats import spearmanr as _spearmanr
from collections import Counter

BENCH_OUTPUT = '/home/x/Dev/rust/zyx/zyx-bench/bench_output.txt'
BENCH_CSV = '/home/x/Dev/rust/zyx/zyx-bench/bench_data.csv'


def parse_bench_output(filename):
    entries = []
    current_section = None
    block_id = 0

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith('===') and line.endswith('==='):
            current_section = line[3:-3].strip()
            block_id += 1
            i += 1
            continue

        if line.startswith('num_groups=') or line.startswith('const=') or line.startswith('cost='):
            # Single-line format: all params + variant_hash + timing
            # cost=..., num_groups=..., ..., max_register_bytes=... variant_hash=X, time ~ GFLOP/s, ...
            m = re.match(
                r'(?:cost=(\d+), )?num_groups=(\d+), wi_per_group=(\d+), wi_ops=(\d+), wi_compute_ops=(\d+), '
                r'wi_barriers=(\d+), wi_global_load_bits=(\d+), wi_global_store_bits=(\d+), '
                r'wi_local_load_bits=(\d+), wi_local_store_bits=(\d+), '
                r'wi_peak_reg_bytes=(\d+), wi_branches=(\d+), '
                r'wi_global_load_lidx_stride=(\d+), wi_global_store_lidx_stride=(\d+), '
                r'wi_local_load_lidx_stride=(\d+), wi_local_store_lidx_stride=(\d+), '
                r'warp_size=(\d+), max_local_threads=(\d+), max_register_bytes=(\d+)',
                line
            )
            if m:
                entry = {
                    'section': current_section,
                    'block_id': block_id,
                    'predicted_cost_us': int(m.group(1)) if m.group(1) and int(m.group(1)) != 18446744073709551615 else None,
                    'num_groups': int(m.group(2)),
                    'wi_per_group': int(m.group(3)),
                    'wi_ops': int(m.group(4)),
                    'wi_compute_ops': int(m.group(5)),
                    'wi_barriers': int(m.group(6)),
                    'wi_global_load_bits': int(m.group(7)),
                    'wi_global_store_bits': int(m.group(8)),
                    'wi_local_load_bits': int(m.group(9)),
                    'wi_local_store_bits': int(m.group(10)),
                    'wi_peak_reg_bytes': int(m.group(11)),
                    'wi_branches': int(m.group(12)),
                    'wi_global_load_lidx_stride': int(m.group(13)) / 10.0,
                    'wi_global_store_lidx_stride': int(m.group(14)) / 10.0,
                    'wi_local_load_lidx_stride': int(m.group(15)) / 10.0,
                    'wi_local_store_lidx_stride': int(m.group(16)) / 10.0,
                    'warp_size': int(m.group(17)),
                    'max_local_threads': int(m.group(18)),
                    'max_register_bytes': int(m.group(19)),
                }
                # Parse variant_hash and timing from the end of the same line
                time_match = re.search(r'variant_hash=(\d+), ([\d.]+)\s*(s|ms|μs)\s*~\s*([\d.]+)\s*[MGT]FLOP/s', line)
                if time_match:
                    entry['variant_hash'] = int(time_match.group(1))
                    time_val = float(time_match.group(2))
                    unit = time_match.group(3)
                    entry['gflops'] = float(time_match.group(4))
                    if unit == 'ms':
                        time_val *= 1000.0
                    elif unit == 's':
                        time_val *= 1_000_000.0
                    entry['time_us'] = time_val
                    entries.append(entry)
                else:
                    # Old format: timing on next line (or no timing)
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
    """Predict rank 0..1 for a single entry (0 = fastest in variant)."""
    feat_vec = np.array([fn(entry) for _, fn in FEATURE_DEFS]).reshape(1, -1)
    dt_pred = dt.predict(feat_vec)
    X_stacked = np.column_stack([feat_vec, dt_pred.reshape(-1, 1)])
    X_scaled = scaler.transform(X_stacked)
    pred = ridge.predict(X_scaled)[0]
    return pred


def predict_raw(lng, lwpg, lops, lcop, lgmem, barr, wr, rr,
                num_groups, wi_per_group, wi_ops, wi_compute_ops,
                wi_global_load_bits, wi_global_store_bits, wi_barriers,
                wi_local_load_bits, wi_local_store_bits, wi_peak_reg_bytes,
                wi_branches, wi_global_load_lidx_stride, wi_global_store_lidx_stride,
                wi_local_load_lidx_stride, wi_local_store_lidx_stride,
                warp_size, max_register_bytes):
    """Predict rank 0..1 (0 fastest in variant) using raw params."""
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


def __detect_variant_groups(entries):
    """Detect variant groups using variant_hash from output, or fallback to ops_bucket."""
    from collections import defaultdict

    # Prefer variant_hash when available (from re-generated output)
    has_hash = any('variant_hash' in e for e in entries)
    if has_hash:
        hash_map = defaultdict(list)
        for i, e in enumerate(entries):
            key = e.get('variant_hash', 0)
            hash_map[key].append(i)
        groups = [indices for indices in hash_map.values() if len(indices) >= 2]
        all_grouped = set(idx for g in groups for idx in g)
        for i in range(len(entries)):
            if i not in all_grouped:
                groups.append([i])
        return groups

    # Fallback: ops_bucket (2 sig figs) + block_id
    def ops_bucket(ops):
        if ops < 100:
            return max(round(ops / 10) * 10, 10)
        s = str(int(ops))
        n = len(s)
        scale = 10 ** (n - 2)
        return (ops // scale) * scale

    bucket_map = defaultdict(list)
    for i, e in enumerate(entries):
        key = (e['section'], e['block_id'], ops_bucket(e['wi_ops']))
        bucket_map[key].append(i)

    groups = [indices for indices in bucket_map.values() if len(indices) >= 2]
    all_grouped = set(idx for g in groups for idx in g)
    for i in range(len(entries)):
        if i not in all_grouped:
            groups.append([i])
    return groups


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

    # === Variant grouping: consecutive entries within same section with similar ops ===
    # Variants of the same kernel have very similar wi_ops counts.
    # Group consecutive entries where ops changes by < 10%.
    variant_groups = __detect_variant_groups(entries)
    print(f"Detected {len(variant_groups)} variant groups")
    group_sizes = Counter(len(g) for g in variant_groups)
    for size in sorted(group_sizes):
        print(f"  size {size}: {group_sizes[size]} groups")

    # === Target: rank 0..1 within each variant group ===
    # 0 = fastest in variant, 1 = slowest in variant
    y = np.empty(len(entries))
    for group in variant_groups:
        if len(group) >= 2:
            times = np.array([entries[i]['time_us'] for i in group])
            ranks = np.argsort(np.argsort(times))  # rank by speed (0=fastest)
            for j, idx in enumerate(group):
                y[idx] = ranks[j] / (len(group) - 1)
        else:
            # Singleton group: put at median rank 0.5
            y[group[0]] = 0.5
    print(f"Target = rank 0..1 within variant (0=fastest)")
    print(f"  y range: {y.min():.3f} - {y.max():.3f}")
    print(f"  y mean: {y.mean():.3f}")

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

    # Sample weights: 50% to barrier>0, 50% to barrier=0, per-kernel equal
    barr_vals = np.array([e['wi_barriers'] for e in entries])
    n_barr0 = np.sum(barr_vals == 0)
    n_barr_gt0 = np.sum(barr_vals > 0)
    sample_weights = np.where(barr_vals > 0, 0.5 / n_barr_gt0, 0.5 / n_barr0)
    sample_weights *= len(entries)
    sample_weights /= np.mean(sample_weights)

    # === Ridge + DT leaves (original architecture) ===
    # DT partitions the space into leaves. Each leaf gets its own intercept (learned by Ridge).
    # Ridge learns per-leaf intercepts + linear effects for selected features.
    # Final: intercept + sum(coef_i * scaled_feature_i) + leaf_bias
    max_dt_leaves = 80
    max_ridge_features = 20
    print(f"\nTraining DT with max {max_dt_leaves} leaves...")
    dt = DecisionTreeRegressor(max_leaf_nodes=max_dt_leaves, random_state=42, min_samples_leaf=5)
    dt.fit(X, y, sample_weight=sample_weights)
    leaf_ids = dt.apply(X)
    tree = dt.tree_
    leaf_node_ids = [i for i in range(tree.node_count) if tree.feature[i] == -2]
    n_leaves = len(leaf_node_ids)
    print(f"DT has {n_leaves} leaves, R²={dt.score(X, y):.4f}")

    # One-hot encode leaf assignments (traversal order = _print_dt_rust order)
    leaf_to_idx = {}
    for node_id in range(tree.node_count):
        if tree.feature[node_id] == -2:
            leaf_to_idx[node_id] = len(leaf_to_idx)
    leaf_ohe = np.zeros((len(entries), n_leaves))
    for i, leaf in enumerate(leaf_ids):
        leaf_ohe[i, leaf_to_idx[leaf]] = 1.0

    # Add leaf features to FEATURE_NAMES for Ridge training
    n_orig_features = X.shape[1]
    for leaf_idx in range(n_leaves):
        FEATURE_NAMES.append(f'dt_leaf_{leaf_idx}')
    X_with_leaves = np.column_stack([X, leaf_ohe])
    n_total_features = X_with_leaves.shape[1]

    # Fit Ridge on all features
    print(f"Total features (eng + DT leaves): {n_total_features}")
    alphas = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_with_leaves)
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X_scaled, y, sample_weight=sample_weights)
    best_alpha = ridge_cv.alpha_

    # Feature selection: keep top N engineered features + ALL DT leaves
    ridge_indices = list(range(n_orig_features))
    dt_leaf_indices = list(range(n_orig_features, n_total_features))

    ridge_coefs = ridge_cv.coef_[ridge_indices]
    top_k = np.argsort(np.abs(ridge_coefs))[-max_ridge_features:]
    selected_ridge_indices = sorted([ridge_indices[i] for i in top_k])
    # Use ALL engineered features (no selection) — within-variant signal is subtle and distributed
    selected_ridge_indices = ridge_indices
    final_indices = selected_ridge_indices + dt_leaf_indices
    n_ridge_sel = len(selected_ridge_indices)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_scaled, y, sample_weight=sample_weights)
    ridge_coef = ridge.coef_
    ridge_intercept = ridge.intercept_
    final_ridge_coefs = ridge_coef[:n_ridge_sel]
    final_dt_coefs = ridge_coef[n_ridge_sel:]

    # Predictions
    stacked_pred = ridge.predict(X_scaled)

    # === Per-kernel R² evaluation ===
    def per_kernel_r2(y_pred, variant_groups):
        r2s = []
        for g in variant_groups:
            if len(g) >= 3:
                y_sec = y[g]
                y_pred_sec = y_pred[g]
                ss_res = np.sum((y_sec - y_pred_sec) ** 2)
                ss_tot = np.sum((y_sec - np.mean(y_sec)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                r2s.append(r2)
        return np.array(r2s)

    def print_per_kernel_report(label, y_pred, variant_groups):
        r2s = per_kernel_r2(y_pred, variant_groups)
        if len(r2s) == 0:
            return
        print(f"\n=== {label} Results ===")
        print(f"  Overall mean R²:   {np.mean(r2s):.4f}  (over {len(r2s)} kernels)")
        print(f"  Median R²:         {np.median(r2s):.4f}")
        print(f"  Worst 5% quantile: {np.quantile(r2s, 0.05):.4f}")
        print(f"  Worst R²:          {np.min(r2s):.4f}")
        print(f"  Best R²:           {np.max(r2s):.4f}")

    print_per_kernel_report("Ridge + DT leaves", stacked_pred, variant_groups)

    # === Cost.rs code generation ===
    # In cost.rs, the model is:
    #   pred = intercept + sum(coef_i * scaled_feature_i) + leaf_bias
    # where leaf_bias comes from DT traversal (returns Ridge coef for that leaf)
    # and features are only the selected engineered features (not DT leaves).
    rust_path = os.path.join(os.path.dirname(__file__), '..', 'zyx', 'src', 'kernel', 'predict_cost.rs')
    with open(rust_path, 'w') as f:
        f.write("// Copyright (C) 2025 zk4x\n")
        f.write("// SPDX-License-Identifier: LGPL-3.0-only\n")
        f.write("//\n")
        f.write("// Auto-generated by regression.py. Do not edit manually.\n")
        f.write(f"// Ridge+DT leaves model (rank 0..1 target): {n_ridge_sel} Ridge features, {n_leaves} DT leaves\n")
        f.write(f"// Total parameters: {n_ridge_sel} Ridge + 1 intercept + {n_leaves} DT leaf biases\n")
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
        # Compute only selected engineered features
        f.write("        // Selected Ridge features\n")
        f.write(f"        let features: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel):
            feat_name = FEATURE_NAMES[final_indices[i]]
            expr = _fix_f64(_feature_to_rust(feat_name))
            f.write(f"            {expr},  // {feat_name}\n")
        f.write("        ];\n")
        f.write("\n")
        f.write("        // DT leaf bias (Ridge coef for the leaf, learned jointly)\n")
        f.write("        let leaf_bias: f64 = {\n")
        import io
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        _print_dt_rust(dt, FEATURE_NAMES, final_dt_coefs)
        sys.stdout = old_stdout
        f.write(buf.getvalue())
        f.write("        };\n")
        f.write("\n")
        # Output scaler means, scales, and coefs for selected features
        scaler_mean = scaler.mean_
        scaler_scale = scaler.scale_
        coefs = ridge.coef_
        f.write("        // StandardScaler parameters (mean, scale)\n")
        f.write(f"        let scaler_mean: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel):
            f.write(f"            {scaler_mean[i]:.8e},\n")
        f.write("        ];\n")
        f.write(f"        let scaler_scale: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel):
            f.write(f"            {scaler_scale[i]:.8e},\n")
        f.write("        ];\n")
        f.write("\n")
        f.write("        // Ridge coefficients (post-scaling)\n")
        f.write(f"        let ridge_coef: [f64; {n_ridge_sel}] = [\n")
        for i in range(n_ridge_sel):
            f.write(f"            {final_ridge_coefs[i]:.8e},\n")
        f.write("        ];\n")
        f.write("\n")
        f.write(f"        let ridge_intercept: f64 = {ridge_intercept:.8e};\n")
        f.write("\n")
        f.write("        // pred = intercept + sum(coef_i * scaled_feature_i) + leaf_bias\n")
        f.write(f"        let mut pred = ridge_intercept;\n")
        f.write(f"        for i in 0..{n_ridge_sel} {{\n")
        f.write("            pred += ridge_coef[i] * (features[i] - scaler_mean[i]) / scaler_scale[i];\n")
        f.write("        }\n")
        f.write("        pred += leaf_bias;\n")
        f.write("\n")
        f.write("        pred * 1_000_000.0\n")
        f.write("    }\n")
        f.write("}\n")
    print(f"\nWrote {rust_path}")

    # === Diagnostic: why is R² low? ===
    print("\n=== Why so bad? Within-group feature variation ===")
    for g in sorted(variant_groups, key=lambda g: len(g))[-10:]:
        if len(g) < 5:
            continue
        times = [entries[i]['time_us'] for i in g]
        t_range = max(times) - min(times)
        t_mean = np.mean(times)
        t_cv = np.std(times) / t_mean if t_mean > 0 else 0
        feat_keys = ['num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops',
                     'wi_barriers', 'wi_global_load_bits', 'wi_global_store_bits',
                     'wi_local_load_bits', 'wi_local_store_bits', 'wi_peak_reg_bytes',
                     'wi_branches']
        n_unique = {}
        for k in feat_keys:
            vals = [entries[i][k] for i in g]
            n_unique[k] = len(set(vals))
        actual = np.array([entries[i]['time_us'] for i in g])
        pred = stacked_pred[g]
        r, _ = _spearmanr(actual, pred) if len(g) >= 3 else (0, 0)
        hash_val = entries[g[0]].get('variant_hash', '?')
        uniq = ', '.join(f"{k}={v}" for k, v in n_unique.items() if v > 1)
        print(f"  hash={hash_val} size={len(g):3d} ρ={r:.3f} t_range={t_range:.1f}us cv={t_cv:.3f}  varying: {uniq}")

    # Count groups where most features are constant
    const_count = 0
    feature_names = ['num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops',
                     'wi_barriers', 'wi_global_load_bits', 'wi_global_store_bits',
                     'wi_local_load_bits', 'wi_local_store_bits', 'wi_peak_reg_bytes',
                     'wi_branches',
                     'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
                     'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
                     'warp_size', 'max_local_threads', 'max_register_bytes']
    for g in variant_groups:
        if len(g) < 3:
            continue
        varying = 0
        for k in feature_names:
            vals = [entries[i][k] for i in g]
            if len(set(vals)) > 1:
                varying += 1
        if varying <= 2:
            const_count += 1
    print(f"\n  Groups with ≤2 varying features: {const_count}/{len(variant_groups)}")

    # How many features actually vary in any group?
    feature_varying_count = {}
    for k in feature_names:
        feature_varying_count[k] = 0
    for g in variant_groups:
        if len(g) < 3:
            continue
        for k in feature_names:
            vals = [entries[i][k] for i in g]
            if len(set(vals)) > 1:
                feature_varying_count[k] += 1
    print("  Features that vary within groups (# groups affected):")
    for k, v in sorted(feature_varying_count.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v}/{len(variant_groups)} groups")

    return dt, entries, ridge, scaler

if __name__ == '__main__':
    main()
