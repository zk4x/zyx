#!/usr/bin/env python3
"""Parse bench_output.txt → bench_data.csv → ridge regression for cost model."""

import re
import csv
import numpy as np
from sklearn.linear_model import Ridge
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

        if line.startswith('num_groups='):
            m = re.match(
                r'num_groups=(\d+), wi_per_group=(\d+), wi_ops=(\d+), wi_compute_ops=(\d+), '
                r'wi_barriers=(\d+), wi_global_load_bits=(\d+), wi_global_store_bits=(\d+)',
                line
            )
            if m:
                entry = {
                    'section': current_section,
                    'num_groups': int(m.group(1)),
                    'wi_per_group': int(m.group(2)),
                    'wi_ops': int(m.group(3)),
                    'wi_compute_ops': int(m.group(4)),
                    'wi_barriers': int(m.group(5)),
                    'wi_global_load_bits': int(m.group(6)),
                    'wi_global_store_bits': int(m.group(7)),
                }
                i += 1
                if i < len(lines):
                    m2 = re.match(
                        r'wi_local_load_bits=(\d+), wi_local_store_bits=(\d+), '
                        r'wi_peak_reg_bytes=(\d+), wi_branches=(\d+)(?:, '
                        r'wi_global_load_lidx_stride=(\d+))?, '
                        r'(?:wi_global_store_lidx_stride=(\d+))?, '
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
                        entry['warp_size'] = int(m2.group(7))
                        entry['max_local_threads'] = int(m2.group(8))
                        entry['max_register_bytes'] = int(m2.group(9))

                i += 1
                if i < len(lines):
                    time_match = re.match(r'([\d.]+)\s*(s|ms|μs)\s*~\s*([\d.]+)\s*[MG]FLOP/s', lines[i].strip())
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
                e.get('warp_size', 32),
                e.get('max_local_threads', 1024), e.get('max_register_bytes', 256),
                e['time_us'], e['gflops']
            ])


def build_features(entries):
    # Single source of truth: (name, callable)
    F = lambda name, fn: (name, fn)
    feature_defs = [
        # --- size & occupancy (7) ---
        F('lng',  lambda e: np.log(e['num_groups'])),
        F('lwpg', lambda e: np.log(e['wi_per_group'] + 1)),
        F('lops', lambda e: np.log(e['wi_ops'])),
        F('barr', lambda e: e['wi_barriers']),
        F('wr',   lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1)),
        F('rr',   lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)),
        F('tot_th', lambda e: e['num_groups'] * e['wi_per_group']),

        # --- stride / coalescing (2) ---
        F('lld_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0))),
        F('lst_st', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0))),

        # --- derived ratios (4) ---
        F('ci',     lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)),
        F('overhd', lambda e: e['wi_ops'] / max(e['wi_compute_ops'], 1)),
        F('opt',    lambda e: e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        F('mpt',    lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1)),

        # --- log of ratios (3) ---
        F('log_ci',  lambda e: np.log1p(e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        F('log_oh',  lambda e: np.log1p(e['wi_ops'] / max(e['wi_compute_ops'], 1))),
        F('log_lm',  lambda e: np.log1p((e.get('wi_local_load_bits',0)+e.get('wi_local_store_bits',0)+1) / max(e['wi_global_load_bits']+e['wi_global_store_bits']+1, 1))),

        # --- barrier one-hot (7) ---
        F('b0', lambda e: 1. if e['wi_barriers']==0 else 0.),
        F('b3', lambda e: 1. if e['wi_barriers']==3 else 0.),
        F('b4', lambda e: 1. if e['wi_barriers']==4 else 0.),
        F('b5', lambda e: 1. if e['wi_barriers']==5 else 0.),
        F('b6', lambda e: 1. if e['wi_barriers']==6 else 0.),
        F('b7', lambda e: 1. if e['wi_barriers']==7 else 0.),
        F('b8', lambda e: 1. if e['wi_barriers']==8 else 0.),

        # --- size × barrier (5) ---
        F('lng*b', lambda e: np.log(e['num_groups']) * e['wi_barriers']),
        F('lwpg*b',lambda e: np.log(e['wi_per_group']+1) * e['wi_barriers']),
        F('lops*b',lambda e: np.log(e['wi_ops']) * e['wi_barriers']),
        F('wr*b',  lambda e: (e['wi_per_group']/max(e.get('warp_size',32),1)) * e['wi_barriers']),
        F('rr*b',  lambda e: (e.get('wi_peak_reg_bytes',0)/max(e.get('max_register_bytes',256),1)) * e['wi_barriers']),

        # --- size × stride (5) ---
        F('lng*s', lambda e: np.log(e['num_groups']) * np.log1p(e.get('wi_global_load_lidx_stride',0))),
        F('lwpg*s',lambda e: np.log(e['wi_per_group']+1) * np.log1p(e.get('wi_global_load_lidx_stride',0))),
        F('lops*s',lambda e: np.log(e['wi_ops']) * np.log1p(e.get('wi_global_load_lidx_stride',0))),
        F('barr*s',lambda e: e['wi_barriers'] * np.log1p(e.get('wi_global_load_lidx_stride',0))),
        F('wr*s',  lambda e: (e['wi_per_group']/max(e.get('warp_size',32),1)) * np.log1p(e.get('wi_global_load_lidx_stride',0))),

        # --- store stride (2) ---
        F('lst*lops',lambda e: np.log1p(e.get('wi_global_store_lidx_stride',0)) * np.log(e['wi_ops'])),
        F('lst*lng', lambda e: np.log1p(e.get('wi_global_store_lidx_stride',0)) * np.log(e['num_groups'])),

        # --- occupancy × ops (4) ---
        F('lng*lops', lambda e: np.log(e['num_groups']) * np.log(e['wi_ops'])),
        F('lwpg*lops',lambda e: np.log(e['wi_per_group']+1) * np.log(e['wi_ops'])),
        F('lng*opt', lambda e: np.log(e['num_groups']) * (e['wi_ops']/max(e['num_groups']*e['wi_per_group'],1))),
        F('lwpg*opt',lambda e: np.log(e['wi_per_group']+1) * (e['wi_ops']/max(e['num_groups']*e['wi_per_group'],1))),

        # --- compute intensity × size (4) ---
        F('lng*ci',  lambda e: np.log(e['num_groups']) * (e['wi_compute_ops']/max(e['wi_global_load_bits']+e['wi_global_store_bits'],1))),
        F('lwpg*ci', lambda e: np.log(e['wi_per_group']+1) * (e['wi_compute_ops']/max(e['wi_global_load_bits']+e['wi_global_store_bits'],1))),
        F('lops*ci', lambda e: np.log(e['wi_ops']) * (e['wi_compute_ops']/max(e['wi_global_load_bits']+e['wi_global_store_bits'],1))),
        F('s*ci',    lambda e: np.log1p(e.get('wi_global_load_lidx_stride',0)) * (e['wi_compute_ops']/max(e['wi_global_load_bits']+e['wi_global_store_bits'],1))),

        # --- register pressure (2) ---
        F('lng*rr',  lambda e: np.log(e['num_groups']) * (e.get('wi_peak_reg_bytes',0)/max(e.get('max_register_bytes',256),1))),
        F('lwpg*rr', lambda e: np.log(e['wi_per_group']+1) * (e.get('wi_peak_reg_bytes',0)/max(e.get('max_register_bytes',256),1))),

        # --- floor / tiny kernel (2) ---
        F('log_o100', lambda e: np.log(e['wi_ops']+100)),
        F('log1p_1kdo', lambda e: np.log1p(1000/max(e['wi_ops'],1))),

        # --- misc (2) ---
        F('spt', lambda e: e['wi_global_store_bits']/max(e['wi_per_group'],1)),
        F('log1p_mp', lambda e: np.log1p((e['wi_global_load_bits']+e['wi_global_store_bits'])/max(e['num_groups']*e['wi_per_group'],1))),
    ]

    FEATURE_NAMES = [name for name, _ in feature_defs]
    features = []
    for e in entries:
        features.append([fn(e) for _, fn in feature_defs])
    return np.array(features), FEATURE_NAMES


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

    print(f"All sections: {len(entries)} entries")

    y = np.log(np.array([e['time_us'] for e in entries]))

    # Build feature matrix
    X, FEATURE_NAMES = build_features(entries)
    print(f"Total features: {X.shape[1]}")

    # Ridge regression with all features (regularization handles overfitting)
    model = Ridge(alpha=1.0)
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\nRidge R² = {r2:.4f}")

    print(f"\nRidge R² = {r2:.4f}")
    print(f"Coefficients:")
    for name, c in zip(FEATURE_NAMES, model.coef_):
        print(f"  {name:20s}  {c:.6f}")
    print(f"  {'intercept':20s}  {model.intercept_:.6f}")

    print("\nPer-section R²:")
    for section in sorted(set(e['section'] for e in entries)):
        mask = [i for i, e in enumerate(entries) if e['section'] == section]
        if len(mask) > 1:
            y_sec = y[mask]
            y_pred_sec = y_pred[mask]
            ss_res_sec = np.sum((y_sec - y_pred_sec) ** 2)
            ss_tot_sec = np.sum((y_sec - np.mean(y_sec)) ** 2)
            r2_sec = 1 - ss_res_sec / ss_tot_sec if ss_tot_sec > 0 else 0
            print(f"  {section}: R²={r2_sec:.4f} (n={len(mask)})")

    print("\n=== Cost.rs constants ===")
    print(f"const LOG_TIME_COEFS: [f64; {len(FEATURE_NAMES)}] = [")
    for c in model.coef_:
        print(f"    {c:.6f},")
    print("];")
    print(f"const LOG_TIME_INTERCEPT: f64 = {model.intercept_:.6f};")

    return model, entries


if __name__ == '__main__':
    main()
