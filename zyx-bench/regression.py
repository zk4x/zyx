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
    # Each "feature" is (name, callable(e) -> value)
    feature_defs = [
        # core transforms (7)
        ('lng', lambda e: np.log(e['num_groups'])),
        ('lwpg', lambda e: np.log(e['wi_per_group'] + 1)),
        ('lops', lambda e: np.log(e['wi_ops'])),
        ('lcop', lambda e: np.log(e['wi_compute_ops'])),
        ('lgmem', lambda e: np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('barr', lambda e: e['wi_barriers']),
        ('lld_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0))),
        # secondary (4)
        ('wr', lambda e: e['wi_per_group'] / e.get('warp_size', 32)),
        ('rr', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)),
        ('lst_st', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0))),
        ('total_threads', lambda e: e['num_groups'] * e['wi_per_group']),
        # tertiary (4)
        ('store_per_thread', lambda e: e['wi_global_store_bits'] / max(e['wi_per_group'], 1)),
        ('ci', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)),
        ('overhead', lambda e: e['wi_ops'] / max(e['wi_compute_ops'], 1)),
        ('mem_per_thread', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1)),
        # log ratios (4)
        ('log1p_ci', lambda e: np.log1p(e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        ('log1p_overhead', lambda e: np.log1p(e['wi_ops'] / max(e['wi_compute_ops'], 1))),
        ('log1p_mp', lambda e: np.log1p((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('log1p_lm', lambda e: np.log1p((e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1, 1))),
        # floor features for tiny kernels (4)
        ('log_ops_100', lambda e: np.log(e['wi_ops'] + 100)),
        ('log_cops_100', lambda e: np.log(e['wi_compute_ops'] + 100)),
        ('log1p_1000_div_ops', lambda e: np.log1p(1000 / max(e['wi_ops'], 1))),
        ('log1p_1000_div_cops', lambda e: np.log1p(1000 / max(e['wi_compute_ops'], 1))),
        # work per resource (6)
        ('ops_per_thread', lambda e: e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('cops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('ops_per_group', lambda e: e['wi_ops'] / max(e['num_groups'], 1)),
        ('log_opt', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('inv_threads', lambda e: 1.0 / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('log_inv_threads', lambda e: np.log1p(1.0 / max(e['num_groups'] * e['wi_per_group'], 1))),
        # barrier interactions (6)
        ('barr*lops', lambda e: e['wi_barriers'] * np.log(e['wi_ops'])),
        ('barr*lcop', lambda e: e['wi_barriers'] * np.log(e['wi_compute_ops'])),
        ('barr*lgmem', lambda e: e['wi_barriers'] * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('barr*lwpg', lambda e: e['wi_barriers'] * np.log(e['wi_per_group'] + 1)),
        ('barr*lng', lambda e: e['wi_barriers'] * np.log(e['num_groups'])),
        ('barr*wr', lambda e: e['wi_barriers'] * (e['wi_per_group'] / e.get('warp_size', 32))),
        # stride core (5)
        ('lld_st*lops', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_ops'])),
        ('lld_st*lcop', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_compute_ops'])),
        ('lld_st*lng', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['num_groups'])),
        ('lld_st*barr', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * e['wi_barriers']),
        ('lld_st*lgmem', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        # stride secondary (4)
        ('lld_st*wr', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * (e['wi_per_group'] / e.get('warp_size', 32))),
        ('lld_st*lst_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log1p(e.get('wi_global_store_lidx_stride', 0))),
        ('lst_st*lops', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)) * np.log(e['wi_ops'])),
        ('lst_st*lng', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)) * np.log(e['num_groups'])),
        # memory ratio (4)
        ('lops*lgmem', lambda e: np.log(e['wi_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('lcop*lgmem', lambda e: np.log(e['wi_compute_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('lng*lgmem', lambda e: np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('lwpg*lgmem', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        # compute intensity interactions (4)
        ('lops*ci', lambda e: np.log(e['wi_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        ('lcop*ci', lambda e: np.log(e['wi_compute_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        ('lng*ci', lambda e: np.log(e['num_groups']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        ('lld_st*ci', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        # thread/group (4)
        ('lng*wr', lambda e: np.log(e['num_groups']) * (e['wi_per_group'] / e.get('warp_size', 32))),
        ('lwpg*wr', lambda e: np.log(e['wi_per_group'] + 1) * (e['wi_per_group'] / e.get('warp_size', 32))),
        ('lng*rr', lambda e: np.log(e['num_groups']) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1))),
        ('lwpg*rr', lambda e: np.log(e['wi_per_group'] + 1) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1))),
        # MLX (2)
        ('lng*lcop', lambda e: np.log(e['num_groups']) * np.log(e['wi_compute_ops'])),
        ('lwpg*lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops'])),
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
