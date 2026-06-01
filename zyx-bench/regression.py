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
                        r'wi_peak_reg_bytes=(\d+), wi_branches=(\d+), '
                        r'warp_size=(\d+), max_local_threads=(\d+), max_register_bytes=(\d+)',
                        lines[i].strip()
                    )
                    if m2:
                        entry['wi_local_load_bits'] = int(m2.group(1))
                        entry['wi_local_store_bits'] = int(m2.group(2))
                        entry['wi_peak_reg_bytes'] = int(m2.group(3))
                        entry['wi_branches'] = int(m2.group(4))
                        entry['warp_size'] = int(m2.group(5))
                        entry['max_local_threads'] = int(m2.group(6))
                        entry['max_register_bytes'] = int(m2.group(7))

                i += 1
                if i < len(lines):
                    time_match = re.match(r'([\d.]+)\s*(μs|ms)\s*~\s*([\d.]+)\s*GFLOP/s', lines[i].strip())
                    if time_match:
                        time_val = float(time_match.group(1))
                        unit = time_match.group(2)
                        entry['gflops'] = float(time_match.group(3))
                        if unit == 'ms':
                            time_val *= 1000.0
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
            'wi_branches', 'warp_size', 'max_local_threads', 'max_register_bytes',
            'time_us', 'gflops'
        ])
        for e in entries:
            writer.writerow([
                e['section'], e['num_groups'], e['wi_per_group'], e['wi_ops'],
                e['wi_compute_ops'], e['wi_barriers'], e['wi_global_load_bits'],
                e['wi_global_store_bits'], e.get('wi_local_load_bits', 0),
                e.get('wi_local_store_bits', 0), e.get('wi_peak_reg_bytes', 0),
                e.get('wi_branches', 0), e.get('warp_size', 32),
                e.get('max_local_threads', 1024), e.get('max_register_bytes', 256),
                e['time_us'], e['gflops']
            ])


FEATURE_NAMES = []


def build_candidates(e):
    """Return dict of candidate features from one entry."""
    ng = e['num_groups']
    wpg = e['wi_per_group']
    ops = e['wi_ops']
    cops = e['wi_compute_ops']
    barr = e['wi_barriers']
    glb = e['wi_global_load_bits']
    gsb = e['wi_global_store_bits']
    llb = e.get('wi_local_load_bits', 0)
    lsb = e.get('wi_local_store_bits', 0)
    preg = e.get('wi_peak_reg_bytes', 0)
    branch = e.get('wi_branches', 0)
    warp = e.get('warp_size', 32)
    mlthr = e.get('max_local_threads', 1024)
    mreg = e.get('max_register_bytes', 256)

    lng = np.log(ng)
    lwpg = np.log(wpg + 1)
    lops = np.log(ops)
    lcop = np.log(cops)
    wr = wpg / warp
    rr = preg / mreg if mreg > 0 else 0

    raw = {
        'ng': ng,
        'wpg': wpg,
        'ops': ops,
        'cops': cops,
        'barr': barr,
        'glb': glb,
        'gsb': gsb,
        'llb': llb,
        'lsb': lsb,
        'preg': preg,
        'branch': branch,
        'warp': warp,
    }

    log = {
        'lng': lng,
        'lwpg': lwpg,
        'lops': lops,
        'lcop': lcop,
        'lglb': np.log(glb) if glb > 0 else 0,
        'lgsb': np.log(gsb) if gsb > 0 else 0,
        'lllb': np.log(llb + 1),
        'llsb': np.log(lsb + 1),
        'lpreg': np.log(preg) if preg > 0 else 0,
    }

    derived = {
        'wr': wr,
        'rr': rr,
        'lgmem': np.log(glb + gsb) if glb + gsb > 0 else 0,
        'lmem': np.log(llb + lsb + 1),
    }

    cand = {}
    cand.update(raw)
    cand.update(log)
    cand.update(derived)

    # interactions
    log_keys = [k for k in log]
    der_keys = [k for k in derived]
    all_keys = log_keys + der_keys + ['barr', 'branch']
    for i, a in enumerate(all_keys):
        for b in all_keys[i:]:
            cand[f'{a}*{b}'] = cand[a] * cand[b]

    return cand


def build_feature_matrix(entries):
    global FEATURE_NAMES
    candidates = [build_candidates(e) for e in entries]
    FEATURE_NAMES = sorted(candidates[0].keys())
    X = np.zeros((len(entries), len(FEATURE_NAMES)))
    for i, c in enumerate(candidates):
        for j, name in enumerate(FEATURE_NAMES):
            X[i, j] = c[name]
    return X


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

    # Build candidate feature matrix with all interactions
    X_all = build_feature_matrix(entries)
    print(f"Total candidate features: {X_all.shape[1]}")

    # Greedy forward selection
    n_total = len(entries)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean) ** 2)

    selected = []
    remaining = list(range(X_all.shape[1]))
    best_r2 = -1e9

    while remaining:
        best_j = None
        best_new_r2 = -1e9
        for j in remaining:
            trial = selected + [j]
            X_trial = X_all[:, trial]
            model = Ridge(alpha=1.0)
            model.fit(X_trial, y)
            y_pred = model.predict(X_trial)
            ss_res = np.sum((y - y_pred) ** 2)
            r2 = 1 - ss_res / ss_tot
            if r2 > best_new_r2:
                best_new_r2 = r2
                best_j = j
        if best_new_r2 <= best_r2 + 0.0001:
            break
        selected.append(best_j)
        remaining.remove(best_j)
        best_r2 = best_new_r2
        print(f"  Added {FEATURE_NAMES[best_j]:20s}  R²={best_new_r2:.4f}")

    print(f"\nSelected {len(selected)} features with R²={best_r2:.4f}")
    print("\nFeature breakdown:")
    for idx in selected:
        print(f"  {FEATURE_NAMES[idx]}")

    # Full model with all selected features
    X = X_all[:, selected]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    y_pred = model.predict(X)

    print(f"\nRidge R² = {best_r2:.4f}")
    print(f"Coefficients:")
    for idx, c in zip(selected, model.coef_):
        print(f"  {FEATURE_NAMES[idx]:20s}  {c:.6f}")
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
    print(f"const LOG_TIME_COEFS: [f64; {len(selected)}] = [")
    for c in model.coef_:
        print(f"    {c:.6f},")
    print("];")
    print(f"const LOG_TIME_INTERCEPT: f64 = {model.intercept_:.6f};")

    return model, entries


if __name__ == '__main__':
    main()
