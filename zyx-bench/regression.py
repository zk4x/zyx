#!/usr/bin/env python3
"""Parse bench_output.txt -> ridge regression for cost model."""

import re
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
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

    # Build feature matrix
    FEATURE_NAMES = []
    X = None

    def f(name, fn):
        nonlocal X
        FEATURE_NAMES.append(name)
        feat = np.array([fn(e) for e in entries]).reshape(-1, 1)
        if X is None:
            X = feat
        else:
            X = np.column_stack([X, feat])

    # === CORE FEATURES ===
    f('lng', lambda e: np.log(e['num_groups']))
    f('lwpg', lambda e: np.log(e['wi_per_group'] + 1))
    f('lops', lambda e: np.log(e['wi_ops']))
    f('lcop', lambda e: np.log(e['wi_compute_ops']))
    f('lgmem', lambda e: np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('barr', lambda e: e['wi_barriers'])
    f('wr', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
    f('rr', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1))
    f('total_threads', lambda e: e['num_groups'] * e['wi_per_group'])
    f('overhead', lambda e: e['wi_ops'] / max(e['wi_compute_ops'], 1))
    f('ci', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))

    # log ratios
    f('log1p_ci', lambda e: np.log1p(e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    f('log1p_overhead', lambda e: np.log1p(e['wi_ops'] / max(e['wi_compute_ops'], 1)))
    f('log1p_mp', lambda e: np.log1p((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1)))
    f('log1p_lm', lambda e: np.log1p((e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1, 1)))

    # floor features for tiny kernels
    f('log_ops_100', lambda e: np.log(e['wi_ops'] + 100))
    f('log_ops_10', lambda e: np.log(e['wi_ops'] + 10))
    f('log_cops_100', lambda e: np.log(e['wi_compute_ops'] + 100))
    f('log1p_1000_div_ops', lambda e: np.log1p(1000 / max(e['wi_ops'], 1)))
    f('log1p_100_div_ops', lambda e: np.log1p(100 / max(e['wi_ops'], 1)))
    f('log1p_1000_div_cops', lambda e: np.log1p(1000 / max(e['wi_compute_ops'], 1)))

    # work per resource
    f('ops_per_thread', lambda e: e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('cops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('ops_per_group', lambda e: e['wi_ops'] / max(e['num_groups'], 1))

    # shuffle depth + interactions
    f('log_lwpg', lambda e: np.log(max(np.log(e['wi_per_group'] + 1), 1e-8)))
    f('lng_log_opt', lambda e: np.log(e['num_groups']) * np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))
    f('lng_log_lwpg', lambda e: np.log(e['num_groups']) * np.log(max(np.log(e['wi_per_group'] + 1), 1e-8)))
    f('lwpg_log_opt', lambda e: np.log(e['wi_per_group']+1) * np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))
    f('raw_ng', lambda e: e['num_groups'])
    f('raw_ng_wpg', lambda e: e['num_groups'] * e['wi_per_group'])
    f('raw_ng_per_ops', lambda e: e['num_groups'] / max(e['wi_ops'], 1))
    f('lbranch', lambda e: np.log1p(e.get('wi_branches', 0)))
    f('barr*lbranch', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_branches', 0)))
    f('lwpg*lbranch', lambda e: np.log(e['wi_per_group'] + 1) * np.log1p(e.get('wi_branches', 0)))
    f('lng*lbranch', lambda e: np.log(e['num_groups']) * np.log1p(e.get('wi_branches', 0)))
    f('lng_lwpg', lambda e: np.log(e['num_groups']) * np.log(e['wi_per_group'] + 1))
    f('lwpg_lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops']))

    # shuffle depth / work balance
    f('log1p_lwpg_div_opt', lambda e: np.log1p(np.log(e['wi_per_group']+1) / max(1e-8, e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))))
    f('lng_log1p_lwpg_div_opt', lambda e: np.log(e['num_groups']) * np.log1p(np.log(e['wi_per_group']+1) / max(1e-8, e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))))

    # ops-per-group
    f('log_ops_per_group', lambda e: np.log(e['wi_ops'] / max(e['num_groups'], 1) + 1))
    f('log_ops_per_grp_barr', lambda e: e['wi_barriers'] * np.log(e['wi_ops'] / max(e['num_groups'], 1) + 1))
    f('log_opt', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))
    f('inv_threads', lambda e: 1.0 / max(e['num_groups'] * e['wi_per_group'], 1))
    f('log_inv_threads', lambda e: np.log1p(1.0 / max(e['num_groups'] * e['wi_per_group'], 1)))

    # barrier regimes
    f('log_opt_barr', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'] * max(e['wi_barriers'], 1), 1)))
    f('log_wpg_barr', lambda e: np.log1p(e['wi_per_group'] * e['wi_barriers']))

    # barrier one-hot
    for bval in [0, 3, 4, 5, 6, 7, 8]:
        f(f'b{bval}', lambda e, b=bval: 1.0 if e['wi_barriers'] == b else 0.0)

    # barrier groups (denser signal)
    f('barr_low', lambda e: 1.0 if e['wi_barriers'] == 0 else 0.0)
    f('barr_med', lambda e: 1.0 if e['wi_barriers'] in [3, 4, 5] else 0.0)
    f('barr_high', lambda e: 1.0 if e['wi_barriers'] >= 6 else 0.0)
    f('barr_nonzero', lambda e: 1.0 if e['wi_barriers'] > 0 else 0.0)

    # barrier one-hot x lng/lwpg/lops
    for bval in [0, 3, 4, 5, 6, 7, 8]:
        f(f'b{bval}*lng', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['num_groups']))
        f(f'b{bval}*lwpg', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['wi_per_group'] + 1))
        f(f'b{bval}*lops', lambda e, b=bval: (1.0 if e['wi_barriers'] == b else 0.0) * np.log(e['wi_ops']))

    # barrier interactions
    f('barr*lops', lambda e: e['wi_barriers'] * np.log(e['wi_ops']))
    f('barr*lcop', lambda e: e['wi_barriers'] * np.log(e['wi_compute_ops']))
    f('barr*lgmem', lambda e: e['wi_barriers'] * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('barr*lwpg', lambda e: e['wi_barriers'] * np.log(e['wi_per_group'] + 1))
    f('barr*lng', lambda e: e['wi_barriers'] * np.log(e['num_groups']))
    f('barr*wr', lambda e: e['wi_barriers'] * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))

    # stride features
    f('lld_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)))
    f('lst_st', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)))
    f('lld_st*lops', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_ops']))
    f('lld_st*lcop', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_compute_ops']))
    f('lld_st*lng', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['num_groups']))
    f('lld_st*barr', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * e['wi_barriers'])
    f('lld_st*lgmem', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('lld_st*wr', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    f('lld_st*lst_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * np.log1p(e.get('wi_global_store_lidx_stride', 0)))
    f('lst_st*lops', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)) * np.log(e['wi_ops']))
    f('lst_st*lng', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0)) * np.log(e['num_groups']))
    f('lld_st_loc', lambda e: np.log1p(e.get('wi_local_load_lidx_stride', 0)))
    f('lst_st_loc', lambda e: np.log1p(e.get('wi_local_store_lidx_stride', 0)))
    f('lld_st_loc*lng', lambda e: np.log1p(e.get('wi_local_load_lidx_stride', 0)) * np.log(e['num_groups']))
    f('lst_st_loc*lng', lambda e: np.log1p(e.get('wi_local_store_lidx_stride', 0)) * np.log(e['num_groups']))

    # memory ratios
    f('lops*lgmem', lambda e: np.log(e['wi_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('lcop*lgmem', lambda e: np.log(e['wi_compute_ops']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('lng*lgmem', lambda e: np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('lwpg*lgmem', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))

    # compute intensity interactions
    f('lops*ci', lambda e: np.log(e['wi_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    f('lcop*ci', lambda e: np.log(e['wi_compute_ops']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    f('lng*ci', lambda e: np.log(e['num_groups']) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    f('lld_st*ci', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0)) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))

    # thread/group
    f('lng*wr', lambda e: np.log(e['num_groups']) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    f('lwpg*wr', lambda e: np.log(e['wi_per_group'] + 1) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    f('lng*rr', lambda e: np.log(e['num_groups']) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)))
    f('lwpg*rr', lambda e: np.log(e['wi_per_group'] + 1) * (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)))

    # MLX
    f('lng*lcop', lambda e: np.log(e['num_groups']) * np.log(e['wi_compute_ops']))
    f('lwpg*lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops']))

    # warp utilization
    f('warp_util', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
    f('log_warp_waste', lambda e: np.log(32 / max(e['wi_per_group'], 1)))
    f('lng_warp_waste', lambda e: np.log(e['num_groups']) * np.log(32 / max(e['wi_per_group'], 1)))
    f('lwpg_warp_waste', lambda e: np.log(e['wi_per_group'] + 1) * np.log(32 / max(e['wi_per_group'], 1)))

    # launch overhead regime
    f('lng_div_lops', lambda e: np.log(max(e['num_groups'] / max(e['wi_ops'], 1), 1e-8)))

    # tree reduction
    f('llm', lambda e: np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)))
    f('barr_wpg', lambda e: e['wi_barriers'] * np.log1p(e['wi_per_group']))

    # barrier-regime interactions
    f('b0*lcop', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_compute_ops']))
    f('b0*lgmem', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('log_cops_100*b0', lambda e: np.log(e['wi_compute_ops'] + 100) * (1.0 if e['wi_barriers'] == 0 else 0.0))
    f('lng*lwpg*b0', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['num_groups']) * np.log1p(e['wi_per_group']))
    f('lwpg*lops*b0', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log1p(e['wi_per_group']) * np.log(e['wi_ops']))

    # GPU execution regime features
    f('coalescing_eff', lambda e: 1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0))
    f('coalescing_eff_st', lambda e: 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0))
    f('reduce_kind', lambda e: e['wi_barriers'] / max(np.log2(e['wi_per_group'] + 1), 1.0))
    f('element_ops', lambda e: e['wi_compute_ops'] / max(e['num_groups'], 1))
    f('warp_div', lambda e: np.log(max(32 / max(e['wi_per_group'], 1), 1e-8)))
    f('sm_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0))
    f('local_occupancy', lambda e: np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0)
    f('launch_overhead', lambda e: np.exp(-e['wi_ops'] / 1000.0) if e['wi_ops'] > 0 else 1.0)
    f('tree_height', lambda e: np.log2(e['wi_barriers'] + 1) if e['wi_barriers'] > 0 else 0)
    f('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)))
    f('element_ops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('compute_intensity_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'] * 32, 1))

    # All GPU-specific features from neural_net.py
    f('barrier_density', lambda e: e['wi_barriers'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('barrier_per_thread', lambda e: e['wi_barriers'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('bw_util_barr', lambda e: ((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max((e['num_groups'] * e['wi_per_group']) * 256.0, 1)) * e['wi_barriers'])
    f('coalescing_barr', lambda e: (1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)) * e['wi_barriers'])
    f('global_coalescing_min', lambda e: min(1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0), 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)))
    f('global_coalescing_product', lambda e: (1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)) * (1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)))
    f('memory_intensity', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    f('occupancy_barr', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0) * e['wi_barriers'])
    f('shared_mem_barr', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) * e['wi_barriers'])
    f('shared_mem_efficiency', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('shared_mem_load_ratio', lambda e: e.get('wi_local_load_bits', 0) / max(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1, 1))
    f('shared_mem_per_thread', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('shared_mem_pressure', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0)
    f('shared_mem_store_ratio', lambda e: e.get('wi_local_store_bits', 0) / max(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1, 1))
    f('shared_mem_util', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0)
    f('thread_efficiency', lambda e: (e['wi_ops'] * e['wi_per_group']) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('warp_divergence', lambda e: np.log(max(32.0 / e['wi_per_group'], 1.0)))
    f('warp_efficiency', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
    f('warp_occupancy', lambda e: (e['num_groups'] * e['wi_per_group']) / 2048.0)
    f('warp_util_barr', lambda e: (e['wi_per_group'] / max(e.get('warp_size', 32), 1)) * e['wi_barriers'])
    f('warp_utilization', lambda e: e['wi_per_group'] / max(e.get('warp_size', 32), 1))
    f('warp_waste', lambda e: (32.0 - e['wi_per_group']) / 32.0)
    f('store_per_thread', lambda e: e['wi_global_store_bits'] / max(e['wi_per_group'], 1))
    f('mem_per_thread', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1))

    # === INTERACTION FEATURES ===

    # ops * barriers * compute intensity
    f('ops_barr_ci', lambda e: np.log(e['wi_ops']) * e['wi_barriers'] * e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    f('threads_barr_gmem', lambda e: np.log(e['num_groups'] * e['wi_per_group']) * e['wi_barriers'] * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('compute_per_barrier', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * max(e['wi_barriers'], 1), 1))
    f('mem_access_pattern', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0) + e.get('wi_global_store_lidx_stride', 0)) / max(np.log1p(e['wi_ops']), 1))
    f('warp_mem_footprint', lambda e: (e['wi_per_group'] / max(e.get('warp_size', 32), 1)) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('shared_mem_per_group', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / max(e['num_groups'], 1))
    f('reg_thread_pressure', lambda e: e.get('wi_peak_reg_bytes', 0) * e['num_groups'] * e['wi_per_group'] / max(e.get('max_register_bytes', 256), 1))
    f('ci_warp_util', lambda e: (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)) * (e['wi_per_group'] / max(e.get('warp_size', 32), 1)))
    f('barrier_overhead_cost', lambda e: e['wi_barriers'] * np.log1p(e['wi_barriers']) / max(np.log1p(e['wi_ops']), 1))
    f('data_reuse_factor', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] / 32.0, 1))

    # sync vs direct overhead ratio (captures barrier=0 vs barrier>0 patterns)
    f('sync_work_ratio', lambda e: e['wi_barriers'] * np.log1p(e['wi_ops']) / max(e['wi_compute_ops'], 1))
    f('barrier_compute_efficiency', lambda e: e['wi_compute_ops'] / max(e['wi_barriers'] * e['num_groups'] * e['wi_per_group'], 1))
    f('group_work_density', lambda e: e['wi_ops'] / max(e['num_groups'] * np.log1p(e['wi_per_group']), 1))
    f('launch_compute_ratio', lambda e: np.exp(-e['wi_ops'] / 500.0) * e['wi_compute_ops'])
    f('tiny_kernel_groups', lambda e: np.exp(-e['wi_ops'] / 1000.0) * e['num_groups'])
    f('barrier_pressure', lambda e: e['wi_barriers'] * np.log1p(e['wi_ops']) / max(e['num_groups'], 1))
    f('memory_locality', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('memory_pressure', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1))

    # compute-bound vs memory-bound interaction features
    f('compute_mem_ratio', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    f('ops_intensity', lambda e: e['wi_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    f('ops_per_barrier', lambda e: e['wi_ops'] / max(e['wi_barriers'], 1))
    f('local_mem_ratio', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))
    f('ops_barr_interaction', lambda e: e['wi_barriers'] * e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))

    # barrier and memory interaction features
    f('barrier_factor', lambda e: e['wi_barriers'] / max(np.log1p(e['wi_ops']), 1))
    f('barrier_efficiency', lambda e: e['wi_barriers'] / max(np.log1p(e['wi_ops']), 1))
    f('ops_compute_factor', lambda e: np.log(e['wi_compute_ops'] + 1) * np.log(e['wi_global_load_bits'] + 1) / max(e['num_groups'], 1))
    f('mem_compute_ratio_log', lambda e: np.log((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1) + 1))
    f('ops_compute_interaction', lambda e: np.log(e['wi_ops'] + 1) * np.log(e['wi_compute_ops'] + 1) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('ops_barrier_factor', lambda e: np.log(e['wi_ops'] + 1) * e['wi_barriers'] / max(e['num_groups'], 1))
    f('mem_compute_factor', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1))
    f('memory_overhead_ratio', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_compute_ops'], 1))
    f('complexity_factor', lambda e: e['wi_ops'] * e['wi_barriers'] / max(e['num_groups'], 1))

    # === EXTRA LOG TRANSFORMS ===
    f('log1p_lng', lambda e: np.log1p(np.abs(np.log(e['num_groups']))))
    f('log1p_lwpg_raw', lambda e: np.log1p(np.abs(np.log(e['wi_per_group'] + 1))))
    f('log1p_lops_raw', lambda e: np.log1p(np.abs(np.log(e['wi_ops']))))
    f('log_ng_per_ops', lambda e: np.log(e['num_groups'] / max(e['wi_ops'], 1) + 1))
    f('log_threads_per_ops', lambda e: np.log((e['num_groups'] * e['wi_per_group']) / max(e['wi_ops'], 1) + 1))
    f('b0_ci', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)))
    f('barr_med_ng', lambda e: (1.0 if e['wi_barriers'] in [3, 4, 5] else 0.0) * np.log(e['num_groups']))
    f('barr_med_lops', lambda e: (1.0 if e['wi_barriers'] in [3, 4, 5] else 0.0) * np.log(e['wi_ops']))
    f('barr_high_ng', lambda e: (1.0 if e['wi_barriers'] >= 6 else 0.0) * np.log(e['num_groups']))
    f('barr_nonzero_lops', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_ops']))
    f('barr_nonzero_lmem', lambda e: (1.0 if e['wi_barriers'] > 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('compute_density', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))
    f('mem_density', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1))
    f('group_aspect', lambda e: e['num_groups'] / max(e['wi_per_group'], 1))
    f('thread_compute', lambda e: np.log(e['wi_compute_ops'] + 1) * np.log(e['num_groups'] * e['wi_per_group'] + 1))

    # === Barrier>0 gate: multiply top features by (barriers > 0) for separate slopes ===
    _barr = lambda e: 1.0 if e['wi_barriers'] > 0 else 0.0
    f('bgt_lng', lambda e, b=_barr: b(e) * np.log(e['num_groups']))
    f('bgt_lwpg', lambda e, b=_barr: b(e) * np.log(e['wi_per_group'] + 1))
    f('bgt_lops', lambda e, b=_barr: b(e) * np.log(e['wi_ops']))
    f('bgt_lcop', lambda e, b=_barr: b(e) * np.log(e['wi_compute_ops']))
    f('bgt_lgmem', lambda e, b=_barr: b(e) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('bgt_lng_lgmem', lambda e, b=_barr: b(e) * np.log(e['num_groups']) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    f('bgt_lng_lcop', lambda e, b=_barr: b(e) * np.log(e['num_groups']) * np.log(e['wi_compute_ops']))
    f('bgt_lwpg_lgmem', lambda e, b=_barr: b(e) * np.log(e['wi_per_group'] + 1) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1))
    # Raw (non-log) features for barrier>0 to capture U-shape
    f('bgt_ng_raw', lambda e, b=_barr: b(e) * e['num_groups'] / 1000.0)
    f('bgt_barr_raw', lambda e, b=_barr: b(e) * e['wi_barriers'])
    f('bgt_barr_ng', lambda e, b=_barr: b(e) * e['wi_barriers'] * e['num_groups'] / 1000.0)

    print(f"Total features: {X.shape[1]}")

    # === DT leaf features: train a shallow DT, use leaf membership as features ===
    # This gives piecewise linear capability: each leaf gets its own bias
    print("\nTraining DT for leaf features...")
    dt_leaf = DecisionTreeRegressor(max_leaf_nodes=200, random_state=42, min_samples_leaf=5)
    dt_leaf.fit(X, y)
    leaf_ids = dt_leaf.apply(X)
    n_leaves = dt_leaf.get_n_leaves()
    print(f"DT has {n_leaves} leaves, R²={dt_leaf.score(X, y):.4f}")

    # One-hot encode leaf membership (sklearn leaves are 1-indexed, may not be contiguous)
    unique_leaves = sorted(set(leaf_ids))
    leaf_to_idx = {leaf: i for i, leaf in enumerate(unique_leaves)}
    n_leaves = len(unique_leaves)
    leaf_onehot = np.zeros((len(entries), n_leaves))
    for i, leaf in enumerate(leaf_ids):
        leaf_onehot[i, leaf_to_idx[leaf]] = 1.0

    # Add to feature matrix
    for leaf_idx in range(n_leaves):
        FEATURE_NAMES.append(f'dt_leaf_{leaf_idx}')
    X = np.column_stack([X, leaf_onehot])
    print(f"Total features after DT leaves: {X.shape[1]}")

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

    # Scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    print(f"\nTotal entries: {len(entries)}, Features: {X.shape[1]}")
    print(f"Section counts: {dict(section_counts)}")

    # Try multiple model approaches
    best_model = None
    best_r2_sections = {}
    best_min_r2 = -999

    print("\n=== Model Search ===")

    # Ridge regression with various alphas
    for alpha in [1e-8, 1e-6, 1e-4, 1e-2]:
        l = Ridge(alpha=alpha, random_state=42, max_iter=10000)
        l.fit(Xs, y, sample_weight=sample_weights)
        section_r2s = evaluate_model(l, Xs, y, section_groups)
        min_r2 = min(section_r2s.values()) if section_r2s else 0
        print(f"   Ridge(alpha={alpha:.8f}): min R²={min_r2:.4f}")
        if min_r2 > best_min_r2:
            best_min_r2 = min_r2
            best_model = l
            best_r2_sections = section_r2s

    # Weight boosting: give extra weight to sections below 0.8
    below_target = [s for s, r in best_r2_sections.items() if r < 0.8]
    if below_target:
        print(f"\n   Boosting weights for {len(below_target)} sections below 0.8...")
        boosted_weights = sample_weights.copy()
        for section in below_target:
            for idx in section_groups[section]:
                boosted_weights[idx] *= 3.0
        boosted_weights = boosted_weights / np.mean(boosted_weights)
        
        for alpha in [1e-8, 1e-6, 1e-4]:
            l = Ridge(alpha=alpha, random_state=42, max_iter=10000)
            l.fit(Xs, y, sample_weight=boosted_weights)
            section_r2s = evaluate_model(l, Xs, y, section_groups)
            min_r2 = min(section_r2s.values()) if section_r2s else 0
            print(f"   Boosted Ridge(alpha={alpha:.8f}): min R²={min_r2:.4f}")
            if min_r2 > best_min_r2:
                best_min_r2 = min_r2
                best_model = l
                best_r2_sections = section_r2s
                print(f"   *** WEIGHT BOOSTING HELPED ***")

    # Feature selection - pick top features by coefficient magnitude
    coef_abs = np.abs(best_model.coef_)
    threshold = np.percentile(coef_abs, 50)
    selected = coef_abs > threshold
    selected_names = [n for n, s in zip(FEATURE_NAMES, selected) if s]
    selected_coefs = best_model.coef_[selected]

    print(f"\n=== Results ===")
    print(f"Selected {len(selected_names)}/{len(FEATURE_NAMES)} features")
    print(f"Minimum per-section R²: {best_min_r2:.4f}")
    print()
    for section in sorted(section_groups.keys()):
        if section in best_r2_sections:
            r2 = best_r2_sections[section]
            flag = " *** ABOVE 0.95 ***" if "MatMul" in section and r2 >= 0.95 else ""
            flag += " *** ABOVE 0.8 ***" if r2 >= 0.8 and section != "MatMul" else ""
            print(f"  {section}: R²={r2:.4f}{flag}")

    if "MatMul" in best_r2_sections and best_r2_sections["MatMul"] >= 0.95 and best_min_r2 >= 0.8:
        print("\n*** All targets MET! ***")
    elif "MatMul" in best_r2_sections:
        print(f"\nMatMul R²={best_r2_sections['MatMul']:.4f} (target: 0.95)")
        print(f"Worst section R²={best_min_r2:.4f} (target: 0.8)")

    # Don't output cost.rs until targets are met
    # print(f"\n=== Cost.rs ({len(selected_names)} selected features) ===")
    # ...

    return best_model, entries, selected_names, selected_coefs


if __name__ == '__main__':
    main()
