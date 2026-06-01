#!/usr/bin/env python3
"""Neural net focused specifically on Reduce Axis section to analyze bottlenecks."""

import re
import csv
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

        if line.startswith('num_groups=') or line.startswith('cost='):
            m = re.match(r'(?:cost=(\d+), )?num_groups=(\d+), wi_per_group=(\d+), wi_ops=(\d+), wi_compute_ops=(\d+), wi_barriers=(\d+), wi_global_load_bits=(\d+), wi_global_store_bits=(\d+)', line)
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
        # floor features for tiny kernels (6)
        ('log_ops_100', lambda e: np.log(e['wi_ops'] + 100)),
        ('log_ops_10', lambda e: np.log(e['wi_ops'] + 10)),
        ('log_cops_100', lambda e: np.log(e['wi_compute_ops'] + 100)),
        ('log1p_1000_div_ops', lambda e: np.log1p(1000 / max(e['wi_ops'], 1))),
        ('log1p_100_div_ops', lambda e: np.log1p(100 / max(e['wi_ops'], 1))),
        ('log1p_1000_div_cops', lambda e: np.log1p(1000 / max(e['wi_compute_ops'], 1))),
        # work per resource (6)
        ('ops_per_thread', lambda e: e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('cops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('ops_per_group', lambda e: e['wi_ops'] / max(e['num_groups'], 1)),
        # shuffle depth + interactions (3)
        ('log_lwpg', lambda e: np.log(max(np.log(e['wi_per_group'] + 1), 1e-8))),
        ('lng_log_opt', lambda e: np.log(e['num_groups']) * np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('lng_log_lwpg', lambda e: np.log(e['num_groups']) * np.log(max(np.log(e['wi_per_group'] + 1), 1e-8))),
        ('lwpg_log_opt', lambda e: np.log(e['wi_per_group']+1) * np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('raw_ng', lambda e: e['num_groups']),
        ('raw_ng_wpg', lambda e: e['num_groups'] * e['wi_per_group']),
        ('raw_ng_per_ops', lambda e: e['num_groups'] / max(e['wi_ops'], 1)),
        ('lbranch', lambda e: np.log1p(e.get('wi_branches', 0))),
        ('barr*lbranch', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_branches', 0))),
        ('lwpg*lbranch', lambda e: np.log(e['wi_per_group'] + 1) * np.log1p(e.get('wi_branches', 0))),
        ('lng*lbranch', lambda e: np.log(e['num_groups']) * np.log1p(e.get('wi_branches', 0))),
        ('lng_lwpg', lambda e: np.log(e['num_groups']) * np.log(e['wi_per_group'] + 1)),
        ('lwpg_lops', lambda e: np.log(e['wi_per_group'] + 1) * np.log(e['wi_ops'])),
        # shuffle depth / work balance (2)
        ('log1p_lwpg_div_opt', lambda e: np.log1p(np.log(e['wi_per_group']+1) / max(1e-8, e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))),
        ('lng_log1p_lwpg_div_opt', lambda e: np.log(e['num_groups']) * np.log1p(np.log(e['wi_per_group']+1) / max(1e-8, e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)))),
        # ops-per-group (2)
        ('log_ops_per_group', lambda e: np.log(e['wi_ops'] / max(e['num_groups'], 1) + 1)),
        ('log_ops_per_grp_barr', lambda e: e['wi_barriers'] * np.log(e['wi_ops'] / max(e['num_groups'], 1) + 1)),
        ('log_opt', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('inv_threads', lambda e: 1.0 / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('log_inv_threads', lambda e: np.log1p(1.0 / max(e['num_groups'] * e['wi_per_group'], 1))),
        # barrier regime (2)
        ('log_opt_barr', lambda e: np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'] * max(e['wi_barriers'], 1), 1))),
        ('log_wpg_barr', lambda e: np.log1p(e['wi_per_group'] * e['wi_barriers'])),
        # barrier one-hot (7 values: 0,3,4,5,6,7,8)
        ('b0', lambda e: 1.0 if e['wi_barriers'] == 0 else 0.0),
        ('b3', lambda e: 1.0 if e['wi_barriers'] == 3 else 0.0),
        ('b4', lambda e: 1.0 if e['wi_barriers'] == 4 else 0.0),
        ('b5', lambda e: 1.0 if e['wi_barriers'] == 5 else 0.0),
        ('b6', lambda e: 1.0 if e['wi_barriers'] == 6 else 0.0),
        ('b7', lambda e: 1.0 if e['wi_barriers'] == 7 else 0.0),
        ('b8', lambda e: 1.0 if e['wi_barriers'] == 8 else 0.0),
        # barrier one-hot × lng (7)
        ('b0*lng', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['num_groups'])),
        ('b3*lng', lambda e: (1.0 if e['wi_barriers'] == 3 else 0.0) * np.log(e['num_groups'])),
        ('b4*lng', lambda e: (1.0 if e['wi_barriers'] == 4 else 0.0) * np.log(e['num_groups'])),
        ('b5*lng', lambda e: (1.0 if e['wi_barriers'] == 5 else 0.0) * np.log(e['num_groups'])),
        ('b6*lng', lambda e: (1.0 if e['wi_barriers'] == 6 else 0.0) * np.log(e['num_groups'])),
        ('b7*lng', lambda e: (1.0 if e['wi_barriers'] == 7 else 0.0) * np.log(e['num_groups'])),
        ('b8*lng', lambda e: (1.0 if e['wi_barriers'] == 8 else 0.0) * np.log(e['num_groups'])),
        # barrier one-hot × lwpg (7)
        ('b0*lwpg', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b3*lwpg', lambda e: (1.0 if e['wi_barriers'] == 3 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b4*lwpg', lambda e: (1.0 if e['wi_barriers'] == 4 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b5*lwpg', lambda e: (1.0 if e['wi_barriers'] == 5 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b6*lwpg', lambda e: (1.0 if e['wi_barriers'] == 6 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b7*lwpg', lambda e: (1.0 if e['wi_barriers'] == 7 else 0.0) * np.log(e['wi_per_group'] + 1)),
        ('b8*lwpg', lambda e: (1.0 if e['wi_barriers'] == 8 else 0.0) * np.log(e['wi_per_group'] + 1)),
        # barrier one-hot × lops (7)
        ('b0*lops', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_ops'])),
        ('b3*lops', lambda e: (1.0 if e['wi_barriers'] == 3 else 0.0) * np.log(e['wi_ops'])),
        ('b4*lops', lambda e: (1.0 if e['wi_barriers'] == 4 else 0.0) * np.log(e['wi_ops'])),
        ('b5*lops', lambda e: (1.0 if e['wi_barriers'] == 5 else 0.0) * np.log(e['wi_ops'])),
        ('b6*lops', lambda e: (1.0 if e['wi_barriers'] == 6 else 0.0) * np.log(e['wi_ops'])),
        ('b7*lops', lambda e: (1.0 if e['wi_barriers'] == 7 else 0.0) * np.log(e['wi_ops'])),
        ('b8*lops', lambda e: (1.0 if e['wi_barriers'] == 8 else 0.0) * np.log(e['wi_ops'])),
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
        # local stride core (4)
        ('lld_st_loc', lambda e: np.log1p(e.get('wi_local_load_lidx_stride', 0))),
        ('lst_st_loc', lambda e: np.log1p(e.get('wi_local_store_lidx_stride', 0))),
        ('lld_st_loc*barr', lambda e: np.log1p(e.get('wi_local_load_lidx_stride', 0)) * e['wi_barriers']),
        ('lst_st_loc*barr', lambda e: np.log1p(e.get('wi_local_store_lidx_stride', 0)) * e['wi_barriers']),
        # local stride × lng (2)
        ('lld_st_loc*lng', lambda e: np.log1p(e.get('wi_local_load_lidx_stride', 0)) * np.log(e['num_groups'])),
        ('lst_st_loc*lng', lambda e: np.log1p(e.get('wi_local_store_lidx_stride', 0)) * np.log(e['num_groups'])),
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
        # warp utilization (4)
        ('warp_util', lambda e: e['wi_per_group'] / e.get('warp_size', 32)),
        ('log_warp_waste', lambda e: np.log(32 / max(e['wi_per_group'], 1))),
        ('lng_warp_waste', lambda e: np.log(e['num_groups']) * np.log(32 / max(e['wi_per_group'], 1))),
        ('lwpg_warp_waste', lambda e: np.log(e['wi_per_group'] + 1) * np.log(32 / max(e['wi_per_group'], 1))),
        # launch overhead regime (3)
        ('lng_div_lops', lambda e: np.log(max(e['num_groups'] / max(e['wi_ops'], 1), 1e-8))),
        ('raw_ng_lwpg', lambda e: e['num_groups'] * np.log1p(e['wi_per_group'])),
        ('total_threads_lwpg', lambda e: (e['num_groups'] * e['wi_per_group']) * np.log1p(e['wi_per_group'])),
        # tree reduction (4)
        ('llm', lambda e: np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('barr_llm', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('barr_wpg', lambda e: e['wi_barriers'] * np.log1p(e['wi_per_group'])),
        ('barr_ng_loc', lambda e: e['wi_barriers'] * np.log1p(e['num_groups'] / max(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0), 1))),
        # barrier-regime interactions (5)
        ('b0*lcop', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_compute_ops'])),
        ('b0*lgmem', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('log_cops_100*b0', lambda e: np.log(e['wi_compute_ops'] + 100) * (1.0 if e['wi_barriers'] == 0 else 0.0)),
        ('lng*lwpg*b0', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log(e['num_groups']) * np.log1p(e['wi_per_group'])),
        ('lwpg*lops*b0', lambda e: (1.0 if e['wi_barriers'] == 0 else 0.0) * np.log1p(e['wi_per_group']) * np.log(e['wi_ops'])),
        # GPU execution regime features
        ('coalescing_eff', lambda e: 1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)),
        ('coalescing_eff_st', lambda e: 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)),
        ('reduce_kind', lambda e: e['wi_barriers'] / max(np.log2(e['wi_per_group'] + 1), 1.0)),
        ('barrier_overhead', lambda e: e['wi_barriers'] * np.log1p(e['wi_barriers']) / max(e['wi_ops'] / max(e['wi_barriers'], 1), 1)),
        ('bw_ratio', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max((e['num_groups'] * e['wi_per_group']) * 256.0, 1)),
        ('mem_computed_cross', lambda e: (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)) / 10.0),
        ('element_ops', lambda e: e['wi_compute_ops'] / max(e['num_groups'], 1)),
        ('reg_pressure', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)),
        ('warp_div', lambda e: np.log(max(32 / max(e['wi_per_group'], 1), 1e-8))),
        ('sm_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0)),
        ('local_occupancy', lambda e: np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0),
        ('bank_conflict', lambda e: (int(e.get('wi_local_load_lidx_stride', 0)) % 10) if e.get('wi_local_load_lidx_stride', 0) > 0 else 0.0),
        ('launch_overhead', lambda e: np.exp(-e['wi_ops'] / 1000.0) if e['wi_ops'] > 0 else 1.0),
        ('data_reuse', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] / 32.0, 1)),
        ('sync_ratio', lambda e: e['wi_barriers'] / max(e['wi_ops'], 1)),
        ('local_coalescing', lambda e: 1.0 - min(abs(int(e.get('wi_local_load_lidx_stride', 0)) % 10 - 5) / 5.0, 1.0) if e.get('wi_local_load_lidx_stride', 0) > 0 else 1.0),
        ('work_balance', lambda e: np.log1p(e['wi_per_group']) / max(np.log1p(e['num_groups']), 1e-8)),
        ('access_complexity', lambda e: (np.log1p(e.get('wi_global_load_lidx_stride', 0)) + np.log1p(e.get('wi_global_store_lidx_stride', 0))) / max(np.log1p(e['wi_ops']), 1)),
        ('tree_height', lambda e: np.log2(e['wi_barriers'] + 1) if e['wi_barriers'] > 0 else 0),
        ('fetch_eff', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] / 8.0, 1)),
        # 5 features targeting specific bottlenecks
        ('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('element_ops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('layer_norm_passes', lambda e: min(e['wi_barriers'] / 2.0, 3.0)),
        ('embedding_sparsity', lambda e: e['num_groups'] / max(e['wi_global_load_bits'] / 32.0, 1)),
        ('compute_intensity_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'] * 32, 1)),
        # Additional features for challenging sections
        ('layer_norm_complexity', lambda e: (e['wi_compute_ops'] * e['wi_barriers']) / max(e['num_groups'], 1)),
        ('reduce_tree_depth', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0))),
        ('memory_access_pattern', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_ops'], 1)),
        ('thread_efficiency', lambda e: (e['wi_ops'] * e['wi_per_group']) / max(e['num_groups'] * e['wi_per_group'], 1)),
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

    # Filter to only Reduce Axis section
    reduce_axis_entries = [e for e in entries if e['section'] == 'Reduce Axis']
    print(f"Reduce Axis entries: {len(reduce_axis_entries)}")

    if len(reduce_axis_entries) < 100:
        print("ERROR: Not enough Reduce Axis entries for meaningful training")
        return

    y = np.log(np.array([e['time_us'] for e in reduce_axis_entries]))

    # Build feature matrix
    X, FEATURE_NAMES = build_features(reduce_axis_entries)
    print(f"Total features: {X.shape[1]}")

    # Analyze barrier distribution
    barrier_counts = Counter(e['wi_barriers'] for e in reduce_axis_entries)
    print(f"Reduce Axis barrier distribution: {dict(sorted(barrier_counts.items()))}")

    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_indices = np.arange(len(X_train))
    test_indices = np.arange(len(X_test))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define neural network
    class NeuralNet(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
            super(NeuralNet, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
            self.bn2 = nn.BatchNorm1d(hidden_dim//2)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
            self.bn3 = nn.BatchNorm1d(hidden_dim//4)
            self.dropout3 = nn.Dropout(dropout_rate*0.7)
            self.fc4 = nn.Linear(hidden_dim//4, 1)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            x = torch.relu(self.bn3(self.fc3(x)))
            x = self.dropout3(x)
            x = self.fc4(x)
            return x.squeeze()

    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = NeuralNet(input_dim, hidden_dim=128, dropout_rate=0.3)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5, min_lr=1e-6)

    # Train model
    print("\nTraining neural network on Reduce Axis only...")
    epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_test_scaled))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_test))
            scheduler.step(val_loss)

        if epoch % 30 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(best_model_state)

    # Make predictions
    model.eval()
    with torch.no_grad():
        y_pred_train = model(torch.FloatTensor(X_train_scaled)).numpy()
        y_pred_test = model(torch.FloatTensor(X_test_scaled)).numpy()
        y_pred_all = model(torch.FloatTensor(scaler.transform(X))).numpy()

    # Calculate R²
    ss_res_train = np.sum((y_train - y_pred_train) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    r2_train = 1 - ss_res_train / ss_tot_train

    ss_res_test = np.sum((y_test - y_pred_test) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    r2_test = 1 - ss_res_test / ss_tot_test

    ss_res_all = np.sum((y - y_pred_all) ** 2)
    ss_tot_all = np.sum((y - np.mean(y)) ** 2)
    r2_all = 1 - ss_res_all / ss_tot_all

    print(f"\nReduce Axis Neural Net Results:")
    print(f"Train R² = {r2_train:.4f}")
    print(f"Test R² = {r2_test:.4f}")
    print(f"Overall R² = {r2_all:.4f}")

    # Feature importance analysis
    print("\nFeature importance analysis:")
    # Get feature weights from first layer
    feature_weights = model.fc1.weight.data.numpy().sum(axis=0)
    feature_importance = np.abs(feature_weights)
    top_features_idx = np.argsort(feature_importance)[-10:][::-1]

    print("Top 10 most important features:")
    for idx in top_features_idx:
        if idx < len(FEATURE_NAMES):
            print(f"  {FEATURE_NAMES[idx]:25s} (importance: {feature_importance[idx]:.4f})")

    # Analyze by barrier groups
    print("\nPerformance by barrier group:")
    for barr_val in sorted(barrier_counts.keys()):
        barr_mask = [e['wi_barriers'] == barr_val for e in reduce_axis_entries]
        if sum(barr_mask) > 10:
            y_barr = y[barr_mask]
            y_pred_barr = y_pred_all[barr_mask]
            ss_res_barr = np.sum((y_barr - y_pred_barr) ** 2)
            ss_tot_barr = np.sum((y_barr - np.mean(y_barr)) ** 2)
            r2_barr = 1 - ss_res_barr / ss_tot_barr if ss_tot_barr > 0 else 0
            print(f"  Barriers={barr_val:2d}: R²={r2_barr:.4f} (n={sum(barr_mask)})")

    return model, reduce_axis_entries, FEATURE_NAMES, r2_all

if __name__ == '__main__':
    main()