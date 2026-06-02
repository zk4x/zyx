#!/usr/bin/env python3
"""Parse bench_output.txt -> bench_data.csv (with variant_hash)."""

import re
import csv
import sys
import os

BENCH_OUTPUT = os.path.join(os.path.dirname(__file__), 'bench_output.txt')
BENCH_CSV = os.path.join(os.path.dirname(__file__), 'bench_data.csv')


def parse_bench_output(filename):
    entries = []
    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('===') and line.endswith('==='):
            i += 1
            continue
        if line.startswith('num_groups=') or line.startswith('const=') or line.startswith('cost='):
            m = re.match(
                r'(?:cost=(\d+), )?num_groups=(\d+), wi_per_group=(\d+), wi_ops=(\d+), wi_compute_ops=(\d+), '
                r'wi_barriers=(\d+), wi_global_load_bits=(\d+), wi_global_store_bits=(\d+), '
                r'wi_local_load_bits=(\d+), wi_local_store_bits=(\d+), '
                r'wi_peak_reg_bytes=(\d+), wi_branches=(\d+), '
                r'wi_global_load_lidx_stride=(\d+), wi_global_store_lidx_stride=(\d+), '
                r'wi_local_load_lidx_stride=(\d+), wi_local_store_lidx_stride=(\d+), '
                r'warp_size=(\d+), max_local_threads=(\d+), max_register_bytes=(\d+), '
                r'wi_register_load_bits=(\d+), wi_register_store_bits=(\d+), '
                r'gws0=(\d+), gws1=(\d+), gws2=(\d+), '
                r'lws0=(\d+), lws1=(\d+), lws2=(\d+), '
                r'max_loop_depth=(\d+), preferred_vector_size=(\d+), local_mem_size=(\d+)',
                line
            )
            if m:
                entry = {
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
                    'wi_register_load_bits': int(m.group(20)),
                    'wi_register_store_bits': int(m.group(21)),
                    'gws0': int(m.group(22)),
                    'gws1': int(m.group(23)),
                    'gws2': int(m.group(24)),
                    'lws0': int(m.group(25)),
                    'lws1': int(m.group(26)),
                    'lws2': int(m.group(27)),
                    'max_loop_depth': int(m.group(28)),
                    'preferred_vector_size': int(m.group(29)),
                    'local_mem_size': int(m.group(30)),
                }
                time_match = re.search(
                    r'variant_hash=(\d+), ([\d.]+)\s*(s|ms|μs)\s*~\s*([\d.]+)\s*[MGT]FLOP/s', line)
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
                    i += 1
                    if i < len(lines):
                        time_match = re.match(
                            r'([\d.]+)\s*(s|ms|μs)\s*~\s*([\d.]+)\s*[MGT]FLOP/s', lines[i].strip())
                        if time_match:
                            time_val = float(time_match.group(1))
                            unit = time_match.group(2)
                            entry['gflops'] = float(time_match.group(3))
                            if unit == 'ms':
                                time_val *= 1000.0
                            elif unit == 's':
                                time_val *= 1_000_000.0
                            entry['time_us'] = time_val
                            entry['variant_hash'] = 0
                            entries.append(entry)
        i += 1
    return entries


def write_csv(entries):
    with open(BENCH_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'variant_hash', 'num_groups', 'wi_per_group', 'wi_ops', 'wi_compute_ops',
            'wi_barriers', 'wi_global_load_bits', 'wi_global_store_bits',
            'wi_local_load_bits', 'wi_local_store_bits', 'wi_peak_reg_bytes',
            'wi_branches', 'wi_global_load_lidx_stride', 'wi_global_store_lidx_stride',
            'wi_local_load_lidx_stride', 'wi_local_store_lidx_stride',
            'warp_size', 'max_local_threads', 'max_register_bytes',
            'wi_register_load_bits', 'wi_register_store_bits',
            'gws0', 'gws1', 'gws2', 'lws0', 'lws1', 'lws2',
            'max_loop_depth', 'preferred_vector_size', 'local_mem_size',
            'time_us', 'gflops'
        ])
        for e in entries:
            writer.writerow([
                e.get('variant_hash', ''),
                e['num_groups'], e['wi_per_group'], e['wi_ops'],
                e['wi_compute_ops'], e['wi_barriers'], e['wi_global_load_bits'],
                e['wi_global_store_bits'],
                e.get('wi_local_load_bits', 0),
                e.get('wi_local_store_bits', 0), e.get('wi_peak_reg_bytes', 0),
                e.get('wi_branches', 0),
                e.get('wi_global_load_lidx_stride', 0),
                e.get('wi_global_store_lidx_stride', 0),
                e.get('wi_local_load_lidx_stride', 0),
                e.get('wi_local_store_lidx_stride', 0),
                e.get('warp_size', 32),
                e.get('max_local_threads', 1024), e.get('max_register_bytes', 256),
                e.get('wi_register_load_bits', 0),
                e.get('wi_register_store_bits', 0),
                e.get('gws0', 1), e.get('gws1', 1), e.get('gws2', 1),
                e.get('lws0', 1), e.get('lws1', 1), e.get('lws2', 1),
                e.get('max_loop_depth', 0),
                e.get('preferred_vector_size', 0), e.get('local_mem_size', 0),
                e['time_us'], e['gflops']
            ])
    print(f"Wrote {len(entries)} entries to {BENCH_CSV}")


if __name__ == '__main__':
    entries = parse_bench_output(BENCH_OUTPUT)
    print(f"Parsed {len(entries)} entries from {BENCH_OUTPUT}")
    write_csv(entries)
