#!/usr/bin/env python3
"""Neural net with enhanced GPU-specific features for Reduce Axis section."""

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

        if line.startswith('cost=') or line.startswith('num_groups='):
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

def build_gpu_features(entries):
    """Build features focused on GPU memory access patterns and performance characteristics."""
    
    # Each "feature" is (name, callable(e) -> value)
    feature_defs = [
        # Core transforms (7)
        ('lng', lambda e: np.log(e['num_groups'])),
        ('lwpg', lambda e: np.log(e['wi_per_group'] + 1)),
        ('lops', lambda e: np.log(e['wi_ops'])),
        ('lcop', lambda e: np.log(e['wi_compute_ops'])),
        ('lgmem', lambda e: np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('barr', lambda e: e['wi_barriers']),
        ('lld_st', lambda e: np.log1p(e.get('wi_global_load_lidx_stride', 0))),
        # Memory coalescing features (8)
        ('global_coalescing_load', lambda e: 1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)),
        ('global_coalescing_store', lambda e: 1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)),
        ('global_coalescing_avg', lambda e: (1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0) + 
                                              1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)) / 2.0),
        ('global_coalescing_min', lambda e: min(1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0),
                                                1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0))),
        ('global_coalescing_product', lambda e: (1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)) *
                                                (1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0))),
        ('local_coalescing_load', lambda e: 1.0 - min(abs(int(e.get('wi_local_load_lidx_stride', 0)) % 32) / 32.0, 1.0) if e.get('wi_local_load_lidx_stride', 0) > 0 else 1.0),
        ('local_coalescing_store', lambda e: 1.0 - min(abs(int(e.get('wi_local_store_lidx_stride', 0)) % 32) / 32.0, 1.0) if e.get('wi_local_store_lidx_stride', 0) > 0 else 1.0),
        ('local_coalescing_avg', lambda e: ((1.0 - min(abs(int(e.get('wi_local_load_lidx_stride', 0)) % 32) / 32.0, 1.0)) if e.get('wi_local_load_lidx_stride', 0) > 0 else 1.0 + 
                                            (1.0 - min(abs(int(e.get('wi_local_store_lidx_stride', 0)) % 32) / 32.0, 1.0)) if e.get('wi_local_store_lidx_stride', 0) > 0 else 1.0) / 2.0),
        # Memory bank conflict features (6)
        ('load_bank_conflict', lambda e: abs(int(e.get('wi_local_load_lidx_stride', 0)) % 32) / 32.0 if e.get('wi_local_load_lidx_stride', 0) > 0 else 0.0),
        ('store_bank_conflict', lambda e: abs(int(e.get('wi_local_store_lidx_stride', 0)) % 32) / 32.0 if e.get('wi_local_store_lidx_stride', 0) > 0 else 0.0),
        ('bank_conflict_load_adj4', lambda e: abs(int(e.get('wi_local_load_lidx_stride', 0)) % 4) / 4.0 if e.get('wi_local_load_lidx_stride', 0) > 0 else 0.0),
        ('bank_conflict_store_adj4', lambda e: abs(int(e.get('wi_local_store_lidx_stride', 0)) % 4) / 4.0 if e.get('wi_local_store_lidx_stride', 0) > 0 else 0.0),
        ('bank_conflict_load_adj8', lambda e: abs(int(e.get('wi_local_load_lidx_stride', 0)) % 8) / 8.0 if e.get('wi_local_load_lidx_stride', 0) > 0 else 0.0),
        ('bank_conflict_store_adj8', lambda e: abs(int(e.get('wi_local_store_lidx_stride', 0)) % 8) / 8.0 if e.get('wi_local_store_lidx_stride', 0) > 0 else 0.0),
        # Shared memory utilization (6)
        ('shared_mem_util', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 
                                       max((e['num_groups'] * e['wi_per_group']) * 16384.0, 1.0)),
        ('shared_mem_load_ratio', lambda e: e.get('wi_local_load_bits', 0) / 
                                           max(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0), 1.0)),
        ('shared_mem_store_ratio', lambda e: e.get('wi_local_store_bits', 0) / 
                                            max(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0), 1.0)),
        ('shared_mem_per_thread', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 
                                             max(e['num_groups'] * e['wi_per_group'], 1.0)),
        ('shared_mem_efficiency', lambda e: min((e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 65536.0, 1.0)),
        ('shared_mem_pressure', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 
                                         max(e.get('max_local_threads', 1024) * 64.0, 1.0)),
        # Warp occupancy and utilization (8)
        ('warp_utilization', lambda e: e['wi_per_group'] / e.get('warp_size', 32)),
        ('warp_efficiency', lambda e: min(e['wi_per_group'] / e.get('warp_size', 32), 1.0)),
        ('warp_waste', lambda e: max(0.0, (e.get('warp_size', 32) - e['wi_per_group']) / e.get('warp_size', 32))),
        ('warp_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0)),
        ('sm_occupancy', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0)),
        ('thread_efficiency', lambda e: (e['wi_ops'] * e['wi_per_group']) / max(e['num_groups'] * e['wi_per_group'], 1.0)),
        ('active_warps', lambda e: e['num_groups'] * max(1, e['wi_per_group'] // e.get('warp_size', 32))),
        ('warp_divergence', lambda e: np.log1p(e.get('wi_branches', 0)) / max(np.log(e['wi_ops'] + 1), 1.0)),
        # Memory bandwidth efficiency (6)
        ('global_bandwidth_util', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / 
                                           max(e['num_groups'] * e['wi_per_group'] * 256.0, 1.0)),
        ('memory_intensity', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1.0)),
        ('bw_ratio', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / 
                              max((e['num_groups'] * e['wi_per_group']) * 256.0, 1.0)),
        ('local_bw_ratio', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / 
                                    max((e['num_groups'] * e['wi_per_group']) * 16384.0, 1.0)),
        ('memory_pressure', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / 
                                    max(e['num_groups'] * e['wi_per_group'] * 32.0, 1.0)),
        ('access_complexity', lambda e: (np.log1p(e.get('wi_global_load_lidx_stride', 0)) + 
                                         np.log1p(e.get('wi_global_store_lidx_stride', 0))) / max(np.log1p(e['wi_ops']), 1)),
        # Register pressure and occupancy (6)
        ('register_pressure', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1.0)),
        ('register_efficiency', lambda e: min(e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1.0), 1.0)),
        ('register_per_thread', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e['num_groups'] * e['wi_per_group'], 1.0)),
        ('register_utilization', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e['wi_per_group'] * 32.0, 1.0)),
        ('register_waste', lambda e: max(0.0, (e.get('max_register_bytes', 256) - e.get('wi_peak_reg_bytes', 0)) / e.get('max_register_bytes', 256))),
        ('register_pressure_score', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1.0) * 
                                            e.get('wi_per_group', 1) / 32.0),
        # Barrier and synchronization features (12)
        ('barrier_per_thread', lambda e: e['wi_barriers'] / max(e['num_groups'] * e['wi_per_group'], 1.0)),
        ('barrier_density', lambda e: e['wi_barriers'] / max(np.log(e['wi_ops'] + 1), 1.0)),
        ('barrier_overhead', lambda e: e['wi_barriers'] * np.log1p(e['wi_barriers']) / max(e['wi_ops'] / max(e['wi_barriers'], 1), 1)),
        ('sync_efficiency', lambda e: e['wi_compute_ops'] / max(e['wi_barriers'] * e['wi_per_group'] + 1, 1.0)),
        ('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('reduce_tree_depth', lambda e: np.log2(e['wi_barriers'] + 1) if e['wi_barriers'] > 0 else 0),
        ('barrier_lwpg', lambda e: e['wi_barriers'] * np.log1p(e['wi_per_group'])),
        ('barrier_lng', lambda e: e['wi_barriers'] * np.log(e['num_groups'])),
        ('barrier_mixed', lambda e: e['wi_barriers'] * np.log(e['wi_ops'] + 1)),
        ('barrier_local', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('barrier_global', lambda e: e['wi_barriers'] * np.log(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1)),
        ('barrier_ratio', lambda e: e['wi_barriers'] / max(e['wi_ops'] / max(e['wi_per_group'], 1), 1.0)),
        # Memory access pattern interactions (10)
        ('coalescing_barr', lambda e: (1.0 - min(e.get('wi_global_load_lidx_stride', 0) / 32.0, 1.0)) * e['wi_barriers']),
        ('coalescing_store_barr', lambda e: (1.0 - min(e.get('wi_global_store_lidx_stride', 0) / 32.0, 1.0)) * e['wi_barriers']),
        ('bank_conflict_barr_load', lambda e: (abs(int(e.get('wi_local_load_lidx_stride', 0)) % 32) / 32.0) * e['wi_barriers'] if e.get('wi_local_load_lidx_stride', 0) > 0 else 0.0),
        ('bank_conflict_barr_store', lambda e: (abs(int(e.get('wi_local_store_lidx_stride', 0)) % 32) / 32.0) * e['wi_barriers'] if e.get('wi_local_store_lidx_stride', 0) > 0 else 0.0),
        ('shared_mem_barr', lambda e: (e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0)) / max(e['num_groups'] * e['wi_per_group'] * 16384.0, 1.0) * e['wi_barriers']),
        ('bw_util_barr', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'] * 256.0, 1.0) * e['wi_barriers']),
        ('reg_pressure_barr', lambda e: (e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1.0)) * e['wi_barriers']),
        ('warp_util_barr', lambda e: (e['wi_per_group'] / e.get('warp_size', 32)) * e['wi_barriers']),
        ('occupancy_barr', lambda e: min((e['num_groups'] * e['wi_per_group']) / 2048.0, 1.0) * e['wi_barriers']),
        ('mem_intensity_barr', lambda e: (e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1.0)) * e['wi_barriers']),
        # Specialized reduction features (6)
        ('reduce_complexity', lambda e: e['wi_barriers'] * np.log1p(e['wi_ops'] / max(e['num_groups'] * e['wi_per_group'], 1.0))),
        ('tree_reduce_cost', lambda e: e['wi_barriers'] * np.log1p(e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0))),
        ('element_ops_per_thread', lambda e: e['wi_compute_ops'] / max(e['num_groups'] * e['wi_per_group'], 1.0)),
        ('layer_norm_complexity', lambda e: (e['wi_compute_ops'] * e['wi_barriers']) / max(e['num_groups'], 1.0)),
        ('memory_access_pattern', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['wi_ops'], 1.0)),
        ('thread_efficiency', lambda e: (e['wi_ops'] * e['wi_per_group']) / max(e['num_groups'] * e['wi_per_group'], 1.0)),
        # Secondary features (keep the most important ones)
        ('wr', lambda e: e['wi_per_group'] / e.get('warp_size', 32)),
        ('rr', lambda e: e.get('wi_peak_reg_bytes', 0) / max(e.get('max_register_bytes', 256), 1)),
        ('lst_st', lambda e: np.log1p(e.get('wi_global_store_lidx_stride', 0))),
        ('total_threads', lambda e: e['num_groups'] * e['wi_per_group']),
        ('store_per_thread', lambda e: e['wi_global_store_bits'] / max(e['wi_per_group'], 1)),
        ('ci', lambda e: e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1)),
        ('overhead', lambda e: e['wi_ops'] / max(e['wi_compute_ops'], 1)),
        ('mem_per_thread', lambda e: (e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1)),
        ('log1p_ci', lambda e: np.log1p(e['wi_compute_ops'] / max(e['wi_global_load_bits'] + e['wi_global_store_bits'], 1))),
        ('log1p_overhead', lambda e: np.log1p(e['wi_ops'] / max(e['wi_compute_ops'], 1))),
        ('log1p_mp', lambda e: np.log1p((e['wi_global_load_bits'] + e['wi_global_store_bits']) / max(e['num_groups'] * e['wi_per_group'], 1))),
        ('log1p_lm', lambda e: np.log1p((e.get('wi_local_load_bits', 0) + e.get('wi_local_store_bits', 0) + 1) / max(e['wi_global_load_bits'] + e['wi_global_store_bits'] + 1, 1))),
        # Barrier one-hot (7 values)
        ('b0', lambda e: 1.0 if e['wi_barriers'] == 0 else 0.0),
        ('b3', lambda e: 1.0 if e['wi_barriers'] == 3 else 0.0),
        ('b4', lambda e: 1.0 if e['wi_barriers'] == 4 else 0.0),
        ('b5', lambda e: 1.0 if e['wi_barriers'] == 5 else 0.0),
        ('b6', lambda e: 1.0 if e['wi_barriers'] == 6 else 0.0),
        ('b7', lambda e: 1.0 if e['wi_barriers'] == 7 else 0.0),
        ('b8', lambda e: 1.0 if e['wi_barriers'] == 8 else 0.0),
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

    # Build enhanced feature matrix
    X, FEATURE_NAMES = build_gpu_features(reduce_axis_entries)
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
    
    # L1 regularization parameters for ~90% sparsity
    l1_lambda = 0.0  # Start with no L1, will prune after training

    # Train model
    print("\nTraining sparse neural network on Reduce Axis only...")
    epochs = 300
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25
    
    # Track sparsity
    sparsity_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Add L1 regularization for sparsity
            l1_loss = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_loss += torch.norm(param, 1)
            loss += l1_lambda * l1_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Track sparsity
        total_weights = 0
        zero_weights = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                zero_weights += (param == 0).sum().item()
        sparsity = zero_weights / total_weights * 100
        sparsity_history.append(sparsity)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_test_scaled))
            val_loss = criterion(val_outputs, torch.FloatTensor(y_test))
            scheduler.step(val_loss)

        if epoch % 30 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.6f}, Val Loss = {val_loss:.6f}, Sparsity = {sparsity:.1f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Final sparsity: {sparsity_history[-1]:.1f}%")
    
    # Apply iterative pruning to achieve ~90% sparsity
    print("\nApplying iterative pruning to achieve 90% sparsity...")
    target_sparsity = 0.9
    current_sparsity = sparsity_history[-1] / 100.0
    
    if current_sparsity < target_sparsity:
        # Calculate pruning threshold
        all_weights = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                all_weights.extend(param.data.abs().cpu().numpy().flatten())
        
        threshold = np.sort(np.array(all_weights))[int(len(all_weights) * (1 - target_sparsity))]
        print(f"Pruning threshold: {threshold:.6f}")
        
        # Apply pruning
        for name, param in model.named_parameters():
            if 'weight' in name:
                mask = param.data.abs() > threshold
                param.data *= mask.float()
        
        # Verify final sparsity
        total_weights = 0
        zero_weights = 0
        for name, param in model.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                zero_weights += (param == 0).sum().item()
        final_sparsity = zero_weights / total_weights * 100
        print(f"Final sparsity after pruning: {final_sparsity:.1f}%")
        
        # Retrain slightly to recover from pruning
        print("Retraining after pruning...")
        for epoch in range(20):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            y_pred_train = model(torch.FloatTensor(X_train_scaled)).numpy()
            y_pred_test = model(torch.FloatTensor(X_test_scaled)).numpy()
            y_pred_all = model(torch.FloatTensor(scaler.transform(X))).numpy()
        
        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
        r2_train = 1 - ss_res_train / ss_tot_train
        
        ss_res_test = np.sum((y_test - y_pred_test) ** 2)
        ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_test = 1 - ss_res_test / ss_tot_test
        
        ss_res_all = np.sum((y - y_pred_all) ** 2)
        ss_tot_all = np.sum((y - np.mean(y)) ** 2)
        r2_all = 1 - ss_res_all / ss_tot_all
        
        print(f"\nResults after pruning and retraining:")
        print(f"Train R² = {r2_train:.4f}")
        print(f"Test R² = {r2_test:.4f}")
        print(f"Overall R² = {r2_all:.4f}")

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

    print(f"\nSparse Reduce Axis Neural Net Results:")
    print(f"Train R² = {r2_train:.4f}")
    print(f"Test R² = {r2_test:.4f}")
    print(f"Overall R² = {r2_all:.4f}")
    
    # Calculate final sparsity
    total_weights = 0
    zero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            zero_weights += (param == 0).sum().item()
    final_sparsity = zero_weights / total_weights * 100
    print(f"Final weight sparsity: {final_sparsity:.1f}%")

    # Feature importance analysis
    print("\nTop 15 most important features:")
    feature_weights = model.fc1.weight.data.numpy().sum(axis=0)
    feature_importance = np.abs(feature_weights)
    top_features_idx = np.argsort(feature_importance)[-15:][::-1]

    for idx in top_features_idx:
        if idx < len(FEATURE_NAMES):
            print(f"  {FEATURE_NAMES[idx]:30s} (importance: {feature_importance[idx]:.4f})")

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