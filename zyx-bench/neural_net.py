# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Tiny MLP cost model, trained with zyx itself.

Everything here is zyx: feature engineering, normalisation, the ranking target,
the model, the optimizer and the metrics. The only non-zyx dependency is
`sqlite3`/`json`, used to read `zyx-bench/bench.db` (written by `zyx-bench`).

The model is deliberately small — `N_FEATURES x HIDDEN + HIDDEN` weights — so it
can be emitted as a few hundred constants in `zyx/src/kernel/predict_cost.rs`.
After training it writes that file as a self-contained `predict_time_us`
(TinyMLP forward pass over the same feature formulas the model was trained on).

Run: `python neural_net.py` (after `cargo run --release` in `zyx-bench`).
"""

import json
import os
import random
import sqlite3
import struct
import sys

import zyx
from zyx import DType, Tensor

BENCH_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.db")

HIDDEN = 5
TRAIN_STEPS = 30000
LR = 3e-3
L1_LAMBDA = 5e-3
TEST_ROWS = 20000
SEED = 42


# ---------------------------------------------------------------- tensor utils

def log1p(x: Tensor) -> Tensor:
    return (x + 1.0).log()


def log2(x: Tensor) -> Tensor:
    return x.log(2.0)


def tmax(a, b) -> Tensor:
    # `hasattr`, not `isinstance(Tensor)`: the same feature builder also runs on
    # `Expr` (the Rust-codegen stand-in) to emit `predict_cost.rs`.
    return a.maximum(b) if hasattr(a, "maximum") else b.maximum(a)


def tmin(a, b) -> Tensor:
    return a.minimum(b) if hasattr(a, "minimum") else a.minimum(b)


def ind(cond: Tensor) -> Tensor:
    """Bool tensor -> 1.0 / 0.0."""
    return cond.where(1.0, 0.0)


# ------------------------------------------------------------------ the data

# Defaults match the extended attributes a measurement may not carry.
DEFAULTS = {
    "warp_size": 32.0,
    "max_register_bytes": 256.0,
    "max_local_threads": 1024.0,
    "gws0": 1.0,
    "gws1": 1.0,
    "gws2": 1.0,
    "lws0": 1.0,
    "lws1": 1.0,
    "lws2": 1.0,
    "max_loop_depth": 0.0,
    "preferred_vector_size": 0.0,
    "local_mem_size": 1.0,
}

# Raw hardware/kernel fields the emitted `predict_time_us` receives and the
# feature builder reads. The emitted signature takes two more
# (`wi_local_{load,store}_lidx_stride`) that no feature uses.
HW_NAMES = [
    "num_groups", "wi_per_group", "wi_ops", "wi_compute_ops", "wi_barriers",
    "wi_global_load_bits", "wi_global_store_bits", "wi_local_load_bits",
    "wi_local_store_bits", "wi_peak_reg_bytes", "wi_branches",
    "wi_global_load_lidx_stride", "wi_global_store_lidx_stride",
    "wi_register_load_bits", "wi_register_store_bits",
    "warp_size", "max_local_threads", "max_register_bytes",
    "gws0", "gws1", "gws2", "lws0", "lws1", "lws2",
    "max_loop_depth", "preferred_vector_size", "local_mem_size",
]


# ------------------------------------------------------- Rust expression DSL
#
# The same `build_features` runs twice: on zyx tensors (training) and on these
# `Expr` leaves (emission). That keeps the emitted Rust feature formulas from
# drifting away from the ones the model was trained on.

def _rust_float(v) -> str:
    if isinstance(v, Expr):
        return v.code
    return repr(float(v))


class Expr:
    """Symbolic `f64` expression that renders as a Rust expression."""

    __slots__ = ("code",)

    def __init__(self, code: str):
        self.code = code

    def __str__(self) -> str:
        return self.code

    def __add__(self, o):
        return Expr(f"({self.code} + {_rust_float(o)})")

    def __radd__(self, o):
        return Expr(f"({_rust_float(o)} + {self.code})")

    def __sub__(self, o):
        return Expr(f"({self.code} - {_rust_float(o)})")

    def __rsub__(self, o):
        return Expr(f"({_rust_float(o)} - {self.code})")

    def __mul__(self, o):
        return Expr(f"({self.code} * {_rust_float(o)})")

    def __rmul__(self, o):
        return Expr(f"({_rust_float(o)} * {self.code})")

    def __truediv__(self, o):
        return Expr(f"({self.code} / {_rust_float(o)})")

    def __rtruediv__(self, o):
        return Expr(f"({_rust_float(o)} / {self.code})")

    def __neg__(self):
        return Expr(f"(-{self.code})")

    # The `Tensor` surface `build_features` uses.
    def log(self, base=None):
        if base is None:
            return Expr(f"({self.code}).ln()")
        if isinstance(base, (int, float)) and float(base) == 2.0:
            return Expr(f"({self.code}).log2()")
        return Expr(f"({self.code}).ln() / ({_rust_float(base)}).ln()")

    def maximum(self, o):
        return Expr(f"({self.code}).max({_rust_float(o)})")

    def minimum(self, o):
        return Expr(f"({self.code}).min({_rust_float(o)})")

    def equal(self, o):
        return Expr(f"({self.code} == {_rust_float(o)})")

    def cmpge(self, o):
        return Expr(f"({self.code} >= {_rust_float(o)})")

    def cmpgt(self, o):
        return Expr(f"({self.code} > {_rust_float(o)})")

    def logical_or(self, o):
        return Expr(f"({self.code} || {_rust_float(o)})")

    def logical_and(self, o):
        return Expr(f"({self.code} && {_rust_float(o)})")

    def reciprocal(self):
        return Expr(f"(1.0 / {self.code})")

    def abs(self):
        return Expr(f"({self.code}).abs()")

    def where(self, if_true, if_false):
        return Expr(f"(if {self.code} {{ {_rust_float(if_true)} }} else {{ {_rust_float(if_false)} }})")


def load_rows():
    if not os.path.exists(BENCH_DB):
        sys.exit(f"missing {BENCH_DB}: run `cargo run --release` in zyx-bench first")
    con = sqlite3.connect(f"file:{BENCH_DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT seed, nanos, features FROM measurements"
        " WHERE status = 'measured' AND nanos IS NOT NULL"
    ).fetchall()
    con.close()
    if not rows:
        sys.exit(f"{BENCH_DB} has no measured rows")
    return [(seed, nanos / 1000.0, json.loads(features)) for seed, nanos, features in rows]


def column(rows, name: str) -> Tensor:
    default = DEFAULTS.get(name, 0.0)
    return Tensor.from_vec([float(f.get(name, default)) for _, _, f in rows], (len(rows),))


# --------------------------------------------------------------- the features
#
# Same set as the previous numpy implementation, expressed as tensor ops. All
# are vectorised over the measurement axis: one tensor per feature.

def build_features(c) -> list:
    f = []
    f.append(("lng", c["num_groups"].log()))
    f.append(("lwpg", log1p(c["wi_per_group"])))
    f.append(("lops", c["wi_ops"].log()))
    f.append(("lcop", c["wi_compute_ops"].log()))
    f.append(("lgmem", log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("barr", c["wi_barriers"]))
    f.append(("wr", c["wi_per_group"] / tmax(c["warp_size"], 1.0)))
    f.append(("rr", c["wi_peak_reg_bytes"] / tmax(c["max_register_bytes"], 1.0)))
    f.append(("total_threads", c["num_groups"] * c["wi_per_group"]))
    f.append(("overhead", c["wi_ops"] / tmax(c["wi_compute_ops"], 1.0)))
    f.append(("ci", c["wi_compute_ops"] / tmax(c["wi_global_load_bits"] + c["wi_global_store_bits"], 1.0)))
    f.append(("log1p_ci", log1p(c["wi_compute_ops"] / tmax(c["wi_global_load_bits"] + c["wi_global_store_bits"], 1.0))))
    f.append(("log1p_overhead", log1p(c["wi_ops"] / tmax(c["wi_compute_ops"], 1.0))))
    f.append(("log1p_mp", log1p((c["wi_global_load_bits"] + c["wi_global_store_bits"]) / tmax(c["num_groups"] * c["wi_per_group"], 1.0))))
    f.append(("log1p_lm", log1p((c["wi_local_load_bits"] + c["wi_local_store_bits"] + 1.0) / tmax(c["wi_global_load_bits"] + c["wi_global_store_bits"] + 1.0, 1.0))))
    f.append(("log_ops_100", (c["wi_ops"] + 100.0).log()))
    f.append(("log_ops_10", (c["wi_ops"] + 10.0).log()))
    f.append(("log_cops_100", (c["wi_compute_ops"] + 100.0).log()))
    f.append(("log1p_1000_div_ops", log1p(tmax(c["wi_ops"], 1.0).reciprocal() * 1000.0)))
    f.append(("log1p_100_div_ops", log1p(tmax(c["wi_ops"], 1.0).reciprocal() * 100.0)))
    f.append(("log1p_1000_div_cops", log1p(tmax(c["wi_compute_ops"], 1.0).reciprocal() * 1000.0)))
    f.append(("ops_per_thread", c["wi_ops"] / tmax(c["num_groups"] * c["wi_per_group"], 1.0)))
    f.append(("cops_per_thread", c["wi_compute_ops"] / tmax(c["num_groups"] * c["wi_per_group"], 1.0)))
    f.append(("ops_per_group", c["wi_ops"] / tmax(c["num_groups"], 1.0)))
    f.append(("log_lwpg", tmax(c["wi_per_group"].log(), 1e-8).log()))
    f.append(("raw_ng", c["num_groups"]))
    f.append(("raw_ng_wpg", c["num_groups"] * c["wi_per_group"]))
    f.append(("lbranch", log1p(c["wi_branches"])))
    f.append(("barr*lbranch", c["wi_barriers"] * log1p(c["wi_branches"])))
    f.append(("lwpg*lbranch", log1p(c["wi_per_group"]) * log1p(c["wi_branches"])))
    f.append(("lng_lwpg", c["num_groups"].log() * log1p(c["wi_per_group"])))
    f.append(("lwpg_lops", log1p(c["wi_per_group"]) * c["wi_ops"].log()))
    f.append(("log_ops_per_group", log1p(c["wi_ops"] / tmax(c["num_groups"], 1.0))))
    f.append(("log_opt", log1p(c["wi_ops"] / tmax(c["num_groups"] * c["wi_per_group"], 1.0))))
    f.append(("inv_threads", tmax(c["num_groups"] * c["wi_per_group"], 1.0).reciprocal()))
    f.append(("log_inv_threads", log1p(tmax(c["num_groups"] * c["wi_per_group"], 1.0).reciprocal())))
    f.append(("log_opt_barr", log1p(c["wi_ops"] / tmax(c["num_groups"] * c["wi_per_group"] * tmax(c["wi_barriers"], 1.0), 1.0))))
    f.append(("log_wpg_barr", log1p(c["wi_per_group"] * c["wi_barriers"])))

    for bval in [0, 3, 4, 5, 6, 7, 8]:
        hit = ind(c["wi_barriers"].equal(float(bval)))
        f.append((f"b{bval}", hit))
    f.append(("barr_low", ind(c["wi_barriers"].equal(0.0))))
    f.append(("barr_med", ind(c["wi_barriers"].equal(3.0).logical_or(c["wi_barriers"].equal(4.0)).logical_or(c["wi_barriers"].equal(5.0)))))
    f.append(("barr_high", ind(c["wi_barriers"].cmpge(6.0))))
    f.append(("barr_nonzero", ind(c["wi_barriers"].cmpgt(0.0))))

    for bval in [0, 3, 4, 5, 6, 7, 8]:
        hit = ind(c["wi_barriers"].equal(float(bval)))
        f.append((f"b{bval}*lng", hit * c["num_groups"].log()))
        f.append((f"b{bval}*lwpg", hit * log1p(c["wi_per_group"])))
        f.append((f"b{bval}*lops", hit * c["wi_ops"].log()))
    f.append(("barr*lops", c["wi_barriers"] * c["wi_ops"].log()))
    f.append(("barr*lcop", c["wi_barriers"] * c["wi_compute_ops"].log()))
    f.append(("barr*lgmem", c["wi_barriers"] * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("barr*lwpg", c["wi_barriers"] * log1p(c["wi_per_group"])))
    f.append(("barr*lng", c["wi_barriers"] * c["num_groups"].log()))
    f.append(("barr*wr", c["wi_barriers"] * (c["wi_per_group"] / tmax(c["warp_size"], 1.0))))
    f.append(("lld_st", log1p(c["wi_global_load_lidx_stride"])))
    f.append(("lst_st", log1p(c["wi_global_store_lidx_stride"])))
    f.append(("lld_st*lops", log1p(c["wi_global_load_lidx_stride"]) * c["wi_ops"].log()))
    f.append(("lld_st*lcop", log1p(c["wi_global_load_lidx_stride"]) * c["wi_compute_ops"].log()))
    f.append(("lld_st*lgmem", log1p(c["wi_global_load_lidx_stride"]) * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lops*lgmem", c["wi_ops"].log() * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lcop*lgmem", c["wi_compute_ops"].log() * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lng*lgmem", c["num_groups"].log() * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lwpg*lgmem", log1p(c["wi_per_group"]) * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lwpg*lops_w", log1p(c["wi_per_group"]) * c["wi_ops"].log()))
    f.append(("lng*lcop", c["num_groups"].log() * c["wi_compute_ops"].log()))
    f.append(("lng*lgmem_w", c["num_groups"].log() * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("warp_util", c["wi_per_group"] / tmax(c["warp_size"], 1.0)))
    f.append(("log_warp_waste", (tmax(c["wi_per_group"], 1.0).reciprocal() * 32.0).log()))
    f.append(("lng_warp_waste", c["num_groups"].log() * (tmax(c["wi_per_group"], 1.0).reciprocal() * 32.0).log()))
    f.append(("lwpg_warp_waste", log1p(c["wi_per_group"]) * (tmax(c["wi_per_group"], 1.0).reciprocal() * 32.0).log()))
    f.append(("coalescing_eff", tmin(c["wi_global_load_lidx_stride"] / 32.0, 1.0) * -1.0 + 1.0))
    f.append(("coalescing_eff_st", tmin(c["wi_global_store_lidx_stride"] / 32.0, 1.0) * -1.0 + 1.0))
    f.append(("reduce_kind", c["wi_barriers"] / tmax(log2(c["wi_per_group"] + 1.0), 1.0)))
    f.append(("element_ops", c["wi_compute_ops"] / tmax(c["num_groups"], 1.0)))
    f.append(("warp_div", tmax(tmax(c["wi_per_group"], 1.0).reciprocal() * 32.0, 1e-8).log()))
    f.append(("sm_occupancy", tmin((c["num_groups"] * c["wi_per_group"]) / 2048.0, 1.0)))
    f.append(("tree_height", ind(c["wi_barriers"].cmpgt(0.0)) * log2(c["wi_barriers"] + 1.0)))
    f.append(("tree_reduce_cost", c["wi_barriers"] * log1p(c["wi_local_load_bits"] + c["wi_local_store_bits"])))
    f.append(("element_ops_per_thread", c["wi_compute_ops"] / tmax(c["num_groups"] * c["wi_per_group"], 1.0)))
    f.append(("shared_mem_pressure", (c["wi_local_load_bits"] + c["wi_local_store_bits"]) / 65536.0))
    f.append(("warp_waste", (c["wi_per_group"] * -1.0 + 32.0) / 32.0))
    f.append(("compute_per_barrier", c["wi_compute_ops"] / tmax(c["num_groups"] * tmax(c["wi_barriers"], 1.0), 1.0)))
    f.append(("barrier_overhead_cost", c["wi_barriers"] * log1p(c["wi_barriers"]) / tmax(log1p(c["wi_ops"]), 1.0)))
    f.append(("group_work_density", c["wi_ops"] / tmax(c["num_groups"] * log1p(c["wi_per_group"]), 1.0)))
    f.append(("ops_per_barrier", c["wi_ops"] / tmax(c["wi_barriers"], 1.0)))
    f.append(("barrier_factor", c["wi_barriers"] / tmax(log1p(c["wi_ops"]), 1.0)))
    f.append(("barrier_efficiency", c["wi_barriers"] / tmax(log1p(c["wi_ops"]), 1.0)))
    f.append(("mem_compute_ratio_log", ((c["wi_global_load_bits"] + c["wi_global_store_bits"]) / tmax(c["wi_compute_ops"], 1.0) + 1.0).log()))
    f.append(("log1p_lng", log1p(c["num_groups"].log().abs())))
    f.append(("log1p_lwpg_raw", log1p(log1p(c["wi_per_group"]).abs())))
    f.append(("log1p_lops_raw", log1p(c["wi_ops"].log().abs())))
    f.append(("log_ng_per_ops", (c["num_groups"] / tmax(c["wi_ops"], 1.0) + 1.0).log()))
    f.append(("log_threads_per_ops", ((c["num_groups"] * c["wi_per_group"]) / tmax(c["wi_ops"], 1.0) + 1.0).log()))
    f.append(("barr_med_lops", ind(c["wi_barriers"].equal(3.0).logical_or(c["wi_barriers"].equal(4.0)).logical_or(c["wi_barriers"].equal(5.0))) * c["wi_ops"].log()))
    f.append(("barr_nonzero_lops", ind(c["wi_barriers"].cmpgt(0.0)) * c["wi_ops"].log()))
    f.append(("barr_nonzero_lmem", ind(c["wi_barriers"].cmpgt(0.0)) * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("compute_density", c["wi_compute_ops"] / tmax(c["num_groups"] * c["wi_per_group"], 1.0)))
    f.append(("thread_compute", log1p(c["wi_compute_ops"]) * log1p(c["num_groups"] * c["wi_per_group"])))
    f.append(("bgt_lng", ind(c["wi_barriers"].cmpgt(0.0)) * c["num_groups"].log()))
    f.append(("bgt_lwpg", ind(c["wi_barriers"].cmpgt(0.0)) * log1p(c["wi_per_group"])))
    f.append(("bgt_lops", ind(c["wi_barriers"].cmpgt(0.0)) * c["wi_ops"].log()))
    f.append(("bgt_lcop", ind(c["wi_barriers"].cmpgt(0.0)) * c["wi_compute_ops"].log()))
    f.append(("bgt_lgmem", ind(c["wi_barriers"].cmpgt(0.0)) * log1p(c["wi_global_load_bits"] + c["wi_global_store_bits"])))
    f.append(("lreg", log1p(c["wi_register_load_bits"])))
    f.append(("lreg_st", log1p(c["wi_register_store_bits"])))
    f.append(("reg_pressure", c["wi_register_load_bits"] / tmax(c["wi_global_load_bits"] + c["wi_global_store_bits"], 1.0)))
    f.append(("lgws0", log1p(c["gws0"])))
    f.append(("lgws1", log1p(c["gws1"])))
    f.append(("lgws2", log1p(c["gws2"])))
    f.append(("llws0", log1p(c["lws0"])))
    f.append(("llws1", log1p(c["lws1"])))
    f.append(("llws2", log1p(c["lws2"])))
    f.append(("lws0_ratio", c["lws0"] / tmax(c["wi_per_group"], 1.0)))
    f.append(("lws1_ratio", c["lws1"] / tmax(c["wi_per_group"], 1.0)))
    f.append(("lws2_ratio", c["lws2"] / tmax(c["wi_per_group"], 1.0)))
    f.append(("lws0xlws1", c["lws0"] * c["lws1"]))
    f.append(("gws0_ratio", c["gws0"] / tmax(c["num_groups"], 1.0)))
    f.append(("gws1_ratio", c["gws1"] / tmax(c["num_groups"], 1.0)))
    f.append(("loop_depth", c["max_loop_depth"]))
    f.append(("loop_depth*lops", c["max_loop_depth"] * tmax(c["wi_ops"], 1.0).log()))
    f.append(("loop_depth*barr", c["max_loop_depth"] * c["wi_barriers"]))
    f.append(("log_local_mem", log1p(c["local_mem_size"])))
    f.append(("pref_vec", c["preferred_vector_size"]))
    f.append(("total_global_load", c["wi_global_load_bits"] * c["num_groups"] * c["wi_per_group"]))
    f.append(("total_global_store", c["wi_global_store_bits"] * c["num_groups"] * c["wi_per_group"]))
    f.append(("total_global_mem", (c["wi_global_load_bits"] + c["wi_global_store_bits"]) * c["num_groups"] * c["wi_per_group"]))
    f.append(("log_total_global_load", tmax(c["wi_global_load_bits"] * c["num_groups"] * c["wi_per_group"], 1.0).log()))
    f.append(("log_total_global_mem", tmax((c["wi_global_load_bits"] + c["wi_global_store_bits"]) * c["num_groups"] * c["wi_per_group"], 1.0).log()))
    f.append(("tile_ratio", (c["wi_local_load_bits"] + c["wi_local_store_bits"]) / tmax(c["wi_global_load_bits"] + c["wi_global_store_bits"], 1.0)))
    f.append(("mem_per_compute", (c["wi_global_load_bits"] + c["wi_global_store_bits"]) * c["num_groups"] * c["wi_per_group"] / tmax(c["wi_compute_ops"] * c["num_groups"] * c["wi_per_group"], 1.0)))
    f.append(("log_mem_per_compute", tmax((c["wi_global_load_bits"] + c["wi_global_store_bits"]) / tmax(c["wi_compute_ops"], 1.0), 1e-8).log()))
    f.append(("occ_regs", c["max_register_bytes"] / tmax(c["wi_peak_reg_bytes"], 1.0)))
    f.append(("occ_threads", c["max_local_threads"] / tmax(c["wi_per_group"], 1.0)))
    f.append(("lmem_global_ratio", c["wi_local_load_bits"] / tmax(c["wi_global_load_bits"], 1.0)))
    f.append(("total_local_traffic", (c["wi_local_load_bits"] + c["wi_local_store_bits"]) * c["num_groups"] * c["wi_per_group"]))
    f.append(("log_total_local", tmax((c["wi_local_load_bits"] + c["wi_local_store_bits"]) * c["num_groups"] * c["wi_per_group"], 1.0).log()))
    f.append(("barrier_intensity", c["wi_barriers"] / tmax(c["wi_compute_ops"], 1.0)))
    f.append(("occ_combined", tmin(c["max_register_bytes"] / tmax(c["wi_peak_reg_bytes"], 1.0), c["max_local_threads"] / tmax(c["wi_per_group"], 1.0))))
    f.append(("log_occ_combined", tmax(tmin(c["max_register_bytes"] / tmax(c["wi_peak_reg_bytes"], 1.0), c["max_local_threads"] / tmax(c["wi_per_group"], 1.0)), 1e-8).log()))
    f.append(("mem_per_warp", (c["wi_global_load_bits"] + c["wi_global_store_bits"]) * 32.0 / tmax(c["wi_per_group"], 1.0)))
    f.append(("compute_per_warp", c["wi_compute_ops"] * 32.0 / tmax(c["wi_per_group"], 1.0)))
    f.append(("log_global_per_local", tmax((c["wi_global_load_bits"] + c["wi_global_store_bits"]) / tmax(c["wi_local_load_bits"] + c["wi_local_store_bits"], 1.0), 1e-8).log()))
    return f


# --------------------------------------------------------------------- ranks

def ranks(t: Tensor) -> Tensor:
    """0-based rank of each element (0 = fastest), by pairwise comparison.

    Ties share the lower rank; there is no argsort in the bindings, and groups
    here are small enough that the O(n^2) form is cheap.
    """
    n = t.shape[0]
    return t.unsqueeze(0).cmplt(t.unsqueeze(1)).cast(DType.F32).sum(1).reshape((n,))


# --------------------------------------------------------------------- model

class TinyMLP:
    """Linear -> LayerNorm -> SELU -> Linear, small enough to emit as Rust."""

    def __init__(self, n_in: int):
        self.n_in = n_in
        self.l1 = zyx.nn.Linear(n_in, HIDDEN, True, DType.F32)
        self.ln = zyx.nn.LayerNorm((HIDDEN,), dtype=DType.F32)
        self.l2 = zyx.nn.Linear(HIDDEN, 1, True, DType.F32)
        self.params = [self.l1.weight, self.l1.bias, self.ln.weight, self.ln.bias, self.l2.weight, self.l2.bias]

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.ln(self.l1(x)).selu()).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.params)


# ------------------------------------------------------------------ emission

# Constants copied from `Tensor::selu` (`zyx/src/tensor/elementwise.rs`) and
# `LayerNorm::new` (`zyx-nn/src/layer_norm.rs`).
SELU_ALPHA = 1.6732632
SELU_SCALE = 1.050701
LAYER_NORM_EPS = 1e-5

PREDICT_COST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "zyx", "src", "kernel", "predict_cost.rs"
)

# Signature of `cost.rs`'s `predict_time_us` call, in order.
SIGNATURE = [
    "num_groups", "wi_per_group", "wi_ops", "wi_compute_ops", "wi_barriers",
    "wi_global_load_bits", "wi_global_store_bits", "wi_local_load_bits",
    "wi_local_store_bits", "wi_peak_reg_bytes", "wi_branches",
    "wi_global_load_lidx_stride", "wi_global_store_lidx_stride",
    "wi_local_load_lidx_stride", "wi_local_store_lidx_stride",
    "warp_size", "max_local_threads", "max_register_bytes",
    "wi_register_load_bits", "wi_register_store_bits",
    "gws0", "gws1", "gws2", "lws0", "lws1", "lws2",
    "max_loop_depth", "preferred_vector_size", "local_mem_size",
]


def feature_rust() -> list:
    """Render the 164 feature formulas as Rust by running `build_features`
    on `Expr` leaves instead of zyx tensors."""
    cols = {name: Expr(name) for name in HW_NAMES}
    return [str(expr) for _, expr in build_features(cols)]


def _floats(t) -> tuple:
    """F32 tensor -> Python floats, without numpy (`numpy()` needs it)."""
    data = t.cast(DType.F32).to_le_bytes()
    return struct.unpack(f"<{len(data) // 4}f", data)


def emit_predict_cost(model: TinyMLP, path: str = PREDICT_COST_PATH):
    """Write the trained TinyMLP as a self-contained `predict_time_us`."""
    features = feature_rust()
    if len(features) != model.n_in:
        sys.exit(f"emission: {len(features)} Rust features for a {model.n_in}-input model")
    if len(SIGNATURE) != 29:
        sys.exit(f"emission: signature has {len(SIGNATURE)} args, cost.rs passes 29")

    w1 = _floats(model.l1.weight)
    b1 = _floats(model.l1.bias)
    ln_w = _floats(model.ln.weight)
    ln_b = _floats(model.ln.bias)
    w2 = _floats(model.l2.weight)
    b2 = _floats(model.l2.bias)
    if len(w1) != HIDDEN * model.n_in or len(w2) != HIDDEN:
        sys.exit("emission: weight shapes do not match the TinyMLP architecture")

    def const(name, values):
        body = ",\n            ".join(f"{v:.9e}" for v in values)
        return f"        const {name}: [f64; {len(values)}] = [\n            {body},\n        ];\n"

    out = []
    out.append("// Copyright (C) 2025 zk4x\n")
    out.append("// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0\n//\n")
    out.append("// Auto-generated by zyx-bench/neural_net.py from bench.db.\n")
    out.append("// TinyMLP: Linear -> LayerNorm -> SELU -> Linear, rank target.\n")
    out.append("#![allow(unused)]\n")
    out.append("//! Cost prediction for kernel selection.\n\n")
    out.append("/// Predict the within-group rank of a kernel (0.0 = fastest).\n")
    out.append("///\n")
    out.append("/// The autotuner only uses this value to order candidates, so the\n")
    out.append("/// rank score is returned directly (lower is better).\n")
    out.append("#[allow(clippy::too_many_arguments)]\n")
    out.append("pub(crate) fn predict_time_us(\n")
    for i, name in enumerate(SIGNATURE):
        comma = "," if i + 1 < len(SIGNATURE) else ","
        out.append(f"    {name}: u32{comma}\n")
    out.append(") -> f64 {\n")
    for name in SIGNATURE:
        out.append(f"    let {name} = {name} as f64;\n")
    out.append("\n    let features: [f64; %d] = [\n" % len(features))
    for expr in features:
        out.append(f"        {expr},\n")
    out.append("    ];\n\n")

    out.append(const("W1", w1))
    out.append(const("B1", b1))
    out.append(const("LN_W", ln_w))
    out.append(const("LN_B", ln_b))
    out.append(const("W2", w2))
    out.append(const("B2", b2))
    out.append(
        f"""
        let n_in = {model.n_in};
        let mut h = [0.0f64; {HIDDEN}];
        for j in 0..{HIDDEN} {{
            let mut v = B1[j];
            for i in 0..n_in {{
                v += W1[j * n_in + i] * features[i];
            }}
            h[j] = v;
        }}
        // zyx LayerNorm: mean over the hidden axis, *biased* variance
        // (correction = 0, same as torch), then `(x - mean) / sqrt(var + eps) * w + b`.
        let mean = h.iter().sum::<f64>() / {HIDDEN}.0;
        let var = h.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / {HIDDEN}.0;
        let inv = 1.0 / (var + {LAYER_NORM_EPS}).sqrt();
        for j in 0..{HIDDEN} {{
            let x = (h[j] - mean) * inv * LN_W[j] + LN_B[j];
            h[j] = {SELU_SCALE} * if x > 0.0 {{ x }} else {{ {SELU_ALPHA} * (x.exp() - 1.0) }};
        }}
        let mut out = B2[0];
        for j in 0..{HIDDEN} {{
            out += W2[j] * h[j];
        }}
        out
    }}
"""
    )
    with open(path, "w") as f:
        f.write("".join(out))
    print(f"Wrote {path}")


# --------------------------------------------------------------------- main

def main():
    rows = load_rows()
    print(f"Read {len(rows)} measurements")

    groups = {}
    for i, (seed, _, _) in enumerate(rows):
        groups.setdefault(seed, []).append(i)
    print(f"Variant groups: {len(groups)}")

    cols = {name: column(rows, name) for name in HW_NAMES}

    feats = build_features(cols)
    n, n_feat = len(rows), len(feats)
    print(f"Features: {n_feat}")

    X = Tensor.cat([t.reshape((n, 1)) for _, t in feats], dim=1)

    # Rank target in [0, 1] within each variant group.
    y = [0.5] * n
    for members in groups.values():
        if len(members) < 2:
            continue
        times = Tensor.from_vec([rows[i][1] for i in members], (len(members),)).cast(DType.F32)
        rv = struct.unpack(f"<{len(members)}f", ranks(times).to_le_bytes())
        denom = float(len(members) - 1)
        for k, i in enumerate(members):
            y[i] = rv[k] / denom

    # Group split: hold out whole variant groups, up to TEST_ROWS measurements.
    rng = random.Random(SEED)
    seeds = list(groups.keys())
    rng.shuffle(seeds)
    test_idx = []
    for s in seeds:
        if len(test_idx) + len(groups[s]) > TEST_ROWS:
            break
        test_idx.extend(groups[s])
    test_mask = [False] * n
    for i in test_idx:
        test_mask[i] = True
    train_idx = [i for i in range(n) if not test_mask[i]]
    test_idx = [i for i in range(n) if test_mask[i]]
    print(f"Train {len(train_idx)} / test {len(test_idx)}")

    def gather(idx):
        return X.index_select(0, Tensor.from_vec([float(i) for i in idx], (len(idx),)).cast(DType.I64))

    # zyx StandardScaler: centre and scale on the training split.
    Xtr_raw = gather(train_idx)
    mu = Xtr_raw.mean(0)
    sd = Xtr_raw.std(0)
    norm = lambda t: (t - mu) / (sd + 1e-8)
    Xtr = norm(Xtr_raw)
    Xte = norm(gather(test_idx))

    model = TinyMLP(n_feat)
    print(f"Model params: {model.n_params()}")
    opt = zyx.optim.AdamW(learning_rate=LR)

    target_tr = Tensor.from_vec([y[i] for i in train_idx], (len(train_idx),))
    # Local row index inside the training block, per group.
    train_pos = {i: k for k, i in enumerate(train_idx)}
    train_groups = []
    for members in groups.values():
        local = [train_pos[i] for i in members if i in train_pos]
        if len(local) >= 4:
            train_groups.append(Tensor.from_vec([float(i) for i in local], (len(local),)).cast(DType.I64))
    print(f"Training groups with >=4 variants: {len(train_groups)}")

    perm = list(range(len(train_groups)))
    step = 0
    while step < TRAIN_STEPS:
        rng.shuffle(perm)
        for gi in perm:
            if step >= TRAIN_STEPS:
                break
            idx = train_groups[gi]
            tape = zyx.Tape()
            tape.add(*model.params)
            x = Xtr.index_select(0, idx)
            target = target_tr.index_select(0, idx)
            pred = model(x)
            diff = pred.unsqueeze(0) - pred.unsqueeze(1)
            s = (target.unsqueeze(0) - target.unsqueeze(1)).sign()
            loss = ((s * diff).exp() + 1.0).log().mean()
            loss = loss + sum(p.abs().sum() for p in model.params) * L1_LAMBDA
            grads = tape.gradient(loss, model.params)
            opt.update(model.params, grads)
            tape.realize([loss, *model.params])
            step += 1
            if step % 5000 == 0:
                print(f"  step {step} loss {loss.item():.4f}")

    evaluate(model, Xte, [y[i] for i in test_idx], test_idx, groups, rows)
    emit_predict_cost(model)


def evaluate(model, Xte, y_test, test_idx, groups, rows):
    n = len(test_idx)
    pos = {i: k for k, i in enumerate(test_idx)}
    pred = model(Xte).detach()
    yt = Tensor.from_vec(y_test, (n,)).cast(DType.F32)

    r2s, rhos, topk = [], [], []
    for members in groups.values():
        local = [pos[i] for i in members if i in pos]
        if len(local) < 3:
            continue
        li = Tensor.from_vec([float(i) for i in local], (len(local),)).cast(DType.I64)
        yg = yt.index_select(0, li)
        pg = pred.index_select(0, li)
        ss_res = ((yg - pg) ** 2).sum()
        ss_tot = ((yg - yg.mean()) ** 2).sum()
        r2s.append(1.0 - (ss_res.item() / ss_tot.item() if ss_tot.item() > 0 else 0.0))
        ra, rp = ranks(yg), ranks(pg)
        ra_c = ra - ra.mean()
        rp_c = rp - rp.mean()
        denom = (ra_c.pow(2.0).sum().sqrt() * rp_c.pow(2.0).sum().sqrt()).item()
        rhos.append((ra_c * rp_c).sum().item() / denom if denom > 0 else 0.0)
        if len(local) >= 20:
            hit = ra.cmplt(10.0).logical_and(rp.cmplt(20.0)).cast(DType.F32).sum()
            topk.append(hit.item() / 10.0)

    import statistics

    print(f"\n=== zyx TinyMLP ({n} test rows, {len(r2s)} groups) ===")
    print(f"  Mean R²:          {statistics.mean(r2s):.4f}")
    print(f"  Median R²:        {statistics.median(r2s):.4f}")
    print(f"  Worst R²:         {min(r2s):.4f}")
    print(f"  Best R²:          {max(r2s):.4f}")
    print(f"  Spearman ρ:       {statistics.mean(rhos):.4f} ± {statistics.pstdev(rhos):.4f}")
    if topk:
        print(f"  Top10-in-top20:   {statistics.mean(topk):.3f} (over {len(topk)} groups)")


if __name__ == "__main__":
    main()
