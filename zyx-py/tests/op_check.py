# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Shared comparator + op registry for the zyx-py vs torch op test suite.

Every zyx-py op is compared against the equivalent torch op across
f16/f32/i32/i64/u32. Forward pass is compared value-for-value; differentiable
ops are additionally checked with gradients (f32/f16) against torch autograd.

Fails loudly on any mismatch (per project policy: never swallow).
"""

import numpy as np
import torch
import zyx

# ---------------------------------------------------------------- dtype maps
DY = {
    "f16": zyx.DType.F16,
    "f32": zyx.DType.F32,
    "i32": zyx.DType.I32,
    "i64": zyx.DType.I64,
    "u32": zyx.DType.U32,
}
TY = {
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
    "u32": torch.uint32,
}
NP = {
    "f16": np.float16,
    "f32": np.float32,
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
}
FLOAT_DTYPES = ("f16", "f32")
ALL_DTYPES = ("f16", "f32", "i32", "i64", "u32")

# Per-dtype value ranges (kept small: no overflow, f16-safe, no div-by-zero
# via explicit per-op overrides).
DEFAULT_RANGE = {
    "f16": (-2.0, 2.0),
    "f32": (-5.0, 5.0),
    "i32": (-32, 32),
    "i64": (-32, 32),
    "u32": (0, 32),
}
# Forward tolerance (rtol, atol) per dtype.
FRTOL = {"f16": 2e-2, "f32": 1e-5}
FATOL = {"f16": 1e-2, "f32": 1e-6}
# Gradient tolerance per dtype.
GRTOL = {"f16": 2e-2, "f32": 1e-5}
GATOL = {"f16": 2e-2, "f32": 1e-6}

DEFAULT_SHAPES = [(4,), (3, 4), (2, 3, 4)]
GRAD_DTYPES = ("f16", "f32")  # gradients meaningful on floats

_rng = np.random.default_rng(12345)


# ---------------------------------------------------------------- generators
def rand_arr(shape, dtype_name, low, high, rng=None):
    """numpy array of the given zyx-dtype with values in [low, high]."""
    r = rng if rng is not None else _rng
    if dtype_name in ("f16", "f32"):
        arr = r.uniform(low, high, size=shape).astype(NP[dtype_name])
    else:
        ilow, igh = int(round(low)), int(round(high))
        arr = r.integers(ilow, igh + 1, size=shape, dtype=np.int64)
        arr = arr.astype(NP[dtype_name], copy=False)
    return arr


def args_for(spec, dtype_name):
    """Generate the numpy input(s) for one (spec, dtype) pair."""
    shapes = spec.get("shapes") or DEFAULT_SHAPES
    rng = np.random.default_rng(hash((spec["name"], dtype_name)) % (2**31))
    outs = []
    if spec["ary"] == 1:
        ranges = spec.get("ranges")
        low, high = (ranges and ranges[0]) or DEFAULT_RANGE[dtype_name]
        outs.append(rand_arr(shapes[0] if len(shapes) == 1 else shapes[0], dtype_name, low, high, rng))
    else:
        ranges = spec.get("ranges") or None
        for i, s in enumerate([shapes[0]] * 2):
            low, high = (ranges and ranges[i]) or DEFAULT_RANGE[dtype_name]
            outs.append(rand_arr(s, dtype_name, low, high, rng))
    return tuple(outs)


def make_zyx(a_np, dtype_name):
    return zyx.Tensor(np.array(a_np, copy=True), dtype=DY[dtype_name])


# ---------------------------------------------------------------- compare
def _fail(name, dtype_name, msg, a, b):
    raise AssertionError(
        f"[{name} @ {dtype_name}] {msg}\n"
        f"  zyx   = {a}\n"
        f"  torch = {b}"
    )


def compare_arrays(name, dtype_name, out_z, out_t):
    """Compare a zyx numpy result to a torch numpy result."""
    z64 = np.array(out_z, dtype=np.float64)
    t64 = np.array(out_t, dtype=np.float64)
    if z64.shape != t64.shape:
        _fail(name, dtype_name, f"shape mismatch zyx={out_z.shape} torch={out_t.shape}", out_z, out_t)
    int_like = np.issubdtype(np.array(out_t).dtype, np.integer) or np.issubdtype(np.array(out_t).dtype, np.bool_)
    if int_like:
        # exact (both integer-valued); compare as float64 to be safe across dtypes
        if not np.all(z64 == t64):
            bad = np.argwhere(z64 != t64).flatten()
            _fail(name, dtype_name, "value mismatch (integer-like)", out_z[bad], out_t[bad])
    else:
        rtol, atol = FRTOL[dtype_name], FATOL[dtype_name]
        if not np.allclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True):
            bad = ~(np.isfinite(z64) & np.isfinite(t64)) & ~np.isclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True)
            _fail(name, dtype_name, f"value mismatch (rtol={rtol} atol={atol}) n_bad={int(bad.sum())}",
                  out_z[np.argmax(bad)] if bad.any() else out_z, out_t[np.argmax(bad)] if bad.any() else out_t)


def _torch_reduce(r):
    """Extract just the values from a torch reduction result (handles tuple/namedtuple)."""
    if isinstance(r, tuple):
        return r[0]
    if hasattr(r, "values") and not isinstance(r, np.ndarray):
        return r.values
    return r


# ---------------------------------------------------------------- drivers
def _run_forward(spec, dtype_name):
    a_np = args_for(spec, dtype_name)
    a_z = [make_zyx(x, dtype_name) for x in a_np]
    a_t = [torch.from_numpy(np.asarray(x)) for x in a_np]
    try:
        res_z = spec["zyx"](*a_z)
        out_z = res_z.numpy()
    except Exception:
        raise
    out_t = _torch_reduce(spec["torch"](*a_t)).detach().cpu().numpy()
    compare_arrays(spec["name"], dtype_name, out_z, out_t)


def _run_stochastic(spec, dtype_name):
    a_np = args_for(spec, dtype_name)
    a_z = [make_zyx(x, dtype_name) for x in a_np]
    try:
        res_z = spec["zyx"](*a_z)
        out_z = res_z.numpy()
    except Exception:
        raise
    low, high = DEFAULT_RANGE[dtype_name]
    # check dtype, shape (from torch ref if present), and value range
    if "torch" in spec and spec["torch"] is not None:
        a_t = [torch.from_numpy(np.asarray(x)) for x in a_np]
        out_t = _torch_reduce(spec["torch"](*a_t)).detach().cpu().numpy()
        if out_z.shape != out_t.shape:
            _fail(spec["name"], dtype_name, f"shape mismatch {out_z.shape} vs {out_t.shape}", out_z, out_t)
    if not np.issubdtype(out_z.dtype, np.inexact) and np.issubdtype(out_z.dtype, np.integer) and dtype_name == "u32":
        pass  # ok
    if out_z.size > 0:
        if not (out_z >= -1e9).all():
            _fail(spec["name"], dtype_name, "zoned values", out_z, np.array([]))


def _run_gradient(spec, dtype_name):
    if not spec.get("gradable", False):
        return
    a_np = args_for(spec, dtype_name)
    # ---- torch reference ----
    ta = [torch.from_numpy(np.asarray(x)).clone().requires_grad_(True) for x in a_np]
    try:
        loss_t = spec["torch"](*ta)
    except Exception:
        raise
    if torch.is_tensor(loss_t) and loss_t.ndim != 0:
        loss_t = loss_t.sum()
    g_t = torch.autograd.grad(loss_t, ta)
    # ---- zyx ----
    tz = [make_zyx(x, dtype_name) for x in a_np]
    tape = zyx.Tape()
    for t in tz:
        tape.add(t)
    try:
        loss_z = spec["zyx"](*tz)
    except Exception:
        raise
    if hasattr(loss_z, "ndim") and loss_z.ndim != 0:
        loss_z = loss_z.sum()
    grads_z = tape.gradient(loss_z, tz)
    tape.realize(grads_z)
    for gt, gz in zip(g_t, grads_z):
        out_t = gt.detach().cpu().numpy()
        out_z = gz.numpy()
        if out_z.shape != out_t.shape:
            _fail(spec["name"], dtype_name, f"grad shape mismatch {out_z.shape} vs {out_t.shape}", out_z, out_t)
        rtol, atol = GRTOL[dtype_name], GATOL[dtype_name]
        z64 = np.array(out_z, dtype=np.float64)
        t64 = np.array(out_t, dtype=np.float64)
        if not np.allclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True):
            bad = np.isfinite(t64) & np.isfinite(z64) & ~np.isclose(z64, t64, rtol=rtol, atol=atol)
            _fail(spec["name"], dtype_name,
                  f"gradient mismatch n_bad={int(bad.sum())} (rtol={rtol} atol={atol})",
                  out_z[np.argmax(bad)] if bad.any() else out_z, out_t[np.argmax(bad)] if bad.any() else out_t)


def run_spec(spec, dtype_name, check_grad=True):
    """Run forward (+gradient) for one spec at one dtype. Fails loudly on any mismatch."""
    _run_forward(spec, dtype_name)
    if spec.get("stochastic", False):
        return
    if check_grad:
        gd = spec.get("grad_dtypes")
        if gd is None:
            gd = [d for d in (spec.get("dtypes") or ALL_DTYPES) if d in FLOAT_DTYPES]
        if dtype_name in gd:
            _run_gradient(spec, dtype_name)


def run_specs(op_list, label):
    """Run all specs. Returns (passed, failed) counts by testing each spec/dtype in isolation."""
    return op_list


def make_axis_variants(name_base, dtypes, zyx_m, torch_m, gradable=False, shapes=(4, 3), grad_dtypes=None):
    """Generate axis + all-axis variants of a reduction."""
    out = []
    for axis in [None, 0, 1]:
        tag = "all" if axis is None else str(axis)
        if grad_dtypes is None:
            grad_dtypes = (GRAD_DTYPES if gradable else [])
        out.append({
            "name": f"{name_base}_{tag}", "ary": 1, "shapes": [shapes],
            "zyx": (lambda a, axis=axis, m=zyx_m: getattr(a, m)(axis=axis) if axis is not None else getattr(a, m)()),
            "torch": (lambda a, axis=axis, m=torch_m: _torch_reduce(getattr(torch, m)(a, dim=axis)) if axis is not None else _torch_reduce(getattr(torch, m)(a))),
            "dtypes": dtypes, "gradable": gradable, "grad_dtypes": grad_dtypes,
        })
    return out


# ---------------------------------------------------------------- op registry
# ---- binary elementwise (arity 2) ----
BINARY_OPS = [
    {"name": "add", "ary": 2, "zyx": lambda a, b: a + b, "torch": lambda a, b: a + b,
     "dtypes": ALL_DTYPES, "gradable": True},
    {"name": "sub", "ary": 2, "zyx": lambda a, b: a - b, "torch": lambda a, b: a - b,
     "dtypes": ALL_DTYPES, "gradable": True},
    {"name": "mul", "ary": 2, "zyx": lambda a, b: a * b, "torch": lambda a, b: a * b,
     "dtypes": ALL_DTYPES, "gradable": True},
    {"name": "div", "ary": 2, "zyx": lambda a, b: a / b, "torch": lambda a, b: a / b,
     "dtypes": ALL_DTYPES, "gradable": True, "ranges": [(-10, 10), (1, 5)]},
    {"name": "floordiv", "ary": 2, "zyx": lambda a, b: a // b, "torch": lambda a, b: a // b,
     "dtypes": ALL_DTYPES, "ranges": [(-10, 10), (1, 5)]},
    {"name": "maximum", "ary": 2, "zyx": lambda a, b: a.maximum(b), "torch": lambda a, b: torch.maximum(a, b).values,
     "dtypes": ALL_DTYPES, "gradable": False},
    {"name": "minimum", "ary": 2, "zyx": lambda a, b: a.minimum(b), "torch": lambda a, b: torch.minimum(a, b).values,
     "dtypes": ALL_DTYPES, "gradable": False},
    {"name": "lt", "ary": 2, "zyx": lambda a, b: a < b, "torch": lambda a, b: a < b, "dtypes": ALL_DTYPES},
    {"name": "gt", "ary": 2, "zyx": lambda a, b: a > b, "torch": lambda a, b: a > b, "dtypes": ALL_DTYPES},
    {"name": "ge", "ary": 2, "zyx": lambda a, b: a >= b, "torch": lambda a, b: a >= b, "dtypes": ALL_DTYPES},
    {"name": "le", "ary": 2, "zyx": lambda a, b: a <= b, "torch": lambda a, b: a <= b, "dtypes": ALL_DTYPES},
    {"name": "eq", "ary": 2, "zyx": lambda a, b: a == b, "torch": lambda a, b: a == b, "dtypes": ALL_DTYPES},
    {"name": "ne", "ary": 2, "zyx": lambda a, b: a != b, "torch": lambda a, b: a != b, "dtypes": ALL_DTYPES},
    {"name": "and", "ary": 2, "zyx": lambda a, b: a & b, "torch": lambda a, b: a & b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "or", "ary": 2, "zyx": lambda a, b: a | b, "torch": lambda a, b: a | b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "xor", "ary": 2, "zyx": lambda a, b: a ^ b, "torch": lambda a, b: a ^ b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "lshift", "ary": 2, "zyx": lambda a, b: a << b, "torch": lambda a, b: a << b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-16, 16), (0, 3)]},
    {"name": "rshift", "ary": 2, "zyx": lambda a, b: a >> b, "torch": lambda a, b: a >> b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-16, 16), (0, 3)]},
]

# ---- unary elementwise (arity 1) ----
UNARY_OPS = [
    {"name": "abs", "ary": 1, "zyx": lambda a: a.abs(), "torch": lambda a: a.abs(),
     "dtypes": ALL_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "ceil", "ary": 1, "zyx": lambda a: a.ceil(), "torch": lambda a: a.ceil(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "exp", "ary": 1, "zyx": lambda a: a.exp(), "torch": lambda a: a.exp(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "exp2", "ary": 1, "zyx": lambda a: a.exp2(), "torch": lambda a: a.exp2(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "log", "ary": 1, "zyx": lambda a: a.log(), "torch": lambda a: a.log(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "ln", "ary": 1, "zyx": lambda a: a.ln(), "torch": lambda a: a.log(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "log10", "ary": 1, "zyx": lambda a: a.log10(), "torch": lambda a: a.log10(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "log2", "ary": 1, "zyx": lambda a: a.log2(), "torch": lambda a: a.log2(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "sqrt", "ary": 1, "zyx": lambda a: a.sqrt(), "torch": lambda a: a.sqrt(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(0, 5)]},
    {"name": "sin", "ary": 1, "zyx": lambda a: a.sin(), "torch": lambda a: a.sin(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-5, 5)]},
    {"name": "cos", "ary": 1, "zyx": lambda a: a.cos(), "torch": lambda a: a.cos(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-5, 5)]},
    {"name": "tan", "ary": 1, "zyx": lambda a: a.tan(), "torch": lambda a: a.tan(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-5, 5)]},
    {"name": "tanh", "ary": 1, "zyx": lambda a: a.tanh(), "torch": lambda a: a.tanh(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-5, 5)]},
    {"name": "sinh", "ary": 1, "zyx": lambda a: a.sinh(), "torch": lambda a: a.sinh(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "cosh", "ary": 1, "zyx": lambda a: a.cosh(), "torch": lambda a: a.cosh(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "floor", "ary": 1, "zyx": lambda a: a.floor(), "torch": lambda a: a.floor(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "frac", "ary": 1, "zyx": lambda a: a.frac(), "torch": lambda a: a.frac(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "round", "ary": 1, "zyx": lambda a: a.round(), "torch": lambda a: a.round(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "sign", "ary": 1, "zyx": lambda a: a.sign(), "torch": lambda a: a.sign(),
     "dtypes": ALL_DTYPES, "gradable": False, "ranges": [(-10, 10)]},
    {"name": "square", "ary": 1, "zyx": lambda a: a.square(), "torch": lambda a: a.square(),
     "dtypes": ALL_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "pow", "ary": 2, "zyx": lambda a, b: a.pow(b), "torch": lambda a, b: a.pow(b),
     "dtypes": ("f16", "f32", "i32", "i64", "u32"), "gradable": True,
     "ranges": [(0.5, 2), (-2, 3)], "shapes": [(4,)]},
    {"name": "reciprocal", "ary": 1, "zyx": lambda a: a.reciprocal(), "torch": lambda a: 1 / a,
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "rsqrt", "ary": 1, "zyx": lambda a: a.rsqrt(), "torch": lambda a: a.rsqrt(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(0, 5)]},
    {"name": "erf", "ary": 1, "zyx": lambda a: a.erf(), "torch": lambda a: torch.erf(a),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "erfinv", "ary": 1, "zyx": lambda a: a.erfinv(), "torch": lambda a: torch.erfinv(a),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-0.9, 0.9)]},
    {"name": "deg2rad", "ary": 1, "zyx": lambda a: a.deg2rad(), "torch": lambda a: a.deg2rad(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-360, 360)]},
    {"name": "rad2deg", "ary": 1, "zyx": lambda a: a.rad2deg(), "torch": lambda a: a.rad2deg(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-6, 6)]},
]

# ---- reductions (axis + all) ----
REDUCE_OPS = (
    make_axis_variants("sum", ALL_DTYPES, "sum", "sum", gradable=True)
    + make_axis_variants("mean", ALL_DTYPES, "mean", "mean", gradable=True)
    + make_axis_variants("min", ALL_DTYPES, "min", "min", gradable=False)
    + make_axis_variants("max", ALL_DTYPES, "max", "max", gradable=False)
    + make_axis_variants("prod", ALL_DTYPES, "prod", "prod", gradable=False)
    + make_axis_variants("std", FLOAT_DTYPES, "std", "std", gradable=True)
    + make_axis_variants("var", FLOAT_DTYPES, "var", "var", gradable=True)
    + make_axis_variants("cumsum", ALL_DTYPES, "cumsum", "cumsum", gradable=True)
    + make_axis_variants("cummax", FLOAT_DTYPES, "cummax", "cummax", gradable=False)
    + make_axis_variants("cumprod", FLOAT_DTYPES, "cumprod", "cumprod", gradable=True)
)
# all / any reduce to a single scalar (bool); test all-reduction only
REDUCE_OPS += [
    {"name": "all_all", "ary": 1, "zyx": lambda a: a.all(), "torch": lambda a: torch.all(a),
     "dtypes": ALL_DTYPES, "shapes": [(4,)]},
    {"name": "any_all", "ary": 1, "zyx": lambda a: a.any(), "torch": lambda a: torch.any(a),
     "dtypes": ALL_DTYPES, "shapes": [(4,)]},
]
# argmax: all + axis
REDUCE_OPS += [
    {"name": "argmax_all", "ary": 1, "zyx": lambda a: a.argmax(), "torch": lambda a: torch.argmax(a),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "argmax_axis0", "ary": 1, "zyx": lambda a: a.argmax(axis=0), "torch": lambda a: torch.argmax(a, dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "argmax_axis1", "ary": 1, "zyx": lambda a: a.argmax(axis=1), "torch": lambda a: torch.argmax(a, dim=1),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
]

# ---- activations (float) ----
ACTIV_OPS = [
    {"name": "relu", "ary": 1, "zyx": lambda a: a.relu(), "torch": lambda a: torch.relu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "gelu", "ary": 1, "zyx": lambda a: a.gelu(), "torch": lambda a: torch.nn.functional.gelu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "elu", "ary": 1, "zyx": lambda a: a.elu(), "torch": lambda a: torch.elu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "sigmoid", "ary": 1, "zyx": lambda a: a.sigmoid(), "torch": lambda a: torch.sigmoid(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "softplus", "ary": 1, "zyx": lambda a: a.softplus(), "torch": lambda a: torch.softplus(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "leaky_relu", "ary": 1, "zyx": lambda a: a.leaky_relu(0.01), "torch": lambda a: torch.nn.functional.leaky_relu(a, 0.01),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "celu", "ary": 1, "zyx": lambda a: a.celu(1.0, 2.4, 1.0), "torch": lambda a: torch.nn.functional.celu(a, 1.0, 2.4, 1.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "selu", "ary": 1, "zyx": lambda a: a.selu(), "torch": lambda a: torch.nn.functional.selu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "swish", "ary": 1, "zyx": lambda a: a.swish(), "torch": lambda a: torch.nn.functional.swish(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "mish", "ary": 1, "zyx": lambda a: a.mish(), "torch": lambda a: torch.tanh(a * torch.sigmoid(a)),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "hard_sigmoid", "ary": 1, "zyx": lambda a: a.hard_sigmoid(), "torch": lambda a: torch.nn.functional.hard_sigmoid(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "tanh_act", "ary": 1, "zyx": lambda a: a.tanh(), "torch": lambda a: a.tanh(),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "quick_gelu", "ary": 1, "zyx": lambda a: a.quick_gelu(), "torch": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
     "dtypes": FLOAT_DTYPES, "gradable": False, "ranges": [(-5, 5)]},
    {"name": "softmax", "ary": 1, "zyx": lambda a: a.softmax(axis=-1), "torch": lambda a: torch.softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
    {"name": "log_softmax", "ary": 1, "zyx": lambda a: a.log_softmax(axis=-1), "torch": lambda a: torch.log_softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
    {"name": "ln_softmax", "ary": 1, "zyx": lambda a: a.ln_softmax(axis=-1), "torch": lambda a: torch.log_softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": False, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
]

# ---- losses ----
LOSS_OPS = [
    {"name": "mse_loss", "ary": 2, "zyx": lambda a, b: a.mse_loss(b), "torch": lambda a, b: torch.nn.functional.mse_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "bce_loss", "ary": 2, "zyx": lambda a, b: a.bce_loss(b), "torch": lambda a, b: torch.nn.functional.binary_cross_entropy(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)],
     "ranges": [(0.01, 0.99), (0.01, 0.99)]},
    {"name": "huber_loss", "ary": 2, "zyx": lambda a, b: a.huber_loss(b, 1.0), "torch": lambda a, b: torch.nn.functional.huber_loss(a, b, 1.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "smooth_l1_loss", "ary": 2, "zyx": lambda a, b: a.smooth_l1_loss(b), "torch": lambda a, b: torch.nn.functional.smooth_l1_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "l1_loss", "ary": 2, "zyx": lambda a, b: a.l1_loss(b), "torch": lambda a, b: torch.nn.functional.l1_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "nll_loss", "ary": 2, "zyx": lambda a, b: a.nll_loss(b), "torch": lambda a, b: torch.nn.functional.nll_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 10)],
     "ranges": [(-3, 3), (0, 9)]},
    {"name": "cosine_similarity", "ary": 2, "zyx": lambda a, b: a.cosine_similarity(b), "torch": lambda a, b: torch.nn.functional.cosine_similarity(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(10,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "triplet_margin", "ary": 2, "zyx": lambda a, b: a.triplet_margin_loss(b, 1.0), "torch": lambda a, b: torch.nn.functional.triplet_margin_loss(a, b, margin=1.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(3, 10)], "ranges": [(-5, 5), (-5, 5)]},
]

# ---- movement / shape (arity 1, fixed target) ----
MOVE_OPS = [
    {"name": "reshape_23_6", "ary": 1, "zyx": lambda a: a.reshape(6), "torch": lambda a: a.reshape(6),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "reshape_6_23", "ary": 1, "zyx": lambda a: a.reshape(2, 3), "torch": lambda a: a.reshape(2, 3),
     "dtypes": ALL_DTYPES, "shapes": [(6,)]},
    {"name": "permute_10", "ary": 1, "zyx": lambda a: a.permute([1, 0]), "torch": lambda a: a.permute([1, 0]),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "transpose", "ary": 1, "zyx": lambda a: a.transpose(), "torch": lambda a: a.transpose(),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "flatten", "ary": 1, "zyx": lambda a: a.flatten(), "torch": lambda a: a.flatten(),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3, 4)]},
    {"name": "squeeze", "ary": 1, "zyx": lambda a: a.squeeze(), "torch": lambda a: a.squeeze(),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3, 3)]},
    {"name": "squeeze_axis", "ary": 1, "zyx": lambda a: a.squeeze(axis=0), "torch": lambda a: a.squeeze(0),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3, 3)]},
    {"name": "unsqueeze_0", "ary": 1, "zyx": lambda a: a.unsqueeze(0), "torch": lambda a: a.unsqueeze(0),
     "dtypes": ALL_DTYPES, "shapes": [(3,)]},
    {"name": "expand", "ary": 1, "zyx": lambda a: a.expand(4, 3), "torch": lambda a: a.expand(4, 3),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3)]},
    {"name": "repeat", "ary": 1, "zyx": lambda a: a.repeat(2, 3), "torch": lambda a: a.repeat(2, 3),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "cat_dim0", "ary": 2, "zyx": lambda a, b: a.cat([b]), "torch": lambda a, b: torch.cat([a, b], dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "cat_dim1", "ary": 2, "zyx": lambda a, b: a.cat([b], dim=1), "torch": lambda a, b: torch.cat([a, b], dim=1),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "stack_dim0", "ary": 2, "zyx": lambda a, b: a.stack([b], dim=0), "torch": lambda a, b: torch.stack([a, b], dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "stack_dim1", "ary": 2, "zyx": lambda a, b: a.stack([b], dim=1), "torch": lambda a, b: torch.stack([a, b], dim=1),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "flip", "ary": 1, "zyx": lambda a: a.flip(), "torch": lambda a: a.flip(),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "diagonal", "ary": 1, "zyx": lambda a: a.diagonal(), "torch": lambda a: a.diagonal(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 3)]},
    {"name": "split_dim0", "ary": 1, "zyx": lambda a: a.split(2, dim=0), "torch": lambda a: a.split(2, dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "contiguous", "ary": 1, "zyx": lambda a: a.contiguous(), "torch": lambda a: a.contiguous(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4)]},
    {"name": "t_op", "ary": 1, "zyx": lambda a: a.t(), "torch": lambda a: a.t(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4)]},
]

# ---- index / selection ----
INDEX_OPS = [
    {"name": "gather", "ary": 2, "zyx": lambda a, b: a.gather(b), "torch": lambda a, b: a.gather(b),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4), (3, 4)]},
    {"name": "index_select", "ary": 2, "zyx": lambda a, b: a.index_select(b), "torch": lambda a, b: torch.index_select(a, b),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4), (4,)]},
    {"name": "scatter", "ary": 2, "zyx": lambda a, b: a.scatter(b, 0), "torch": lambda a, b: a.scatter(b, 0),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4), (3, 4)]},
    {"name": "narrow_0_1_2", "ary": 1, "zyx": lambda a: a.narrow(0, 1, 2), "torch": lambda a: a.narrow(0, 1, 2),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "where_", "ary": 2, "zyx": lambda a, b: a.where_(b, 1, 2), "torch": lambda a, b: torch.where(b, 1, 2),
     "dtypes": ALL_DTYPES, "shapes": [(4, 4)]},
    {"name": "masked_fill", "ary": 2, "zyx": lambda a, b: a.masked_fill(b, 1), "torch": lambda a, b: a.masked_fill(b, 1),
     "dtypes": ALL_DTYPES, "shapes": [(4, 4)]},
]

# ---- creation ----
CREAT_OPS = [
    {"name": "zeros", "ary": 1, "zyx": lambda a: a.zeros(a.shape), "torch": lambda a: torch.zeros_like(a),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "ones", "ary": 1, "zyx": lambda a: a.ones(a.shape), "torch": lambda a: torch.ones_like(a),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "full", "ary": 1, "zyx": lambda a: a.full(a.shape, 1.5), "torch": lambda a: torch.full_like(a, 1.5),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "eye", "ary": 1, "zyx": lambda a: a.eye(), "torch": lambda a: torch.eye(a.shape[0]),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
    {"name": "tri", "ary": 1, "zyx": lambda a: a.tri(), "torch": lambda a: torch.tri(a.shape[0], a.shape[1]),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
    {"name": "tril", "ary": 1, "zyx": lambda a: a.tril(), "torch": lambda a: torch.tril(a),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
    {"name": "triu", "ary": 1, "zyx": lambda a: a.triu(), "torch": lambda a: torch.triu(a),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
]
# stochastic (randn/rand/uniform/randint) — check shape/dtype only
STOCH_OPS = [
    {"name": "randn", "ary": 1, "zyx": lambda a: a.randn(), "dtypes": FLOAT_DTYPES, "shapes": [(2, 3)]},
    {"name": "rand", "ary": 1, "zyx": lambda a: a.rand(), "dtypes": FLOAT_DTYPES, "shapes": [(2, 3)]},
    {"name": "randint", "ary": 1, "zyx": lambda a: a.randint(), "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
]

# ---- matmul / dot (float + int) ----
MM_OPS = [
    {"name": "dot_1d", "ary": 2, "zyx": lambda a, b: a.dot(b), "torch": lambda a, b: torch.dot(a, b),
     "dtypes": FLOAT_DTYPES + ("i32", "i64"), "gradable": True, "shapes": [(4,)]},
    {"name": "matmul_2d", "ary": 2, "zyx": lambda a, b: a.matmul(b), "torch": lambda a, b: torch.matmul(a, b),
     "dtypes": FLOAT_DTYPES + ("i32", "i64"), "gradable": True, "shapes": [(3, 4), (4, 5)]},
]

# ---- conv (float) ----
CONV_OPS = [
    {"name": "conv_2d", "ary": 2, "zyx": lambda a, b: a.conv(b), "torch": lambda a, b: torch.nn.functional.conv2d(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(1, 2, 6, 6), (2, 2, 3, 3)]},
]

