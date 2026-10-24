# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Single-file op test suite: every zyx-py op compared against the equivalent
torch op across f16/f32/i32/i64/u32. Forward pass is compared value-for-value;
differentiable ops are additionally checked with gradients (f32/f16) against
torch autograd.

Where zyx intentionally follows tinygrad semantics (gelu/quick_gelu
approximations, integer prod wrapping), the reference is tinygrad instead.

Fails loudly on any mismatch (per project policy: never swallow).
"""

import math

import hashlib
import numpy as np
import pytest
import torch
import zyx
from tinygrad import Tensor as TinyTensor

# ---------------------------------------------------------------- dtype maps
DY = {
    "f16": zyx.DType.F16,
    "f32": zyx.DType.F32,
    "i32": zyx.DType.I32,
    "i64": zyx.DType.I64,
    "u32": zyx.DType.U32,
    "bool": zyx.DType.Bool,
}
TY = {
    "f16": torch.float16,
    "f32": torch.float32,
    "i32": torch.int32,
    "i64": torch.int64,
    "u32": torch.uint32,
    "bool": torch.bool,
}
NP = {
    "f16": np.float16,
    "f32": np.float32,
    "i32": np.int32,
    "i64": np.int64,
    "u32": np.uint32,
    "bool": np.bool_,
}
FLOAT_DTYPES = ("f16", "f32")
ALL_DTYPES = ("f16", "f32", "i32", "i64", "u32")
# torch CPU has essentially no uint32 kernels (no arith/compare/flip/gather);
# specs torch cannot express exclude u32 (zyx runs them fine).
NOU32_DTYPES = ("f16", "f32", "i32", "i64")

# Per-dtype value ranges (kept small: no overflow, f16-safe, no div-by-zero
# via explicit per-op overrides).
# f32 default widened to (-10, 10): only linear elementwise ops use it
# (add/sub/mul/min/max/comparisons), all safe at that magnitude in f32.
# f16 stays tight: mul/pow/exp chains overflow f16 above small magnitudes.
DEFAULT_RANGE = {
    "f16": (-2.0, 2.0),
    "f32": (-10.0, 10.0),
    "i32": (-32, 32),
    "i64": (-32, 32),
    "u32": (0, 32),
    "bool": (0, 1),
}
# Forward tolerance (rtol, atol) per dtype. Integer inputs can produce float
# outputs (div, mean-ref); they compare at f32 precision.
FRTOL = {"f16": 2e-2, "f32": 1e-5, "i32": 1e-5, "i64": 1e-5, "u32": 1e-5}
FATOL = {"f16": 1e-2, "f32": 1e-6, "i32": 1e-6, "i64": 1e-6, "u32": 1e-6}
# Gradient tolerance per dtype.
GRTOL = {"f16": 2e-2, "f32": 1e-5}
GATOL = {"f16": 2e-2, "f32": 1e-6}

# Prime/odd geometries (not powers of two) stress tiling and edge handling;
# each new shape extends the old draws (same seed stream prefix), so coverage
# only grows, never shifts. Per-test runtime unchanged.
DEFAULT_SHAPES = [(7,), (5, 7), (2, 5, 7)]
GRAD_DTYPES = ("f16", "f32")  # gradients meaningful on floats

_rng = np.random.default_rng(12345)


# ---------------------------------------------------------------- generators
def rand_arr(shape, dtype_name, low, high, rng=None):
    """numpy array of the given zyx-dtype with values in [low, high]."""
    r = rng if rng is not None else _rng
    if dtype_name in ("f16", "f32"):
        arr = r.uniform(low, high, size=shape).astype(NP[dtype_name])
    elif dtype_name == "bool":
        arr = (r.integers(0, 2, size=shape) != 0)
    else:
        ilow, igh = int(round(low)), int(round(high))
        arr = r.integers(ilow, igh + 1, size=shape, dtype=np.int64)
        arr = arr.astype(NP[dtype_name], copy=False)
    return arr


def input_dtype(spec, i, dtype_name):
    """Per-input dtype name: spec input_dtypes[i] or the case dtype."""
    idtypes = spec.get("input_dtypes") or []
    return (idtypes[i] if i < len(idtypes) and idtypes[i] else dtype_name)


def args_for(spec, dtype_name):
    """Generate the numpy input(s) for one (spec, dtype) pair."""
    shapes = spec.get("shapes") or DEFAULT_SHAPES
    # Stable across processes (unlike builtin hash(), which is salted per
    # process and made every run draw different inputs): deterministic draws
    # turn flakes into reproducible pass/fail.
    digest = hashlib.sha256(f"{spec['name']}|{dtype_name}".encode()).hexdigest()
    rng = np.random.default_rng(int(digest, 16) % (2**31))
    outs = []
    n = spec["ary"]
    for i in range(n):
        s = shapes[i] if i < len(shapes) else shapes[0]
        idt = input_dtype(spec, i, dtype_name)
        # scatter-style index input: unique permutation per slice along dim
        # (duplicate indices make parallel scatter order-undefined)
        if spec.get("permute_index") is not None and i == spec["permute_index"]:
            dim = spec["permute_axis"]
            other = [k for k in range(len(s)) if k != dim]
            perms = [rng.permutation(s[dim]) for _ in range(int(np.prod([s[k] for k in other])) or 1)]
            arr = np.stack(perms, axis=1).reshape(s) if len(s) == 2 else np.stack(perms).reshape(s)
            outs.append(arr.astype(np.int64))
            continue
        ranges = spec.get("ranges") or None
        low, high = (ranges[i] if ranges and i < len(ranges) else None) or DEFAULT_RANGE[idt]
        outs.append(rand_arr(s, idt, low, high, rng))
    return tuple(outs)


def make_zyx(a_np, dtype_name):
    if dtype_name is None:
        return zyx.Tensor(np.array(a_np, copy=True))
    return zyx.Tensor(np.array(a_np, copy=True), dtype=DY[dtype_name])


# ---------------------------------------------------------------- compare
def _fail(name, dtype_name, msg, a, b):
    raise AssertionError(
        f"[{name} @ {dtype_name}] {msg}\n"
        f"  zyx   = {a}\n"
        f"  torch = {b}"
    )


def _pick(o, idx):
    """Index a result for failure display; flattens N-D, 0-d scalars pass through."""
    o = np.asarray(o).reshape(-1)
    return o[idx] if o.size > 0 else o


def compare_arrays(name, dtype_name, out_z, out_t, rtol=None, atol=None):
    """Compare a zyx numpy result to a torch numpy result.

    Per-spec overrides (rtol/atol) loosen but never tighten: the
    effective tolerance is max(table, override), so f16 behavior is
    unchanged when only f32 needs room (e.g. approximation gaps).
    """
    if rtol is None:
        rtol = FRTOL[dtype_name]
    else:
        rtol = max(FRTOL[dtype_name], rtol)
    if atol is None:
        atol = FATOL[dtype_name]
    else:
        atol = max(FATOL[dtype_name], atol)
    z64 = np.array(out_z, dtype=np.float64)
    t64 = np.array(out_t, dtype=np.float64)
    if z64.shape != t64.shape:
        _fail(name, dtype_name, f"shape mismatch zyx={out_z.shape} torch={out_t.shape}", out_z, out_t)
    int_like = np.issubdtype(np.array(out_t).dtype, np.integer) or np.issubdtype(np.array(out_t).dtype, np.bool_)
    if int_like:
        # exact (both integer-valued); compare as float64 to be safe across dtypes
        if not np.all(z64 == t64):
            bad = np.argwhere(z64 != t64).flatten()
            _fail(name, dtype_name, "value mismatch (integer-like)", _pick(out_z, bad), _pick(out_t, bad))
    else:
        if not np.allclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True):
            bad = ~(np.isfinite(z64) & np.isfinite(t64)) & ~np.isclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True)
            _fail(name, dtype_name, f"value mismatch (rtol={rtol} atol={atol}) n_bad={int(bad.sum())}",
                  _pick(out_z, np.argmax(bad)) if bad.any() else out_z, _pick(out_t, np.argmax(bad)) if bad.any() else out_t)


def _torch_reduce(r):
    """Extract just the values from a torch reduction result (handles tuple/namedtuple)."""
    if isinstance(r, tuple):
        return r[0]
    if hasattr(r, "values") and not callable(r.values) and not isinstance(r, np.ndarray):
        return r.values
    return r


# ---------------------------------------------------------------- drivers
def _run_forward(spec, dtype_name):
    a_np = args_for(spec, dtype_name)
    a_z = [make_zyx(x, input_dtype(spec, i, dtype_name)) for i, x in enumerate(a_np)]
    a_t = [torch.from_numpy(np.asarray(x)) for x in a_np]
    try:
        res_z = spec["zyx"](*a_z)
    except Exception:
        raise
    if isinstance(res_z, (list, tuple)):
        # multi-output (split): compare elementwise against torch's tuple
        out_t = spec["torch"](*a_t)
        assert isinstance(out_t, (list, tuple)) and len(out_t) == len(res_z), \
            f"{spec['name']} output count zyx={len(res_z)} torch={len(out_t) if isinstance(out_t, (list, tuple)) else '?'}"
        for rz, rt in zip(res_z, out_t):
            compare_arrays(spec["name"], dtype_name,
                           rz.numpy(), _torch_reduce(rt).detach().cpu().numpy(),
                           rtol=spec.get("frtol"), atol=spec.get("fratol"))
        return
    out_z = res_z.numpy()
    if spec.get("tinygrad") is not None:
        # zyx intentionally follows tinygrad semantics here (gelu/quick_gelu
        # approximations, integer prod wrapping): the reference is tinygrad,
        # compared at table tolerance with no loosening overrides.
        a_g = [TinyTensor(np.asarray(x)) for x in a_np]
        out_t = np.asarray(spec["tinygrad"](*a_g).numpy())
    else:
        out_t = _torch_reduce(spec["torch"](*a_t)).detach().cpu().numpy()
    compare_arrays(spec["name"], dtype_name, out_z, out_t,
                   rtol=spec.get("frtol"), atol=spec.get("fratol"))


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
    # gradients flow only through float inputs on all sides
    float_mask = [np.issubdtype(np.asarray(x).dtype, np.floating) for x in a_np]
    if spec.get("tinygrad") is not None:
        # ---- tinygrad reference (zyx follows tinygrad semantics) ----
        ga = [TinyTensor(np.asarray(x)) for x in a_np]
        for t, f in zip(ga, float_mask):
            if f:
                t.requires_grad_(True)
        loss_g = spec["tinygrad"](*ga)
        if loss_g.ndim != 0:
            loss_g = loss_g.sum()
        loss_g.backward()
        g_t = [t.grad.numpy() for t, f in zip(ga, float_mask) if f]
    else:
        # ---- torch reference (only float inputs take gradients) ----
        ta = [torch.from_numpy(np.asarray(x)).clone().requires_grad_(True) if np.issubdtype(np.asarray(x).dtype, np.floating)
              else torch.from_numpy(np.asarray(x)) for x in a_np]
        try:
            loss_t = spec["torch"](*ta)
        except Exception:
            raise
        if torch.is_tensor(loss_t) and loss_t.ndim != 0:
            loss_t = loss_t.sum()
        g_t = torch.autograd.grad(loss_t, [t for t, f in zip(ta, float_mask) if f])
    # ---- zyx ----
    tz = [make_zyx(x, input_dtype(spec, i, dtype_name)) for i, x in enumerate(a_np)]
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
    grads_z = [g for g, f in zip(grads_z, float_mask) if f]
    tz = [t for t, f in zip(tz, float_mask) if f]
    for gt, gz in zip(g_t, grads_z):
        out_t = gt.detach().cpu().numpy() if torch.is_tensor(gt) else np.asarray(gt)
        out_z = gz.numpy()
        if out_z.shape != out_t.shape:
            _fail(spec["name"], dtype_name, f"grad shape mismatch {out_z.shape} vs {out_t.shape}", out_z, out_t)
        rtol, atol = GRTOL[dtype_name], GATOL[dtype_name]
        if spec.get("grtol") is not None:
            rtol = max(rtol, spec["grtol"])
        if spec.get("gratol") is not None:
            atol = max(atol, spec["gratol"])
        z64 = np.array(out_z, dtype=np.float64)
        t64 = np.array(out_t, dtype=np.float64)
        if not np.allclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True):
            bad = np.isfinite(t64) & np.isfinite(z64) & ~np.isclose(z64, t64, rtol=rtol, atol=atol)
            _fail(spec["name"], dtype_name,
                  f"gradient mismatch n_bad={int(bad.sum())} (rtol={rtol} atol={atol})",
                  _pick(out_z, np.argmax(bad)) if bad.any() else out_z, _pick(out_t, np.argmax(bad)) if bad.any() else out_t)


def run_spec(spec, dtype_name, check_grad=True):
    """Run forward (+gradient) for one spec at one dtype. Fails loudly on any mismatch."""
    if spec.get("stochastic", False):
        _run_stochastic(spec, dtype_name)
        return
    _run_forward(spec, dtype_name)
    if check_grad:
        gd = spec.get("grad_dtypes")
        if gd is None:
            gd = [d for d in (spec.get("dtypes") or ALL_DTYPES) if d in FLOAT_DTYPES]
        if dtype_name in gd:
            _run_gradient(spec, dtype_name)


def run_specs(op_list, label):
    """Run all specs. Returns (passed, failed) counts by testing each spec/dtype in isolation."""
    return op_list


def make_axis_variants(name_base, dtypes, zyx_m, torch_m, gradable=False, shapes=(4, 3), grad_dtypes=None, skip_all=False):
    """Generate axis + all-axis variants of a reduction."""
    out = []
    for axis in [None, 0, 1]:
        if axis is None and skip_all:
            continue
        tag = "all" if axis is None else str(axis)
        if grad_dtypes is None:
            grad_dtypes = (GRAD_DTYPES if gradable else [])
        out.append({
            "name": f"{name_base}_{tag}", "ary": 1, "shapes": [shapes],
            "zyx": (lambda a, axis=axis, m=zyx_m: getattr(a, m)(dim=axis) if axis is not None else getattr(a, m)()),
            "torch": (lambda a, axis=axis, m=torch_m: _torch_reduce(getattr(torch, m)(a, dim=axis)) if axis is not None else _torch_reduce(getattr(torch, m)(a))),
            "dtypes": dtypes, "gradable": gradable, "grad_dtypes": grad_dtypes,
        })
    return out


# ---------------------------------------------------------------- op registry
# ---- binary elementwise (arity 2) ----
BINARY_OPS = [
    {"name": "add", "ary": 2, "zyx": lambda a, b: a + b, "torch": lambda a, b: a + b,
     "dtypes": NOU32_DTYPES, "gradable": True},
    {"name": "sub", "ary": 2, "zyx": lambda a, b: a - b, "torch": lambda a, b: a - b,
     "dtypes": NOU32_DTYPES, "gradable": True},
    {"name": "mul", "ary": 2, "zyx": lambda a, b: a * b, "torch": lambda a, b: a * b,
     "dtypes": ALL_DTYPES, "gradable": True},
    {"name": "div", "ary": 2, "zyx": lambda a, b: a / b, "torch": lambda a, b: a / b,
     "dtypes": ALL_DTYPES, "gradable": True, "ranges": [(-10, 10), (1, 5)]},
    {"name": "floordiv", "ary": 2, "zyx": lambda a, b: a // b, "torch": lambda a, b: a // b,
     "dtypes": NOU32_DTYPES, "ranges": [(-10, 10), (1, 5)]},
    {"name": "maximum", "ary": 2, "zyx": lambda a, b: a.maximum(b), "torch": lambda a, b: torch.maximum(a, b),
     "dtypes": NOU32_DTYPES, "gradable": False},
    {"name": "minimum", "ary": 2, "zyx": lambda a, b: a.minimum(b), "torch": lambda a, b: torch.minimum(a, b),
     "dtypes": NOU32_DTYPES, "gradable": False},
    {"name": "lt", "ary": 2, "zyx": lambda a, b: a < b, "torch": lambda a, b: a < b, "dtypes": NOU32_DTYPES},
    {"name": "gt", "ary": 2, "zyx": lambda a, b: a > b, "torch": lambda a, b: a > b, "dtypes": NOU32_DTYPES},
    {"name": "ge", "ary": 2, "zyx": lambda a, b: a >= b, "torch": lambda a, b: a >= b, "dtypes": NOU32_DTYPES},
    {"name": "le", "ary": 2, "zyx": lambda a, b: a <= b, "torch": lambda a, b: a <= b, "dtypes": NOU32_DTYPES},
    {"name": "eq", "ary": 2, "zyx": lambda a, b: a == b, "torch": lambda a, b: a == b, "dtypes": ALL_DTYPES},
    {"name": "ne", "ary": 2, "zyx": lambda a, b: a != b, "torch": lambda a, b: a != b, "dtypes": ALL_DTYPES},
    {"name": "and", "ary": 2, "zyx": lambda a, b: a & b, "torch": lambda a, b: a & b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "or", "ary": 2, "zyx": lambda a, b: a | b, "torch": lambda a, b: a | b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "xor", "ary": 2, "zyx": lambda a, b: a ^ b, "torch": lambda a, b: a ^ b,
     "dtypes": ("i32", "i64", "u32"), "ranges": [(-8, 8), (-8, 8)]},
    {"name": "lshift", "ary": 2, "zyx": lambda a, b: a << b, "torch": lambda a, b: a << b,
     "dtypes": ("i32", "i64"), "ranges": [(-16, 16), (0, 3)]},
    {"name": "rshift", "ary": 2, "zyx": lambda a, b: a >> b, "torch": lambda a, b: a >> b,
     "dtypes": ("i32", "i64"), "ranges": [(-16, 16), (0, 3)]},
]

# ---- unary elementwise (arity 1) ----
UNARY_OPS = [
    {"name": "abs", "ary": 1, "zyx": lambda a: a.abs(), "torch": lambda a: a.abs(),
     "dtypes": NOU32_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "ceil", "ary": 1, "zyx": lambda a: a.ceil(), "torch": lambda a: a.ceil(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "exp", "ary": 1, "zyx": lambda a: a.exp(), "torch": lambda a: a.exp(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "exp2", "ary": 1, "zyx": lambda a: a.exp2(), "torch": lambda a: a.exp2(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(-3, 3)]},
    {"name": "log", "ary": 1, "zyx": lambda a: a.log(math.e), "torch": lambda a: a.log(),
     "dtypes": ("f16", "f32"), "gradable": True, "ranges": [(1, 5)]},
    {"name": "ln", "ary": 1, "zyx": lambda a: a.ln(), "torch": lambda a: torch.log(a),
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
    {"name": "round", "ary": 1, "zyx": lambda a: a.round(), "torch": lambda a: a.round(),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "sign", "ary": 1, "zyx": lambda a: a.sign(), "torch": lambda a: a.sign(),
     "dtypes": NOU32_DTYPES, "gradable": False, "ranges": [(-10, 10)]},
    {"name": "neg", "ary": 1, "zyx": lambda a: -a, "torch": lambda a: torch.neg(a),
     "dtypes": NOU32_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "trunc", "ary": 1, "zyx": lambda a: a.trunc(), "torch": lambda a: torch.trunc(a),
     "dtypes": ("f16", "f32"), "gradable": False, "ranges": [(-5, 5)]},
    {"name": "clamp", "ary": 1, "zyx": lambda a: a.clamp(-1.0, 1.0), "torch": lambda a: torch.clamp(a, -1.0, 1.0),
     "dtypes": NOU32_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "square", "ary": 1, "zyx": lambda a: a.square(), "torch": lambda a: a.square(),
     "dtypes": NOU32_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "pow", "ary": 2, "zyx": lambda a, b: a.pow(b), "torch": lambda a, b: a.pow(b),
     # u32 excluded: torch has no uint32 pow (NotImplementedError), so no
     # reference exists. zyx u32 lowering is exact (wrapping, verified
     # against Python big-int modular arithmetic).
     "dtypes": ("f16", "f32", "i32", "i64"), "gradable": True,
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
    + make_axis_variants("mean", FLOAT_DTYPES, "mean", "mean", gradable=True)
    + make_axis_variants("min", NOU32_DTYPES, "min", "min", gradable=False)
    + make_axis_variants("max", NOU32_DTYPES, "max", "max", gradable=False)
    + [dict(s, tinygrad=(lambda a, tag=s["name"].rsplit("_", 1)[-1]: a.prod() if tag == "all" else a.prod(int(tag)))) for s in
       make_axis_variants("prod", ALL_DTYPES, "prod", "prod", gradable=False)]
    + make_axis_variants("std", FLOAT_DTYPES, "std", "std", gradable=True)
    + make_axis_variants("var", FLOAT_DTYPES, "var", "var", gradable=True)
    + make_axis_variants("cumsum", ALL_DTYPES, "cumsum", "cumsum", gradable=True, skip_all=True)
    + make_axis_variants("cummax", FLOAT_DTYPES, "cummax", "cummax", gradable=False, skip_all=True)
    + make_axis_variants("cumprod", FLOAT_DTYPES, "cumprod", "cumprod", gradable=True, skip_all=True)
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
     "dtypes": NOU32_DTYPES, "shapes": [(4, 3)]},
    {"name": "argmax_axis0", "ary": 1, "zyx": lambda a: a.argmax(dim=0), "torch": lambda a: torch.argmax(a, dim=0),
     "dtypes": NOU32_DTYPES, "shapes": [(4, 3)]},
    {"name": "argmax_axis1", "ary": 1, "zyx": lambda a: a.argmax(dim=1), "torch": lambda a: torch.argmax(a, dim=1),
     "dtypes": NOU32_DTYPES, "shapes": [(4, 3)]},
]

# ---- activations (float) ----
ACTIV_OPS = [
    {"name": "relu", "ary": 1, "zyx": lambda a: a.relu(), "torch": lambda a: torch.relu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "gelu", "ary": 1, "zyx": lambda a: a.gelu(), "torch": lambda a: torch.nn.functional.gelu(a),
     "tinygrad": lambda a: a.gelu(),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "elu", "ary": 1, "zyx": lambda a: a.elu(1.0), "torch": lambda a: torch.nn.functional.elu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "sigmoid", "ary": 1, "zyx": lambda a: a.sigmoid(), "torch": lambda a: torch.sigmoid(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "softplus", "ary": 1, "zyx": lambda a: a.softplus(1.0, 20.0), "torch": lambda a: torch.nn.functional.softplus(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-10, 10)]},
    {"name": "leaky_relu", "ary": 1, "zyx": lambda a: a.leaky_relu(0.01), "torch": lambda a: torch.nn.functional.leaky_relu(a, 0.01),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "celu", "ary": 1, "zyx": lambda a: a.celu(1.0), "torch": lambda a: torch.nn.functional.celu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "selu", "ary": 1, "zyx": lambda a: a.selu(), "torch": lambda a: torch.nn.functional.selu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "swish", "ary": 1, "zyx": lambda a: a.swish(), "torch": lambda a: torch.nn.functional.silu(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "mish", "ary": 1, "zyx": lambda a: a.mish(), "torch": lambda a: a * torch.tanh(torch.nn.functional.softplus(a)),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "hard_sigmoid", "ary": 1, "zyx": lambda a: a.hard_sigmoid(), "torch": lambda a: torch.nn.functional.hardsigmoid(a),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "tanh_act", "ary": 1, "zyx": lambda a: a.tanh(), "torch": lambda a: a.tanh(),
     "dtypes": FLOAT_DTYPES, "gradable": True, "ranges": [(-5, 5)]},
    {"name": "quick_gelu", "ary": 1, "zyx": lambda a: a.quick_gelu(), "torch": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
     "tinygrad": lambda a: a.quick_gelu(),
     "dtypes": FLOAT_DTYPES, "gradable": False, "ranges": [(-5, 5)]},
    {"name": "softmax", "ary": 1, "zyx": lambda a: a.softmax(dim=-1), "torch": lambda a: torch.softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
    {"name": "log_softmax", "ary": 1, "zyx": lambda a: a.log_softmax(dim=-1), "torch": lambda a: torch.log_softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
    {"name": "ln_softmax", "ary": 1, "zyx": lambda a: a.ln_softmax(dim=-1), "torch": lambda a: torch.log_softmax(a, dim=-1),
     "dtypes": FLOAT_DTYPES, "gradable": False, "shapes": [(4, 5)], "ranges": [(-5, 5)]},
]

# ---- losses ----
LOSS_OPS = [
    {"name": "mse_loss", "ary": 2, "zyx": lambda a, b: a.mse_loss(b), "torch": lambda a, b: torch.nn.functional.mse_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "bce_loss", "ary": 2, "zyx": lambda a, b: a.bce_loss(b, 1e-7), "torch": lambda a, b: torch.nn.functional.binary_cross_entropy(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)],
     "ranges": [(0.01, 0.99), (0.01, 0.99)]},
    {"name": "huber_loss", "ary": 2, "zyx": lambda a, b: a.huber_loss(b, delta=1.0), "torch": lambda a, b: torch.nn.functional.huber_loss(a, b, delta=1.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "smooth_l1_loss", "ary": 2, "zyx": lambda a, b: a.smooth_l1_loss(b), "torch": lambda a, b: torch.nn.functional.smooth_l1_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "l1_loss", "ary": 2, "zyx": lambda a, b: a.l1_loss(b), "torch": lambda a, b: torch.nn.functional.l1_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "nll_loss", "ary": 2, "zyx": lambda a, b: a.nll_loss(b), "torch": lambda a, b: torch.nn.functional.nll_loss(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 10), (4,)], "input_dtypes": [None, "i64"],
     "ranges": [(-3, 3), (0, 9)]},
    {"name": "cross_entropy", "ary": 2, "zyx": lambda a, b: a.cross_entropy(b, zyx.Reduction.Mean),
     "torch": lambda a, b: torch.nn.functional.cross_entropy(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(4, 10), (4,)], "input_dtypes": [None, "i64"],
     "ranges": [(-3, 3), (0, 9)]},
    {"name": "cosine_similarity", "ary": 2, "zyx": lambda a, b: a.cosine_similarity(b), "torch": lambda a, b: torch.nn.functional.cosine_similarity(a, b, dim=0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(10,)], "ranges": [(-5, 5), (-5, 5)]},
    {"name": "triplet_margin", "ary": 3, "zyx": lambda a, b, c: a.triplet_margin_loss(b, c), "torch": lambda a, b, c: torch.nn.functional.triplet_margin_loss(a, b, c, margin=1.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(3, 10), (3, 10), (3, 10)], "ranges": [(-5, 5), (-5, 5), (-5, 5)]},
]

# ---- movement / shape (arity 1, fixed target) ----
MOVE_OPS = [
    {"name": "reshape_23_6", "ary": 1, "zyx": lambda a: a.reshape(6), "torch": lambda a: a.reshape(6),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "reshape_6_23", "ary": 1, "zyx": lambda a: a.reshape(2, 3), "torch": lambda a: a.reshape(2, 3),
     "dtypes": ALL_DTYPES, "shapes": [(6,)]},
    {"name": "permute_10", "ary": 1, "zyx": lambda a: a.permute([1, 0]), "torch": lambda a: a.permute([1, 0]),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "transpose", "ary": 1, "zyx": lambda a: a.transpose(0, 1), "torch": lambda a: a.transpose(0, 1),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "flatten", "ary": 1, "zyx": lambda a: a.flatten(), "torch": lambda a: a.flatten(),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3, 4)]},
    {"name": "squeeze", "ary": 1, "zyx": lambda a: a.squeeze(), "torch": lambda a: a.squeeze(),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3, 3)]},
    {"name": "squeeze_axis", "ary": 1, "zyx": lambda a: a.squeeze(dim=0), "torch": lambda a: a.squeeze(0),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3, 3)]},
    {"name": "unsqueeze_0", "ary": 1, "zyx": lambda a: a.unsqueeze(0), "torch": lambda a: a.unsqueeze(0),
     "dtypes": ALL_DTYPES, "shapes": [(3,)]},
    {"name": "expand", "ary": 1, "zyx": lambda a: a.expand(4, 3), "torch": lambda a: a.expand(4, 3),
     "dtypes": ALL_DTYPES, "shapes": [(1, 3)]},
    {"name": "repeat", "ary": 1, "zyx": lambda a: a.repeat(2, 3), "torch": lambda a: a.repeat(2, 3),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "cat_dim0", "ary": 2, "zyx": lambda a, b: zyx.Tensor.cat([a, b], 0), "torch": lambda a, b: torch.cat([a, b], dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "cat_dim1", "ary": 2, "zyx": lambda a, b: zyx.Tensor.cat([a, b], 1), "torch": lambda a, b: torch.cat([a, b], dim=1),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "stack_dim0", "ary": 2, "zyx": lambda a, b: zyx.Tensor.stack([a, b], 0), "torch": lambda a, b: torch.stack([a, b], dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "stack_dim1", "ary": 2, "zyx": lambda a, b: zyx.Tensor.stack([a, b], 1), "torch": lambda a, b: torch.stack([a, b], dim=1),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3), (2, 3)]},
    {"name": "flip", "ary": 1, "zyx": lambda a: a.flip((0, 1)), "torch": lambda a: a.flip((0, 1)),
     "dtypes": NOU32_DTYPES, "shapes": [(2, 3)]},
    {"name": "diagonal", "ary": 1, "zyx": lambda a: a.diagonal(), "torch": lambda a: a.diagonal(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 3)]},
    {"name": "split_dim0", "ary": 1, "zyx": lambda a: a.split((2, 2), dim=0), "torch": lambda a: a.split(2, dim=0),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "contiguous", "ary": 1, "zyx": lambda a: a.contiguous(), "torch": lambda a: a.contiguous(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4)]},
    {"name": "t_op", "ary": 1, "zyx": lambda a: a.t(), "torch": lambda a: a.t(),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4)]},
]

# ---- index / selection ----
INDEX_OPS = [
    {"name": "gather", "ary": 2, "zyx": lambda a, b: a.gather(0, b), "torch": lambda a, b: a.gather(0, b),
     "dtypes": NOU32_DTYPES, "shapes": [(3, 4), (3, 4)], "input_dtypes": [None, "i64"], "ranges": [(-5, 5), (0, 2)]},
    {"name": "index_select", "ary": 2, "zyx": lambda a, b: a.index_select(0, b), "torch": lambda a, b: torch.index_select(a, 0, b),
     "dtypes": ALL_DTYPES, "shapes": [(3, 4), (4,)], "input_dtypes": [None, "i64"], "ranges": [(-5, 5), (0, 2)]},
    {"name": "scatter", "ary": 3, "zyx": lambda a, b, c: a.scatter(0, b, c), "torch": lambda a, b, c: torch.scatter(a, 0, b, c),
     "dtypes": NOU32_DTYPES, "shapes": [(3, 4), (3, 4), (3, 4)], "input_dtypes": [None, "i64", None], "ranges": [(-5, 5), (0, 2), (-5, 5)],
     "permute_index": 1, "permute_axis": 0},
    {"name": "narrow_0_1_2", "ary": 1, "zyx": lambda a: a.narrow(0, 1, 2), "torch": lambda a: a.narrow(0, 1, 2),
     "dtypes": ALL_DTYPES, "shapes": [(4, 3)]},
    {"name": "where_", "ary": 2, "zyx": lambda a, b: (a > b).where(1, 2), "torch": lambda a, b: torch.where(a > b, 1, 2),
     "dtypes": NOU32_DTYPES, "shapes": [(4, 4)]},
    {"name": "masked_fill", "ary": 2, "zyx": lambda a, b: a.masked_fill(b, 1), "torch": lambda a, b: a.masked_fill(b, 1),
     "dtypes": NOU32_DTYPES, "shapes": [(4, 4)], "input_dtypes": [None, "bool"]},
]

# ---- creation ----
CREAT_OPS = [
    {"name": "zeros", "ary": 1, "zyx": lambda a: a.zeros(a.shape), "torch": lambda a: torch.zeros_like(a),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "ones", "ary": 1, "zyx": lambda a: a.ones(a.shape), "torch": lambda a: torch.ones_like(a),
     "dtypes": ALL_DTYPES, "shapes": [(2, 3)]},
    {"name": "full", "ary": 1, "zyx": lambda a: a.full(*a.shape, fill_value=1.5), "torch": lambda a: torch.full_like(a, 1.5),
     "dtypes": FLOAT_DTYPES, "shapes": [(2, 3)]},
    {"name": "eye", "ary": 1, "zyx": lambda a: zyx.Tensor.eye(a.shape[0], a.dtype), "torch": lambda a: torch.eye(a.shape[0], dtype=a.dtype),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
    {"name": "arange", "ary": 1, "zyx": lambda a: zyx.Tensor.arange(0, 8, 2), "torch": lambda a: torch.arange(0, 8, 2),
     "dtypes": ("i64",), "shapes": [(4,)]},
    {"name": "tril", "ary": 1, "zyx": lambda a: a.tril(0), "torch": lambda a: torch.tril(a),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
    {"name": "triu", "ary": 1, "zyx": lambda a: a.triu(0), "torch": lambda a: torch.triu(a),
     "dtypes": FLOAT_DTYPES, "shapes": [(3, 3)]},
]
# stochastic (randn/rand/uniform/randint) — check shape/dtype only
STOCH_OPS = [
    {"name": "randn", "ary": 1, "zyx": lambda a: a.randn(2, 3), "dtypes": FLOAT_DTYPES, "shapes": [(2, 3)],
     "stochastic": True},
    {"name": "rand", "ary": 1, "zyx": lambda a: a.rand(2, 3), "dtypes": FLOAT_DTYPES, "shapes": [(2, 3)],
     "stochastic": True},
    {"name": "randint", "ary": 1, "zyx": lambda a: a.randint(2, 3), "dtypes": ALL_DTYPES, "shapes": [(2, 3)],
     "stochastic": True},
]

# ---- matmul / dot (float + int) ----
MM_OPS = [
    {"name": "dot_1d", "ary": 2, "zyx": lambda a, b: a.dot(b), "torch": lambda a, b: torch.dot(a, b),
     "dtypes": FLOAT_DTYPES + ("i32", "i64"), "gradable": True, "shapes": [(4,)]},
    {"name": "matmul_2d", "ary": 2, "zyx": lambda a, b: a.matmul(b), "torch": lambda a, b: torch.matmul(a, b),
     "dtypes": FLOAT_DTYPES + ("i32", "i64"), "gradable": True, "shapes": [(3, 4), (4, 5)]},
    {"name": "matmul_batched", "ary": 2, "zyx": lambda a, b: a.matmul(b), "torch": lambda a, b: torch.matmul(a, b),
     "dtypes": FLOAT_DTYPES + ("i32", "i64"), "gradable": True, "shapes": [(2, 3, 4), (2, 4, 5)]},
    {"name": "max_pool_2d", "ary": 1, "zyx": lambda a: a.max_pool((2, 2), (2, 2), (0, 0)),
     "torch": lambda a: torch.nn.functional.max_pool2d(a, 2, 2, 0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(1, 1, 6, 6)], "ranges": [(-5, 5)]},
    {"name": "pad_const", "ary": 1, "zyx": lambda a: a.pad((1, 1, 1, 1), 0.0),
     "torch": lambda a: torch.nn.functional.pad(a, (1, 1, 1, 1), value=0.0),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(2, 3)], "ranges": [(-5, 5)]},
]

# ---- conv (float) ----
CONV_OPS = [
    {"name": "conv_2d", "ary": 2, "zyx": lambda a, b: a.conv(b), "torch": lambda a, b: torch.nn.functional.conv2d(a, b),
     "dtypes": FLOAT_DTYPES, "gradable": True, "shapes": [(1, 2, 6, 6), (2, 2, 3, 3)]},
]


# ---------------------------------------------------------------- tests
def param(op_list):
    cases = [(s, d) for s in op_list for d in (s.get("dtypes") or ALL_DTYPES)]
    ids = [f"{s['name']}_{d}" for s, d in cases]
    return pytest.mark.parametrize("spec,dtype", cases, ids=ids)


@param(BINARY_OPS)
def test_binary(spec, dtype):
    run_spec(spec, dtype)


@param(UNARY_OPS)
def test_unary(spec, dtype):
    run_spec(spec, dtype)


@param(REDUCE_OPS)
def test_reduce(spec, dtype):
    run_spec(spec, dtype)


@param(ACTIV_OPS)
def test_acts(spec, dtype):
    run_spec(spec, dtype)


@param(LOSS_OPS)
def test_losses(spec, dtype):
    run_spec(spec, dtype)


@param(MOVE_OPS)
def test_move(spec, dtype):
    run_spec(spec, dtype)


@param(INDEX_OPS)
def test_index(spec, dtype):
    run_spec(spec, dtype)


@param(CREAT_OPS)
def test_creat(spec, dtype):
    run_spec(spec, dtype)


@param(STOCH_OPS)
def test_stoch(spec, dtype):
    run_spec(spec, dtype)


@param(MM_OPS)
def test_mm(spec, dtype):
    run_spec(spec, dtype)


@param(CONV_OPS)
def test_conv(spec, dtype):
    run_spec(spec, dtype)
