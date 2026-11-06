# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""NN layers + optimizers compared against torch. Forward + gradient (f32)."""
import numpy as np
import torch
import zyx

DT = zyx.DType.F32
TORCH = torch.float32
SEED = 2024


def _torch_grad(loss_fn, th, th_w):
    # loss_fn(a, w) -> scalar
    loss = loss_fn(th, th_w)
    if torch.is_tensor(loss) and loss.ndim != 0:
        loss = loss.sum()
    a_g = torch.autograd.grad(loss, th)[0]
    w_g = torch.autograd.grad(loss, th_w)[0]
    return loss.detach().cpu().numpy(), a_g.detach().cpu().numpy(), w_g.detach().cpu().numpy()


def _zyx_grad(loss_fn, xa, xw):
    tape = zyx.Tape()
    tape.add(xa)
    tape.add(xw)
    loss = loss_fn(xa, xw)
    if hasattr(loss, "ndim") and loss.ndim != 0:
        loss = loss.sum()
    ga = tape.gradient(loss, xa)
    gw = tape.gradient(loss, xw)
    tape.realize([ga, gw])
    return loss.numpy(), ga.numpy(), gw.numpy()


def _cmp(name, name_z, name_t, zt, tt, rtol=1e-5, atol=1e-6):
    z64, t64 = np.asarray(name_z, dtype=np.float64), np.asarray(name_t, dtype=np.float64)
    if z64.shape != t64.shape:
        raise AssertionError(f"[{name}] shape {z64.shape} vs {t64.shape}")
    if not np.allclose(z64, t64, rtol=rtol, atol=atol, equal_nan=True):
        raise AssertionError(f"[{name}] {name_t} mismatch; z={name_z} t={name_t}")


# ---------------------------------------------------------------- helpers
def _init(name, zyx_factory, torch_factory, zyx_layer, torch_layer, in_shape,
          out_dtype=TORCH, grad=True):
    # generate random weights + inputs
    def rand(*sz, scale=1.0, dtype=DT):
        r = np.random.default_rng(2024 + hash(name) % 9999)
        if dtype in (zyx.DType.F16, zyx.DType.F32):
            return zyx.Tensor(r.uniform(-scale, scale, sz).astype(np.float32), dtype=dtype)
        return zyx.Tensor(r.uniform(-scale, scale, sz).astype(np.float32), dtype=dtype)
    # weights for the zyx layer (via factory)
    return zyx_factory, torch_factory, zyx_layer, torch_layer


def test_linear():
    n, m = 4, 6
    # zyx
    xz = zyx.Tensor(np.random.default_rng(0).uniform(-2, 2, (1, n)).astype(np.float32), dtype=DT)
    zyx_w = zyx.Tensor(np.random.default_rng(1).uniform(-2, 2, (m, n)).astype(np.float32), dtype=DT)
    zyx_b = zyx.Tensor(np.random.default_rng(2).uniform(-2, 2, (m,)).astype(np.float32), dtype=DT)
    Lz = zyx.nn.Linear(n, m)
    Lz.weight = zyx_w
    Lz.bias = zyx_b
    # torch
    xt = torch.from_numpy(np.array(xz.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    wt = torch.from_numpy(np.array(zyx_w.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    bt = torch.from_numpy(np.array(zyx_b.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    Lt = torch.nn.Linear(n, m, bias=True)
    Lt.weight.data.copy_(wt)
    Lt.bias.data.copy_(bt)

    def f_z(zx, wz):  # placeholder; uses loaded Lz
        return zx
    # Compare forward (using Lz with loaded weights)
    yz = Lz(xz)
    yt = Lt(xt)
    _cmp("linear", yz.numpy(), yt.detach().cpu().numpy(), 1e-4, 1e-4)

    # Gradient: loss over aligned shapes, flows through L into x
    tape = zyx.Tape()
    tape.add(xz)
    yz_tape = Lz(xz)
    loss_z = (yz_tape * yz_tape).sum()
    gxz = tape.gradient(loss_z, [xz])
    tape.realize(gxz)

    def f_t(tx, tw, tb):
        return (Lt(tx) * Lt(tx)).sum()
    yt = f_t(xt, wt, bt)
    gx = torch.autograd.grad(yt, xt)
    _cmp("linear_grad", gxz[0].numpy(), gx[0].detach().cpu().numpy(), 1e-4, 1e-4)


def test_layernorm_parameters():
    """LayerNorm must expose weight/bias, or no optimizer can update it."""
    ln = zyx.nn.LayerNorm((6,), dtype=DT)
    assert hasattr(ln, "weight"), "LayerNorm exposes no weight"
    assert hasattr(ln, "bias"), "LayerNorm exposes no bias"
    assert tuple(ln.weight.shape) == (6,), ln.weight.shape
    assert tuple(ln.bias.shape) == (6,), ln.bias.shape


def test_layernorm_matches_torch():
    """LayerNorm must use torch's biased variance (divide by n), not n - 1."""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, -3.0, 0.5, 7.0, 2.0]], dtype=np.float32)
    zln = zyx.nn.LayerNorm((5,), dtype=DT)
    tln = torch.nn.LayerNorm((5,))
    with torch.no_grad():
        tln.weight.copy_(torch.from_numpy(np.array(zln.weight.numpy())))
        tln.bias.copy_(torch.from_numpy(np.array(zln.bias.numpy())))
    out_z = np.array(zln(zyx.Tensor(x.copy(), dtype=DT)).numpy())
    out_t = tln(torch.from_numpy(x.copy())).detach().numpy()
    np.testing.assert_allclose(out_z, out_t, rtol=1e-5, atol=1e-5,
                               err_msg="zyx LayerNorm differs from torch")


def test_optimizer_update_in_place():
    """Every optimizer assigns into the parameter: the caller's tensor changes.

    The parameter is the same tensor object across the step (that is the
    `assign` contract); a handle-replacing update would leave `p` untouched.
    """
    for opt_cls in (zyx.optim.SGD, zyx.optim.Adam, zyx.optim.AdamW, zyx.optim.RMSprop):
        p = zyx.Tensor(np.ones((3,), dtype=np.float32), dtype=DT)
        before = np.array(p.numpy())
        tape = zyx.Tape()
        tape.add(p)
        loss = (p * p).sum()
        (g,) = tape.gradient(loss, [p])
        opt_cls().update([p], [g])
        tape.realize([loss, p])
        after = np.array(p.numpy())
        assert not np.allclose(before, after), f"{opt_cls.__name__} did not update the parameter in place"


def test_optimizer_eager_update():
    """`update` outside a tape must work, not panic mid-materialize."""
    import subprocess
    import sys

    code = (
        "import zyx\n"
        "p = zyx.Tensor.from_vec([1.0, 2.0], (2,))\n"
        "zyx.optim.SGD(learning_rate=0.5).update([p], [zyx.Tensor.from_vec([1.0, 1.0], (2,))])\n"
        "assert p.item() == 0.0, p.item()\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, "eager optimizer update aborted:\n" + r.stderr[-1500:]


if __name__ == "__main__":
    test_linear()
