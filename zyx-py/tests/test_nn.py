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
    n, m, k = 4, 6, 8
    # zyx
    xz = zyx.Tensor(np.random.default_rng(0).uniform(-2, 2, (1, n)).astype(np.float32), dtype=DT)
    zyx_w = zyx.Tensor(np.random.default_rng(1).uniform(-2, 2, (m, k)).astype(np.float32), dtype=DT)
    zyx_b = zyx.Tensor(np.random.default_rng(2).uniform(-2, 2, (m,)).astype(np.float32), dtype=DT)
    Lz = zyx.nn.Linear(m, n)
    Lz.load(zyx_w, zyx_b)  # weight shape (out, in)? adjust
    # torch
    xt = torch.from_numpy(np.array(xz.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    wt = torch.from_numpy(np.array(zyx_w.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    bt = torch.from_numpy(np.array(zyx_b.numpy(), dtype=np.float32)).clone().requires_grad_(True)
    Lt = torch.nn.Linear(m, n, bias=True)
    Lt.weight.data.copy_(wt)
    Lt.bias.data.copy_(bt)

    def f_z(zx, wz):  # placeholder; uses loaded Lz
        return zx
    # Compare forward (using Lz with loaded weights)
    yz = Lz(xz)
    yt = Lt(xt)
    _cmp("linear", yz.numpy(), yt.detach().cpu().numpy(), 1e-4, 1e-4)

    # Gradient: build a small tape loss = (L(x) * x).sum()
    # forward through Lz inside a tape
    tape = zyx.Tape()
    tape.add(xz)
    yz_tape = Lz(xz)
    loss_z = (yz_tape * xz).sum()
    gxz = tape.gradient(loss_z, xz)
    tape.realize(gxz)

    def f_t(tx, tw, tb):
        return (Lt(tx) * tx).sum()
    yt = f_t(xt, wt, bt)
    gx = torch.autograd.grad(yt, xt)
    _cmp("linear_grad", gxz.numpy(), gx.detach().cpu().numpy(), 1e-4, 1e-4)


if __name__ == "__main__":
    test_linear()
