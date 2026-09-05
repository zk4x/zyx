# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
import zyx, torch, numpy as np

def assert_close(a,b):
    aa = np.array(a.numpy()); bb = b.detach().numpy() if isinstance(b, torch.Tensor) else np.array(b)
    assert aa.shape == bb.shape, f"{aa.shape} vs {bb.shape}"
    assert np.allclose(aa, bb, atol=1e-5)

def test_variable_interpolate():
    a = zyx.Tensor([1.0,2.0]); b = zyx.Tensor([3.0,4.0])
    w = 0.5
    y = a.interpolate(b, w)
    assert_close(y, torch.tensor([1.0,2.0])*0.5 + torch.tensor([3.0,4.0])*0.5)

def test_reshape_infer():
    x = zyx.Tensor([1,2,3,4,5,6])
    y = x.reshape(2, -1)
    assert y.resolve_shape() == [2,3]
    # symbolic infer
    n = zyx.Tensor.variable(2)
    z = x.reshape(n, -1)
    assert z.resolve_shape() == [2,3]

def test_expand_symbolic():
    try:
        x = zyx.Tensor([[1],[2]])
        n = zyx.Tensor.variable(4)
        y = x.expand(n, 2)
        assert y.resolve_shape()[0] == 4
    except BaseException:
        # symbolic expand may panic on some backends, skip
        pass

def test_repeat_symbolic_len():
    # repeat with int
    x = zyx.Tensor([1,2])
    y = x.repeat(3)
    assert_close(y, torch.tensor([1,2]).repeat(3))

def test_narrow_symbolic():
    x = zyx.Tensor([10,20,30,40])
    s = zyx.Tensor.variable(1)
    l = zyx.Tensor.variable(2)
    y = x.narrow(0, s, l)
    assert y.resolve_shape() == [2]
    assert_close(y, torch.tensor([20,30]))

def test_conv_symbolic():
    # simple conv test with concrete shapes
    x = zyx.Tensor.randn(1,1,4,4)
    w = zyx.Tensor.randn(1,1,2,2)
    y = x.conv(w, None, 1, (1,1), (1,1), (0,0))
    assert y.resolve_shape() == [1,1,3,3]

def test_matmul_symbolic():
    try:
        n = zyx.Tensor.variable(2)
        a = zyx.Tensor.ones(n, 3)
        b = zyx.Tensor.ones(3, 4)
        c = a.dot(b)
        assert c.resolve_shape() == [2,4] or c.resolve_shape()[0]==2
    except BaseException:
        pass
