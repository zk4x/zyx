import zyx
import torch
import numpy as np

def assert_close(zyx_t, torch_t, atol=1e-5, rtol=1e-5):
    a = np.array(zyx_t.numpy())
    b = torch_t.detach().numpy() if isinstance(torch_t, torch.Tensor) else np.array(torch_t)
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    assert np.allclose(a, b, atol=atol, rtol=rtol), f"values differ max {np.abs(a-b).max()}"

def test_zeros():
    t = zyx.Tensor.zeros(2, 3)
    assert t.resolve_shape() == [2, 3]
    assert_close(t, torch.zeros(2, 3))

def test_ones():
    t = zyx.Tensor.ones(2, 3)
    assert_close(t, torch.ones(2, 3))

def test_full():
    t = zyx.Tensor.full(2, 2, a=3.14)
    assert_close(t, torch.full((2, 2), 3.14))

def test_eye():
    t = zyx.Tensor.eye(3)
    assert_close(t, torch.eye(3))

def test_arange():
    t = zyx.Tensor.arange(0, 5, 1)
    assert_close(t, torch.arange(0, 5, 1))

def test_rand():
    zyx.Tensor.manual_seed(0)
    t = zyx.Tensor.rand(2, 3)
    assert t.resolve_shape() == [2, 3]

def test_randn():
    zyx.Tensor.manual_seed(1)
    t = zyx.Tensor.randn(2, 3)
    assert t.resolve_shape() == [2, 3]

def test_zeros_like():
    a = zyx.Tensor([1.0, 2.0, 3.0])
    z = zyx.Tensor.zeros_like(a)
    assert_close(z, torch.zeros_like(torch.tensor([1.0,2.0,3.0])))

def test_ones_like():
    a = zyx.Tensor([1.0, 2.0])
    o = zyx.Tensor.ones_like(a)
    assert_close(o, torch.ones_like(torch.tensor([1.0,2.0])))

def test_from_vec():
    t = zyx.Tensor.from_vec([1,2,3,4], [2,2])
    assert_close(t, torch.tensor([[1,2],[3,4]]))

def test_variable_and_symbolic_shape():
    n = zyx.Tensor.variable(4)
    t = zyx.Tensor.zeros(n, 3)
    # symbolic shape: first dim is variable
    shapes = t.shape()
    assert len(shapes) == 2
    assert shapes[0].item() == 4
    assert t.resolve_shape() == [4, 3]

def test_rand_symbolic():
    n = zyx.Tensor.variable(2)
    t = zyx.Tensor.rand(n, 3)
    assert t.resolve_shape() == [2, 3]

def test_reshape_symbolic():
    n = zyx.Tensor.variable(2)
    t = zyx.Tensor.ones(n, 4)
    reshaped = t.reshape(8)
    assert reshaped.resolve_shape() == [8]
    reshaped2 = t.reshape(n, 2, 2)
    assert reshaped2.resolve_shape() == [2, 2, 2]
