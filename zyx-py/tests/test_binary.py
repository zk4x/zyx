import zyx
import torch
import numpy as np

def assert_close(a,b):
    aa = np.array(a.numpy())
    bb = b.detach().numpy()
    assert aa.shape == bb.shape
    assert np.allclose(aa, bb, atol=1e-5, rtol=1e-5)

def test_add():
    x = zyx.Tensor([1.0,2.0]); y = zyx.Tensor([3.0,4.0])
    assert_close(x + y, torch.tensor([1.0,2.0])+torch.tensor([3.0,4.0]))
    assert_close(x + 1.0, torch.tensor([1.0,2.0])+1.0)

def test_sub():
    x = zyx.Tensor([5.0,6.0]); y = zyx.Tensor([1.0,2.0])
    assert_close(x - y, torch.tensor([5.0,6.0])-torch.tensor([1.0,2.0]))

def test_mul():
    x = zyx.Tensor([2.0,3.0]); y = zyx.Tensor([4.0,5.0])
    assert_close(x * y, torch.tensor([2.0,3.0])*torch.tensor([4.0,5.0]))

def test_div():
    x = zyx.Tensor([6.0,8.0]); y = zyx.Tensor([2.0,4.0])
    assert_close(x / y, torch.tensor([6.0,8.0])/torch.tensor([2.0,4.0]))

def test_pow():
    x = zyx.Tensor([2.0,3.0]); y = zyx.Tensor([3.0,2.0])
    assert_close(x.pow(y), torch.pow(torch.tensor([2.0,3.0]), torch.tensor([3.0,2.0])))

def test_maximum_minimum():
    x = zyx.Tensor([1.0,5.0]); y = zyx.Tensor([3.0,2.0])
    assert_close(x.maximum(y), torch.maximum(torch.tensor([1.0,5.0]), torch.tensor([3.0,2.0])))
    assert_close(x.minimum(y), torch.minimum(torch.tensor([1.0,5.0]), torch.tensor([3.0,2.0])))

def test_cmplt_cmpgt():
    x = zyx.Tensor([1.0,5.0]); y = zyx.Tensor([3.0,2.0])
    assert_close(x.cmplt(y), torch.tensor([1.0,5.0]) < torch.tensor([3.0,2.0]))
    assert_close(x.cmpgt(y), torch.tensor([1.0,5.0]) > torch.tensor([3.0,2.0]))

def test_equal_ne():
    x = zyx.Tensor([1.0,2.0]); y = zyx.Tensor([1.0,3.0])
    assert_close(x.equal(y), torch.tensor([1.0,2.0]) == torch.tensor([1.0,3.0]))
    assert_close(x.ne(y), torch.tensor([1.0,2.0]) != torch.tensor([1.0,3.0]))

def test_logical():
    x = zyx.Tensor([True, False]); y = zyx.Tensor([True, True])
    assert_close(x.logical_and(y), torch.tensor([True, False]) & torch.tensor([True, True]))
    assert_close(x.logical_or(y), torch.tensor([True, False]) | torch.tensor([True, True]))

def test_where():
    cond = zyx.Tensor([True, False, True])
    a = zyx.Tensor([1.0,2.0,3.0]); b = zyx.Tensor([10.0,20.0,30.0])
    assert_close(cond.where_(a,b), torch.where(torch.tensor([True,False,True]), torch.tensor([1.0,2.0,3.0]), torch.tensor([10.0,20.0,30.0])))

def test_dot_matmul():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]]); y = zyx.Tensor([[5.0,6.0],[7.0,8.0]])
    assert_close(x.dot(y), torch.matmul(torch.tensor([[1.0,2.0],[3.0,4.0]]), torch.tensor([[5.0,6.0],[7.0,8.0]])))
    assert_close(x.matmul(y), torch.matmul(torch.tensor([[1.0,2.0],[3.0,4.0]]), torch.tensor([[5.0,6.0],[7.0,8.0]])))
    assert_close(x @ y, torch.tensor([[1.0,2.0],[3.0,4.0]]) @ torch.tensor([[5.0,6.0],[7.0,8.0]]))

def test_cast():
    x = zyx.Tensor([1.0,2.0])
    y = x.cast(zyx.DType.I32)
    assert y.dtype() == zyx.DType.I32

def test_interpolate():
    x = zyx.Tensor([1.0,2.0]); y = zyx.Tensor([3.0,4.0])
    z = x.interpolate(y, 0.5)
    # interpolate is linear: x*(1-w)+y*w
    expected = torch.tensor([1.0,2.0])*0.5 + torch.tensor([3.0,4.0])*0.5
    assert_close(z, expected)
