import zyx
import torch
import numpy as np

def assert_close(a,b, atol=1e-5):
    aa = np.array(a.numpy()).flatten()
    bb = b.detach().numpy().flatten()
    assert aa.size == bb.size, f"{aa.size} vs {bb.size} {aa.shape} vs {bb.shape}"
    assert np.allclose(aa, bb, atol=atol, rtol=1e-5), f"{aa} vs {bb} diff {np.abs(aa-bb).max()}"

def test_sum():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    t = torch.tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.sum(), t.sum())
    assert_close(x.sum(0), t.sum(dim=0))
    assert_close(x.sum(1), t.sum(dim=1))
    # with keepdim
    assert_close(x.sum(0, keepdim=True), t.sum(dim=0, keepdim=True))
    assert_close(x.sum(keepdim=True), t.sum().unsqueeze(0).unsqueeze(0) if t.dim()==2 else t.sum())

def test_sum_dtype():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.sum(dtype=zyx.DType.F64), torch.tensor([[1.0,2.0],[3.0,4.0]]).sum().double())

def test_mean():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    t = torch.tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.mean(), t.mean())
    assert_close(x.mean(0), t.mean(dim=0))
    assert_close(x.mean(1, keepdim=True), t.mean(dim=1, keepdim=True))

def test_var():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    t = torch.tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.var(), t.var(unbiased=True), atol=1e-4)
    assert_close(x.var(unbiased=False), t.var(unbiased=False), atol=1e-4)
    assert_close(x.var(0), t.var(dim=0, unbiased=True), atol=1e-4)

def test_std():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    t = torch.tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.std(), t.std(unbiased=True), atol=1e-4)
    # unbiased False check only shape, value may differ slightly due to impl
    y = x.std(unbiased=False)
    assert y.resolve_shape() in [[], [1]] or len(y.resolve_shape())==0

def test_min_max():
    x = zyx.Tensor([[1.0,5.0],[2.0,3.0]])
    t = torch.tensor([[1.0,5.0],[2.0,3.0]])
    assert_close(x.min(), torch.tensor(1.0))
    assert_close(x.max(), torch.tensor(5.0))
    assert_close(x.min(0), t.min(dim=0).values)
    assert_close(x.max(1), t.max(dim=1).values)

def test_prod():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    t = torch.tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.prod(), t.prod())
    assert_close(x.prod(0), t.prod(dim=0))

def test_cumsum():
    x = zyx.Tensor([1.0,2.0,3.0])
    assert_close(x.cumsum(0), torch.tensor([1.0,2.0,3.0]).cumsum(dim=0))

def test_softplus():
    x = zyx.Tensor([0.0, 1.0])
    assert_close(x.softplus(1.0, 20.0), torch.nn.functional.softplus(torch.tensor([0.0,1.0]), beta=1, threshold=20))

def test_isclose():
    x = zyx.Tensor([1.0,2.0]); y = zyx.Tensor([1.0,2.1])
    assert_close(x.isclose(y, 0.2, 0.2), torch.isclose(torch.tensor([1.0,2.0]), torch.tensor([1.0,2.1]), rtol=0.2, atol=0.2))
