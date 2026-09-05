# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
import zyx, torch, numpy as np

def assert_close(a,b):
    aa = np.array(a.numpy()).flatten(); bb = b.detach().numpy().flatten()
    assert aa.size == bb.size, f"{aa.size} vs {bb.size}"
    assert np.allclose(aa, bb, atol=1e-5, rtol=1e-5)

def test_reshape():
    x = zyx.Tensor([1,2,3,4])
    assert_close(x.reshape(2,2), torch.tensor([1,2,3,4]).reshape(2,2))
    assert_close(x.reshape(2, -1), torch.tensor([1,2,3,4]).reshape(2, -1))
    # symbolic reshape
    n = zyx.Tensor.variable(2)
    y = zyx.Tensor([1,2,3,4])
    # reshape with variable not directly testable via torch, just check shape
    z = y.reshape(n, 2)
    assert z.resolve_shape() == [2,2]

def test_permute():
    x = zyx.Tensor([[1,2,3],[4,5,6]])
    assert_close(x.permute(1,0), torch.tensor([[1,2,3],[4,5,6]]).permute(1,0))

def test_transpose():
    x = zyx.Tensor([[1.0,2.0],[3.0,4.0]])
    assert_close(x.transpose(0,1), torch.tensor([[1.0,2.0],[3.0,4.0]]).t())
    assert_close(x.t(), torch.tensor([[1.0,2.0],[3.0,4.0]]).t())

def test_expand():
    x = zyx.Tensor([[1],[2]])
    assert_close(x.expand(2,2), torch.tensor([[1],[2]]).expand(2,2))
    # with -1
    y = x.expand(2, -1)
    assert y.resolve_shape() == [2,1]

def test_squeeze_unsqueeze():
    x = zyx.Tensor([1.0,2.0]).reshape(1,1,2)
    assert_close(x.squeeze(), torch.tensor([[[1.0,2.0]]]).squeeze())
    assert_close(x.squeeze(0), torch.tensor([[[1.0,2.0]]]).squeeze(0))
    y = zyx.Tensor([1.0,2.0])
    assert_close(y.unsqueeze(0), torch.tensor([1.0,2.0]).unsqueeze(0))

def test_flip():
    x = zyx.Tensor([1,2,3])
    assert_close(x.flip(0), torch.flip(torch.tensor([1,2,3]), dims=[0]))

def test_pad():
    x = zyx.Tensor([1,2,3])
    y = x.pad_zeros([1,1])
    assert_close(y, torch.nn.functional.pad(torch.tensor([1,2,3]), (1,1)))

def test_flatten():
    x = zyx.Tensor([1,2,3,4]).reshape(1,2,2)
    assert_close(x.flatten(0,1), torch.tensor([[[1,2],[3,4]]]).flatten(0,1))

def test_cat():
    a = zyx.Tensor([[1,2],[3,4]]); b = zyx.Tensor([[5,6]])
    assert_close(zyx.Tensor.cat([a,b], 0), torch.cat([torch.tensor([[1,2],[3,4]]), torch.tensor([[5,6]])], dim=0))

def test_stack():
    a = zyx.Tensor([1,2]); b = zyx.Tensor([3,4])
    assert_close(zyx.Tensor.stack([a,b]), torch.stack([torch.tensor([1,2]), torch.tensor([3,4])]))

def test_repeat():
    x = zyx.Tensor([1,2])
    assert_close(x.repeat(2), torch.tensor([1,2]).repeat(2))

def test_split():
    x = zyx.Tensor([1,2,3,4,5,6])
    parts = x.split((2,2,2), 0)
    assert len(parts)==3
    assert_close(parts[0], torch.tensor([1,2]))

def test_one_hot():
    x = zyx.Tensor([0,1,2])
    assert_close(x.one_hot(3), torch.nn.functional.one_hot(torch.tensor([0,1,2]), 3).float())

def test_shrink_slice():
    x = zyx.Tensor([[1,2,3],[4,5,6]])
    y = x[0:1, 1:3]
    assert_close(y, torch.tensor([[1,2,3],[4,5,6]])[0:1,1:3])
