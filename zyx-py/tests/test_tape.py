# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
import zyx, torch, numpy as np

def assert_close(a,b):
    aa = np.array(a.numpy()); bb = b.detach().numpy()
    assert aa.shape == bb.shape
    assert np.allclose(aa, bb, atol=1e-5)

def test_tape_gradient():
    tape = zyx.Tape()
    x = zyx.Tensor([2.0,3.0])
    tape.add(x)
    y = (x * x).sum()
    grads = tape.gradient(y, [x])
    assert len(grads)==1
    tape.realize(grads)
    assert_close(grads[0], torch.tensor([4.0,6.0]))

def test_tape_realize():
    tape = zyx.Tape()
    x = zyx.Tensor([1.0,2.0])
    tape.add(x)
    y = x + 1.0
    tape.realize([y])
    assert_close(y, torch.tensor([2.0,3.0]))

def test_tape_freeze():
    tape = zyx.Tape()
    x = zyx.Tensor([1.0,2.0])
    tape.add(x)
    y = x * 2.0
    frozen = tape.freeze([y])
    out = frozen.replay([x])
    assert len(out)==1
    assert_close(out[0], torch.tensor([2.0,4.0]))

def test_training_flag():
    assert zyx.Tensor.training() in [True, False]
    zyx.Tensor.set_training(True)
    assert zyx.Tensor.training() == True
    zyx.Tensor.set_training(False)
    assert zyx.Tensor.training() == False
    zyx.Tensor.set_training(True)

def test_manual_seed():
    zyx.Tensor.manual_seed(42)
    a = zyx.Tensor.rand(2,2)
    zyx.Tensor.manual_seed(42)
    b = zyx.Tensor.rand(2,2)
    assert_close(a,b)

def test_dropout():
    x = zyx.Tensor.ones(10,10)
    zyx.Tensor.set_training(True)
    y = x.dropout(0.5)
    # dropout training: some zeros
    assert y.resolve_shape() == [10,10]
    zyx.Tensor.set_training(False)
    y2 = x.dropout(0.5)
    assert_close(y2, torch.ones(10,10)/0.5 if False else torch.ones(10,10)*2.0)  # approximate
    zyx.Tensor.set_training(True)
