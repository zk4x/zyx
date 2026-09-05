# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
import zyx

def test_sgd():
    opt = zyx.optim.SGD(learning_rate=0.01)
    assert opt is not None

def test_adam():
    opt = zyx.optim.Adam(learning_rate=0.001)
    assert opt is not None

def test_adamw():
    opt = zyx.optim.AdamW(learning_rate=0.001)
    assert opt is not None

def test_rmsprop():
    opt = zyx.optim.RMSprop(learning_rate=0.01)
    assert opt is not None

def test_sgd_update_with_tape():
    zyx.Tensor.manual_seed(0)
    a = zyx.Tensor.randn(2,2)
    b = zyx.Tensor.randn(2,2)
    tape = zyx.Tape()
    tape.add(a)
    tape.add(b)
    c = (a * b).sum()
    grads = tape.gradient(c, [a, b])
    tape.realize(grads + [c])
    assert c.resolve_shape() == [] or c.resolve_shape() == [1]
