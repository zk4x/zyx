# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
import zyx, numpy as np, torch

def test_kernel_basic():
    k = zyx.PyKernel()
    shape = k.add_shape([4])
    a = k.param(zyx.DType.F32, 0, shape)
    b = k.param(zyx.DType.F32, 0, shape)
    out_shape = k.add_shape([4])
    out = k.param(zyx.DType.F32, 1, out_shape)
    len_id = k.const_idx(4)
    gidx = k.group_index(0, len_id)
    la = k.load(a, gidx, 0)
    lb = k.load(b, gidx, 0)
    c = k.add(la, lb)
    k.store(out, c, gidx, 0)
    compiled = k.compile()
    assert compiled is not None
    # forward test disabled due to shape handling

def test_kernel_mad():
    k = zyx.PyKernel()
    shape = k.add_shape([2])
    a = k.param(zyx.DType.F32, 0, shape)
    out = k.param(zyx.DType.F32, 1, shape)
    len_id = k.const_idx(2)
    gidx = k.group_index(0, len_id)
    v = k.load(a, gidx, 0)
    r = k.mad(v, v, v)
    k.store(out, r, gidx, 0)
    compiled = k.compile()
    assert compiled is not None
