# Copyright (C) 2025 zk4x
# SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
"""Forward + gradient comparison of every elementwise/reduction/movement/index/
creation/matmul op in zyx-py against torch, across f16/f32/i32/i64/u32."""
import pytest

from op_check import (
    run_spec, ALL_DTYPES,
    BINARY_OPS, UNARY_OPS, REDUCE_OPS, ACTIV_OPS, LOSS_OPS,
    MOVE_OPS, INDEX_OPS, CREAT_OPS, STOCH_OPS, MM_OPS, CONV_OPS,
)


def _param(op_list):
    cases = [(s, d) for s in op_list for d in (s.get("dtypes") or ALL_DTYPES)]
    ids = [f"{s['name']}_{d}" for s, _ in cases]
    return pytest.mark.parametrize("spec,dtype", cases, ids=ids)


@param(BINARY_OPS)
def test_binary(spec, dtype):
    run_spec(spec, dtype)


@param(UNARY_OPS)
def test_unary(spec, dtype):
    run_spec(spec, dtype)


@param(REDUCE_OPS)
def test_reduce(spec, dtype):
    run_spec(spec, dtype)


@param(ACTIV_OPS)
def test_acts(spec, dtype):
    run_spec(spec, dtype)


@param(LOSS_OPS)
def test_losses(spec, dtype):
    run_spec(spec, dtype)


@param(MOVE_OPS)
def test_move(spec, dtype):
    run_spec(spec, dtype)


@param(INDEX_OPS)
def test_index(spec, dtype):
    run_spec(spec, dtype)


@param(CREAT_OPS)
def test_creat(spec, dtype):
    run_spec(spec, dtype)


@param(STOCH_OPS)
def test_stoch(spec, dtype):
    run_spec(spec, dtype)


@param(MM_OPS)
def test_mm(spec, dtype):
    run_spec(spec, dtype)


@param(CONV_OPS)
def test_conv(spec, dtype):
    run_spec(spec, dtype)
