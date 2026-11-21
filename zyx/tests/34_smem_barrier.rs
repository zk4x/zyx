// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Regression for `common_subexpression_elimination` and barriers.
//!
//! A barrier publishes the block's writes: a slot read before it and the same
//! slot read after a rewrite are two different values. CSE pools loads by
//! buffer and index, so without a purge at `Op::Barrier` the second read is
//! merged into the first and the kernel returns the pre-barrier value. This
//! kernel writes one local buffer, reads it, rewrites the same address, and
//! reads it again across a second barrier; the value it stores must be the
//! rewritten one.

use zyx::kernel::{Kernel, MemScope};
use zyx::{DType, Dev, Tensor, ZyxError};

/// Local slots (and block threads).
const N: i64 = 512;

#[test]
fn local_buffer_reused_across_barrier() -> Result<(), ZyxError> {
    let Some(dev) = Dev::all().into_iter().find(|d| !matches!(d, Dev::C | Dev::Host)) else {
        return Ok(());
    };
    let mut k = Kernel::new(dev);
    let out = k.param_mut(DType::F32);
    let t = k.local_range(0, u32::try_from(N).expect("block fits u32"));
    let smem = k.storage(DType::F32, MemScope::Local, N);

    let tv = k.cast(t, DType::F32);
    k.store(smem, tv, t);
    k.barrier();
    let first = k.load(smem, t);
    let one = k.const_val(1.0f32);
    let bumped = k.add(first, one);
    k.store(smem, bumped, t);
    k.barrier();
    let second = k.load(smem, t);
    k.store(out, second, t);

    k.verify();
    let got: Vec<f32> = k.compile()?.forward1(&[], vec![Tensor::from(N)])?.try_into()?;
    let want: Vec<f32> = (0..N).map(|i| i as f32 + 1.0).collect();
    for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g, e, "slot {i}: zyx {g} != {e}");
    }
    Ok(())
}
