// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tensor, ZyxError};

// Reproduces: `index_select` (via `randint` indices) produces a `-1` dimension
// (printed as `r4294967295` / ~4.29×10⁹) on a `Param` shape pre-linearize. After
// linearization that becomes a negative loop / group-index length, which hangs
// the kernel (4.29×10⁹-element loop). `kernel::verify` now catches it loudly
// (panics on a resolvable negative loop/group length) instead of hanging.
//
// With the bug present this test panics in `verify`; once the `-1` dimension is
// eliminated upstream, it completes.
#[test]
fn index_select_randint_negative_dim() -> Result<(), ZyxError> {
    let x = Tensor::randn([60000, 784], DType::F32)?;
    let idx = Tensor::randint([128], 0..60000)?;
    let out = x.index_select(0, &idx)?;
    // Force execution: triggers linearization + verify.
    let _ = out.sum([0, 1])?.item::<f32>();
    Ok(())
}

// Reshape `-1` inference must build a symbolic `numel / product(others)` dim
// expression, never read concrete values at construction time (a variable slot
// may be unbound and would launder garbage into the graph as a const dim).
#[test]
fn reshape_infer_symbolic_dim() -> Result<(), ZyxError> {
    let x = Tensor::randn([4, 64], DType::F32)?;
    // Narrow with variable bounds: output dim is symbolic (variable-backed).
    let start = Tensor::variable(0i64);
    let len = Tensor::variable(2i64);
    let y = x.narrow(0, start, len)?;
    assert_eq!(y.shape(), [2, 64]);
    // -1 inference over a symbolic dim: inferred = (2*64)/64 = 2.
    let z = y.reshape([-1])?;
    assert_eq!(z.shape(), [128]);
    let _ = z.item::<f32>();
    Ok(())
}

// KV-cache pattern: assign into a narrowed region of a preallocated cache,
// with variable-backed bounds and a squeeze+transpose source chain (llama).
#[test]
fn kv_cache_narrow_assign_symbolic() -> Result<(), ZyxError> {
    let cache = Tensor::zeros([1024, 8, 128], DType::F32);
    let start = Tensor::variable(0i64);
    let len = Tensor::variable(2i64);
    println!("len={}", len.id());
    // len is the shared dynamic dim: k's shape AND the narrow use the same
    // dim tensor, so the assign's provability check passes.
    let k = Tensor::randn([Tensor::from(1), Tensor::from(8), len.clone(), Tensor::from(128)], DType::F32)?;
    let k_assign = k.squeeze([0]).transpose(0, 1).unwrap();
    // Invalid: the cache's zeros kernel is a pure const fill with no backing
    // buffer — assign through a view of it would write into an orphaned copy.
    // The error tells the user to materialize with `.contiguous()` first.
    assert!(
        matches!(cache.narrow(0, start.clone(), len.clone())?.assign(&k_assign), Err(ZyxError::ShapeError(_))),
        "assign through a view of an unmaterialized base must be rejected"
    );
    // Happy path: materialize the base, then assign writes through the view
    // into the cache itself. The len variable is the SHARED dim tensor of
    // both k's shape and the narrow — assign requires provably equal shapes
    // (same dim tensor in both operands, or concrete in both).
    let cache = cache.contiguous()?;
    cache.narrow(0, start.clone(), len.clone())?.assign(&k_assign).unwrap();
    // The assigned slice must land inside the cache: sum differs from zero and
    // equals the source region's sum (mod layout changes from the transpose).
    assert_ne!(cache.sum_all().item::<f32>(), 0.0);
    assert!(cache.narrow(0, start.clone(), len.clone())?.sum_all().item::<f32>().is_equal(k_assign.sum_all().item::<f32>()));
    Ok(())
}
