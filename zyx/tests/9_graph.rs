// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::result::Result;
use zyx::{DType, ReduceOp, Scalar, Tape, Tensor, ZyxError};

// ============================================================================
// Symbolic-dim / kernelize invariants
// ============================================================================

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
    assert_eq!(y.resolve_shape(), [2, 64]);
    // -1 inference over a symbolic dim: inferred = (2*64)/64 = 2.
    let z = y.reshape([-1])?;
    assert_eq!(z.resolve_shape(), [128]);
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

// ============================================================================
// shrink.rs — log-softmax + axis-sum loss with grad + param update
// ============================================================================

fn run(loss: impl FnOnce(&Tensor) -> Result<Tensor, ZyxError>) -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = loss(&logits)?;
    tape.realize([&loss])?;
    Ok(())
}

// A: manual log-softmax CE mean, no one-hot
#[test]
fn shrink_a_manual_ce() -> Result<(), ZyxError> {
    run(|l: &Tensor| {
        let m = l - &l.max_keepdim([1])?;
        let ls = &m - m.exp().sum_keepdim([1])?.ln();
        Ok((-&ls).sum_all())
    })
}

// B: keepdim reduce then axis reduce, no div/ln
#[test]
fn shrink_b_keepdim_then_axis() -> Result<(), ZyxError> {
    run(|l: &Tensor| {
        let s = l.clone().exp().sum_keepdim([1])?;
        (&s * l).sum([1]).map(|t| t.sum_all())
    })
}

// C: full-reduce after keepdim reduce
#[test]
fn shrink_c_full_reduce() -> Result<(), ZyxError> {
    run(|l: &Tensor| {
        let s = l.clone().exp().sum_keepdim([1])?;
        Ok((&s * l).sum_all())
    })
}

// D: exp-sum-keepdim minus input (backward touches sub of reduced)
#[test]
fn shrink_d_exp_sum_sub() -> Result<(), ZyxError> {
    run(|l: &Tensor| {
        let s = l.clone().exp().sum_keepdim([1])?;
        Ok(&s - l)
    })
}

// E: single keepdim sum
#[test]
fn shrink_e_single_keepdim_sum() -> Result<(), ZyxError> {
    run(|l: &Tensor| l.clone().exp().sum_keepdim([1]))
}

// ============================================================================
// shrink2.rs — manual CE with one-hot, full / loss-only / grad
// ============================================================================

// F1: manual CE with one-hot target, realize loss only
#[test]
fn f1_onehot_loss_only() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let y = Tensor::from([0u32, 1]);
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let ls = &m - m.exp().sum_keepdim([1])?.ln();
    let oh = y.unsqueeze(1)?.one_hot_along_dim(2, 1)?;
    let loss = (&-&ls * &oh).sum([1])?;
    tape.realize([&loss])?;
    Ok(())
}

// F2: manual CE with one-hot + gradient + realize param update
#[test]
fn f2_onehot_grad_realize() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let y = Tensor::from([0u32, 1]);
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let ls = &m - m.exp().sum_keepdim([1])?.ln();
    let oh = y.unsqueeze(1)?.one_hot_along_dim(2, 1)?;
    let loss = (-&ls * &oh).sum([1])?;
    let grads = tape.gradient(&loss.mean_all(), [&w]);
    let nw = &w + &grads[0] * -0.01f32;
    tape.realize([&nw])?;
    Ok(())
}

// G1: NO one-hot: use full-rank target via broadcasting scalar; grad + realize update
#[test]
fn g1_no_onehot_grad_realize() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let ls = &m - m.exp().sum_keepdim([1])?.ln();
    let loss = (-&ls).sum([1])?;
    let grads = tape.gradient(&loss.mean_all(), [&w]);
    let nw = &w + &grads[0] * -0.01f32;
    tape.realize([&nw])?;
    Ok(())
}

// H: original cross_entropy call, realize loss only (isolate gradient machinery)
#[test]
fn h_cross_entropy_loss_only() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let y = Tensor::from([0u32, 1]);
    let logits = x.dot(w.t())?;
    let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
    tape.realize([&loss])?;
    Ok(())
}

// ============================================================================
// shrink3.rs — grad and update with various reduce chains
// ============================================================================

// Common: log-softmax + axis-sum loss over logits from x.dot(w.t()).
fn build(w: &Tensor) -> Result<(Tape, Tensor, Tensor), ZyxError> {
    let tape = Tape::new([w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let ls = &m - m.exp().sum_keepdim([1])?.ln();
    let loss = (-&ls).sum([1])?;
    Ok((tape, w.clone(), loss))
}

// G1 baseline (known failing)
#[test]
fn g1_baseline() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let (tape, _, loss) = build(&w)?;
    let grads = tape.gradient(&loss.mean_all(), [&w]);
    let nw = &w + &grads[0] * -0.01f32;
    tape.realize([&nw])?;
    Ok(())
}

// I3: no mean_all
#[test]
fn i3_no_mean() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let (tape, _, loss) = build(&w)?;
    let grads = tape.gradient(&loss, [&w]);
    let nw = &w + &grads[0] * -0.01f32;
    tape.realize([&nw])?;
    Ok(())
}

// I2: realize only the gradient itself
#[test]
fn i2_grad_only() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let (tape, _, loss) = build(&w)?;
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&grads[0]])?;
    Ok(())
}

// I4: skip ln entirely: plain softmax numerator path
#[test]
fn i4_no_ln() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let loss = m.exp().sum([1])?;
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// I5: reduce-free loss, just to test grad+update with dot/sub/exp
#[test]
fn i5_reduce_free() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = logits.exp().sum_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// ============================================================================
// shrink4.rs — pure reduce / mean chains (no softmax machinery)
// ============================================================================

// J1: axis-sum then mean_all, plain negation instead of softmax
#[test]
fn j1_axis_sum_then_mean() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = (-&logits).sum([1])?.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// J2: only two sums, nothing else
#[test]
fn j2_two_sums() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = logits.sum([1])?.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// J3: single matmul input reduce-free loss with mean_all only
#[test]
fn j3_mean_only() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = logits.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// K1: keep two-reduce structure but make them same-shaped axes ([1] then [0])
#[test]
fn k1_sum_chain_axes() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let loss = logits.sum([1])?.sum([0])?;
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// ============================================================================
// shrink5.rs — g1 with axis flips and stripped softmax pieces
// ============================================================================

// L1: like g1 but two explicit full sums instead of mean_all
#[test]
fn l1_two_explicit_sums() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([1])?;
    let ls = &m - m.exp().sum_keepdim([1])?.ln();
    let loss = (-&ls).sum([1])?.sum_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// L2: g1 without max_keepdim step
#[test]
fn l2_no_max() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let ls = &logits - logits.clone().exp().sum_keepdim([1])?.ln();
    let loss = (-&ls).sum([1])?.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// L3: g1 but only ln-part reduced twice: mean over ln of summed exp
#[test]
fn l3_ln_only() -> Result<(), ZyxError> {
    let w = Tensor::randn([2, 3], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([2, 3], DType::F32)?;
    let logits = x.dot(w.t())?;
    let s = logits.exp().sum_keepdim([1])?;
    let loss = (-s.ln()).sum([1])?.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// L4: g1 but axis 0 everywhere (stresses assign/Contiguous symmetry for the
// in-place store contract when a leaf is consumed as a backward-read)
#[test]
fn l4_axis0() -> Result<(), ZyxError> {
    let w = Tensor::randn([3, 2], DType::F32)?;
    let tape = Tape::new([&w])?;
    let x = Tensor::randn([3, 2], DType::F32)?;
    let logits = x.dot(w.t())?;
    let m = &logits - logits.max_keepdim([0])?;
    let ls = &m - m.exp().sum_keepdim([0])?.ln();
    let loss = (-&ls).sum([0])?.mean_all();
    let grads = tape.gradient(&loss, [&w]);
    tape.realize([&(&w + &grads[0] * -0.01f32)])?;
    Ok(())
}

// ============================================================================
// 12_repro.rs — small_net reducers (cross-entropy mean loss with grad + update)
// ============================================================================

// Minimal reducer of small_net: one linear layer + cross-entropy mean loss.
#[test]
fn ce_one_layer() -> Result<(), ZyxError> {
    let w1 = Tensor::randn([3, 4], DType::F32)?;
    let b1 = Tensor::randn([3], DType::F32)?;

    let tape = Tape::new([&w1, &b1])?;
    let x = Tensor::randn([2, 4], DType::F32)?;
    let y = Tensor::from([0u32, 1]);
    let logits = x.dot(w1.t())? + &b1;
    let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
    let grads = tape.gradient(&loss, [&w1, &b1]);
    tape.realize([&(&w1 + &grads[0] * -0.01f32), &(&b1 + &grads[1] * -0.01f32)])?;
    Ok(())
}

// Even smaller: no bias, no relu; straight CE on raw logits.
#[test]
fn ce_bare() -> Result<(), ZyxError> {
    let w1 = Tensor::randn([3, 4], DType::F32)?;

    let tape = Tape::new([&w1])?;
    let x = Tensor::randn([2, 4], DType::F32)?;
    let y = Tensor::from([0u32, 1]);
    let logits = x.dot(w1.t())?;
    let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
    let grads = tape.gradient(&loss, [&w1]);
    tape.realize([&(&w1 + &grads[0] * -0.01f32)])?;
    Ok(())
}

// Suspect: iterations reuse/promote tensors across tape scopes.
#[test]
fn ce_iterated() -> Result<(), ZyxError> {
    let w1 = Tensor::randn([3, 4], DType::F32)?;
    let b1 = Tensor::randn([3], DType::F32)?;

    for _ in 0..3 {
        let tape = Tape::new([&w1, &b1])?;
        let x = Tensor::randn([2, 4], DType::F32)?;
        let y = Tensor::from([0u32, 1]);
        let logits = x.dot(w1.t())? + &b1;
        let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
        let grads = tape.gradient(&loss, [&w1, &b1]);
        let lr = 0.01f32;
        let nw = &w1 - &grads[0] * lr;
        let nb = &b1 - &grads[1] * lr;
        tape.realize([&nw, &nb])?;
    }
    Ok(())
}

// Same structure as small_net, tiny dims.
#[test]
fn ce_two_layer() -> Result<(), ZyxError> {
    let w1 = Tensor::randn([3, 4], DType::F32)?;
    let b1 = Tensor::randn([3], DType::F32)?;
    let w2 = Tensor::randn([2, 3], DType::F32)?;
    let b2 = Tensor::randn([2], DType::F32)?;

    for _ in 0..3 {
        let tape = Tape::new([&w1, &b1, &w2, &b2])?;
        let x = Tensor::randn([2, 4], DType::F32)?;
        let y = Tensor::from([0u32, 1]);
        let h = (x.dot(w1.t())? + &b1).relu();
        let logits = h.dot(w2.t())? + &b2;
        let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
        let grads = tape.gradient(&loss, [&w1, &b1, &w2, &b2]);
        let lr = 0.01f32;
        let nw1 = &w1 - &grads[0] * lr;
        let nb1 = &b1 - &grads[1] * lr;
        let nw2 = &w2 - &grads[2] * lr;
        let nb2 = &b2 - &grads[3] * lr;
        tape.realize([&nw1, &nb1, &nw2, &nb2])?;
    }
    Ok(())
}

#[test]
#[should_panic]
fn promote_dead_graph() {
    let w = Tensor::randn([2, 3], DType::F32).unwrap();
    let derived = {
        let tape = Tape::new([&w]).unwrap();
        let x = Tensor::randn([2, 3], DType::F32).unwrap();
        let out = x.dot(w.t()).unwrap(); // `out` is born in `Graph` state on tape's graph.
        // Drop tape without realizing → tape's graph goes dead; `out` is now
        // a `Graph { .. }` tensor whose graph is dead.
        drop(tape);
        out
    };
    // Use the dead-graph `derived` in a new tape. The implicit promote of
    // `derived` into the new tape is the call expected to hit line 1599.
    let tape2 = Tape::new([&w]).unwrap();
    let _ = &derived + 1.0f32; // triggers promote_to_graph on `derived`.
    tape2.realize([]).unwrap();
}
