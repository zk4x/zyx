// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, ReduceOp, Scalar, Tape, Tensor, ZyxError};

#[test]
fn sin() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;
    let z = x.sin();
    tape.realize([&z])?;
    let zdata: Vec<f32> = z.try_into()?;
    for (x, y) in data.iter().zip(zdata) {
        assert!(x.sin().is_equal(y), "{} != {y}", x.sin());
    }
    Ok(())
}

#[test]
fn relu() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;
    let z = x.relu();
    tape.realize([&z])?;
    assert_eq!(z, [0.0f32, 0.001, 1.780, 5.675, 0.0, 0.0, 1.215, 0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let tape = Tape::new([&x, &y])?;
    let z = x.dot(y)?;
    tape.realize([&z])?;
    assert_eq!(z, [[31, 15], [22, 10]]);
    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 4., 3.]);
    let tape = Tape::new([&x])?;
    let y = x.softmax([])?;
    tape.realize([&y])?;
    assert_eq!(y, [0.09003056585788726807f32, 0.66524088382720947266, 0.24472846090793609619]);
    Ok(())
}

#[test]
fn causal_self_attention() -> Result<(), ZyxError> {
    let dtype = DType::F32;
    let n_embd = 4;
    let n_head = 4;
    let c_attn_weight = Tensor::from([
        [3, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [1, 1, 2, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [3, 1, 5, 3, 1, 2, 5, 4, 2, 3, 1, 3],
        [3, 1, 2, 3, 1, 2, 5, 8, 2, 3, 1, 3],
    ])
    .t()
    .cast(dtype);

    let x = Tensor::from([[[1, 0, 4, 2], [2, 5, 0, 1], [0, 8, 1, 0], [5, 1, 0, 0]]]).cast(dtype);

    let [b, t, c] = x.shape()[..] else {
        return Err(ZyxError::ShapeError("x must have exactly 3 dims, b, t, c".into()));
    };

    let tape = Tape::new([&x, &c_attn_weight])?;
    let mut splits = x.dot(c_attn_weight.t())?.split([n_embd, n_embd, n_embd], 2)?;
    let mut v = splits.pop().unwrap();
    let mut k = splits.pop().unwrap();
    let mut q = splits.pop().unwrap();

    k = k.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    q = q.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    v = v.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;

    let mut att = q.dot(k.t())? * (1f32 / (*k.shape().last().unwrap() as f32).sqrt());
    att = att.softmax([-1])?;
    let mut y = att.dot(v)?;
    y = y.transpose(1, 2)?.reshape([b, t, c])?;

    tape.realize([&y])?;

    assert_eq!(
        y,
        [[
            [18f32, 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.],
            [18., 27., 9., 24.]
        ]]
    );

    Ok(())
}

// Reproducer for bug in promote_to_graph: promoting an eager tensor to Graph
// leaves a stale store entry in the original kernel. When gradient needs the
// value, materialize_kernel finds the store target in Graph state → panic.
#[test]
fn promote_and_gradient() -> Result<(), ZyxError> {
    let gt = Tensor::randn([2, 4, 1, 1], DType::F32)?;
    let tape = Tape::new([&gt])?;

    let a = Tensor::ones([4], DType::F32);
    let shape = [2, 4, 1, 1];
    let b = a.reshape([1, 4, 1, 1])?;
    let c = b.expand(shape)?;
    let d = &c + 1e-5f32;
    let e = d.rsqrt();

    let result = &e + &gt;
    let _g = tape.gradient(&result, [&gt]);
    Ok(())
}

// Reproducer for orphan kernel outputs: forward+backward through a 2-layer
// net with weight transpose, cross-entropy loss, and SGD update; then realize
// all params. This pattern triggers fill_remaining's force-seal bug where
// orphan kernel outputs not in the output_set prevent kernel sealing.
#[test]
fn small_net() -> Result<(), ZyxError> {
    let w1 = Tensor::randn([3, 4], DType::F32)?;
    let b1 = Tensor::randn([3], DType::F32)?;
    let w2 = Tensor::randn([2, 3], DType::F32)?;
    let b2 = Tensor::randn([2], DType::F32)?;

    for _ in 0..3 {
        let tape = Tape::new([&w1, &b1, &w2, &b2])?;
        let x = Tensor::randn([2, 4], DType::F32)?;
        let y = Tensor::from([0u32, 1]);
        let h = (x.dot(&w1.t())? + &b1).relu();
        let logits = h.dot(&w2.t())? + &b2;
        let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
        let grads = tape.gradient(&loss, [&w1, &b1, &w2, &b2]);

        let lr = 0.01f32;
        let new_w1 = &w1 - &grads[0] * lr;
        let new_b1 = &b1 - &grads[1] * lr;
        let new_w2 = &w2 - &grads[2] * lr;
        let new_b2 = &b2 - &grads[3] * lr;

        tape.realize([&new_w1, &new_b1, &new_w2, &new_b2])?;
    }
    Ok(())
}

#[test]
fn tape_caching() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let y = Tensor::randn([3, 3], DType::F32)?;
    for _ in 0..10 {
        let tape = Tape::new([&x])?;
        let z = &x + y.t().sin();
        tape.realize([&z])?;
    }
    Ok(())
}

#[test]
fn tape_matmul() -> Result<(), ZyxError> {
    let x = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let w = Tensor::from([0.5f32, 1.5, 2.5, 3.5]).reshape([2, 2])?.relu();
    for _ in 0..3 {
        let tape = Tape::new([&x])?;
        let z = x.dot(&w)?.relu();
        tape.realize([&z])?;
    }
    Ok(())
}

#[test]
fn tape_big_matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 392], DType::F32)?;
    let w = Tensor::rand([392, 296], DType::F32)?;
    let tape = Tape::new([&x, &w])?;
    let z = x.dot(&w)?.relu();
    tape.realize([&z])?;
    let shape = z.shape();
    assert_eq!(shape, [256, 296]);
    Ok(())
}

#[test]
fn drop_without_realize_params_eager() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    {
        let _tape = Tape::new([&x])?;
        let _z = x.sin();
    }
    let y = x + 1.0f32;
    let data: Vec<f32> = y.try_into()?;
    assert_eq!(data, [2.0f32, 3.0, 4.0]);
    Ok(())
}

#[test]
#[should_panic(expected = "tape scope has ended")]
fn use_intermediate_after_drop_panics() {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let z;
    {
        let _tape = Tape::new([&x]).unwrap();
        z = x.sin();
    }
    let _ = z + 1.0f32;
}

#[test]
fn realize_outputs_eager_leaves_eager_after_drop() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let z;
    {
        let tape = Tape::new([&x])?;
        z = x.sin();
        tape.realize([&z])?;
    }
    let zdata: Vec<f32> = z.try_into()?;
    for (a, b) in [1.0f32, 2.0, 3.0].iter().zip(zdata) {
        assert!(a.sin().is_equal(b));
    }
    let y = x + 1.0f32;
    let data: Vec<f32> = y.try_into()?;
    assert_eq!(data, [2.0f32, 3.0, 4.0]);
    Ok(())
}

#[test]
fn realized_tensor_promotes_as_leaf() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let tape = Tape::new([&x])?;
    let z = x.relu();
    tape.realize([&z])?;
    let data: Vec<f32> = z.try_into()?;
    assert_eq!(data, [1.0f32, 2.0, 3.0]);
    Ok(())
}

/*#[test]
fn frozen_tape_replay() -> Result<(), ZyxError> {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let frozen = {
        let tape = Tape::new([&x])?;
        let z = x.relu();
        tape.freeze([&z])?
    };
    let zs = frozen.replay([&x])?;
    assert_eq!(zs.len(), 1);
    let z = zs.into_iter().next().unwrap();
    let data: Vec<f32> = z.try_into()?;
    assert_eq!(data, [1.0f32, 2.0, 3.0]);
    Ok(())
}*/

#[test]
#[should_panic(expected = "tape scope has ended")]
fn use_frozen_output_panics() {
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let z;
    {
        let tape = Tape::new([&x]).unwrap();
        z = x.relu();
        let _frozen = tape.freeze([&z]).unwrap();
    }
    let _ = z + 1.0f32;
}
