// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tape, Tensor, ZyxError};

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
fn matmul_f32() -> Result<(), ZyxError> {
    let x = Tensor::from([[2f32, 4., 3.], [1., 5., 1.]]);
    let y = Tensor::from([[2f32, 4.], [3., 1.], [5., 1.]]);
    let tape = Tape::new([&x, &y])?;
    let z = x.dot(y)?;
    tape.realize([&z])?;
    assert_eq!(z, [[31f32, 15.], [22., 10.]]);
    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 4., 3.]);
    let tape = Tape::new([&x])?;
    let y = x.softmax([])?;
    tape.realize([&y])?;
    assert_eq!(y, [0.090_030_566_f32, 0.665_240_9, 0.244_728_46]);
    Ok(())
}

#[test]
fn self_attention() -> Result<(), ZyxError> {
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

    let [b, t, c] = x.dims::<3>()?;
    let (b, t, c) = (b.item::<i64>() as u64, t.item::<i64>() as u64, c.item::<i64>() as u64);

    let tape = Tape::new([&x, &c_attn_weight])?;
    let mut splits = x.dot(c_attn_weight.t())?.split([n_embd, n_embd, n_embd], 2)?;
    let mut v = splits.pop().unwrap();
    let mut k = splits.pop().unwrap();
    let mut q = splits.pop().unwrap();

    k = k.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    q = q.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;
    v = v.reshape([b, t, n_head, c / n_head])?.transpose(1, 2)?;

    let mut att = q.dot(k.t())? * (1f32 / ((c / n_head) as f32).sqrt());
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

// Gather through the full pipeline (kernelizer + fold_loops). The kernelizer
// fuses the arange produced by one_hot_along_dim into constants, making the
// mask's loop operand analyzable so the gather loop can fold to a direct gather.
#[test]
fn narrow_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    let tape = Tape::new([&x])?;
    let y = x.narrow(0, 1, 2)?;
    tape.realize([&y])?;
    assert_eq!(y, [[5, 6, 7, 8], [9, 10, 11, 12]]);
    Ok(())
}

#[test]
fn narrow_2() -> Result<(), ZyxError> {
    let x = Tensor::from([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]);
    let tape = Tape::new([&x])?;
    let y = x.narrow(0, 1, 2)?;
    let z = y.narrow(-1, 1, 2)?;
    tape.realize([&z])?;
    assert_eq!(z, [[6, 7], [10, 11]]);
    Ok(())
}

#[test]
fn gather() -> Result<(), ZyxError> {
    let x = Tensor::from([10, 20, 30, 40, 50]);
    let indices = Tensor::from([0u32, 2, 4, 1]);
    let tape = Tape::new([&x, &indices])?;
    let gathered = x.gather(0, &indices)?;
    tape.realize([&gathered])?;
    assert_eq!(gathered, [10, 30, 50, 20]);
    Ok(())
}

#[test]
fn assign_graph() -> Result<(), ZyxError> {
    let x = Tensor::from([0f32, 0f32, 0f32, 0f32]);
    let src = Tensor::from([1f32, 2f32, 3f32, 4f32]);
    let tape = Tape::new([&x])?;
    x.clone().assign(&src)?;
    tape.realize([&x])?;
    let out: Vec<f32> = x.try_into()?;
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    Ok(())
}

#[test]
fn assign_graph_computed_src() -> Result<(), ZyxError> {
    let x = Tensor::from([0f32, 0f32, 0f32, 0f32]);
    let a = Tensor::from([1f32, 2f32, 3f32, 4f32]);
    let tape = Tape::new([&x, &a])?;
    let src = (&a * 2.0f32) + 1.0f32;
    x.clone().assign(&src)?;
    tape.realize([&x])?;
    let out: Vec<f32> = x.try_into()?;
    assert_eq!(out, vec![3.0, 5.0, 7.0, 9.0]);
    Ok(())
}

#[test]
fn assign_graph_movement_dst() -> Result<(), ZyxError> {
    let base = Tensor::from([1.6f32, 0f32, 0f32, 2.3f32, 4.7f32]);
    let src = Tensor::from([7f32, 8f32, 9f32]);
    let tape = Tape::new([&base])?;
    let dst = base.slice(1..4)?;
    dst.assign(&src)?;
    tape.realize([&base])?;
    let out: Vec<f32> = base.try_into()?;
    assert_eq!(out, vec![1.6f32, 7.0, 8.0, 9.0, 4.7]);
    Ok(())
}

#[test]
fn assign_multiple_same_root() -> Result<(), ZyxError> {
    let base = Tensor::from([0f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
    let src1 = Tensor::from([1f32, 2f32, 3f32, 4f32, 5f32, 6f32]);
    let src2 = Tensor::from([7f32, 8f32]);
    let src3 = Tensor::from([9f32, 10f32]);
    let tape = Tape::new([&base])?;
    base.clone().assign(&src1)?;
    base.slice(0..2)?.assign(&src2)?;
    base.slice(4..6)?.assign(&src3)?;
    tape.realize([&base])?;
    let out: Vec<f32> = base.try_into()?;
    assert_eq!(out, vec![7.0, 8.0, 3.0, 4.0, 9.0, 10.0]);
    Ok(())
}

#[test]
fn assign_narrow_same_root() -> Result<(), ZyxError> {
    let base = Tensor::from([0f32, 0f32, 0f32, 0f32, 0f32, 0f32]);
    let src1 = Tensor::from([1f32, 2f32]);
    let src2 = Tensor::from([3f32, 4f32]);
    let tape = Tape::new([&base])?;
    base.narrow(0, 0, 2)?.assign(&src1)?;
    base.narrow(0, 2, 2)?.assign(&src2)?;
    tape.realize([&base])?;
    let out: Vec<f32> = base.try_into()?;
    assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0, 0.0, 0.0]);
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
fn big_matmul() -> Result<(), ZyxError> {
    let x = Tensor::rand([256, 392], DType::F32)?;
    let w = Tensor::rand([392, 296], DType::F32)?;
    let tape = Tape::new([&x, &w])?;
    let z = x.dot(&w)?.relu();
    tape.realize([&z])?;
    let shape = z.resolve_shape();
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

// A tensor promoted into a tape as a leaf must stay alive for the whole tape
// scope even after the caller drops its handle: the tape holds a reference.
// Without that, `drop(x)` releases the underlying tensor and `realize` then
// reads a freed tensor.
#[test]
fn drop_leaf_handle_before_realize() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 1]).expand([2, 3])?;
    let tape = Tape::new([&x])?;
    // `x + 2` borrows then `drop(x)` drops the `x` handle while `x` is a tape
    // leaf; with no tape-held reference the leaf tensor is freed and `realize`
    // reads a dangling tensor.
    let z = &x + 2;
    drop(x);
    tape.realize([&z])?;
    println!("{z}");
    Ok(())
}
