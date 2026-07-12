// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, Scalar, Tape, Tensor, ZyxError};

#[test]
fn sin() -> Result<(), ZyxError> {
    let data: [f32; 10] = [-3.285, 0.001, 1.780, 5.675, -8.521, -0.456, 1.215, -3.474, -4.128, -7.657];
    let x = Tensor::from(data);
    let tape = Tape::new();
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
    let tape = Tape::new();
    let z = x.relu();
    tape.realize([&z])?;
    assert_eq!(z, [0.0f32, 0.001, 1.780, 5.675, 0.0, 0.0, 1.215, 0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn matmul() -> Result<(), ZyxError> {
    let x = Tensor::from([[2, 4, 3], [1, 5, 1]]);
    let y = Tensor::from([[2, 4], [3, 1], [5, 1]]);
    let tape = Tape::new();
    let z = x.dot(y)?;
    tape.realize([&z])?;
    assert_eq!(z, [[31, 15], [22, 10]]);
    Ok(())
}

#[test]
fn softmax() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 4., 3.]);
    let tape = Tape::new();
    let y = x.softmax([])?;
    tape.realize([&y])?;
    assert_eq!(y, [0.09003056585788726807f32, 0.66524088382720947266, 0.24472846090793609619,]);
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

    let tape = Tape::new();
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
