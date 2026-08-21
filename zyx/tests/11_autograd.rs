// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, ReduceOp, Tape, Tensor, ZyxError};

#[test]
fn grad_relu_1() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 0, -1]);
    let tape = Tape::new([&x])?;
    let z = x.relu();
    tape.realize([&z])?;
    assert_eq!(z, [3, 0, 0]);
    //println!("{z}");
    Ok(())
}

#[test]
fn grad_relu_2() -> Result<(), ZyxError> {
    let x = Tensor::from([3, -2, 0]);
    let tape = Tape::new([&x])?;
    let z = x.relu();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [1, 0, 0]);
    Ok(())
}

#[test]
fn grad_reciprocal() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let tape = Tape::new([&x])?;
    let z = x.reciprocal();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [-0.111_111_11_f32, -0.25, -0.0625]);
    Ok(())
}

#[test]
fn grad_exp2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 0.5];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = x.exp2();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    let expected: Vec<_> = data.iter().map(|&x| 2f32.powf(x) * std::f32::consts::LN_2).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_reciprocal_2() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::F64).any() {
        return Ok(());
    }
    // Input tensor
    let x = Tensor::from([2.0, -1.0, 0.5]);

    // Create gradient tape
    let tape = Tape::new([&x])?;

    // Forward pass: y = 1 / x
    let y = x.reciprocal();

    // Compute gradients
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    // Expected gradients: dy/dx = -1 / x^2
    let expected = [-1.0 / 4.0, -1.0 / 1.0, -1.0 / 0.25]; // [-0.25, -1.0, -4.0]

    // Compare
    assert_eq!(x_grad, expected);

    Ok(())
}

#[test]
fn grad_contiguous() -> Result<(), ZyxError> {
    // Graph path: contiguous is a fusion break inside a tape scope, and the
    // gradient flows through it unchanged (d(y)/dx with y = contiguous(x) is 1).
    let x = Tensor::from([1.0, 2.0, 3.0]);
    let tape = Tape::new([&x])?;
    let z = x.contiguous()?.relu();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [1.0, 1.0, 1.0]);
    Ok(())
}

#[test]
fn grad_floor() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::F64).any() {
        return Ok(());
    }
    let x = Tensor::from([0.5, 1.5, -0.5, -1.5, 0.1, 0.9, -0.1, -0.9, 2.3, -2.3]);
    let tape = Tape::new([&x])?;
    let y = x.floor();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, vec![0.0; 10]);
    Ok(())
}

#[test]
fn grad_trunc() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::F64).any() {
        return Ok(());
    }
    let x = Tensor::from([0.5, 1.5, -0.5, -1.5, 0.1, 0.9, -0.1, -0.9, 2.3, -2.3]);
    let tape = Tape::new([&x])?;
    let y = x.trunc();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, vec![0.0; 10]);
    Ok(())
}

#[test]
fn grad_pow_2() -> Result<(), ZyxError> {
    // Input tensors
    let x = Tensor::from([2.0f32, 3.0, 4.0]);
    let y = Tensor::from([3.0f32, 2.0, 0.5]);

    // Forward pass: z = x ^ y
    let tape = Tape::new([&x, &y])?;
    let z = x.pow(&y)?;

    // Compute gradients
    let mut grads = tape.gradient(&z, [&x, &y]);
    let x_grad = grads.remove(0);
    let y_grad = grads.remove(0);
    tape.realize([&x_grad, &y_grad])?;

    // Expected gradients
    // dz/dx = y * x^(y-1)
    let expected_x = [
        3.0 * 2.0f32.powf(2.0),  // 3 * 2^(3-1) = 3 * 4 = 12
        2.0 * 3.0f32.powf(1.0),  // 2 * 3^(2-1) = 2 * 3 = 6
        0.5 * 4.0f32.powf(-0.5), // 0.5 * 4^(-0.5) = 0.5 * 0.5 = 0.25
    ];

    // dz/dy = x^y * ln(x)
    let expected_y = [
        2.0f32.powf(3f32) * 2.0f32.ln(), // 8 * ln(2)
        3.0f32.powf(2f32) * 3.0f32.ln(), // 9 * ln(3)
        4.0f32.powf(0.5) * 4.0f32.ln(),  // 2 * ln(4)
    ];

    assert_eq!(x_grad, expected_x);
    assert_eq!(y_grad, expected_y);

    Ok(())
}

#[test]
fn grad_pow_3() -> Result<(), ZyxError> {
    if !Tensor::dtype_capability(DType::F64).log2() {
        return Ok(());
    }
    // Use non-round numbers to expose log2 -> ln approximation errors
    let x = Tensor::from([1.5, 2.3, 5.7]);
    let y = Tensor::from([0.7, 1.2, 0.3]);

    // Forward pass: z = x ^ y
    let tape = Tape::new([&x, &y])?;
    let z = x.pow(&y)?;

    // Compute gradients
    let mut grads = tape.gradient(&z, [&x, &y]);
    let x_grad = grads.remove(0);
    let y_grad = grads.remove(0);
    tape.realize([&x_grad, &y_grad])?;

    // Convert tensors to Vec<f64> for comparison
    let x_vec: Vec<f64> = x_grad.clone().try_into().unwrap();
    let y_vec: Vec<f64> = y_grad.clone().try_into().unwrap();
    let x_val: Vec<f64> = x.clone().try_into().unwrap();
    let y_val: Vec<f64> = y.clone().try_into().unwrap();

    // Expected gradients
    let expected_x_vec: Vec<f64> =
        x_val.iter().zip(y_val.iter()).map(|(&xv, &yv): (&f64, &f64)| yv * xv.powf(yv - 1.0)).collect();

    let expected_y_vec: Vec<f64> = x_val.iter().zip(y_val.iter()).map(|(&xv, &yv): (&f64, &f64)| xv.powf(yv) * xv.ln()).collect();

    // Compare element-wise with tolerance
    let tol: f64 = 1e-12;
    for (a, b) in x_vec.iter().zip(expected_x_vec.iter()) {
        assert!((a - b).abs() < tol, "x_grad mismatch: {} != {}", a, b);
    }
    for (a, b) in y_vec.iter().zip(expected_y_vec.iter()) {
        assert!((a - b).abs() < tol, "y_grad mismatch: {} != {}", a, b);
    }

    Ok(())
}

#[test]
fn grad_cos_2() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let tape = Tape::new([&x])?;
    let z = x.cos();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [-0.141_12_f32, -0.909_297_4, 0.756_802_5]);
    Ok(())
}

#[test]
fn grad_add_1() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = Tape::new([&x, &y])?;
    let z = &x + &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [1, 1, 1]);
    assert_eq!(y_grad, [1, 1, 1]);
    Ok(())
}

#[test]
fn grad_sub() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = Tape::new([&x, &y])?;
    let z = &x - &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [1, 1, 1]);
    assert_eq!(y_grad, [-1, -1, -1]);
    Ok(())
}

#[test]
fn grad_mul() -> Result<(), ZyxError> {
    let x = Tensor::from([3i32, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = Tape::new([&x, &y])?;
    let z = &x * &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [3i32, 1, 5]);
    assert_eq!(y_grad, [3, 2, 4]);
    Ok(())
}

#[test]
fn grad_div_1() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let y = Tensor::from([3f32, 1., 5.]);
    let tape = Tape::new([&x, &y])?;
    let z = &x / &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [0.333_333_34_f32, 1., 0.2]);
    assert_eq!(y_grad, [-0.333_333_34_f32, -2., -0.16]);
    Ok(())
}

#[test]
fn grad_pow() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let y = Tensor::from([3f32, 1., 5.]);
    let tape = Tape::new([&x, &y])?;
    let z = x.pow(&y)?;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [27f32, 1., 1280.]);
    assert_eq!(y_grad, [29.662_53_f32, 1.386_294_4, 1_419.565_4]);
    Ok(())
}

#[test]
fn grad_reshape() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = Tape::new([&x])?;
    let z = x.reshape([1, 3, 1, 1])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [[1i32], [1], [1]]);
    Ok(())
}

#[test]
fn grad_expand_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = Tape::new([&x])?;
    let z = x.expand([3, 4])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [[4], [4], [4]]);
    Ok(())
}

#[test]
fn grad_expand_2() -> Result<(), ZyxError> {
    let x = Tensor::from([4i32, 3, 1]);
    let tape = Tape::new([&x])?;
    let z = x.reshape([3, 1])?.expand([3, 4])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [4, 4, 4]);
    Ok(())
}

#[test]
fn grad_permute() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = Tape::new([&x])?;
    let z = x.permute([1, 0])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [[1], [1], [1]]);
    Ok(())
}

#[test]
fn grad_flip() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32, 5], [3, 1]]);
    let tape = Tape::new([&x])?;
    let z = x.flip([0, 1])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;
    assert_eq!(x_grad, [[1, 1], [1, 1]]);
    Ok(())
}

#[test]
fn grad_dot() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([2, 3, 1]).reshape([3, 1]).unwrap();
    let tape = Tape::new([&x, &y])?;
    let z = x.dot(&y).unwrap();
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;
    assert_eq!(x_grad, [2, 3, 1]);
    assert_eq!(y_grad, [[2], [3], [1]]);
    Ok(())
}

#[test]
fn grad_linear_1() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let w = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4]).reshape([3, 5])?;
    let b = Tensor::from([4, 1, 5, 7, 6]);

    let tape = Tape::new([&x, &w, &b])?;

    let z = x.matmul(&w)? + &b;

    let mut grads = tape.gradient(&z, [&w, &b]);
    let b_grad = grads.pop().unwrap();
    let w_grad = grads.pop().unwrap();
    tape.realize([&w_grad, &b_grad])?;

    assert_eq!(w_grad, [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]);
    assert_eq!(b_grad, [1, 1, 1, 1, 1]);

    Ok(())
}

#[test]
fn grad_mse() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 3., 1.]);
    let y = Tensor::from([5f32, 1., 1.]);
    let tape = Tape::new([&x, &y])?;
    let z = &x - &y;
    let z = &z * &z;
    let mut grads = tape.gradient(&z, [&x, &y]);

    let y_grad = grads.pop().unwrap();
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad, &y_grad])?;

    assert_eq!(x_grad, [-6f32, 4., 0.]);
    assert_eq!(y_grad, [6f32, -4., 0.]);

    Ok(())
}

#[test]
fn grad_linear_2() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([5, 4, 5, 2]);
    let w1 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4]).reshape([3, 5])?;
    let b1 = Tensor::from([4, 1, 5, 7, 6]);

    let w2 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4, 5, 1, 2, 4, 1]).reshape([5, 4])?;
    let b2 = Tensor::from([4, 1, 5, 7]);

    let tape = Tape::new([&x, &y, &w1, &b1, &w2, &b2])?;

    let x = x.matmul(&w1)? + &b1;
    let x = x.relu();
    let x = x.matmul(&w2)? + &b2;
    //let x = x.sigmoid();
    //let x = x.mse_loss(y)?;
    let x = x - y;
    let x = (x.clone() * x).sum_all();
    //println!("{x:?}");

    let mut grads = tape.gradient(&x, [&w1, &b1, &w2, &b2]);

    let b2_grad = grads.pop().unwrap();
    let w2_grad = grads.pop().unwrap();
    let b1_grad = grads.pop().unwrap();
    let w1_grad = grads.pop().unwrap();

    tape.realize([&w1_grad, &b1_grad, &w2_grad, &b2_grad])?;

    //println!("{w1_grad}");
    //println!("{b1_grad}");
    //println!("{w2_grad}");
    //println!("{b2_grad}");

    assert_eq!(
        w1_grad,
        [
            [11528, 21316, 18580, 19872, 11476],
            [17292, 31974, 27870, 29808, 17214],
            [5764, 10658, 9290, 9936, 5738]
        ]
    );
    assert_eq!(b1_grad, [5764, 10658, 9290, 9936, 5738]);
    assert_eq!(
        w2_grad,
        [
            [11628, 5542, 16082, 10506],
            [18468, 8802, 25542, 16686],
            [11628, 5542, 16082, 10506],
            [17100, 8150, 23650, 15450],
            [15732, 7498, 21758, 14214]
        ]
    );
    assert_eq!(b2_grad, [684, 326, 946, 618]);

    Ok(())
}

#[test]
fn grad_t6() -> Result<(), ZyxError> {
    let x = Tensor::randn([8, 10, 10], DType::F32).unwrap();
    let y = Tensor::uniform([8, 10, 10], -1f32..4f32).unwrap();
    let b = Tensor::zeros([10], DType::F32);
    let tape = Tape::new([&x, &y, &b])?;
    let _z = &x + &y;
    let z = x.dot(&y).unwrap() + &b;
    let z = z.gelu(); // TODO there is some numeric instability in gelu

    // Zyx allows for arbitrary differentiation
    let _b_grad = tape.gradient(&z, [&b])[0].clone();
    //println!("{bb_grad}");

    Ok(())
}

#[test]
fn grad_t7() -> Result<(), ZyxError> {
    let x = Tensor::rand([8, 10, 10], DType::F32).unwrap();
    let tape = Tape::new([&x])?;

    let z = x.sum_all();

    let grads = tape.gradient(&z, [&z]);
    let g = grads[0].clone();
    tape.realize([&g])?;

    assert_eq!(g, [1f32]);

    Ok(())
}

#[test]
fn grad_add_2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![4f32, 5., 6.]);
    let tape = Tape::new([&x, &y])?;
    let z = &x + y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();

    tape.realize([&x_grad])?;

    let expected = vec![1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sub_2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![4f32, 5., 6.]);
    let tape = Tape::new([&x, &y])?;
    let z = &x - y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected = vec![1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_mul_2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y_data = vec![4f32, 5., 6.];
    let y = Tensor::from(y_data.clone());
    let tape = Tape::new([&x, &y])?;
    let z = &x * y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    assert_eq!(x_grad, y_data);
    Ok(())
}

#[test]
fn grad_div_2() -> Result<(), ZyxError> {
    let data = vec![2f32, 4., 6.];
    let x = Tensor::from(data.clone());
    let y_data = vec![1f32, 2., 3.];
    let y = Tensor::from(y_data.clone());
    let tape = Tape::new([&x, &y])?;
    let z = &x / y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = y_data.iter().map(|v| 1.0 / v).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_pow_4() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![2f32; 3]);
    let tape = Tape::new([&x, &y])?;
    let z = x.pow(&y)?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| 2.0 * x).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_neg() -> Result<(), ZyxError> {
    let data = vec![1f32, -2., 3.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = -&x;
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected = vec![-1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_log2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = x.log2();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / (x * std::f32::consts::LN_2)).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_ln() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = x.ln();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / x).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_reciprocal_3() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;

    let y = x.reciprocal();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| -1.0 / (x * x)).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sqrt() -> Result<(), ZyxError> {
    let data = vec![1f32, 4., 9.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;

    let y = x.sqrt();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / (2.0 * x.sqrt())).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sin_1() -> Result<(), ZyxError> {
    let data = vec![0f32, 1., 2.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;

    let y = x.sin();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();

    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| x.cos()).collect();

    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_cos() -> Result<(), ZyxError> {
    let data = vec![0f32, 1., 2.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = x.cos();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected: Vec<_> = data.iter().map(|&x| -x.sin()).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sum() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let tape = Tape::new([&x])?;
    let y = x.sum_all();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected = vec![1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_max_1() -> Result<(), ZyxError> {
    let data = vec![1f32, 3., 2.];
    let x = Tensor::from(data);
    let tape = Tape::new([&x])?;

    let y = x.max_all();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected = vec![0f32, 1., 0.];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_cmplt_none() -> Result<(), ZyxError> {
    let x = Tensor::from(vec![1f32, 2., 3.]);
    let y = Tensor::from(vec![2f32, 2., 2.]);
    let tape = Tape::new([&x, &y])?;

    let z = x.cmplt(&y)?;
    let mut grads = tape.gradient(&z, [&x]);
    let g = grads.pop().unwrap();
    tape.realize([&g])?;

    assert_eq!(g, [0f32, 0., 0.]);
    Ok(())
}

#[test]
fn grad_maximum() -> Result<(), ZyxError> {
    let x_data = vec![1f32, 5., 2.];
    let y_data = vec![2f32, 3., 3.];

    let x = Tensor::from(x_data.clone());
    let y = Tensor::from(y_data.clone());
    let tape = Tape::new([&x, &y])?;
    let z = x.maximum(&y)?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap();
    tape.realize([&x_grad])?;

    let expected = vec![0f32, 1., 0.];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad7() -> Result<(), ZyxError> {
    let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0]]);
    let w1 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    let b1 = Tensor::from([0.0f32, 0.0]);
    let th = Tensor::from([1.0f32]);
    let w2 = Tensor::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b2 = Tensor::from([0.0f32, 0.0, 0.0]);
    let w3 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let b3 = Tensor::from([0.0f32, 0.0]);

    let tape = Tape::new([&x, &w1, &b1, &th, &w2, &b2, &w3, &b3])?;

    let v_pre1 = x.matmul(&w1).unwrap() + b1;
    let spike1 = v_pre1.cmpgt(&th).unwrap().cast(DType::F32);

    let v_pre2 = spike1.matmul(&w2).unwrap() + b2;
    let spike2 = v_pre2.cmpgt(&th).unwrap().cast(DType::F32);

    let out = spike2.matmul(&w3).unwrap() + b3;
    let loss = out.sum_all();

    // First call: gradient of loss wrt w3
    let _d_w3 = tape.gradient(&loss, std::slice::from_ref(&w3));
    //println!("d_w3: {:?}", d_w3);

    // Second call: gradient of loss wrt spike2
    let _d_spike2 = tape.gradient(&loss, std::slice::from_ref(&spike2));
    //println!("d_spike2: {:?}", d_spike2);

    // Third call: gradient of loss wrt spike1 - this is where the crash happens
    let _d_spike1 = tape.gradient(&loss, std::slice::from_ref(&spike1));
    //println!("d_spike1: {:?}", d_spike1);

    drop(tape);
    Ok(())
}

#[test]
fn grad_cmpgt_source() -> Result<(), ZyxError> {
    // Test gradient when source is INPUT to cmpgt (non-differentiable op)
    // d_loss/dx should be None because cmpgt is non-differentiable
    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let th = Tensor::from([2.0f32]);
    let w = Tensor::from([1.0f32, 1.0, 1.0]);

    let tape = Tape::new([&x, &th, &w])?;

    // spike is output of cmpgt - non-differentiable
    let spike = x.cmpgt(&th).unwrap();

    // Use spike in a differentiable operation
    let spike_f32 = spike.cast(DType::F32);
    let out = spike_f32 * w.clone();
    let loss = out.sum_all();

    // Gradient through cmpgt (w.r.t. input) is zero since cmpgt is non-differentiable
    let d_x = tape.gradient(&loss, &[x])[0].clone();
    // Gradient w.r.t. w is spike_f32 = [0, 0, 1]
    let d_w = tape.gradient(&loss, std::slice::from_ref(&w))[0].clone();

    tape.realize([&d_x, &d_w])?;

    assert_eq!(d_x, [0f32, 0., 0.]);
    assert_eq!(d_w, [0f32, 0., 1.]);

    Ok(())
}

#[test]
fn grad_2_tapes() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let tape1 = Tape::new([&x])?;
    let z1 = x.reciprocal();

    tape1.realize([&z1])?;

    let y = Tensor::from([1f32, 2., 3.]);
    let tape2 = Tape::new([&y])?;
    let z2 = y.ln();

    let z3 = z1.clone() + z2.clone();

    tape2.realize([&z2, &z3])?;

    Ok(())
}

/*#[test]
#[should_panic(expected = "tensor was never realized")]
fn grad_orphan_then_use_directly() {
    let x = Tensor::from([3f32, 2., 4.]);
    let y;

    {
        let _tape = Tape::new_empty().unwrap();
        y = x.relu();
    }

    let _z = y.ln();
}*/

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
        let h = (x.dot(w1.t())? + &b1).relu();
        let logits = h.dot(w2.t())? + &b2;
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
fn zz_bw_relu_matmul() -> Result<(), ZyxError> {
    let x = Tensor::randn([64, 784], DType::F32)?;
    let w1 = Tensor::randn([128, 784], DType::F32)?;
    let b1 = Tensor::randn([128], DType::F32)?;
    let w2 = Tensor::randn([10, 128], DType::F32)?;
    let b2 = Tensor::randn([10], DType::F32)?;
    let tape = Tape::new([&w1, &b1, &w2, &b2])?;
    let l1 = (x.matmul(w1.t())? + &b1).relu();
    let logits = l1.matmul(w2.t())? + &b2;
    let loss = logits.sum_all();
    let grads = tape.gradient(&loss, [&w1, &b1, &w2, &b2, &loss]);
    let lr = 0.01f32;
    let n1 = &w1 - &grads[0] * lr;
    let n2 = &b1 - &grads[1] * lr;
    let n3 = &w2 - &grads[2] * lr;
    let n4 = &b2 - &grads[3] * lr;
    tape.realize([&n1, &n2, &n3, &n4, &loss])?;
    let v: Vec<f32> = loss.try_into()?;
    println!("loss {}", v[0]);
    Ok(())
}
