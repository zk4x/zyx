// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, GradientTape, Tensor, ZyxError};

#[cfg(not(feature = "wgpu"))]
#[test]
fn grad_relu_1() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 0, -1]);
    let _tape = GradientTape::new();
    let z = x.relu();
    assert_eq!(z, [3, 0, 0]);
    //println!("{z}");
    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn grad_relu_2() -> Result<(), ZyxError> {
    let x = Tensor::from([3, -2, 0]);
    let tape = GradientTape::new();
    let z = x.relu();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [1, 0, 0]);
    Ok(())
}

#[test]
fn grad_reciprocal() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let tape = GradientTape::new();
    let z = x.reciprocal();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [-0.1111111111f32, -0.25, -0.0625]);
    Ok(())
}

#[test]
fn grad_exp2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 0.5];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();
    let y = x.exp2();
    let mut grads: Vec<Option<Tensor>> = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    let expected: Vec<_> = data.iter().map(|&x| 2f32.powf(x) * std::f32::consts::LN_2).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_reciprocal_2() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::F64) {
        return Ok(());
    }
    // Input tensor
    let x = Tensor::from([2.0, -1.0, 0.5]);

    // Create gradient tape
    let tape = GradientTape::new();

    // Forward pass: y = 1 / x
    let y = x.reciprocal();

    // Compute gradients
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    // Expected gradients: dy/dx = -1 / x^2
    let expected = [-1.0 / 4.0, -1.0 / 1.0, -1.0 / 0.25]; // [-0.25, -1.0, -4.0]

    // Compare
    assert_eq!(x_grad, expected);

    Ok(())
}

#[test]
fn grad_floor() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::F64) {
        return Ok(());
    }
    let x = Tensor::from([0.5, 1.5, -0.5, -1.5, 0.1, 0.9, -0.1, -0.9, 2.3, -2.3]);
    let tape = GradientTape::new();
    let y = x.floor();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, vec![0.0; 10]);
    Ok(())
}

#[test]
fn grad_trunc() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::F64) {
        return Ok(());
    }
    let x = Tensor::from([0.5, 1.5, -0.5, -1.5, 0.1, 0.9, -0.1, -0.9, 2.3, -2.3]);
    let tape = GradientTape::new();
    let y = x.trunc();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, vec![0.0; 10]);
    Ok(())
}

#[test]
fn grad_pow_2() -> Result<(), ZyxError> {
    // Input tensors
    let x = Tensor::from([2.0f32, 3.0, 4.0]);
    let y = Tensor::from([3.0f32, 2.0, 0.5]);

    // Forward pass: z = x ^ y
    let tape = GradientTape::new();
    let z = x.pow(&y)?;

    // Compute gradients
    let mut grads = tape.gradient(&z, [&x, &y]);
    let x_grad = grads.remove(0).unwrap();
    let y_grad = grads.remove(0).unwrap();

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

#[cfg(not(feature = "wgpu"))]
#[test]
fn grad_pow_3() -> Result<(), ZyxError> {
    if !Tensor::supports(DType::F64) {
        return Ok(());
    }
    // Use non-round numbers to expose log2 -> ln approximation errors
    let x = Tensor::from([1.5, 2.3, 5.7]);
    let y = Tensor::from([0.7, 1.2, 0.3]);

    // Forward pass: z = x ^ y
    let tape = GradientTape::new();
    let z = x.pow(&y)?;

    // Compute gradients
    let mut grads = tape.gradient(&z, [&x, &y]);
    let x_grad = grads.remove(0).unwrap();
    let y_grad = grads.remove(0).unwrap();

    // Convert tensors to Vec<f64> for comparison
    let x_vec: Vec<f64> = x_grad.clone().try_into().unwrap();
    let y_vec: Vec<f64> = y_grad.clone().try_into().unwrap();
    let x_val: Vec<f64> = x.clone().try_into().unwrap();
    let y_val: Vec<f64> = y.clone().try_into().unwrap();

    // Expected gradients
    let expected_x_vec: Vec<f64> = x_val
        .iter()
        .zip(y_val.iter())
        .map(|(&xv, &yv): (&f64, &f64)| yv * xv.powf(yv - 1.0))
        .collect();

    let expected_y_vec: Vec<f64> = x_val
        .iter()
        .zip(y_val.iter())
        .map(|(&xv, &yv): (&f64, &f64)| xv.powf(yv) * xv.ln())
        .collect();

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
    let tape = GradientTape::new();
    let z = x.cos();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    //println!("{x_grad:.10}");
    assert_eq!(x_grad, [-0.1411200017f32, -0.9092974067, 0.7568024993]);
    Ok(())
}

#[test]
fn grad_add() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = GradientTape::new();
    let z = &x + &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [1, 1, 1]);
    assert_eq!(y_grad, [1, 1, 1]);
    Ok(())
}

#[test]
fn grad_sub() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = GradientTape::new();
    let z = &x - &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [1, 1, 1]);
    assert_eq!(y_grad, [-1, -1, -1]);
    Ok(())
}

#[test]
fn grad_mul() -> Result<(), ZyxError> {
    let x = Tensor::from([3i32, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = GradientTape::new();
    let z = &x * &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [3i32, 1, 5]);
    assert_eq!(y_grad, [3, 2, 4]);
    Ok(())
}

#[test]
fn grad_div() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let y = Tensor::from([3f32, 1., 5.]);
    let tape = GradientTape::new();
    let z = &x / &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [0.3333333333f32, 1., 0.2]);
    assert_eq!(y_grad, [-0.3333333333f32, -2., -0.16]);
    Ok(())
}

#[test]
fn grad_pow() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let y = Tensor::from([3f32, 1., 5.]);
    let tape = GradientTape::new();
    let z = x.pow(&y)?;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [27f32, 1., 1280.]);
    assert_eq!(y_grad, [29.6625317940f32, 1.3862943611, 1419.5654257868]);
    Ok(())
}

#[test]
fn grad_reshape() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = GradientTape::new();
    let z = x.reshape([1, 3, 1, 1])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [[1i32], [1], [1]]);
    Ok(())
}

#[test]
fn grad_expand_1() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = GradientTape::new();
    let z = x.expand([3, 4])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [[4], [4], [4]]);
    Ok(())
}

#[test]
fn grad_expand_2() -> Result<(), ZyxError> {
    let x = Tensor::from([4i32, 3, 1]);
    let tape = GradientTape::new();
    let z = x.reshape([3, 1])?.expand([3, 4])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [4, 4, 4]);
    Ok(())
}

#[test]
fn grad_permute() -> Result<(), ZyxError> {
    let x = Tensor::from([[4i32], [3], [1]]);
    let tape = GradientTape::new();
    let z = x.permute([1, 0])?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [[1], [1], [1]]);
    Ok(())
}

#[test]
fn grad_dot() {
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([2, 3, 1]).reshape([3, 1]).unwrap();
    let tape = GradientTape::new();
    let z = x.dot(&y).unwrap();
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [2, 3, 1]);
    assert_eq!(y_grad, [[2], [3], [1]]);
}

#[test]
fn grad_linear_1() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let w = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4]).reshape([3, 5])?;
    let b = Tensor::from([4, 1, 5, 7, 6]);

    let tape = GradientTape::new();

    let z = x.matmul(&w)? + &b;

    let mut grads = tape.gradient(&z, [&w, &b]);
    let b_grad = grads.pop().unwrap().unwrap();
    let w_grad = grads.pop().unwrap().unwrap();

    assert_eq!(w_grad, [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]);
    assert_eq!(b_grad, [1, 1, 1, 1, 1]);

    Ok(())
}

#[test]
fn grad_mse() -> Result<(), ZyxError> {
    let x = Tensor::from([2f32, 3., 1.]);
    let y = Tensor::from([5f32, 1., 1.]);
    let tape = GradientTape::new();
    let z = &x - &y;
    let z = &z * &z;
    let mut grads = tape.gradient(&z, [&x, &y]);

    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();

    assert_eq!(x_grad, [-6f32, 4., 0.]);
    assert_eq!(y_grad, [6f32, -4., 0.]);

    Ok(())
}

#[cfg(not(feature = "wgpu"))]
#[test]
fn grad_linear_2() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([5, 4, 5, 2]);
    let w1 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4]).reshape([3, 5])?;
    let b1 = Tensor::from([4, 1, 5, 7, 6]);

    let w2 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4, 5, 1, 2, 4, 1]).reshape([5, 4])?;
    let b2 = Tensor::from([4, 1, 5, 7]);

    let tape = GradientTape::new();

    let x = x.matmul(&w1)? + &b1;
    let x = x.relu();
    let x = x.matmul(&w2)? + &b2;
    //let x = x.sigmoid();
    //let x = x.mse_loss(y)?;
    let x = x - y;
    let x = (x.clone() * x).sum_all();
    //println!("{x:?}");

    let mut grads = tape.gradient(&x, [&w1, &b1, &w2, &b2]);

    let b2_grad = grads.pop().unwrap().unwrap();
    let w2_grad = grads.pop().unwrap().unwrap();
    let b1_grad = grads.pop().unwrap().unwrap();
    let w1_grad = grads.pop().unwrap().unwrap();

    Tensor::realize([&w1_grad, &b1_grad, &w2_grad, &b2_grad])?;

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
    use zyx::{DType, GradientTape};
    let x = Tensor::randn([8, 10, 10], DType::F32).unwrap();
    let y = Tensor::uniform([8, 10, 10], -1f32..4f32).unwrap();
    let b = Tensor::zeros([10], DType::F32);
    let tape = GradientTape::new();
    let _z = &x + &y;
    let z = x.dot(&y).unwrap() + &b;
    let z = z.gelu(); // TODO there is some numeric instability in gelu

    // Zyx allows for arbitrary differentiation
    let b_grad = tape.gradient_persistent(&z, [&b])[0].clone().unwrap();
    //panic!();
    //println!("{b_grad}");
    // Also higher order derivatives
    let _bb_grad = tape.gradient(&b_grad, [&b])[0].clone().unwrap();
    //println!("{bb_grad}");

    Ok(())
}

#[test]
fn grad_t7() -> Result<(), ZyxError> {
    use zyx::{DType, GradientTape};
    let x = Tensor::rand([8, 10, 10], DType::F32).unwrap();
    let tape = GradientTape::new();

    let z = x.sum_all();

    let grads = tape.gradient(&z, [&z]);

    assert_eq!(grads[0].clone().unwrap(), [1f32]);

    Ok(())
}

#[test]
fn grad_add_2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![4f32, 5., 6.]);
    let tape = GradientTape::new();

    let z = &x + y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected = vec![1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sub_2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![4f32, 5., 6.]);
    let tape = GradientTape::new();

    let z = &x - y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

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
    let tape = GradientTape::new();

    let z = &x * y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    assert_eq!(x_grad, y_data);
    Ok(())
}

#[test]
fn grad_div_2() -> Result<(), ZyxError> {
    let data = vec![2f32, 4., 6.];
    let x = Tensor::from(data.clone());
    let y_data = vec![1f32, 2., 3.];
    let y = Tensor::from(y_data.clone());
    let tape = GradientTape::new();

    let z = &x / y;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = y_data.iter().map(|v| 1.0 / v).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_pow_4() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let y = Tensor::from(vec![2f32; 3]);
    let tape = GradientTape::new();

    let z = x.pow(&y)?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| 2.0 * x).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_neg() -> Result<(), ZyxError> {
    let data = vec![1f32, -2., 3.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = -&x;
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected = vec![-1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_log2() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.log2();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / (x * std::f32::consts::LN_2)).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_ln() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.ln();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / x).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_reciprocal_3() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 4.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.reciprocal();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| -1.0 / (x * x)).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sqrt() -> Result<(), ZyxError> {
    let data = vec![1f32, 4., 9.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.sqrt();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| 1.0 / (2.0 * x.sqrt())).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sin() -> Result<(), ZyxError> {
    let data = vec![0f32, 1., 2.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.sin();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| x.cos()).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_cos() -> Result<(), ZyxError> {
    let data = vec![0f32, 1., 2.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.cos();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected: Vec<_> = data.iter().map(|&x| -x.sin()).collect();
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_sum() -> Result<(), ZyxError> {
    let data = vec![1f32, 2., 3.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.sum_all();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected = vec![1f32; data.len()];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_max() -> Result<(), ZyxError> {
    let data = vec![1f32, 3., 2.];
    let x = Tensor::from(data.clone());
    let tape = GradientTape::new();

    let y = x.max_all();
    let mut grads = tape.gradient(&y, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected = vec![0f32, 1., 0.];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad_cmplt_none() -> Result<(), ZyxError> {
    let x = Tensor::from(vec![1f32, 2., 3.]);
    let y = Tensor::from(vec![2f32, 2., 2.]);
    let tape = GradientTape::new();

    let z = x.cmplt(&y)?;
    let mut grads = tape.gradient(&z, [&x]);

    assert!(grads.pop().unwrap().is_none());
    Ok(())
}

#[test]
fn grad_maximum() -> Result<(), ZyxError> {
    let x_data = vec![1f32, 5., 2.];
    let y_data = vec![2f32, 3., 3.];

    let x = Tensor::from(x_data.clone());
    let y = Tensor::from(y_data.clone());
    let tape = GradientTape::new();

    let z = x.maximum(&y)?;
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();

    let expected = vec![0f32, 1., 0.];
    assert_eq!(x_grad, expected);
    Ok(())
}

#[test]
fn grad7() {
    let tape = GradientTape::new();

    let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0]]);
    let w1 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    let b1 = Tensor::from([0.0f32, 0.0]);

    let v_pre1 = x.matmul(&w1).unwrap() + b1;
    let th = Tensor::from([1.0f32]);
    let spike1 = v_pre1.cmpgt(&th).unwrap().cast(DType::F32);

    let w2 = Tensor::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b2 = Tensor::from([0.0f32, 0.0, 0.0]);

    let v_pre2 = spike1.matmul(&w2).unwrap() + b2;
    let spike2 = v_pre2.cmpgt(&th).unwrap().cast(DType::F32);

    let w3 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let b3 = Tensor::from([0.0f32, 0.0]);

    let out = spike2.matmul(&w3).unwrap() + b3;
    let loss = out.sum_all();

    // First call: gradient of loss wrt w3
    let d_w3 = tape.gradient_persistent(&loss, &[w3.clone()]);
    println!("d_w3: {:?}", d_w3);

    // Second call: gradient of loss wrt spike2
    let d_spike2 = tape.gradient_persistent(&loss, &[spike2.clone()]);
    println!("d_spike2: {:?}", d_spike2);

    // Third call: gradient of loss wrt spike1 - this is where the crash happens
    let d_spike1 = tape.gradient_persistent(&loss, &[spike1.clone()]);
    println!("d_spike1: {:?}", d_spike1);

    drop(tape);
}

#[test]
fn grad_cmpgt_source() {
    // Test gradient when source is INPUT to cmpgt (non-differentiable op)
    // d_loss/dx should be None because cmpgt is non-differentiable
    let tape = GradientTape::new();

    let x = Tensor::from([1.0f32, 2.0, 3.0]);
    let th = Tensor::from([2.0f32]);

    // spike is output of cmpgt - non-differentiable
    let spike = x.cmpgt(&th).unwrap();

    // Use spike in a differentiable operation
    let spike_f32 = spike.cast(DType::F32);
    let w = Tensor::from([1.0f32, 1.0, 1.0]);
    let out = spike_f32 * w.clone();
    let loss = out.sum_all();

    // Request gradient w.r.t. x (INPUT to cmpgt - non-differentiable)
    // Since cmpgt is non-differentiable, gradient should be None
    let d_x = tape.gradient_persistent(&loss, &[x]);
    assert!(d_x[0].is_none(), "Gradient through cmpgt (w.r.t. input) should be None");

    // Also request gradient w.r.t. w (should work - differentiable path)
    let d_w = tape.gradient_persistent(&loss, &[w.clone()]);
    assert!(d_w[0].is_some(), "Gradient of w should exist");

    drop(tape);
}

#[test]
fn grad8() {
    let tape = GradientTape::new();

    let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0]]);
    let w1 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]);
    let b1 = Tensor::from([0.0f32, 0.0]);

    let v_pre1 = x.matmul(&w1).unwrap() + b1.clone();
    let th = Tensor::from([1.0f32]);
    let spike1 = v_pre1.cmpgt(&th).unwrap().cast(DType::F32);

    let w2 = Tensor::from([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let b2 = Tensor::from([0.0f32, 0.0, 0.0]);

    let v_pre2 = spike1.matmul(&w2).unwrap() + b2.clone();
    let spike2 = v_pre2.cmpgt(&th).unwrap().cast(DType::F32);

    let w3 = Tensor::from([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let b3 = Tensor::from([0.0f32, 0.0]);

    let out = spike2.matmul(&w3).unwrap() + b3.clone();
    let loss = out.sum_all();

    let d_w3 = tape.gradient_persistent(&loss, &[w3.clone()])[0].clone().unwrap();
    let d_b3 = tape.gradient_persistent(&loss, &[b3])[0].clone().unwrap();

    let sigma_t = Tensor::from([0.5f32]);
    let neg_one = Tensor::from([-1.0f32]);
    let oma_t = Tensor::from([0.1f32]);

    let diff2 = v_pre2 - th.clone();
    let abs_diff2 = diff2.abs();
    let surr2 = sigma_t.clone() * (neg_one.clone() * sigma_t.clone() * abs_diff2).exp();
    let d_v2_pre = Tensor::from([[0.5f32, 0.5, 0.5]]);
    let d_pre2 = oma_t.clone() * d_v2_pre;

    let d_w2_acc = spike1.transpose(0, 1).unwrap().matmul(&d_pre2).unwrap();
    let d_b2_acc = d_pre2.sum([0]).unwrap();
    let d_spike1 = d_pre2.matmul(&w2.transpose(0, 1).unwrap()).unwrap();

    let diff1 = v_pre1 - th;
    let abs_diff1 = diff1.abs();
    let surr1 = sigma_t.clone() * (neg_one.clone() * sigma_t.clone() * abs_diff1).exp();
    let d_v1_pre = d_spike1 * surr1;
    let d_pre1 = oma_t * d_v1_pre;

    let d_w1_acc = x.transpose(0, 1).unwrap().matmul(&d_pre1).unwrap();
    let d_b1_acc = d_pre1.sum([0]).unwrap();

    println!("d_w3 shape: {:?}", d_w3.shape());
    println!("d_b3 shape: {:?}", d_b3.shape());
    println!("d_w2_acc shape: {:?}", d_w2_acc.shape());
    println!("d_b2_acc shape: {:?}", d_b2_acc.shape());
    println!("d_w1_acc shape: {:?}", d_w1_acc.shape());
    println!("d_b1_acc shape: {:?}", d_b1_acc.shape());

    Tensor::realize_all().unwrap();

    println!("All tensors realized successfully");

    drop(tape);
}
