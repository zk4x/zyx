use zyx::{GradientTape, Tensor, ZyxError};

#[test]
fn grad_relu() -> Result<(), ZyxError> {
    let x = Tensor::from([3, 2, 4]);
    let tape = GradientTape::new();
    let z = x.relu();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [1, 1, 1]);
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
fn grad_cos() -> Result<(), ZyxError> {
    let x = Tensor::from([3f32, 2., 4.]);
    let tape = GradientTape::new();
    let z = x.cos();
    let mut grads = tape.gradient(&z, [&x]);
    let x_grad = grads.pop().unwrap().unwrap();
    println!("{x_grad:.10}");
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
    let x = Tensor::from([3, 2, 4]);
    let y = Tensor::from([3, 1, 5]);
    let tape = GradientTape::new();
    let z = &x * &y;
    let mut grads = tape.gradient(&z, [&x, &y]);
    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();
    assert_eq!(x_grad, [3, 1, 5]);
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
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([5, 1, 1]);
    let tape = GradientTape::new();
    let z = (&x - &y).pow(2)?;
    let mut grads = tape.gradient(&z, [&x, &y]);

    let y_grad = grads.pop().unwrap().unwrap();
    let x_grad = grads.pop().unwrap().unwrap();

    assert_eq!(x_grad, [-6, 4, 0]);
    assert_eq!(y_grad, [6, -4, 0]);

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

    let tape = GradientTape::new();

    let x = x.matmul(&w1)? + &b1;
    let x = x.relu();
    let x = x.matmul(&w2)? + &b2;
    //let x = x.sigmoid();
    //let x = x.mse_loss(y)?;
    let x = x - y;
    let x = (x.clone() * x).sum([])?;
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

    assert_eq!(w1_grad, [[11528, 21316, 18580, 19872, 11476], [17292, 31974, 27870, 29808, 17214], [5764, 10658, 9290, 9936, 5738]]);
    assert_eq!(b1_grad,  [5764, 10658, 9290, 9936, 5738]);
    assert_eq!(w2_grad, [[11628, 5542, 16082, 10506], [18468, 8802, 25542, 16686], [11628, 5542, 16082, 10506], [17100, 8150, 23650, 15450], [15732, 7498, 21758, 14214]]);
    assert_eq!(b2_grad,  [684, 326, 946, 618]);

    //assert_eq!(w1_grad, [[20, 28, 24, 28, 16], [30, 42, 36, 42, 24], [10, 14, 12, 14,  8]]);
    //assert_eq!(b1_grad, [10, 14, 12, 14, 8]);
    //assert_eq!(w2_grad, [[17, 17, 17, 17], [27, 27, 27, 27], [17, 17, 17, 17], [25, 25, 25, 25], [23, 23, 23, 23]]);
    //assert_eq!(b2_grad, [1, 1, 1, 1]);

    //assert_eq!(b1_grad, [1441, 2664, 2322, 2484, 1434]);
    //assert_eq!(w1_grad, [[2882, 5329, 4645, 4968, 2869], [4323, 7993, 6967, 7452, 4303], [1441, 2664, 2322, 2484, 1434]]);
    //assert_eq!(w2_grad, [[2907, 1385, 4020, 2626], [4617, 2200, 6385, 4171], [2907, 1385, 4020, 2626], [4275, 2037, 5912, 3862], [3933, 1874, 5439, 3553]]);

    Ok(())
}

// TODO this fails likely due to runtime realize graph creation issue, but perhaps it's scheduler
/*#[test]
fn grad_t6() -> Result<(), ZyxError> {
    use zyx::GradientTape;
    let x = Tensor::randn([8, 1024, 1024], DType::F32).unwrap();
    let y = Tensor::uniform([8, 1024, 1024], -1f32..4f32).unwrap();
    let b = Tensor::zeros([1024], DType::F32);
    let tape = GradientTape::new();
    let _z = &x + &y;
    let z = (x.dot(&y).unwrap() + &b).gelu();
    // Zyx allows for arbitrary differentiation
    let b_grad = tape.gradient(&z, [&b])[0].clone().unwrap();
    //panic!();
    println!("{b_grad}");
    // Also higher order derivatives
    let bb_grad = tape.gradient(&b_grad, [&b])[0].clone().unwrap();
    println!("{bb_grad}");

    Ok(())
}*/
