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
fn grad_linear_2() -> Result<(), ZyxError> {
    let x = Tensor::from([2, 3, 1]);
    let y = Tensor::from([5, 4, 5, 2]);
    let w1 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4]).reshape([3, 5])?;
    let b1 = Tensor::from([4, 1, 5, 7, 6]);

    let w2 = Tensor::from([2, 3, 1, 4, 5, 1, 6, 2, 3, 1, 6, 2, 4, 1, 4, 5, 1, 2, 4, 1]).reshape([5, 4])?;
    let b2 = Tensor::from([4, 1, 5, 7]);

    let tape = GradientTape::new();

    let x = x.matmul(&w1)? + &b1;
    //let x = x.relu();
    let x = x.matmul(&w2)? + &b2;
    //let x = x.sigmoid();
    let x = x.mse_loss(y)?;

    let mut grads = tape.gradient(&x, [&w1, &b1, &w2, &b2]);
    let b2_grad = grads.pop().unwrap().unwrap();
    let w2_grad = grads.pop().unwrap().unwrap();
    let b1_grad = grads.pop().unwrap().unwrap();
    let w1_grad = grads.pop().unwrap().unwrap();

    println!("{w1_grad}");
    println!("{b1_grad}");
    println!("{w2_grad}");
    println!("{b2_grad}");

    assert_eq!(w1_grad, [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [1, 1, 1, 1, 1]]);
    assert_eq!(b1_grad, [1, 1, 1, 1, 1]);

    Ok(())
}
