use zyx::prelude::*;

#[test]
fn binary() -> Result<(), OutOfMemoryError> {
    add(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    add(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    add(&mut Context::torch())?;

    sub(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    sub(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    sub(&mut Context::torch())?;

    mul(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    mul(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    mul(&mut Context::torch())?;

    div(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    div(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    div(&mut Context::torch())?;

    pow(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    pow(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    pow(&mut Context::torch())?;

    dot(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    dot(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    dot(&mut Context::torch())?;

    Ok(())
}

fn add(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    // Same size
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
    let mut z = &x + &y;
    z.realize()?;
    std::println!("Z tensor {z}");
    assert_eq!(z, [[4, 5, 6], [7, 4, 8]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    std::println!("{}\n{}", x.grad().unwrap(), y.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
    assert_eq!(y.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);

    // broadcast right
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut y = ctx.tensor([2, 1, 3]);
    let mut z = &x + &y;
    z.realize()?;
    assert_eq!(z, [[4, 5, 6], [7, 3, 7]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    std::println!("{}\n{}", x.grad().unwrap(), y.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
    assert_eq!(y.grad().unwrap(), [2, 2, 2]);

    // broadcast left
    let mut x = ctx.tensor([2, 4, 3]);
    let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
    let mut z = &x + &y;
    z.realize()?;
    assert_eq!(z, [[4, 5, 6], [4, 6, 7]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    std::println!("{}\n{}", x.grad().unwrap(), y.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [2, 2, 2]);
    assert_eq!(y.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);

    // Same value
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut z = &x + &x;
    z.realize()?;
    assert_eq!(z, [[4, 8, 6], [10, 4, 8]]);
    z.backward(&mut x);
    x.realize_grad()?;
    std::println!("{}", x.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[2, 2, 2], [2, 2, 2]]);

    Ok(())
}

#[test]
fn add2() -> Result<(), OutOfMemoryError> {
    let ctx = Context::new();
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    for i in 0..10 {
        let i = i as i32 + 1;
        let mut z = &x + &x;
        z.realize()?;
        assert_eq!(z, [[4, 8, 6], [10, 4, 8]]);
        z.backward(&mut x);
        x.realize_grad()?;
        std::println!("{}", x.grad().unwrap());
        assert_eq!(
            x.grad().unwrap(),
            [[2 * i, 2 * i, 2 * i], [2 * i, 2 * i, 2 * i]]
        );
    }

    Ok(())
}

fn sub(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
    let mut z = &x - &y;
    z.realize()?;
    assert_eq!(z, [[0, 3, 0], [3, 0, 0]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
    assert_eq!(y.grad().unwrap(), [[-1, -1, -1], [-1, -1, -1]]);
    Ok(())
}

fn mul(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
    let mut z = &x * &y;
    z.realize()?;
    assert_eq!(z, [[4, 4, 9], [10, 4, 16]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[2, 1, 3], [2, 2, 4]]);
    assert_eq!(y.grad().unwrap(), [[2, 4, 3], [5, 2, 4]]);
    Ok(())
}

fn div(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut y = ctx.tensor([[2., 1., 3.], [2., 2., 4.]]);
    let mut z = &x / &y;
    z.realize()?;
    assert_eq!(z, [[1., 4., 1.], [2.5, 1., 1.]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[0.5, 1., 0.3333333], [0.5, 0.5, 0.25]]);
    assert_eq!(
        y.grad().unwrap(),
        [[-0.5, -4., -0.3333333], [-1.25, -0.5, -0.25]]
    );
    Ok(())
}

fn pow(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut y = ctx.tensor([[2., 1., 3.], [2., 2., 4.]]);
    let mut z = x.pow(&y);
    z.realize()?;
    assert_eq!(z, [[4., 4., 27.], [25., 4., 256.]]);
    z.backward((&mut x, &mut y));
    (&mut x, &mut y).realize_grads()?;
    std::println!("{:.5}", x.grad().unwrap());
    std::println!("{:.5}", y.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[4., 1., 27.], [10., 4., 256.]]);
    assert_eq!(
        y.grad().unwrap(),
        [[2.77259, 5.54518, 29.66253], [40.23595, 2.77259, 354.89136]]
    );
    Ok(())
}

fn dot(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut y = ctx.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
    let mut z = x.dot(&y);
    [&mut x, &mut y, &mut z].realize()?;
    std::println!("{}", x);
    std::println!("{}", y);
    std::println!("{}", z);
    assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);
    z.backward((&mut x, &mut y));
    //for node in ctx.debug_nodes() { println!("{}", node); }
    (&mut x.grad().unwrap(), &mut y.grad().unwrap()).realize()?;
    std::println!("{}", x.grad().unwrap());
    std::println!("{}", y.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[8, 4, 9], [8, 4, 9]]);
    assert_eq!(y.grad().unwrap(), [[7, 7, 7], [6, 6, 6], [7, 7, 7]]);
    Ok(())
}
