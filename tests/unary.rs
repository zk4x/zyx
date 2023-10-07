use zyx::prelude::*;

#[test]
fn unary() -> Result<(), OutOfMemoryError> {
    exp(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    exp(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    exp(&mut Context::torch())?;

    ln(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    ln(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    ln(&mut Context::torch())?;

    relu(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    relu(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    relu(&mut Context::torch())?;

    neg(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    neg(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    neg(&mut Context::torch())?;

    tanh(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    tanh(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    tanh(&mut Context::torch())?;

    cast(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    cast(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    cast(&mut Context::torch())?;

    Ok(())
}

fn exp(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut z = x.exp();
    z.realize()?;
    assert_eq!(
        z,
        [
            [7.389056, 54.59815, 20.085537],
            [148.41316, 7.389056, 54.59815]
        ]
    );
    z.backward(&mut x);
    x.realize_grads()?;
    assert_eq!(
        x.grad().unwrap(),
        [
            [7.389056, 54.59815, 20.085537],
            [148.41316, 7.389056, 54.59815]
        ]
    );
    Ok(())
}

#[allow(clippy::approx_constant)]
fn ln(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut z = x.ln();
    z.realize()?;
    assert_eq!(
        z,
        [
            [core::f32::consts::LN_2, 1.38629, 1.09861],
            [1.60944, core::f32::consts::LN_2, 1.38629]
        ]
    );
    z.backward(&mut x);
    x.realize_grads()?;
    std::println!("{:.5}", x.grad().unwrap());
    assert_eq!(x.grad().unwrap(), [[0.5, 0.25, 0.333333], [0.2, 0.5, 0.25]]);
    Ok(())
}

fn relu(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[-2., 4., -3.], [-5., 2., 4.]]);
    let mut z = x.relu();
    z.realize()?;
    assert_eq!(z, [[0., 4., 0.], [0., 2., 4.]]);
    z.backward(&mut x);
    x.realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[0., 1., 0.], [0., 1., 1.]]);
    Ok(())
}

fn neg(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut z = -&x;
    z.realize()?;
    assert_eq!(z, [[-2., -4., -3.], [-5., -2., -4.]]);
    z.backward(&mut x);
    x.realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[-1., -1., -1.], [-1., -1., -1.]]);
    Ok(())
}

fn tanh(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut z = x.tanh();
    z.realize()?;
    assert_eq!(
        z,
        [
            [0.9640276, 0.9993293, 0.9950548],
            [0.9999092, 0.9640276, 0.9993293]
        ]
    );
    z.backward(&mut x);
    x.realize_grads()?;
    assert_eq!(
        x.grad().unwrap(),
        [
            [0.0706508, 0.0013409, 0.0098660],
            [0.0001816, 0.0706508, 0.0013409]
        ]
    );
    Ok(())
}

fn cast(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
    let mut z = x.cast(DType::I32);
    z.realize()?;
    assert_eq!(z, [[2, 4, 3], [5, 2, 4]]);
    z.backward(&mut x);
    x.realize_grads()?;
    assert_eq!(x.grad().unwrap(), [[1., 1., 1.], [1., 1., 1.]]);
    Ok(())
}

#[test]
fn dropout() {
    let ctx = Context::new();
    let x = ctx.randn_i32((2, 8));
    let mut z = x.dropout(0.9);
    z.realize().unwrap();
    std::println!("{z}");
}
