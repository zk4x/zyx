use zyx::prelude::*;

#[test]
fn reduce() -> Result<(), OutOfMemoryError> {
    sum(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    sum(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    sum(&mut Context::torch())?;

    max(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    max(&mut Context::opencl().unwrap())?;
    #[cfg(feature = "torch")]
    max(&mut Context::torch())?;

    Ok(())
}

fn sum(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut z = x.sum(1);
    assert_eq!(z.shape(), (2, 1));
    z.realize()?;
    std::println!("{z}");
    assert_eq!(z, [[9], [11]]);
    let mut z = x.sum(0);
    assert_eq!(z.shape(), (1, 3));
    z.realize()?;
    std::println!("{z}");
    assert_eq!(z, [[7, 6, 7]]);
    z.backward(&mut x);
    x.realize_grad()?;
    assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
    Ok(())
}

fn max(ctx: &mut Context) -> Result<(), OutOfMemoryError> {
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    let mut z = x.max(1);
    assert_eq!(z.shape(), (2, 1));
    z.realize()?;
    assert_eq!(z, [[4], [5]]);
    let mut z = x.max(0);
    assert_eq!(z.shape(), (1, 3));
    z.realize()?;
    assert_eq!(z, [[5, 4, 4]]);
    z.backward(&mut x);
    x.realize_grad()?;
    assert_eq!(x.grad().unwrap(), [[0, 1, 0], [1, 0, 1]]);
    Ok(())
}
