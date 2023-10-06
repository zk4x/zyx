use zyx::prelude::*;

#[test]
fn mul() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
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
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
