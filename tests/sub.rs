use zyx::prelude::*;

#[test]
fn sub() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
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
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
