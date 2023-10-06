use zyx::prelude::*;

#[test]
fn div() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
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
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
