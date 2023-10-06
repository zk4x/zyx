use zyx::{OutOfMemoryError, context::Context, parameters::IntoParameters};

#[test]
fn exp() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
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
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
