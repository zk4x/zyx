use zyx::prelude::*;

#[test]
fn cast() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut z = x.cast(DType::I32);
        z.realize()?;
        assert_eq!(z, [[2, 4, 3], [5, 2, 4]]);
        z.backward(&mut x);
        x.realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[1., 1., 1.], [1., 1., 1.]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
