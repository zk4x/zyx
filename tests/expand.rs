use zyx::prelude::*;

#[test]
fn expand() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.expand((4, 2, 3));
        z.realize()?;
        assert_eq!(
            z,
            [
                [[2, 4, 3], [5, 2, 4]],
                [[2, 4, 3], [5, 2, 4]],
                [[2, 4, 3], [5, 2, 4]],
                [[2, 4, 3], [5, 2, 4]]
            ]
        );
        z.backward(&mut x);
        x.realize_grad()?;
        assert_eq!(x.grad().unwrap(), [[4, 4, 4], [4, 4, 4]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
