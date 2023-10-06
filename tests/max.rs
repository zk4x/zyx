use zyx::prelude::*;

#[test]
fn max() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.max(1);
        assert_eq!(z.shape(), (2, 1));
        z.realize()?;
        assert_eq!(z, [[4], [5]]);
        let mut z = x.max(0);
        assert_eq!(z.shape(), (1, 3));
        z.realize()?;
        assert_eq!(z, [[5, 4, 4]]);
        // TODO max backward
        //z.backward(&mut x);
        //x.realize_grad()?;
        //assert_eq!(x.grad().unwrap(), [[0, 1, 0], [1, 0, 1]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
