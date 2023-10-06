use zyx::prelude::*;

#[test]
fn sum() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
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
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
