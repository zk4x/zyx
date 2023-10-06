use zyx::prelude::*;

#[test]
fn pow() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut y = ctx.tensor([[2., 1., 3.], [2., 2., 4.]]);
        let mut z = x.pow(&y);
        z.realize()?;
        assert_eq!(z, [[4., 4., 27.], [25., 4., 256.]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        std::println!("{:.5}", x.grad().unwrap());
        std::println!("{:.5}", y.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[4., 1., 27.], [10., 4., 256.]]);
        assert_eq!(
            y.grad().unwrap(),
            [[2.77259, 5.54518, 29.66253], [40.23595, 2.77259, 354.89136]]
        );
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
