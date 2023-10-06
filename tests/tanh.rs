use zyx::prelude::*;

#[test]
fn tanh() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut z = x.tanh();
        z.realize()?;
        assert_eq!(
            z,
            [
                [0.9640276, 0.9993293, 0.9950548],
                [0.9999092, 0.9640276, 0.9993293]
            ]
        );
        z.backward(&mut x);
        x.realize_grads()?;
        assert_eq!(
            x.grad().unwrap(),
            [
                [0.0706508, 0.0013409, 0.0098660],
                [0.0001816, 0.0706508, 0.0013409]
            ]
        );
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
