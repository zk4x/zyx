use zyx::prelude::*;

#[allow(clippy::approx_constant)]
#[test]
fn ln() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut z = x.ln();
        z.realize()?;
        assert_eq!(
            z,
            [
                [core::f32::consts::LN_2, 1.38629, 1.09861],
                [1.60944, core::f32::consts::LN_2, 1.38629]
            ]
        );
        z.backward(&mut x);
        x.realize_grads()?;
        std::println!("{:.5}", x.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[0.5, 0.25, 0.333333], [0.2, 0.5, 0.25]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

