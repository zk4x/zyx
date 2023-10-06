use zyx::prelude::*;

#[test]
fn dot() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        use zyx::parameters::IntoParameters;
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
        let mut z = x.dot(&y);
        [&mut x, &mut y, &mut z].realize()?;
        std::println!("{}", x);
        std::println!("{}", y);
        std::println!("{}", z);
        assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);
        z.backward((&mut x, &mut y));
        //for node in ctx.debug_nodes() { println!("{}", node); }
        (&mut x.grad().unwrap(), &mut y.grad().unwrap()).realize()?;
        std::println!("{}", x.grad().unwrap());
        std::println!("{}", y.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[8, 4, 9], [8, 4, 9]]);
        assert_eq!(y.grad().unwrap(), [[7, 7, 7], [6, 6, 6], [7, 7, 7]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
