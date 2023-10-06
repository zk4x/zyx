use zyx::prelude::*;

#[test]
fn permute() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.permute((-1, -2));
        z.realize()?;
        std::println!("{z}");
        assert_eq!(z, [[2, 5], [4, 2], [3, 4]]);
        let mut x = ctx.tensor([[[2, 4, 3]], [[5, 2, 4]]]);
        let mut z = x.permute((2, 1, 0));
        z.realize()?;
        assert_eq!(z, [[[2, 5]], [[4, 2]], [[3, 4]]]);
        z.backward(&mut x);
        x.realize_grad()?;
        assert_eq!(x.grad().unwrap(), [[[1, 1, 1]], [[1, 1, 1]]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

#[cfg(feature = "opencl")]
#[test]
fn permute_2() -> Result<(), OutOfMemoryError> {
    use zyx::context::Context;
    let (dim1, dim2) = (256, 256);

    let ctx = Context::new();
    let mut x = ctx.randn((dim1, dim2));
    let mut z = x.transpose();
    x.realize()?;
    z.realize()?;
    let vecx = x.to_vec().unwrap();
    let vecz = z.to_vec().unwrap();
    //std::println!("{x}");
    //std::println!("{z}");

    let ctx = Context::opencl().unwrap();
    let x = ctx.tensor_from_iter_f32((dim1, dim2), vecx);
    let mut z = x.transpose();
    z.realize()?;
    //std::println!("{z}");
    let cl_vecz = z.to_vec().unwrap();
    for (x, y) in vecz.into_iter().zip(cl_vecz) {
        assert!((x - y).abs() < 0.001, "{x} != {y}");
    }
    Ok(())
}

#[test]
fn permute_add() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let y = ctx.tensor([[1, 2], [1, 4], [2, 3]]);
        let mut z = x.permute((-1, -2)) + y;
        z.realize()?;
        //std::println!("{z}");
        assert_eq!(z, [[3, 7], [5, 6], [5, 7]]);
        let mut x = ctx.tensor([[[2, 4, 3]], [[5, 2, 4]]]);
        let mut z = x.permute((2, 1, 0));
        z.realize()?;
        assert_eq!(z, [[[2, 5]], [[4, 2]], [[3, 4]]]);
        z.backward(&mut x);
        x.realize_grad()?;
        assert_eq!(x.grad().unwrap(), [[[1, 1, 1]], [[1, 1, 1]]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}
