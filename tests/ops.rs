// Tests for all tensor operations

use zyx::{context::Context, dtype::DType, parameters::IntoParameters, OutOfMemoryError};

// TODO test expanding with binary ops

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

#[test]
fn neg() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut z = -&x;
        z.realize()?;
        assert_eq!(z, [[-2., -4., -3.], [-5., -2., -4.]]);
        z.backward(&mut x);
        x.realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[-1., -1., -1.], [-1., -1., -1.]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

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

#[test]
fn add() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
        let mut z = &x + &y;
        z.realize()?;
        assert_eq!(z, [[4, 5, 6], [7, 4, 8]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        assert_eq!(y.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

#[test]
fn sub() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
        let mut z = &x - &y;
        z.realize()?;
        assert_eq!(z, [[0, 3, 0], [3, 0, 0]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        assert_eq!(y.grad().unwrap(), [[-1, -1, -1], [-1, -1, -1]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

#[test]
fn mul() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
        let mut z = &x * &y;
        z.realize()?;
        assert_eq!(z, [[4, 4, 9], [10, 4, 16]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[2, 1, 3], [2, 2, 4]]);
        assert_eq!(y.grad().unwrap(), [[2, 4, 3], [5, 2, 4]]);
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

#[test]
fn div() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2., 4., 3.], [5., 2., 4.]]);
        let mut y = ctx.tensor([[2., 1., 3.], [2., 2., 4.]]);
        let mut z = &x / &y;
        z.realize()?;
        assert_eq!(z, [[1., 4., 1.], [2.5, 1., 1.]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[0.5, 1., 0.3333333], [0.5, 0.5, 0.25]]);
        assert_eq!(
            y.grad().unwrap(),
            [[-0.5, -4., -0.3333333], [-1.25, -0.5, -0.25]]
        );
        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

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

#[test]
fn sum() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.sum(1);
        assert_eq!(z.shape(), (2, 1));
        z.realize()?;
        assert_eq!(z, [[9], [11]]);
        let mut z = x.sum(0);
        assert_eq!(z.shape(), (1, 3));
        z.realize()?;
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

#[test]
fn permute() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.permute((-1, -2));
        z.realize()?;
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

#[cfg(all(feature = "rand", feature = "opencl"))]
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

#[test]
fn reshape() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = x.reshape(6);
        z.realize()?;
        assert_eq!(z, [2, 4, 3, 5, 2, 4]);
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

#[test]
fn backpropagation() -> Result<(), OutOfMemoryError> {
    let ctx = Context::new();
    let mut x = ctx.tensor([[3, 4, 2], [4, 2, 3]]);
    let z = (&x + &x).relu();
    for _ in 0..3 {
        z.backward(&mut x);
        std::println!();
        for n in ctx.debug_nodes() { std::println!("{n}"); }
        use std::io::Write;
        std::fs::File::create("graph.dot").unwrap().write_all(ctx.dot_graph().as_bytes()).unwrap();
        x.realize_grad()?;
    }
    Ok(())
}
