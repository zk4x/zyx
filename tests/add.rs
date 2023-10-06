use zyx::prelude::*;

#[test]
fn add() -> Result<(), OutOfMemoryError> {
    let t = |ctx: &mut Context| -> Result<(), OutOfMemoryError> {
        // Same size
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
        let mut z = &x + &y;
        z.realize()?;
        assert_eq!(z, [[4, 5, 6], [7, 4, 8]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        assert_eq!(y.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        
        // broadcast right
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut y = ctx.tensor([2, 1, 3]);
        let mut z = &x + &y;
        z.realize()?;
        assert_eq!(z, [[4, 5, 6], [7, 3, 7]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        std::println!("{}\n{}", x.grad().unwrap(), y.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);
        assert_eq!(y.grad().unwrap(), [2, 2, 2]);

        // broadcast left
        let mut x = ctx.tensor([2, 4, 3]);
        let mut y = ctx.tensor([[2, 1, 3], [2, 2, 4]]);
        let mut z = &x + &y;
        z.realize()?;
        assert_eq!(z, [[4, 5, 6], [4, 6, 7]]);
        z.backward((&mut x, &mut y));
        (&mut x, &mut y).realize_grads()?;
        std::println!("{}\n{}", x.grad().unwrap(), y.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [2, 2, 2]);
        assert_eq!(y.grad().unwrap(), [[1, 1, 1], [1, 1, 1]]);

        // Same value
        let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
        let mut z = &x + &x;
        z.realize()?;
        assert_eq!(z, [[4, 8, 6], [10, 4, 8]]);
        z.backward(&mut x);
        x.realize_grad()?;
        std::println!("{}", x.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[2, 2, 2], [2, 2, 2]]);

        Ok(())
    };
    t(&mut Context::new())?;
    #[cfg(feature = "opencl")]
    t(&mut Context::opencl().unwrap())?;
    Ok(())
}

#[test]
fn add2() -> Result<(), OutOfMemoryError> {
    let ctx = Context::new();
    let mut x = ctx.tensor([[2, 4, 3], [5, 2, 4]]);
    for i in 0..10 {
        let i = i as i32 + 1;
        let mut z = &x + &x;
        z.realize()?;
        assert_eq!(z, [[4, 8, 6], [10, 4, 8]]);
        z.backward(&mut x);
        x.realize_grad()?;
        std::println!("{}", x.grad().unwrap());
        assert_eq!(x.grad().unwrap(), [[2*i, 2*i, 2*i], [2*i, 2*i, 2*i]]);
    }

    Ok(())
}
