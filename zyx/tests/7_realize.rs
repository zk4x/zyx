use zyx::{Tensor, ZyxError};

#[test]
fn t01() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);

    for _ in 0..1 {
        let y = x.exp2();
        x = y.log2();

        println!("x rc = {}", x.ref_count());
        println!("y rc = {}", y.ref_count());

        Tensor::debug_graph();
        Tensor::realize([&x])?;
        Tensor::debug_graph();

        println!("x rc = {}", x.ref_count());
        println!("y rc = {}", y.ref_count());
    }

    Tensor::debug_graph();

    Ok(())
}

#[test]
fn t02() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);
    let z = Tensor::from(6);

    for _ in 0..20 {
        let y0 = x.exp2();
        let y1 = y0.exp2() * &z;
        let y2 = y1.exp2() + 3;
        let _y3 = y2.exp2();
        x = y2.log2();
        Tensor::realize([&x])?;
    }

    Ok(())
}

#[test]
fn t03() -> Result<(), ZyxError> {
    let mut x = Tensor::from([[2f32, 3., 4.], [5., 6., 7.]]);
    let z = Tensor::from(6);

    for _ in 0..2 {
        let y0 = x.exp2();
        let y1 = y0.exp2() * &z;
        let y2 = y1.exp2() + 3;
        let _y3 = y2.exp2();
        x = y2.log2();
        Tensor::realize([&x])?;
    }

    Ok(())
}
