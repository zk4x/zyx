use zyx_core::{error::ZyxError, backend::Backend, scalar::Scalar};
use zyx_core::dtype::DType;

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.randn([2, 3, 4], DType::I32)?;
    let z = &x + &x;
    let x_grad = z.backward([&x]).into_iter().flatten().next().unwrap();
    //std::fs::write("graph.dot", dev.plot_graph([&x, &x_grad])).unwrap();
    //println!("{x}");
    assert_eq!(x_grad, [[[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]], [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]]);
    Ok(())
}

pub fn t1<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([2, 3, 4])?;
    let z = x.sum(..);
    let x_grad = z.backward([&x]).into_iter().flatten().next().unwrap();
    assert_eq!(x_grad, [1, 1, 1]);
    Ok(())
}
