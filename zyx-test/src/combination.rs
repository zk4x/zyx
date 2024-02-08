use zyx_core::{backend::Backend, error::ZyxError, scalar::Scalar};

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[4, 3, 4], [4, 2, 5]]);
    assert_eq!(x.sum(0), [[8, 5, 9]]);
    assert_eq!(x.sum(1), [[11], [11]]);
    println!("{}", x.sum(()));
    assert_eq!(x.sum(()), [[22]]);
    Ok(())
}

pub fn t1<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.tensor([[2, 4, 3], [5, 2, 4]]);
    let y = dev.tensor([[2, 2, 4], [1, 2, 1], [3, 4, 2]]);
    let z = x.dot(&y);
    //let _ = std::fs::write("matmul.dot", dev.plot_graph([&x, &y, &z]));
    //std::println!("\n{x}\n\n{y}");
    //std::println!("\n{}\n", x.reshape([2, 1, 3]).transpose().expand([2, 3, 3]));
    //std::println!("{}\n", y.reshape([1, 3, 3]).expand([2, 3, 3]));
    //std::println!("{}", z);
    assert_eq!(z, [[17, 24, 18], [24, 30, 30]]);
    Ok(())
}
