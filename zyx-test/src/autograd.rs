use zyx_core::{error::ZyxError, backend::Backend, scalar::Scalar};

pub fn t0<T: Scalar>(dev: impl Backend, _: T) -> Result<(), ZyxError> {
    let x = dev.randn([2, 3, 4], T::dtype());
    let z = &x + &x;
    let xgrad = z.backward([&x]).next().unwrap().unwrap();
    println!("xgrad: {xgrad}");
    std::fs::write("t0.dot", dev.plot_graph([&xgrad, &x, &z])).unwrap();
    Ok(())
}

