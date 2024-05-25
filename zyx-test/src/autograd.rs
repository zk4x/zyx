use zyx::{DType, Tensor};
use crate::assert_eq;

pub fn t0(dtype: DType) {
    assert_eq!(dtype, DType::I32);
    let x = Tensor::randn([2, 3, 4], dtype)?;
    let z = &x + &x;
    let x_grad = z.backward([&x]).into_iter().flatten().next().unwrap();
    //std::fs::write("graph.dot", dev.plot_graph([&x, &x_grad])).unwrap();
    //println!("{x}");
    assert_eq!(
        x_grad,
        [
            [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]],
            [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        ]
    );
    Ok(())
}

pub fn t1(dtype: DType) {
    assert_eq!(dtype, DType::I32);
    let x = Tensor::from([2i32, 3, 4])?;
    let z = x.sum(..);
    let x_grad = z.backward([&x]).into_iter().flatten().next().unwrap();
    assert_eq!(x_grad, [1i32, 1, 1]);
    Ok(())
}
