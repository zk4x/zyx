use zyx::{DType, Tensor, ZyxError};

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

    for _ in 0..200 {
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
fn t04() -> Result<(), ZyxError> {
    struct MnistNet {
        l1_weight: Tensor,
        l1_bias: Tensor,
        l2_weight: Tensor,
        l2_bias: Tensor,
    }

    impl MnistNet {
        fn forward(&self, x: &Tensor) -> Tensor {
            let x = x.reshape([0, 784]).unwrap();
            let x = x.matmul(&self.l1_weight.t()).unwrap() + &self.l1_bias;
            let x = x.relu();
            let x = x.matmul(&self.l2_weight.t()).unwrap() + &self.l2_bias;
            x
        }
    }

    let state_dict = Tensor::load("../zyx-examples/models/mnist.safetensors")?;

    let net = MnistNet {
        l1_weight: state_dict["l1.weight"].clone(),
        l1_bias: state_dict["l1.bias"].clone(),
        l2_weight: state_dict["l2.weight"].clone(),
        l2_bias: state_dict["l2.bias"].clone(),
    }; // l1: Linear::new(784, 128, true, dtype)?, l2: Linear::new(128, 10, true, dtype)? };

    let x = Tensor::rand([784], DType::F32)?;
    let x = net.forward(&x);
    println!("{x}");

    Ok(())
}
