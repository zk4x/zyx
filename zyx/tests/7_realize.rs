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
fn pad_1() -> Result<(), ZyxError> {
    let x = Tensor::arange(0, 20, 1)?.reshape([4, 5])?;
    assert_eq!(x.rslice(3)?, [[3], [8], [13], [18]]);
    Ok(())
}

#[test]
fn t_15() {
    let mut x = Tensor::from([[2, 3, 1], [2, 4, 1]]);
    for _ in 0..10 {
        x = &x + &x;
        //println!("{x}");
        //Tensor::plot_graph([], &format!("graph{i}"));
        //Tensor::realize([&x]).unwrap();
    }
    //println!("{x}");
    assert_eq!(x, [[2048, 3072, 1024], [2048, 4096, 1024]]);
}

#[test]
fn iter1() -> Result<(), ZyxError> {
    let mut x = Tensor::randn([64, 64], DType::F32)?;
    let y = Tensor::randn([64, 64], DType::F32)?;

    for _ in 0..20 {
        x = x.dot(&y)?.softmax([-1])?;
        Tensor::realize([&x])?;
        //println!("{}", x.is_realized());
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
    };

    let x = Tensor::arange(0f32, 784. * 184., 1.)?;
    let x = net.forward(&x);

    Tensor::realize([&x])?;
    //println!("{x}");

    Ok(())
}
