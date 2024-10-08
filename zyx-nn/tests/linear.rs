use zyx::{ZyxError, DType, Tensor};

#[test]
fn linear() -> Result<(), ZyxError> {
    use zyx_nn::Linear;

    let l0 = Linear::new(4, 16, true, DType::F32)?;
    println!("{}\n{}", l0.weight, l0.bias.as_ref().unwrap());
    let x = Tensor::randn([8, 4], DType::F32)?;
    let y = l0.forward(x)?.relu();
    println!("{y}");

    Ok(())
}
