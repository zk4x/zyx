use zyx::{Tensor, DType};
use zyx_nn::{Linear, LayerNorm};

#[test]
fn linear_norm_example() -> Result<(), zyx::ZyxError> {
    // Linear layer example
    let linear = Linear::new(128, 64, true, DType::F32)?;
    println!("Linear created: in=128, out=64, bias=true");
    
    let x = Tensor::randn([32, 128], DType::F32)?;
    let y = linear.forward(&x)?;
    println!("Linear output shape: {:?}", y.shape());
    
    // LayerNorm example
    let norm = LayerNorm::new([64], 1e-5, true, true, DType::F32)?;
    println!("LayerNorm created: shape=[64], eps=1e-5, affine=true, bias=true");
    
    let z = norm.forward(&y)?;
    println!("LayerNorm output shape: {:?}", z.shape());
    
    Ok(())
}
