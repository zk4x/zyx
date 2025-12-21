use std::collections::HashMap;

use zyx::Module;
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

#[test]
fn t0() -> Result<(), ZyxError> {
    #[derive(Module)]
    struct Linear {
        weight: Tensor,
        bias: Option<Tensor>,
        blah: f32,
    }

    #[derive(Module)]
    struct Net {
        l1: Linear,
    }

    let net = Net {
        l1: Linear {
            weight: Tensor::rand([3, 2], DType::F32)?,
            bias: Some(Tensor::rand([3, 2], DType::F32)?),
            blah: 3.2,
        },
    };

    let blah: HashMap<String, &Tensor> = net.iter_tensors().collect();
    assert_eq!(blah.len(), 2);

    //net.set_params(blah.into_iter());

    Ok(())
}
