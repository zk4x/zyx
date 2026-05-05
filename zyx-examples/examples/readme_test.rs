// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::{DType, GradientTape, Module, Tensor};
use zyx_nn::{Linear, LayerNorm, Module, MultiheadAttention};
use zyx_optim::AdamW;

#[derive(Module)]
struct TransformerBlock {
    attn: MultiheadAttention,
    mlp: Linear,
    mlp2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    fn new(dim: u64, num_heads: u64, dtype: DType) -> Result<Self, zyx::ZyxError> {
        Ok(Self {
            attn: MultiheadAttention::new(dim, num_heads, 0.0, true, false, false, None, None, true, dtype)?,
            mlp: Linear::new(dim, dim * 4, true, dtype)?,
            mlp2: Linear::new(dim * 4, dim, true, dtype)?,
            norm1: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
            norm2: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, zyx::ZyxError> {
        let attn_out = self.attn.forward(x, x, x, None::<Tensor>, false, None::<Tensor>, true, false)?.0;
        let x = self.norm1.forward(&(x + attn_out))?;
        let mlp_out = self.mlp.forward(&x)?.gelu();
        let mlp_out = self.mlp2.forward(&mlp_out)?;
        Ok(self.norm2.forward(&(x + mlp_out))?)
    }
}

fn main() -> Result<(), zyx::ZyxError> {
    let mut model = TransformerBlock::new(64, 4, DType::F32)?;
    let mut optim = AdamW::default();
    let x = Tensor::randn([2, 8, 64], DType::F32)?;

    let tape = GradientTape::new();
    let out = model.forward(&x)?;
    let grads = tape.gradient(&out, &model);

    // Update parameters with gradients
    optim.update(model.iter_mut(), grads);

    // Realize model to trigger computation (zyx uses lazy evaluation)
    model.realize()?;
    println!("Training step completed!");
    Ok(())
}
