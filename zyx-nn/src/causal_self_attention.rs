use crate::Linear;
use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Causal self attention
#[derive(Debug, Module)]
pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    dropout_p: f32,
}

impl CausalSelfAttention {
    /// New causal self attention
    pub fn init(
        n_embd: usize,
        n_head: usize,
        bias: bool,
        dropout_p: f32,
        dtype: DType,
    ) -> Result<CausalSelfAttention, ZyxError> {
        Ok(CausalSelfAttention {
            c_attn: Linear::init(n_embd, 3 * n_embd, bias, dtype)?,
            c_proj: Linear::init(n_embd, n_embd, bias, dtype)?,
            n_head,
            dropout_p,
        })
    }

    /// Forward pass of causal self attention
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x: Tensor = x.into();
        let [b, t, c] = x.shape()[..] else {
            return Err(ZyxError::ShapeError(
                "x must have exactly 3 dims, b, t, c".into(),
            ));
        };
        let mut splits = self.c_attn.forward(x)?.split([c, c, c], 2)?;
        let mut v = splits.pop().unwrap();
        let mut k = splits.pop().unwrap();
        let mut q = splits.pop().unwrap();

        k = k
            .reshape([b, t, self.n_head, c / self.n_head])?
            .transpose(1, 2)?;
        q = q
            .reshape([b, t, self.n_head, c / self.n_head])?
            .transpose(1, 2)?;
        v = v
            .reshape([b, t, self.n_head, c / self.n_head])?
            .transpose(1, 2)?;

        let scale = (1.0 / (*k.shape().last().unwrap() as f64).sqrt()) as f32;
        //println!("scale = {scale}");
        let mut att = q.dot(k.t())? * scale;
        //println!("{att}");
        //panic!();

        // TODO rewrite this
        //att = att.masked_fill(self.bias.get((.., .., ..T, ..T)) == 0, f32::INF);
        att = att.softmax([-1])?;
        //println!("{att}");
        // TODO enable dropout
        //att = att.dropout(self.dropout_p)?;
        let mut y = att.dot(v)?;
        y = y.transpose(1, 2)?.reshape([b, t, c])?;
        y = self.c_proj.forward(y)?;
        //y = y.dropout(self.dropout_p)?;
        return Ok(y);
    }
}

#[test]
fn attention1() -> Result<(), ZyxError> {
    Tensor::manual_seed(49340293);
    let n_head = 2;
    let dropout_p = 0.0;

    let attn = CausalSelfAttention {
        c_attn: Linear {
            weight: [
                [-0.495788f32, 0.119697, -0.139357, 0.059328],
                [0.407094, -0.065494, -0.129729, -0.074552],
                [0.324870, 0.155732, 0.297099, -0.412060],
                [0.020193, -0.336263, -0.009602, 0.116321],
                [-0.453359, -0.220178, 0.232500, 0.120824],
                [-0.457052, -0.312347, -0.267674, 0.344709],
                [-0.262033, -0.192330, -0.090726, -0.405672],
                [-0.472127, -0.110653, -0.040921, -0.487143],
                [-0.459970, 0.357617, 0.109131, 0.214290],
                [0.296274, 0.091488, 0.121792, -0.081484],
                [-0.097352, -0.116311, -0.033035, 0.236983],
                [0.078229, 0.294886, 0.363787, -0.383411],
            ]
            .into(),
            bias: None,
        },
        c_proj: Linear {
            weight: [
                [-0.202461f32, -0.263050, -0.244990, 0.044416],
                [-0.398643, 0.219820, 0.253934, 0.204294],
                [-0.323065, 0.195841, -0.106940, 0.142828],
                [0.233007, -0.026790, -0.293228, 0.118043],
            ]
            .into(),
            bias: None,
        },
        n_head,
        dropout_p,
    };

    let mut x = Tensor::from([[
        [-1.363837f32, -0.801618, -1.304842, -1.664811],
        [-0.385430, -0.955608, -1.003842, 0.073811],
        [-0.785831, 1.030346, 0.593785, -0.214361],
    ]]);

    for _ in 0..5 {
        x = attn.forward(x)?;
        Tensor::realize([&x])?;
    }

    //println!("{x:.8}");

    /*assert_eq!(
        x,
        [[[ 0.04401812f32, -0.14199661, -0.11446018,  0.03237676],
         [ 0.05587596, -0.12779452, -0.10237779,  0.02773934],
         [ 0.03065444, -0.15511249, -0.12694199,  0.03479434]]]
    );*/

    // after 5 iterations
    assert_eq!(
        x,
        [[[-1.34166388e-04f32, -3.10145377e-04, -3.39602208e-04,  2.14193460e-05],
         [-1.34166388e-04, -3.10145377e-04, -3.39602208e-04,  2.14193460e-05],
         [-1.34166388e-04, -3.10145377e-04, -3.39602208e-04,  2.14193460e-05]]]
    );

    Ok(())
}
