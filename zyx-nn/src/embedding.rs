use zyx::{DType, Tensor, ZyxError};
use zyx_derive::Module;

/// Embedding layer
#[derive(Debug, Module)]
pub struct Embedding {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Embedding size
    pub embed_size: usize,
    /// Weight
    pub weight: Tensor,
    /// Arange
    pub arange: Tensor,
}

impl Embedding {
    /// new embedding layer
    pub fn new(vocab_size: usize, embed_size: usize, dtype: DType) -> Result<Embedding, ZyxError> {
        Ok(Embedding {
            vocab_size,
            embed_size,
            weight: Tensor::glorot_uniform([vocab_size, embed_size], dtype)?
                .reshape([1, 1, vocab_size, embed_size])?,
            arange: Tensor::arange(0, vocab_size as i64, 1)?
                .reshape([1, 1, vocab_size, 1])?
                .cast(dtype),
        })
    }

    /// Initialize embedding using only weight
    pub fn from_params(weight: Tensor) -> Result<Embedding, ZyxError> {
        let sh = weight.shape();
        assert_eq!(sh.len(), 2);
        Ok(Embedding {
            vocab_size: sh[0],
            embed_size: sh[1],
            arange: Tensor::arange(0, sh[0] as i64, 1)?
                .reshape([1, 1, sh[0], 1])?
                .cast(weight.dtype()),
            weight,
        })
    }

    /// Forward embedding layer
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x: Tensor = x.into();
        let x_sh = x.shape();
        if x.numel() == 0 {
            return Ok(Tensor::zeros(
                x_sh.iter()
                    .copied()
                    .chain([self.embed_size])
                    .collect::<Vec<usize>>(),
                x.dtype(),
            ));
        }
        let xdt = x.dtype();
        let wdt = self.weight.dtype();
        if xdt != wdt {
            return Err(ZyxError::DTypeError(
                format!("Embedding::forward input x has dtype {xdt} but weight has dtype {wdt}")
                    .into(),
            ));
        }
        let big_shp: Vec<usize> = x_sh
            .iter()
            .copied()
            .chain([self.vocab_size, self.embed_size])
            .collect();
        let arange = self.arange.expand(big_shp.clone())?;
        let idx = x
            .reshape(x_sh.into_iter().chain([1, 1]).collect::<Vec<usize>>())?
            .expand(big_shp.clone())?;
        let vals = self.weight.expand(big_shp)?;
        (arange.equal(idx)?.cast(xdt) * vals).sum([2])
    }
}
