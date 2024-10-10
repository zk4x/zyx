use zyx::{Tensor, ZyxError, DType};
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
    pub fn init(vocab_size: usize, embed_size: usize, dtype: DType) -> Result<Embedding, ZyxError> {
        Ok(Embedding {
            vocab_size,
            embed_size,
            weight: Tensor::glorot_uniform([vocab_size, embed_size], dtype)?.reshape([1, 1, vocab_size, embed_size])?,
            arange: Tensor::arange(0, vocab_size as i64, 1)?.reshape([1, 1, vocab_size, 1])?.cast(dtype),
        })
    }

    /// Forward embedding layer
    pub fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let x = x.into();
        let x_sh = x.shape();
        if x.numel() == 0 {
            return Ok(Tensor::zeros(x_sh.iter().copied().chain([self.embed_size]).collect::<Vec<usize>>(), x.dtype()))
        }
        let big_shp: Vec<usize> = x_sh.iter().copied().chain([self.vocab_size, self.embed_size]).collect();
        let arange = self.arange.expand(big_shp.clone())?;
        let idx = x.reshape(x_sh.into_iter().chain([1, 1]).collect::<Vec<usize>>())?.expand(big_shp.clone())?;
        let vals = self.weight.expand(big_shp)?;
        return (arange.equal(idx)? * vals).sum([2])
    }
}
