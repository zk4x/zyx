use zyx::{Tensor, DType};
use crate::Linear;

/// Causal self attention
pub struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    n_embd: usize,
    dropout_p: f32,
}

impl CausalSelfAttention {
    pub fn new(n_embd: usize, n_head: usize, bias: bool, dropout_p: f32, dtype: DType) -> CausalSelfAttention {
        CausalSelfAttention {
            c_attn: Linear::new(n_embd, 3*n_embd, bias, dtype),
            c_proj: Linear::new(n_embd, n_embd, bias, dtype),
            n_head,
            n_embd,
            dropout_p,
        }
    }

    pub fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let mut x: Tensor = x.into();
        let [b, t, c] = x.shape()[..] else { panic!("x must have 3 dims") };
        let mut splits = self.c_attn.forward(x).split(self.n_embd, 2);
        let mut v = splits.pop().unwrap();
        let mut k = splits.pop().unwrap();
        let mut q = splits.pop().unwrap();

        k = k.reshape([b, t, self.n_head, c/self.n_head]).transpose(1, 2);
        q = q.reshape([b, t, self.n_head, c/self.n_head]).transpose(1, 2);
        v = v.reshape([b, t, self.n_head, c/self.n_head]).transpose(1, 2);

        let mut att = q.dot(k.t()) * (1.0/ (*k.shape().last().unwrap() as f64).sqrt());
        // TODO rewrite this
        //att = att.masked_fill(self.bias.get((.., .., ..T, ..T)) == 0, f32::INF);
        att = att.softmax(1);
        att = att.dropout(self.dropout_p);
        let mut y = att.dot(v);
        y = y.transpose(1, 2).reshape([b, t, c]);
        y = self.c_proj.forward(y);
        y = y.dropout(self.dropout_p);
        return y;
    }
}