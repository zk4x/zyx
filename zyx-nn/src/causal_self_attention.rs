use zyx::Tensor;
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
    fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let mut x: Tensor = x.into();
        let [B, T, C] = x.shape()[..] else { panic!("x must have 3 dims") };
        let [q, k, v] = self.c_attn.forward(x).split(self.n_embd, 2)[..] else { panic!() };

        k = k.reshape([B, T, self.n_head, C/self.n_head]).transpose(1, 2);
        q = q.reshape([B, T, self.n_head, C/self.n_head]).transpose(1, 2);
        v = v.reshape([B, T, self.n_head, C/self.n_head]).transpose(1, 2);

        let mut att = q.dot(k.t()) * (1.0/k.shape().last().unwrap().sqrt());
        // TODO rewrite this
        //att = att.masked_fill(self.bias.get((.., .., ..T, ..T)) == 0, f32::INF);
        att = att.softmax(1);
        att = att.dropout(self.dropout_p);
        let mut y = att.dot(v);
        y = y.transpose(1, 2).reshape([B, T, C]);
        y = self.c_proj.forward(y);
        y = y.dropout(self.dropout_p);
        return y;
    }
}

