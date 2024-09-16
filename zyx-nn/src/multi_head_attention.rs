use zyx::Tensor;
use zyx_derive::Module;

struct MultiHeadAttention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    embed_dim: usize,
    kdim: usize,
    vdim: usize,
    num_heads: usize,
    dropout_p: f32,
    head_dim: usize,
    add_zero_attn: bool,
}
