use zyx::Tensor;
use zyx_derive::Module;

struct ScaledDotProductAttention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    d: usize,
    mask: bool,
}

impl ScaledDotProductAttention {
    fn new(dtype: DType) -> Self {
        Self {
            x: Tensor::rand([1024, 1024], dtype),
            y: Tensor::rand([1024, 1024], dtype),
            z: Tensor::rand([1024, 1024], dtype),
        }
    }

    fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let mut x: Tensor = x.into();
        x = q.dot(k).div(d);
        if self.mask {
            //x = x.mask()
            todo!()
        }
        x = x.softmax();
        x = x.dot(v);
        return x
    }
}

#[derive(Module)]
struct CausalSelfAttention {
}

impl CausalSelfAttention {
    fn new(n_embd: usize, n_head: usize) -> Self {
        let c_attn = Linear::new(n_embd, 3 * n_embd);
    }

    fn forward(&self) {
        todo!()
    }
}

