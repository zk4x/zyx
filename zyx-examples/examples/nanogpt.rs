// nanoGPT, credit goes to Andrej Karpathy and great minds who invented parts of this model
// https://github.com/karpathy/nanoGPT

use zyx::{DType, Tensor, ZyxError};
use zyx_nn::{CausalSelfAttention, Embedding, LayerNorm, Linear, Module};

#[derive(Module)]
struct GPTConfig {
    block_size: usize,
    vocab_size: usize,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    dropout: f32,
    bias: bool,
    dtype: DType,
    eps: f64,
}

impl Default for GPTConfig {
    fn default() -> Self {
        return GPTConfig {
            block_size: 1024,
            vocab_size: 50304, // GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            dropout: 0.0,
            bias: true, // True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            dtype: DType::F32,
            eps: 1e-5,
        };
    }
}

#[derive(Module)]
struct MLP {
    c_fc: Linear,
    c_proj: Linear,
    dropout: f32,
}

impl MLP {
    fn init(config: &GPTConfig) -> Result<MLP, ZyxError> {
        Ok(MLP {
            c_fc: Linear::new(config.n_embd, 4 * config.n_embd, config.bias, config.dtype)?,
            c_proj: Linear::new(4 * config.n_embd, config.n_embd, config.bias, config.dtype)?,
            dropout: config.dropout,
        })
    }

    fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let mut x = self.c_fc.forward(x)?;
        x = x.gelu();
        x = self.c_proj.forward(x)?;
        x = x.dropout(self.dropout);
        return Ok(x);
    }
}

#[derive(Module)]
struct Block {
    ln_1: LayerNorm,
    attn: CausalSelfAttention,
    ln_2: LayerNorm,
    mlp: MLP,
}

impl Block {
    fn init(config: &GPTConfig) -> Result<Block, ZyxError> {
        Ok(Block {
            ln_1: LayerNorm::new(config.n_embd, config.eps, true, config.bias, config.dtype)?,
            attn: CausalSelfAttention::new(
                config.n_embd,
                config.n_head,
                config.bias,
                config.dropout,
                config.dtype,
            )?,
            ln_2: LayerNorm::new(config.n_embd, config.eps, true, config.bias, config.dtype)?,
            mlp: MLP::init(config)?,
        })
    }

    fn forward(&self, x: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let mut x = x.into();
        x = &x + self.attn.forward(self.ln_1.forward(&x)?)?;
        x = &x + self.mlp.forward(self.ln_2.forward(&x)?)?;
        return Ok(x);
    }
}

#[derive(Module)]
struct GPT {
    config: GPTConfig,
    wte: Embedding,
    wpe: Embedding,
    h: Vec<Block>,
    ln_f: LayerNorm,
    lm_head: Linear,
}

impl GPT {
    fn init(config: GPTConfig) -> Result<GPT, ZyxError> {
        assert!(config.vocab_size > 0);
        assert!(config.block_size > 0);

        let mut gpt = GPT {
            h: (0..config.n_layer)
                .map(|_| Block::init(&config).unwrap())
                .collect(),
            wte: Embedding::new(config.vocab_size, config.n_embd, config.dtype)?,
            wpe: Embedding::new(config.block_size, config.n_embd, config.dtype)?,
            ln_f: LayerNorm::new(config.n_embd, config.eps, true, config.bias, config.dtype)?,
            lm_head: Linear::new(config.n_embd, config.vocab_size, config.bias, config.dtype)?,
            config,
        };

        gpt.wte.weight = gpt.lm_head.weight.clone();

        // TODO initialize weights
        // if isinstance(module, nn.Linear):
        //    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        // if module.bias is not None:
        //    torch.nn.init.zeros_(module.bias)
        // elif isinstance(module, nn.Embedding):
        //    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        // TODO scaled initialization to residual projections
        //c_proj.weight = torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        Ok(gpt)
    }

    fn get_num_params(&self, non_embedding: bool) -> usize {
        let mut n_params = 0;
        for p in self.into_iter() {
            n_params += p.numel();
        }
        if non_embedding {
            n_params -= self.wpe.weight.numel();
        }
        return n_params;
    }

    fn forward(&self, idx: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let idx = idx.into();
        let [_, t] = idx.shape()[..] else {
            panic!("Input must have 2d shape batch x time")
        };
        assert!(
            t <= self.config.block_size,
            "Time dimensions must be <= block size"
        );
        let pos = Tensor::arange(0, t as i64, 1)?.cast(self.config.dtype);

        let tok_emb = self.wte.forward(idx)?; // [b, t, n_embd]
        let pos_emb = self.wpe.forward(pos)?; // [t, n_embd]
        let mut x = tok_emb + pos_emb;
        for block in &self.h {
            x = block.forward(x)?;
        }
        x = self.ln_f.forward(x)?;

        let logits = self.lm_head.forward(x)?;

        Ok(logits)
    }

    fn generate(
        &self,
        idx: impl Into<Tensor>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Result<Tensor, ZyxError> {
        let mut idx = idx.into();
        for _ in 0..max_new_tokens {
            let idx_cond = if idx.shape()[1] <= self.config.block_size {
                idx.clone()
            } else {
                idx.rslice(-(self.config.block_size as i32)..)?
            };
            let mut logits = self.forward(idx_cond)?;
            logits = logits.slice((.., -1, ..))? / temperature;
            /*if let Some(top_k) = top_k {
                v = logits.topk(top_k.min(logits.shape().last().unwrap()));
                // TODO, probably use where_:
                // logits[logits < v[:, [-1]]] = -float('Inf')
            }*/
            let probs = logits.softmax([-1])?;
            let idx_next = probs.multinomial(1, false)?;
            idx = Tensor::cat([&idx, &idx_next], 1)?;
        }
        Ok(idx)
    }
}

fn main() {}
