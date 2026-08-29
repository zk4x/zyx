// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// nanoGPT, credit goes to Andrej Karpathy and great minds who invented parts of this model
// https://github.com/karpathy/nanoGPT

#![allow(unused)]

use std::collections::HashMap;
use zyx::{DType, Module, ReduceOp, Tape, Tensor, ZyxError};
use zyx_nn::{CausalSelfAttention, Embedding, LayerNorm, Linear, Module};
use zyx_optim::AdamW;

#[derive(Module)]
struct GPTConfig {
    block_size: i64,
    vocab_size: i64,
    n_layer: i64,
    n_head: i64,
    n_embd: i64,
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
            ln_1: LayerNorm::new([config.n_embd], config.eps, true, config.bias, config.dtype)?,
            attn: CausalSelfAttention::new(
                config.n_embd,
                config.n_head,
                config.bias,
                config.dropout,
                config.dtype,
            )?,
            ln_2: LayerNorm::new([config.n_embd], config.eps, true, config.bias, config.dtype)?,
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
            ln_f: LayerNorm::new([config.n_embd], config.eps, true, config.bias, config.dtype)?,
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

    fn get_num_params(&self, non_embedding: bool) -> i64 {
        let mut n_params = 0;
        for p in self.into_iter() {
            n_params += p.numel().item::<i64>();
        }
        if non_embedding {
            n_params -= self.wpe.weight.numel().item::<i64>();
        }
        return n_params;
    }

    fn forward(&self, idx: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let idx = idx.into();
        let shape = idx.shape();
        let t = match shape.as_slice() {
            [_, t] => t.clone(),
            _ => panic!("Input must have 2d shape batch x time"),
        };
        assert!(
            t.item::<i64>() <= self.config.block_size,
            "Time dimensions must be <= block size"
        );
        let pos = Tensor::arange(0, t.item::<i64>(), 1)?.cast(self.config.dtype);

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
            let idx_cond = if idx.shape()[1].item::<i64>() <= self.config.block_size {
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

fn main() -> Result<(), ZyxError> {
    let config = GPTConfig {
        block_size: 256,
        vocab_size: 65,
        n_layer: 6,
        n_head: 6,
        n_embd: 384,
        dropout: 0.2,
        bias: false,
        dtype: DType::F32,
        eps: 1e-5,
    };

    let data = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/data/input.txt"))
        .unwrap();
    let chars: Vec<char> = {
        let mut chars: Vec<char> = data.chars().collect();
        chars.sort();
        chars.dedup();
        chars
    };
    let vocab_size = chars.len() as i64;
    let stoi: HashMap<char, i64> = chars
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i as i64))
        .collect();
    println!("vocab size {vocab_size}");

    let encoded: Vec<i64> = data.chars().map(|c| stoi[&c]).collect();
    let encoded = Tensor::from(encoded);
    let n = encoded.numel().item::<i64>();
    let train_data = encoded.rslice(..(9 * n / 10))?;
    let val_data = encoded.rslice((9 * n / 10)..)?;

    let block_size = config.block_size;
    let mut model = GPT::init(config)?;
    println!("num params (non embedding) {}", model.get_num_params(true));

    let batch_size: i64 = 12;
    let mut optim = AdamW::default();

    // estimate loss
    for (name, split_data) in [("train", &train_data), ("val", &val_data)] {
        let samples = Tensor::randint([batch_size], 0..(split_data.numel().item::<i64>() - block_size - 1))?;
        let ix: Vec<i64> = Vec::try_from(samples.clone())?;
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        for &i in &ix {
            let x_i = split_data.rslice(i..(i + block_size))?;
            let y_i = split_data.rslice((i + 1)..(i + 1 + block_size))?;
            x_batch.push(x_i.reshape([1, block_size])?);
            y_batch.push(y_i.reshape([1, block_size])?);
        }
        let x = Tensor::cat(&x_batch, 0)?;
        let y = Tensor::cat(&y_batch, 0)?;
        let logits = model.forward(&x)?;
        let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
        println!("{name} loss {:.4}", loss.item::<f32>());
    }

    // training loop
    for step in 0..5 {
        let tape = Tape::new(&model)?;
        let samples =
            Tensor::randint([batch_size], 0..(train_data.numel().item::<i64>() - block_size - 1))?;
        let ix: Vec<i64> = Vec::try_from(samples)?;
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        for &i in &ix {
            let x_i = train_data.rslice(i..(i + block_size))?;
            let y_i = train_data.rslice((i + 1)..(i + 1 + block_size))?;
            x_batch.push(x_i.reshape([1, block_size])?);
            y_batch.push(y_i.reshape([1, block_size])?);
        }
        let x = Tensor::cat(&x_batch, 0)?;
        let y = Tensor::cat(&y_batch, 0)?;

        let logits = model.forward(&x)?;
        let loss = logits.cross_entropy(y, ReduceOp::Mean)?;
        let grads = tape.gradient(&loss, &model);
        optim.update(model.iter_mut(), grads);
        tape.realize(model.iter().chain(optim.iter()).chain([&loss]))?;
        println!("step {step}, loss {:.4}", loss.item::<f32>());
    }

    Ok(())
}
