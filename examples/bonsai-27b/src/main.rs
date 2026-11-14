// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! CLI for Ternary-Bonsai-2-27B.

use std::io::Write;

use bonsai_27b::Bonsai;
use clap::Parser;
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::SeedableRng;
use tokenizers::Tokenizer;
use zyx::{DType, Tensor, ZyxError};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// GGUF model path.
    model: String,
    /// Tokenizer JSON path. Without it the prompt is read as integer ids.
    #[arg(long)]
    tokenizer_file: Option<String>,
    /// Raw prompt (no chat template).
    #[arg(long, default_value = "Hello")]
    prompt: String,
    /// Tokens to generate.
    #[arg(long, short = 'n', default_value_t = 128)]
    sample_len: usize,
    /// Context window; sizes the full-attention K/V cache.
    #[arg(long, default_value_t = 8192)]
    max_context: i64,
    /// Prompt tokens per forward call.
    #[arg(long, default_value_t = 2048)]
    batch_size: usize,
    /// 0 means greedy.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long, default_value_t = 0.95)]
    top_p: f64,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 299_792_458)]
    seed: u64,
    /// Print the prompt token ids and nothing else.
    #[arg(long)]
    ids_only: bool,
}

struct Sampler {
    rng: rand::rngs::StdRng,
    temperature: f64,
    top_p: f64,
    top_k: usize,
}

impl Sampler {
    fn new(seed: u64, temperature: f64, top_p: f64, top_k: usize) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            top_p,
            top_k,
        }
    }

    fn sample(&mut self, logits: &[f32]) -> u32 {
        let best = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .expect("logits is non-empty");
        if self.temperature <= 0.0 {
            return best;
        }
        let inv = 1.0 / self.temperature;
        let max = logits[best as usize] as f64;
        let mut probs: Vec<(usize, f64)> = logits
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, ((x as f64 - max) * inv).exp()))
            .collect();
        probs.sort_by(|a, b| b.1.total_cmp(&a.1));
        probs.truncate(self.top_k.min(probs.len()));
        let sum: f64 = probs.iter().map(|(_, p)| p).sum();
        for (_, p) in probs.iter_mut() {
            *p /= sum;
        }
        if self.top_p > 0.0 && self.top_p < 1.0 {
            let mut cum = 0.0;
            let mut keep = probs.len();
            for (i, (_, p)) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            let sum: f64 = probs.iter().map(|(_, p)| p).sum();
            for (_, p) in probs.iter_mut() {
                *p /= sum;
            }
        }
        let weights: Vec<f64> = probs.iter().map(|(_, p)| *p).collect();
        let dist = WeightedIndex::new(&weights).expect("positive sampling weights");
        probs[dist.sample(&mut self.rng)].0 as u32
    }
}

fn main() -> Result<(), ZyxError> {
    let args = Args::parse();

    let tokenizer = args
        .tokenizer_file
        .as_deref()
        .map(|path| Tokenizer::from_file(path).map_err(|e| ZyxError::parse_error(e.to_string().into())))
        .transpose()?;
    let ids: Vec<i64> = if let Some(tok) = &tokenizer {
        tok.encode(args.prompt.clone(), true)
            .map_err(|e| ZyxError::parse_error(e.to_string().into()))?
            .get_ids()
            .iter()
            .map(|&id| i64::from(id))
            .collect()
    } else {
        args.prompt
            .split_whitespace()
            .map(|x| x.parse::<i64>().expect("integer token id"))
            .collect()
    };
    if ids.is_empty() {
        return Err(ZyxError::parse_error("prompt produced no token ids".into()));
    }
    if args.ids_only {
        println!("{ids:?}");
        return Ok(());
    }

    let mut model = Bonsai::from_gguf(&args.model, args.max_context, args.batch_size as i64)?;
    let mut sampler = Sampler::new(args.seed, args.temperature, args.top_p, args.top_k);
    let started = std::time::Instant::now();

    let mut pos = 0i64;
    let mut logits: Option<Tensor> = None;
    for chunk in ids.chunks(args.batch_size) {
        let input = Tensor::from_vec(chunk.to_vec(), [chunk.len() as i64])?;
        logits = Some(model.forward(&input, pos)?);
        pos += chunk.len() as i64;
    }
    let logits = logits.expect("at least one prompt chunk");
    let last = logits.squeeze([0]).cast(DType::F32).to_vec::<f32>()?;
    let mut next = sampler.sample(&last);

    let eos = tokenizer
        .as_ref()
        .and_then(|t| t.token_to_id("<|im_end|>"))
        .unwrap_or(248_046);
    let mut generated = 0usize;
    for _ in 0..args.sample_len {
        if let Some(tok) = &tokenizer {
            print!(
                "{}",
                tok.decode(&[next], true)
                    .map_err(|e| ZyxError::parse_error(e.to_string().into()))?
            );
            std::io::stdout().flush().expect("stdout");
        } else {
            print!("{next} ");
            std::io::stdout().flush().expect("stdout");
        }
        generated += 1;
        if next == eos {
            break;
        }
        if generated == args.sample_len {
            break;
        }
        let input = Tensor::from_vec(vec![i64::from(next)], [1i64])?;
        let out = model.forward(&input, pos)?;
        pos += 1;
        let row = out.squeeze([0]).cast(DType::F32).to_vec::<f32>()?;
        next = sampler.sample(&row);
    }
    println!();
    let _ = std::io::stdout().flush();
    eprintln!(
        "{generated} tokens in {:.2}s ({:.2} tok/s)",
        started.elapsed().as_secs_f64(),
        generated as f64 / started.elapsed().as_secs_f64()
    );
    Ok(())
}
