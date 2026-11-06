// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Llama CLI: loads weights, samples tokens, prints text.
//!
//! The model itself lives in the `llama` library target (`src/lib.rs`), so
//! benchmarks and other examples can reuse it without going through the CLI.

use std::collections::HashMap;

use clap::Parser;
use llama::{Llama, LlamaConfig, load_weights, remap_gguf_weights};
use rand::{distr::Distribution, SeedableRng};
use tokenizers::Tokenizer;
use zyx::{DType, Tensor, ZyxError};

fn parse_dtype(s: &str) -> Result<DType, String> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        _ => Err(format!("unsupported dtype: {s} (use f32, f16, bf16)")),
    }
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    fn sample_argmax(&mut self, logits: &Tensor) -> u32 {
        let logits_v: Vec<f32> = logits.clone().cast(DType::F32).try_into().unwrap();
        logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> u32 {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).unwrap();
        distr.sample(&mut self.rng) as u32
    }

    fn sample_topp(&mut self, prs: &mut [f32], top_p: f32) -> u32 {
        let mut argsort: Vec<usize> = (0..prs.len()).collect();
        argsort.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
        let mut cumsum = 0.0;
        for &i in &argsort {
            if cumsum >= top_p {
                prs[i] = 0.0;
            } else {
                cumsum += prs[i];
            }
        }
        self.sample_multinomial(prs)
    }

    fn sample_topk(&mut self, prs: &mut [f32], top_k: usize) -> u32 {
        if top_k >= prs.len() {
            return self.sample_multinomial(prs);
        }
        let mut argsort: Vec<usize> = (0..prs.len()).collect();
        let (indices, _, _) = argsort.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
        let top_prs: Vec<f32> = indices.iter().map(|&i| prs[i]).collect();
        let index = self.sample_multinomial(&top_prs);
        indices[index as usize] as u32
    }

    pub fn sample(&mut self, logits: &Tensor) -> u32 {
        let logits = logits.cast(DType::F32);
        match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(&logits),
            Sampling::All { temperature: _ } => {
                let prs = logits.softmax([-1]).unwrap();
                let prs: Vec<f32> = prs.try_into().unwrap();
                self.sample_multinomial(&prs)
            }
            Sampling::TopK { k, temperature } => {
                let logits = &logits / *temperature;
                let prs = logits.softmax([-1]).unwrap();
                let mut prs: Vec<f32> = prs.try_into().unwrap();
                self.sample_topk(&mut prs, *k)
            }
            Sampling::TopP { p, temperature } => {
                let logits = &logits / *temperature;
                let prs = logits.softmax([-1]).unwrap();
                let mut prs: Vec<f32> = prs.try_into().unwrap();
                if *p <= 0.0 || *p >= 1.0 {
                    self.sample_multinomial(&prs)
                } else {
                    self.sample_topp(&mut prs, *p as f32)
                }
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let logits = &logits / *temperature;
                let prs = logits.softmax([-1]).unwrap();
                let prs: Vec<f32> = prs.try_into().unwrap();
                let mut argsort: Vec<usize> = (0..prs.len()).collect();
                let (indices, _, _) = argsort.select_nth_unstable_by(*k, |&i, &j| prs[j].total_cmp(&prs[i]));
                let mut prs_topk = vec![0.0f32; prs.len()];
                let sum: f32 = indices.iter().map(|&i| prs[i]).sum();
                for &i in indices.iter() {
                    prs_topk[i] = prs[i] / sum;
                }
                if *p <= 0.0 || *p >= 1.0 {
                    self.sample_multinomial(&prs_topk)
                } else {
                    self.sample_topp(&mut prs_topk, *p as f32)
                }
            }
        }
    }
}

pub fn apply_repeat_penalty(logits: &Tensor, penalty: f32, context: &[u32]) -> Tensor {
    let mut logits: Vec<f32> = logits.cast(DType::F32).try_into().unwrap();
    let mut seen = std::collections::HashSet::new();
    for &tid in context {
        if seen.contains(&tid) {
            continue;
        }
        seen.insert(tid);
        if let Some(logit) = logits.get_mut(tid as usize) {
            if *logit >= 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
    Tensor::from(logits)
}

struct TextGeneration {
    model: Llama,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    fn new(
        model: Llama,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        verbose_prompt: bool,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<(), ZyxError> {
        use std::io::Write;
        let encoding = self.tokenizer.encode(prompt, true).unwrap();
        if encoding.is_empty() {
            panic!("Empty prompts are not supported.");
        }
        if self.verbose_prompt {
            for (token, id) in encoding.get_tokens().iter().zip(encoding.get_ids().iter()) {
                println!("{id:7} -> '{}'", token.replace('▁', " ").replace("<0x0A>", "\n"));
            }
        }
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let eos_token = self.tokenizer.token_to_id("<|end_of_text|>").unwrap_or(2);
        let mut generated_tokens = 0usize;
        println!("prompt:\n{prompt}");
        std::io::stdout().flush().unwrap();
        let start_gen = std::time::Instant::now();

        let mut start_pos = 0usize;
        // prefill: process the entire prompt at once
        let input = Tensor::from(tokens.clone()).unsqueeze(0).unwrap();
        let mut logits = self.model.forward(&input, start_pos);
        logits = logits.squeeze([0]);
        let next_token = self.logits_processor.sample(&logits);
        let token_str = self.tokenizer.decode(&[next_token], true).unwrap();
        println!("{token_str}={next_token}");
        std::io::stdout().flush().unwrap();
        start_pos = tokens.len();
        let mut last_tok = next_token;

        // generate one token at a time
        for _ in 1..sample_len {
            let input = Tensor::from(vec![last_tok]).unsqueeze(0).unwrap();
            let mut logits = self.model.forward(&input, start_pos);
            logits = logits.squeeze([0]);
            let logits = if self.repeat_penalty == 1.0 {
                logits
            } else {
                apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[tokens.len().saturating_sub(self.repeat_last_n)..])
            };
            let next_token = self.logits_processor.sample(&logits);
            let token_str = self.tokenizer.decode(&[next_token], true).unwrap();
            println!("{token_str}={next_token}");
            std::io::stdout().flush().unwrap();
            start_pos += 1;
            last_tok = next_token;
            generated_tokens += 1;
            if next_token == eos_token {
                break;
            }
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64()
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    verbose_prompt: bool,
    #[arg(long)]
    prompt: Option<String>,
    #[arg(long)]
    temperature: Option<f64>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 299792458)]
    seed: u64,
    #[arg(long, short = 'n', default_value_t = 500)]
    sample_len: usize,
    #[arg(long)]
    model_id: Option<String>,
    #[arg(long)]
    weight_file: Option<String>,
    #[arg(long)]
    tokenizer_file: Option<String>,
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    #[arg(long, default_value_t = 4096)]
    max_context: usize,

    #[arg(long, value_parser = parse_dtype, default_value_t = DType::F16)]
    dtype: DType,
    #[arg(long)]
    config_file: Option<String>,
}

fn main() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    let args = Args::parse();
    let weight_file = args.weight_file.unwrap_or_else(|| "model.safetensors".to_string());
    let tokenizer_file = args.tokenizer_file.unwrap_or_else(|| "tokenizer.json".to_string());
    let tokenizer = Tokenizer::from_file(&tokenizer_file).unwrap();

    let cfg: LlamaConfig = match &args.config_file {
        Some(path) => {
            let content = std::fs::read_to_string(path).unwrap();
            serde_json::from_str(&content).unwrap()
        }
        None => LlamaConfig {
            max_context: args.max_context,
            ..LlamaConfig::default()
        },
    };

    let mut vb: HashMap<String, Tensor> = if weight_file.ends_with(".gguf") {
        let (_metadata, tensors) = Tensor::load_gguf(&weight_file).unwrap();
        let mapped = remap_gguf_weights(tensors);
        mapped
    } else {
        load_weights(&weight_file)
    };
    let model = Llama::new(&cfg, &mut vb, args.dtype);

    let prompt = args.prompt.clone().unwrap_or_else(|| "Hello".to_string());
    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        args.verbose_prompt,
    );
    pipeline.run(&prompt, args.sample_len).unwrap();
    Ok(())
}
