use std::collections::HashMap;

use clap::Parser;
use rand::{distr::Distribution, SeedableRng};
use serde::Deserialize;
use tokenizers::Tokenizer;
use zyx::{DType, Tape, Tensor, ZyxError};
use zyx_nn::{Linear, RMSNorm};

fn parse_dtype(s: &str) -> Result<DType, String> {
    match s.to_lowercase().as_str() {
        "f32" => Ok(DType::F32),
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::BF16),
        _ => Err(format!("unsupported dtype: {s} (use f32, f16, bf16)")),
    }
}

fn repeat_kv(xs: Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return xs;
    }
    let [b_sz, n_kv_head, seq_len, head_dim] = xs.dims::<4>().unwrap();
    // seq_len may be symbolic: pass its dim tensor directly (NOT -1) so the
    // inferred dim keeps the original TensorId — the merge-time provability
    // checks in binary/assign require the SAME dim tensor, not just equal
    // values.
    xs.unsqueeze(2)
        .unwrap()
        .expand([-1i64, -1, n_rep as i64, -1, -1])
        .unwrap()
        .reshape([
            b_sz.clone(),
            n_kv_head * (n_rep as i64),
            seq_len.clone(),
            head_dim.clone(),
        ])
        .unwrap()
}

fn get_mask(size: u64) -> Tensor {
    let size_u = size as usize;
    let mask: Vec<f32> = (0..size_u)
        .flat_map(|i| (0..size_u).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    Tensor::from(mask).reshape([size, size]).unwrap()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub factor: f64,
    pub high_freq_factor: f64,
    pub low_freq_factor: f64,
    pub original_max_position_embeddings: usize,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub head_dim: Option<usize>,
    pub rope_scaling: Option<RopeScaling>,
    pub tie_word_embeddings: bool,
    #[serde(default = "default_max_context")]
    pub max_context: usize,
}

#[derive(Deserialize)]
struct ShardedIndex {
    weight_map: HashMap<String, String>,
}

const fn default_max_context() -> usize {
    4096
}

impl LlamaConfig {
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

impl Default for LlamaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128256,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            head_dim: Some(128),
            rope_scaling: None,
            tie_word_embeddings: false,
            max_context: 4096,
        }
    }
}

fn precompute_rope_freqs(cfg: &LlamaConfig) -> (Tensor, Tensor) {
    let head_dim = cfg.head_dim();
    let theta = cfg.rope_theta;
    let mut inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / theta.powf(i as f64 / head_dim as f64) as f32)
        .collect();

    if let Some(scaling) = &cfg.rope_scaling {
        let low_wavelen = scaling.original_max_position_embeddings as f64 / scaling.low_freq_factor;
        let high_wavelen =
            scaling.original_max_position_embeddings as f64 / scaling.high_freq_factor;
        let factor = scaling.factor;
        for freq in inv_freq.iter_mut() {
            let wavelen = 2.0 * std::f64::consts::PI / *freq as f64;
            if wavelen > low_wavelen {
                *freq = (*freq as f64 / factor) as f32;
            } else if wavelen < high_wavelen {
                // keep original
            } else {
                let smooth = (scaling.original_max_position_embeddings as f64 / wavelen
                    - scaling.low_freq_factor)
                    / (scaling.high_freq_factor - scaling.low_freq_factor);
                *freq = ((1.0 - smooth) * *freq as f64 / factor + smooth * *freq as f64) as f32;
            }
        }
    }

    let inv_freq_len = inv_freq.len() as i64;
    let inv_freq = Tensor::from(inv_freq).reshape([1, inv_freq_len]).unwrap();
    let max_pos = cfg.max_position_embeddings as i64;
    let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, 1)
        .unwrap()
        .cast(DType::F32)
        .reshape([max_pos, 1])
        .unwrap();
    let freqs = t.matmul(&inv_freq).unwrap();
    (freqs.cos(), freqs.sin())
}

fn apply_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor, seqlen_offset: &Tensor) -> Tensor {
    let [seq_len, _hd] = xs.rdims::<2>().unwrap();
    let c = cos.narrow(0, seqlen_offset, &seq_len).unwrap();
    let s = sin.narrow(0, seqlen_offset, &seq_len).unwrap();
    xs.rope(c, s).unwrap()
}

trait VarMap {
    fn remove_prefix(&mut self, prefix: &str) -> Self;
    fn take(&mut self, key: &str) -> Tensor;
}

impl VarMap for HashMap<String, Tensor> {
    fn remove_prefix(&mut self, prefix: &str) -> Self {
        let p = format!("{prefix}.");
        let mut res = HashMap::new();
        let paths: Vec<String> = self.keys().filter(|k| k.starts_with(&p)).cloned().collect();
        for path in paths {
            let t = self.remove(&path).unwrap();
            let stripped = path.strip_prefix(&p).unwrap().to_string();
            res.insert(stripped, t);
        }
        res
    }

    fn take(&mut self, key: &str) -> Tensor {
        self.remove(key)
            .unwrap_or_else(|| panic!("key {key} not found in weight map"))
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    cache_k: Tensor,
    cache_v: Tensor,
    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    cos: Tensor,
    sin: Tensor,
}

impl Attention {
    fn new(
        cfg: &LlamaConfig,
        cos: &Tensor,
        sin: &Tensor,
        vb: &mut HashMap<String, Tensor>,
        dtype: DType,
    ) -> Self {
        let num_heads = cfg.num_attention_heads as i64;
        let num_kv_heads = cfg.num_key_value_heads() as i64;
        let head_dim = cfg.head_dim() as i64;
        let q_proj = Linear {
            weight: vb.take("q_proj.weight"),
            bias: None,
        };
        let k_proj = Linear {
            weight: vb.take("k_proj.weight"),
            bias: None,
        };
        let v_proj = Linear {
            weight: vb.take("v_proj.weight"),
            bias: None,
        };
        let o_proj = Linear {
            weight: vb.take("o_proj.weight"),
            bias: None,
        };
        let max_context = cfg.max_context as i64;
        let cache_k = Tensor::zeros([max_context, num_kv_heads, head_dim], dtype)
            .contiguous()
            .unwrap();
        let cache_v = Tensor::zeros([max_context, num_kv_heads, head_dim], dtype)
            .contiguous()
            .unwrap();
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            cache_k,
            cache_v,
            num_heads,
            num_kv_heads,
            head_dim,
            cos: cos.clone(),
            sin: sin.clone(),
        }
    }

    fn forward(&mut self, xs: &Tensor, start_pos: &Tensor, cache_len: &Tensor) -> Tensor {
        let [b_size, seq_len, _n_embd] = xs.dims::<3>().unwrap();
        let q = self.q_proj.forward(xs).unwrap();
        let k = self.k_proj.forward(xs).unwrap();
        let v = self.v_proj.forward(xs).unwrap();

        let q = q
            .reshape([
                b_size.clone(),
                seq_len.clone(),
                self.num_heads.into(),
                self.head_dim.into(),
            ])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let k = k
            .reshape([
                b_size.clone(),
                seq_len.clone(),
                self.num_kv_heads.into(),
                self.head_dim.into(),
            ])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = v
            .reshape([
                b_size.clone(),
                seq_len.clone(),
                self.num_kv_heads.into(),
                self.head_dim.into(),
            ])
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let q = apply_rope(&q, &self.cos, &self.sin, start_pos);
        let k = apply_rope(&k, &self.cos, &self.sin, start_pos);

        // update the preallocated cache using assign
        // (offset is a variable, so the kernel shape never changes)
        let k_assign = k.squeeze([0]).transpose(0, 1).unwrap();
        let v_assign = v.squeeze([0]).transpose(0, 1).unwrap();
        self.cache_k
            .narrow(0, start_pos, &seq_len)
            .unwrap()
            .assign(&k_assign)
            .unwrap();
        self.cache_v
            .narrow(0, start_pos, &seq_len)
            .unwrap()
            .assign(&v_assign)
            .unwrap();

        // read back the full cache up to start_pos+seq_len.
        // cache_len is a variable: the read shape [1, H, L_var, D] is identical
        // on every decode step, so the kernels compile exactly once.
        let k = self
            .cache_k
            .narrow(0, 0i64, cache_len.clone())
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let v = self
            .cache_v
            .narrow(0, 0i64, cache_len.clone())
            .unwrap()
            .unsqueeze(0)
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        let n_kv_groups = (self.num_heads / self.num_kv_heads) as usize;
        let k = repeat_kv(k, n_kv_groups);
        let v = repeat_kv(v, n_kv_groups);

        let scale = Tensor::from(1.0f32 / (self.head_dim as f32).sqrt()).cast(q.dtype());
        let attn = q.matmul(k.transpose(2, 3).unwrap()).unwrap() * scale;

        let seq_len_i = seq_len.item::<i64>();
        let attn = if seq_len_i <= 1 {
            attn
        } else {
            let mask = get_mask(seq_len_i as u64).cast(attn.dtype());
            attn + mask
        };
        let attn = attn.softmax([-1]).unwrap();
        let out = attn.matmul(&v).unwrap();
        let out = out.transpose(1, 2).unwrap();
        let d = self.num_heads * self.head_dim;
        let out = out
            .reshape([b_size.clone(), seq_len.clone(), d.into()])
            .unwrap();
        let out = self.o_proj.forward(out).unwrap();
        out
    }
}

struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn new(vb: &mut HashMap<String, Tensor>) -> Self {
        let gate_proj = Linear {
            weight: vb.take("gate_proj.weight"),
            bias: None,
        };
        let up_proj = Linear {
            weight: vb.take("up_proj.weight"),
            bias: None,
        };
        let down_proj = Linear {
            weight: vb.take("down_proj.weight"),
            bias: None,
        };
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(xs).unwrap().swish();
        let up = self.up_proj.forward(xs).unwrap();
        self.down_proj.forward(gate * up).unwrap()
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn new(
        cfg: &LlamaConfig,
        cos: &Tensor,
        sin: &Tensor,
        vb: &mut HashMap<String, Tensor>,
        dtype: DType,
    ) -> Self {
        let self_attn = Attention::new(cfg, cos, sin, &mut vb.remove_prefix("self_attn"), dtype);
        let mlp = MLP::new(&mut vb.remove_prefix("mlp"));
        let input_layernorm = RMSNorm {
            scale: vb.take("input_layernorm.weight").cast(dtype),
            eps: cfg.rms_norm_eps,
        };
        let post_attention_layernorm = RMSNorm {
            scale: vb.take("post_attention_layernorm.weight").cast(dtype),
            eps: cfg.rms_norm_eps,
        };
        Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        }
    }

    fn forward(&mut self, xs: &Tensor, start_pos: &Tensor, cache_len: &Tensor) -> Tensor {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs).unwrap();
        let attn_out = self.self_attn.forward(&xs, start_pos, cache_len);
        let xs = attn_out + residual;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(xs).unwrap();
        let mlp_out = self.mlp.forward(&xs);
        let out = mlp_out + residual;
        out
    }
}

fn embedding_forward(weight: &Tensor, input: &Tensor) -> Tensor {
    let [b_size, seq_len] = input.dims::<2>().unwrap();
    let [vocab_size, embed_size] = weight.dims::<2>().unwrap();
    let idx = input
        .cast(DType::F32)
        .reshape([b_size, seq_len, 1i64.into(), 1i64.into()])
        .unwrap();
    let arange = Tensor::arange(0, vocab_size.item::<i64>(), 1)
        .unwrap()
        .reshape([1i64.into(), 1i64.into(), vocab_size.clone(), 1i64.into()])
        .unwrap()
        .cast(DType::F32);
    let w = weight
        .reshape([1i64.into(), 1i64.into(), vocab_size, embed_size])
        .unwrap();
    let one_hot = arange.equal(idx).unwrap().cast(w.dtype());
    (one_hot * w).sum([2]).unwrap()
}

struct Llama {
    embed_weight: Tensor,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Linear,
}

impl Llama {
    fn new(cfg: &LlamaConfig, vb: &mut HashMap<String, Tensor>, dtype: DType) -> Self {
        let (cos, sin) = precompute_rope_freqs(cfg);
        let cos = cos.cast(dtype);
        let sin = sin.cast(dtype);

        let embed_weight = vb.take("model.embed_tokens.weight");
        let lm_head_weight = if cfg.tie_word_embeddings {
            embed_weight.clone()
        } else {
            vb.take("lm_head.weight")
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut vb_layers = vb.remove_prefix("model.layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(
                cfg,
                &cos,
                &sin,
                &mut vb_layers.remove_prefix(&i.to_string()),
                dtype,
            );
            layers.push(layer);
        }
        let norm = RMSNorm {
            scale: vb.take("model.norm.weight").cast(dtype),
            eps: cfg.rms_norm_eps,
        };
        let lm_head = Linear {
            weight: lm_head_weight,
            bias: None,
        };
        Self {
            embed_weight,
            layers,
            norm,
            lm_head,
        }
    }

    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> Tensor {
        let [_b_size, seq_len] = input_ids.dims::<2>().unwrap();
        let tape = Tape::empty();
        tape.add(&self.embed_weight).unwrap();
        for layer in &self.layers {
            tape.add(&layer.input_layernorm.scale).unwrap();
            tape.add(&layer.post_attention_layernorm.scale).unwrap();
            tape.add(&layer.self_attn.q_proj.weight).unwrap();
            tape.add(&layer.self_attn.k_proj.weight).unwrap();
            tape.add(&layer.self_attn.v_proj.weight).unwrap();
            tape.add(&layer.self_attn.o_proj.weight).unwrap();
            tape.add(&layer.self_attn.cos).unwrap();
            tape.add(&layer.self_attn.sin).unwrap();
            tape.add(&layer.self_attn.cache_k).unwrap();
            tape.add(&layer.self_attn.cache_v).unwrap();
            tape.add(&layer.mlp.gate_proj.weight).unwrap();
            tape.add(&layer.mlp.up_proj.weight).unwrap();
            tape.add(&layer.mlp.down_proj.weight).unwrap();
        }
        tape.add(&self.norm.scale).unwrap();
        tape.add(&self.lm_head.weight).unwrap();
        let mut xs = embedding_forward(&self.embed_weight, input_ids);
        // Symbolic positions: fresh variables each call, but identical kernel
        // IR every step (params hash by ordinal, not value) — one compile total.
        let pos = Tensor::variable(start_pos as i64);
        let cache_len = Tensor::variable(start_pos as i64 + seq_len.item::<i64>());
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &pos, &cache_len);
        }
        xs = self.norm.forward(xs).unwrap();
        xs = xs.narrow(1, seq_len - 1i64, 1i64).unwrap().squeeze([1]);
        let out = self.lm_head.forward(xs).unwrap();
        let mut realize_args: Vec<&Tensor> = vec![&out];
        for layer in &self.layers {
            realize_args.push(&layer.self_attn.cache_k);
            realize_args.push(&layer.self_attn.cache_v);
        }
        tape.realize(realize_args).unwrap();
        out
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
        let (indices, _, _) =
            argsort.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
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
                let (indices, _, _) =
                    argsort.select_nth_unstable_by(*k, |&i, &j| prs[j].total_cmp(&prs[i]));
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
                println!(
                    "{id:7} -> '{}'",
                    token.replace('▁', " ").replace("<0x0A>", "\n")
                );
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
                apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[tokens.len().saturating_sub(self.repeat_last_n)..],
                )
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

fn remap_gguf_weights(mut gguf: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    if let Some(t) = gguf.remove("token_embd.weight") {
        // GGUF stores [embed_dim, vocab_size]; we need [vocab_size, embed_dim]
        out.insert("model.embed_tokens.weight".to_string(), t.t());
    }
    if let Some(t) = gguf.remove("output_norm.weight") {
        out.insert("model.norm.weight".to_string(), t);
    }
    if let Some(t) = gguf.remove("output.weight") {
        out.insert("lm_head.weight".to_string(), t);
    }
    let layer_keys: Vec<String> = gguf
        .keys()
        .filter(|k| k.starts_with("blk."))
        .cloned()
        .collect();
    for key in layer_keys {
        let parts: Vec<&str> = key.splitn(4, '.').collect();
        if parts.len() < 4 {
            continue;
        }
        let layer: usize = parts[1].parse().unwrap_or(0);
        let tensor = gguf.remove(&key).unwrap();
        // GGUF stores weights as [in_features, out_features]; zyx-nn Linear
        // does x.dot(weight.t()) expecting [out_features, in_features]. Transpose.
        let tensor = match parts[2] {
            "attn_norm" | "ffn_norm" => tensor,
            _ => tensor.t(),
        };
        let hf_key = match parts[2] {
            "attn_q" => format!("model.layers.{layer}.self_attn.q_proj.weight"),
            "attn_k" => format!("model.layers.{layer}.self_attn.k_proj.weight"),
            "attn_v" => format!("model.layers.{layer}.self_attn.v_proj.weight"),
            "attn_output" => format!("model.layers.{layer}.self_attn.o_proj.weight"),
            "attn_norm" => format!("model.layers.{layer}.input_layernorm.weight"),
            "ffn_gate" => format!("model.layers.{layer}.mlp.gate_proj.weight"),
            "ffn_down" => format!("model.layers.{layer}.mlp.down_proj.weight"),
            "ffn_up" => format!("model.layers.{layer}.mlp.up_proj.weight"),
            "ffn_norm" => format!("model.layers.{layer}.post_attention_layernorm.weight"),
            _ => continue,
        };
        out.insert(hf_key, tensor);
    }
    out
}

fn load_weights(weight_path: &str) -> HashMap<String, Tensor> {
    let path = std::path::Path::new(weight_path);
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "json" => {
            let content = std::fs::read_to_string(weight_path).unwrap();
            let index: ShardedIndex = serde_json::from_str(&content).unwrap();
            let base = path.parent().unwrap();
            let mut shard_files: Vec<&String> = index.weight_map.values().collect();
            shard_files.sort();
            shard_files.dedup();
            let mut all_weights = HashMap::new();
            for f in shard_files {
                let shard_path = base.join(f);
                let weights = Tensor::load(&shard_path).unwrap();
                all_weights.extend(weights);
            }
            all_weights
        }
        _ => Tensor::load(weight_path).unwrap(),
    }
}

fn main() -> Result<(), ZyxError> {
    Tensor::set_implicit_casts(false);
    let args = Args::parse();
    let weight_file = args
        .weight_file
        .unwrap_or_else(|| "model.safetensors".to_string());
    let tokenizer_file = args
        .tokenizer_file
        .unwrap_or_else(|| "tokenizer.json".to_string());
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
