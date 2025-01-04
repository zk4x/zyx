// Implementation based on candle rust
// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py

use std::collections::HashMap;
use tokenizers::Tokenizer;
use zyx::{DType, Tensor, ZyxError};
use zyx_nn::{Embedding, LayerNorm, Linear};

/*fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor, ZyxError> {
    let [bs, seqlen, n_kv_heads, head_dim] = x.shape()[..] else {
        panic!()
    };
    if n_rep == 1 {
        return Ok(x);
    }
    return x
        .repeat([1, 1, 1, n_rep])
        .unwrap()
        .reshape([bs, seqlen, n_kv_heads * n_rep, head_dim]);
}*/

fn repeat_kv(xs: Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        xs
    } else {
        let [b_sz, n_kv_head, seq_len, head_dim] = xs.shape()[..] else { panic!() };
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(vec![&xs; n_rep], 2).unwrap().reshape([b_sz, n_kv_head * n_rep, seq_len, head_dim]).unwrap()
    }
}

trait VarMap {
    fn g(&mut self, p: &str) -> Self;
    fn t(&mut self, t: &str) -> Tensor;
}

impl VarMap for HashMap<String, Tensor> {
    fn g(&mut self, p: &str) -> Self {
        // TODO fix this to work with dots
        let p = format!("{p}.");
        let mut res = HashMap::new();
        let paths: Vec<String> = self.keys().filter(|k| k.starts_with(&p)).cloned().collect();
        //println!("Filtered paths: {:?}", paths);
        for mut path in paths {
            let t = self.remove(&path).unwrap();
            for _ in 0..p.len() {
                path.remove(0);
            }
            //println!("Inserting at {path:?}");
            res.insert(path, t);
        }
        res
    }

    fn t(&mut self, t: &str) -> Tensor {
        if let Some(x) = self.remove(t) {
            return x;
        } else {
            let mut keys: Vec<String> = self.keys().cloned().collect();
            keys.sort();
            //println!("{:?}", keys);
            panic!("The above keys did not contain {t}");
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
enum Activation {
    ReLU,
}

impl Activation {
    fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let x = x.into();
        match self {
            Activation::ReLU => x.relu(),
        }
    }
}

/// Phi model.
/// https://huggingface.co/microsoft/phi-2
/// There is an alternative implementation of the phi model in mixformers.rs.
/// This corresponds to the model update made with the following commit:
/// https://huggingface.co/microsoft/phi-2/commit/cb2f4533604d8b67de604e7df03bfe6f3ca22869
use serde::Deserialize;

// https://huggingface.co/microsoft/phi-2/blob/main/configuration_phi.py
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: Option<usize>,
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) layer_norm_eps: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) partial_rotary_factor: f64,
    pub(crate) qk_layernorm: bool,
}

impl Config {
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    dim: usize,
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &Config) -> Result<Self, ZyxError> {
        let dim = (cfg.partial_rotary_factor * cfg.head_dim() as f64) as usize;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from(inv_freq).reshape([1, inv_freq_len]).unwrap();
        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, 1)
            .unwrap()
            .cast(DType::F32)
            .reshape((cfg.max_position_embeddings, 1))
            .unwrap();
        let freqs = t.matmul(&inv_freq).unwrap();
        Ok(Self {
            dim,
            sin: freqs.sin(),
            cos: freqs.cos(),
        })
    }

    fn apply_rotary_emb(&self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor, ZyxError> {
        let [_b_size, _num_heads, seq_len, _headdim] = xs.shape()[..] else {
            panic!()
        };
        let xs_rot = xs.get((.., .., .., ..self.dim as isize)).unwrap();
        let xs_pass = xs.get((.., .., .., self.dim as isize..)).unwrap();
        let c = self.cos.narrow(0, seqlen_offset, seq_len).unwrap();
        let s = self.sin.narrow(0, seqlen_offset, seq_len).unwrap();
        //let xs_rot = candle_nn::rotary_emb::rope(&xs_rot, &c, &s).unwrap();
        let xs_rot = xs_rot.rope(c, s).unwrap();
        Tensor::cat([&xs_rot, &xs_pass], -1)
    }
}

#[derive(Debug)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
    act: Activation,
}

impl MLP {
    fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        //let fc1 = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1")).unwrap();
        let fc1 = Linear {
            weight: vb.t("fc1.weight"),
            bias: Some(vb.t("fc1.bias")),
        };
        //let fc2 = linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2")).unwrap();
        let fc2 = Linear {
            weight: vb.t("fc2.weight"),
            bias: Some(vb.t("fc2.bias")),
        };
        Ok(Self {
            fc1,
            fc2,
            // This does not match the mixformers implementation where Gelu is used rather than
            // GeluNew.
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, ZyxError> {
        let x = self.fc1.forward(xs).unwrap();
        let x = self.act.forward(x);
        let x = self.fc2.forward(x).unwrap();
        Ok(x)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
    q_layernorm: Option<LayerNorm>,
    k_layernorm: Option<LayerNorm>,
    rotary_emb: RotaryEmbedding,
    softmax_scale: f64,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

fn get_mask(size: usize) -> Result<Tensor, ZyxError> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from(mask).reshape([size, size])
}

/*fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor, ZyxError> {
    mask.where_(on_true, on_false)
}*/

impl Attention {
    fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads();
        let head_dim = cfg.head_dim();
        //let q_proj = linear(cfg.hidden_size, num_heads * head_dim, vb.pp("q_proj")).unwrap();
        let q_proj = Linear {
            weight: vb.t("q_proj.weight"),
            bias: Some(vb.t("q_proj.bias")),
        };
        //let k_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("k_proj")).unwrap();
        let k_proj = Linear {
            weight: vb.t("k_proj.weight"),
            bias: Some(vb.t("k_proj.bias")),
        };
        //let v_proj = linear(cfg.hidden_size, num_kv_heads * head_dim, vb.pp("v_proj")).unwrap();
        let v_proj = Linear {
            weight: vb.t("v_proj.weight"),
            bias: Some(vb.t("v_proj.bias")),
        };
        //let dense = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("dense")).unwrap();
        let dense = Linear {
            weight: vb.t("dense.weight"),
            bias: Some(vb.t("dense.bias")),
        };
        // Alternative rope scalings are not supported.
        let rotary_emb = RotaryEmbedding::new(cfg).unwrap();
        let (q_layernorm, k_layernorm) = if cfg.qk_layernorm {
            //let q_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("q_layernorm")).unwrap();
            let q_layernorm = LayerNorm {
                weight: Some(vb.t("q_layernorm.weight")),
                bias: Some(vb.t("q_layernorm.bias")),
                eps: cfg.layer_norm_eps,
                d_dims: head_dim,
            };
            //let k_layernorm = layer_norm(head_dim, cfg.layer_norm_eps, vb.pp("k_layernorm")).unwrap();
            let k_layernorm = LayerNorm {
                weight: Some(vb.t("k_layernorm.weight")),
                bias: Some(vb.t("k_layernorm.bias")),
                eps: cfg.layer_norm_eps,
                d_dims: head_dim,
            };
            (Some(q_layernorm), Some(k_layernorm))
        } else {
            (None, None)
        };
        let softmax_scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj: dense,
            kv_cache: None,
            q_layernorm,
            k_layernorm,
            rotary_emb,
            softmax_scale,
            num_heads,
            num_kv_heads,
            head_dim,
        })
    }

    /*fn repeat_kv(&self, xs: Tensor) -> Result<Tensor, ZyxError> {
        repeat_kv(xs, self.num_heads / self.num_kv_heads)
    }*/

    fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let [b_size, seq_len, _n_embd] = xs.shape()[..] else {
            panic!()
        };
        let query_states = self.q_proj.forward(xs).unwrap();
        let key_states = self.k_proj.forward(xs).unwrap();
        let value_states = self.v_proj.forward(xs).unwrap();

        let query_states = match &self.q_layernorm {
            None => query_states,
            Some(ln) => ln.forward(query_states).unwrap(),
        };
        let key_states = match &self.k_layernorm {
            None => key_states,
            Some(ln) => ln.forward(key_states).unwrap(),
        };

        let query_states = query_states
            .reshape([b_size, seq_len, self.num_heads, self.head_dim])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let key_states = key_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let value_states = value_states
            .reshape([b_size, seq_len, self.num_kv_heads, self.head_dim])
            .unwrap()
            .transpose(1, 2)
            .unwrap();

        // Rotary embeddings.
        let seqlen_offset = match &self.kv_cache {
            None => 0,
            Some((prev_k, _)) => prev_k.shape()[2],
        };
        let query_states = self
            .rotary_emb
            .apply_rotary_emb(&query_states, seqlen_offset)
            .unwrap();
        let key_states = self
            .rotary_emb
            .apply_rotary_emb(&key_states, seqlen_offset)
            .unwrap();

        // KV cache.
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat([prev_k, &key_states], 2).unwrap();
                let v = Tensor::cat([prev_v, &value_states], 2).unwrap();
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Repeat kv.
        /*let key_states = self.repeat_kv(key_states).unwrap();
        let value_states = self.repeat_kv(value_states).unwrap();

        let attn_weights = query_states
            .cast(DType::F32)
            .matmul(&key_states.cast(DType::F32).t())
            .unwrap()
            * self.softmax_scale;
        let attn_weights = match mask {
            None => attn_weights,
            Some(mask) => {
                let mut sh = vec![b_size, self.num_heads];
                sh.append(&mut mask.shape());
                masked_fill(&attn_weights, &mask.expand(sh).unwrap(), f32::NEG_INFINITY).unwrap()
            }
        };
        let attn_weights = attn_weights
            .softmax([-3])
            .unwrap()
            .cast(value_states.dtype());
        Tensor::realize([&attn_weights, &value_states]).unwrap();
        let attn_output = attn_weights.matmul(&value_states).unwrap();
        let attn_output = attn_output.transpose(1, 2).unwrap();
        let d: usize = attn_output.shape()[2..].iter().product();
        let attn_output = attn_output.reshape([b_size, seq_len, d]).unwrap();
        let attn_output = self.dense.forward(attn_output).unwrap();
        attn_output*/

        let num_kv_groups = self.num_heads/self.num_kv_heads;
        let key_states = repeat_kv(key_states, num_kv_groups);
        let value_states = repeat_kv(value_states, num_kv_groups);

        let attn_output = {
            let scale = half::f16::from_f64(1f64 / f64::sqrt(self.head_dim as f64));
            let attn_weights = query_states.matmul(key_states.transpose(2, 3).unwrap()).unwrap() * scale;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights + mask,
            };
            let attn_weights = attn_weights.softmax([-1]).unwrap();
            attn_weights.matmul(&value_states).unwrap()
        };
        let attn_output = attn_output.transpose(1, 2).unwrap();
        let d: usize = attn_output.shape()[2..].iter().product();
        let attn_output = attn_output.reshape([b_size, seq_len, d]).unwrap();
        self.o_proj.forward(attn_output).unwrap()
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: LayerNorm,
}

impl DecoderLayer {
    fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let self_attn = Attention::new(cfg, &mut vb.g("self_attn")).unwrap();
        let mlp = MLP::new(cfg, &mut vb.g("mlp")).unwrap();
        /*let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("input_layernorm"),
        )?;*/
        let weight = vb.t("input_layernorm.weight");
        let input_layernorm = LayerNorm {
            d_dims: weight.rank(),
            weight: Some(weight),
            bias: Some(vb.t("input_layernorm.bias")),
            eps: cfg.layer_norm_eps,
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
        })
    }

    fn forward(&mut self, xs: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs).unwrap();
        let attn_outputs = self.self_attn.forward(&xs, mask);
        //println!("{attn_outputs}");
        let feed_forward_hidden_states = self.mlp.forward(&xs).unwrap();
        let res = attn_outputs + feed_forward_hidden_states + residual;
        res
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    final_layernorm: LayerNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let mut vb_m = vb.g("model");
        //let embed_tokens = Embedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let embed_tokens = Embedding::from_params(vb_m.t("embed_tokens.weight")).unwrap();
        /*let final_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb_m.pp("final_layernorm"),
        )?;*/
        let weight = vb_m.t("final_layernorm.weight");
        let final_layernorm = LayerNorm {
            d_dims: weight.rank(),
            weight: Some(weight),
            bias: Some(vb_m.t("final_layernorm.bias")),
            eps: cfg.layer_norm_eps,
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut vb_m = vb_m.g("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(cfg, &mut vb_m.g(&format!("{layer_idx}"))).unwrap();
            layers.push(layer);
        }
        //let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head")).unwrap();
        let lm_head = Linear {
            weight: vb.t("lm_head.weight"),
            bias: Some(vb.t("lm_head.bias")),
        };
        Ok(Self {
            embed_tokens,
            layers,
            final_layernorm,
            lm_head,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Tensor {
        let [_b_size, seq_len] = xs.shape()[..] else {
            panic!()
        };
        let mut xs = self.embed_tokens.forward(xs.cast(DType::F16)).unwrap();
        let mask = if seq_len <= 1 {
            None
        } else {
            Some(get_mask(seq_len).unwrap())
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, mask.as_ref());
            println!("{xs}");
        }
        println!("{xs}");
        panic!();
        //Tensor::plot_graph([], "graph").unwrap();
        let xs = self
            .final_layernorm
            .forward(xs)
            .unwrap()
            .narrow(1, seq_len - 1, 1)
            .unwrap();
        self.lm_head.forward(xs).unwrap().squeeze(0).unwrap()
    }

    pub fn clear_kv_cache(&mut self) {
        self.layers.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}

use clap::Parser;

//use anyhow::{Error as E, Result};
//use candle_examples::token_output_stream::TokenOutputStream;
//use candle_transformers::models::mixformer::{Config, MixFormerSequentialForCausalLM as MixFormer};
//use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi};
//use candle_transformers::models::phi3::{Config as Phi3Config, Model as Phi3};
//use candle_transformers::models::quantized_mixformer::MixFormerSequentialForCausalLM as QMixFormer;
//use candle::{DType, Device, IndexOp, Tensor};
//use candle_nn::VarBuilder;
//use candle_transformers::generation::LogitsProcessor;
//use tokenizers::Tokenizer;
//use hf_hub::{api::sync::Api, Repo, RepoType};

use rand::{distributions::Distribution, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32, ZyxError> {
        let logits_v: Vec<f32> = logits.try_into().unwrap();
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32, ZyxError> {
        let distr = rand::distributions::WeightedIndex::new(prs).unwrap();
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32, ZyxError> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&mut self, prs: &mut Vec<f32>, top_k: usize) -> Result<u32, ZyxError> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs).unwrap();
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(
        &mut self,
        prs: &mut Vec<f32>,
        top_k: usize,
        top_p: f32,
    ) -> Result<u32, ZyxError> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs).unwrap()
            } else {
                self.sample_topp(&mut prs, top_p).unwrap()
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32, ZyxError> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(
        &mut self,
        logits: &Tensor,
        f: impl FnOnce(&mut [f32]),
    ) -> Result<u32, ZyxError> {
        let logits = logits.cast(DType::F32);
        let prs = |temperature: f64| -> Result<Vec<f32>, ZyxError> {
            let logits = &logits / temperature;
            let prs = logits.softmax([-1]).unwrap();
            let mut prs: Vec<f32> = prs.try_into().unwrap();
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits).unwrap(),
            Sampling::All { temperature } => {
                let prs = prs(*temperature).unwrap();
                self.sample_multinomial(&prs).unwrap()
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature).unwrap();
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs).unwrap()
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, *p as f32).unwrap()
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature).unwrap();
                self.sample_topk(&mut prs, *k).unwrap()
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature).unwrap();
                self.sample_topk_topp(&mut prs, *k, *p as f32).unwrap()
            }
        };
        Ok(next_token)
    }
}

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
pub struct TokenOutputStream {
    tokenizer: tokenizers::Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

impl TokenOutputStream {
    pub fn new(tokenizer: tokenizers::Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    pub fn into_inner(self) -> tokenizers::Tokenizer {
        self.tokenizer
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, ZyxError> {
        match self.tokenizer.decode(tokens, true) {
            Ok(str) => Ok(str),
            Err(err) => panic!("cannot decode: {err}"),
        }
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    pub fn next_token(&mut self, token: u32) -> Result<Option<String>, ZyxError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens).unwrap()
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..]).unwrap();
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_rest(&self) -> Result<Option<String>, ZyxError> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens).unwrap()
        };
        let text = self.decode(&self.tokens[self.prev_index..]).unwrap();
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    pub fn decode_all(&self) -> Result<String, ZyxError> {
        self.decode(&self.tokens)
    }

    pub fn get_token(&self, token_s: &str) -> Option<u32> {
        self.tokenizer.get_vocab(true).get(token_s).copied()
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn clear(&mut self) {
        self.tokens.clear();
        self.prev_index = 0;
        self.current_index = 0;
    }
}

pub fn apply_repeat_penalty(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
) -> Result<Tensor, ZyxError> {
    let mut logits: Vec<f32> = logits.cast(DType::F32).try_into().unwrap();
    let mut already_seen = std::collections::HashSet::new();
    for token_id in context {
        if already_seen.contains(token_id) {
            continue;
        }
        already_seen.insert(token_id);
        if let Some(logit) = logits.get_mut(*token_id as usize) {
            if *logit >= 0. {
                *logit /= penalty
            } else {
                *logit *= penalty
            }
        }
    }
    Ok(Tensor::from(logits))
}

struct TextGeneration {
    model: Model,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    verbose_prompt: bool,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
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
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            verbose_prompt,
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<(), ZyxError> {
        use std::io::Write;
        println!("starting the inference loop");
        let tokens = self.tokenizer.tokenizer().encode(prompt, true).unwrap();
        if tokens.is_empty() {
            panic!("Empty prompts are not supported in the phi model.")
        }
        if self.verbose_prompt {
            for (token, id) in tokens.get_tokens().iter().zip(tokens.get_ids().iter()) {
                let token = token.replace('▁', " ").replace("<0x0A>", "\n");
                println!("{id:7} -> '{token}'");
            }
        }
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => panic!("cannot find the endoftext token"),
        };
        println!("{prompt}");
        std::io::stdout().flush().unwrap();
        let start_gen = std::time::Instant::now();
        //let mut pos = 0;
        for index in 0..sample_len {
            let context_size: usize = if index > 0 { 1 } else { tokens.len() };
            let ctxt: Vec<u32> = tokens[tokens.len().saturating_sub(context_size)..].into();
            let input = Tensor::from(ctxt).unsqueeze(0).unwrap();
            let logits = self.model.forward(&input);
            //println!("Realizing.");
            //Tensor::realize([&logits]).unwrap();
            //let res: String = self.tokenizer.decode_all().unwrap();
            //Tensor::plot_graph([], "phi")?;
            let logits = logits.squeeze(0).unwrap().cast(DType::F32);
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                apply_repeat_penalty(&logits, self.repeat_penalty, &tokens[start_at..]).unwrap()
            };

            let next_token = self.logits_processor.sample(&logits).unwrap();
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest().unwrap() {
                    print!("{t}");
                    std::io::stdout().flush().unwrap();
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token).unwrap() {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            //pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Display the token for the specified prompt.
    #[arg(long)]
    verbose_prompt: bool,

    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    mmlu_dir: Option<String>,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 5000)]
    sample_len: usize,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.0)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,

    /// The dtype to be used for running the model, e.g. f32, bf16, or f16.
    #[arg(long)]
    dtype: Option<String>,
}

fn main() -> Result<(), ZyxError> {
    let args = Args::parse();
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );
    let tokenizer = Tokenizer::from_file("../tokenizer.json").unwrap();

    let model = {
        //let vb = unsafe { VarBuilder::from_mmaped_safetensors(filename, dtype)? };
        let mut vb: HashMap<String, Tensor> =
            Tensor::load("/home/x/Dev/rust/zyx/model.safetensors").unwrap();
        //let mut keys: Vec<String> = vb.keys().cloned().collect();
        //keys.sort();
        //println!("{:?}", keys);
        //let config_filename = repo.get("config.json")?;
        //let config = std::fs::read_to_string(config_filename)?;
        let config = Config {
            vocab_size: 51200,
            hidden_size: 2048,
            intermediate_size: 8192,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: None,
            hidden_act: Activation::ReLU,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            tie_word_embeddings: false,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            qk_layernorm: false,
        };
        Model::new(&config, &mut vb).unwrap()
    };

    match (args.prompt, args.mmlu_dir) {
        (None, None) | (Some(_), Some(_)) => {
            panic!("exactly one of --prompt and --mmlu-dir must be specified")
        }
        (Some(prompt), None) => {
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
        }
        (None, Some(mmlu_dir)) => mmlu(model, tokenizer, mmlu_dir).unwrap(),
    }
    Ok(())
}

fn mmlu<P: AsRef<std::path::Path>>(
    mut model: Model,
    tokenizer: Tokenizer,
    mmlu_dir: P,
) -> Result<(), ZyxError> {
    for dir_entry in mmlu_dir.as_ref().read_dir()?.flatten() {
        let dir_entry = dir_entry.path();
        let theme = match dir_entry.file_stem().and_then(|v| v.to_str()) {
            None => "".to_string(),
            Some(v) => match v.strip_suffix("_test") {
                None => v.replace('_', " "),
                Some(v) => v.replace('_', " "),
            },
        };
        if dir_entry.extension().as_ref().and_then(|v| v.to_str()) != Some("csv") {
            continue;
        }
        println!("reading {dir_entry:?}");
        let dir_entry = std::fs::File::open(dir_entry)?;
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(dir_entry);
        let token_a = tokenizer.token_to_id("A").unwrap();
        let token_b = tokenizer.token_to_id("B").unwrap();
        let token_c = tokenizer.token_to_id("C").unwrap();
        let token_d = tokenizer.token_to_id("D").unwrap();
        for row in reader.records() {
            let row = match row {
                Err(_) => continue,
                Ok(row) => row,
            };
            if row.len() < 5 {
                continue;
            }
            let question = row.get(0).unwrap();
            let answer_a = row.get(1).unwrap();
            let answer_b = row.get(2).unwrap();
            let answer_c = row.get(3).unwrap();
            let answer_d = row.get(4).unwrap();
            let answer = row.get(5).unwrap();
            let prompt = format!(
                    "{} {theme}.\n{question}\nA. {answer_a}\nB. {answer_b}\nC. {answer_c}\nD. {answer_d}\nAnswer:\n",
                    "The following are multiple choice questions (with answers) about"
                );
            let tokens = tokenizer.encode(prompt.as_str(), true).unwrap();
            let tokens = tokens.get_ids().to_vec();
            let input = Tensor::from(tokens).unsqueeze(0).unwrap();
            model.clear_kv_cache();
            let logits = model.forward(&input);
            let logits = logits.squeeze(0)?.cast(DType::F32);
            let logits_v: Vec<f32> = logits.try_into().unwrap();
            let pr_a = logits_v[token_a as usize];
            let pr_b = logits_v[token_b as usize];
            let pr_c = logits_v[token_c as usize];
            let pr_d = logits_v[token_d as usize];
            let model_answer = if pr_a > pr_b && pr_a > pr_c && pr_a > pr_d {
                "A"
            } else if pr_b > pr_c && pr_b > pr_d {
                "B"
            } else if pr_c > pr_d {
                "C"
            } else {
                "D"
            };
            println!("{prompt}\n -> {model_answer} vs {answer}");
        }
    }
    Ok(())
}
