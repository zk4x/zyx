// Implementation based on candle rust
// This implementation is based on:
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/modeling_phi3.py

use std::collections::HashMap;
use std::sync::Arc;
use zyx::{Tensor, DType, ZyxError};
use zyx_nn::{Linear, RMSNorm, Embedding};

fn repeat_kv(x: Tensor, n_rep: usize) -> Result<Tensor, ZyxError> {
    let [bs, seqlen, n_kv_heads, head_dim] = x.shape()[..] else { panic!() };
    if n_rep == 1 {
        return Ok(x)
    }
    return x.repeat([1, 1, 1, n_rep])?.reshape([bs, seqlen, n_kv_heads * n_rep, head_dim])
}

trait VarMap {
    fn g(&mut self, p: &str) -> Self;
    fn t(&mut self, t: &str) -> Tensor;
}

impl VarMap for HashMap<String, Tensor> {
    fn g(&mut self, p: &str) -> Self {
        // TODO fix this to work with dots
        let mut res = HashMap::new();
        let paths: Vec<String> = self.keys().filter(|k| k.starts_with(p)).cloned().collect();
        for p in paths {
            let t = self.remove(&p).unwrap();
            res.insert(p, t);
        }
        res
    }

    fn t(&mut self, t: &str) -> Tensor {
        self.remove(t).unwrap()
    }
}

// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/config.json
#[derive(Debug, Clone)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub rope_scaling: Option<String>,
    pub max_position_embeddings: usize,
    pub dtype: DType,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config) -> Result<Self, ZyxError> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from(inv_freq).reshape([1, inv_freq_len])?.cast(dtype);
        let t = Tensor::arange(0u32, max_seq_len as u32, 1)?
            .cast(dtype)
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin(),
            cos: freqs.cos(),
        })
    }

    pub fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor), ZyxError> {
        let [_b_sz, _h, seq_len, _n_embd] = q.shape()[..] else { panic!() };
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = q.rope(&cos, &sin)?;
        let k_embed = k.rope(&cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug)]
struct Attention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        //let op_size = num_heads * head_dim + 2 * num_kv_heads * head_dim;
        //let qkv_proj = linear(cfg.hidden_size, op_size, vb.pp("qkv_proj"))?;
        let qkv_proj = Linear {
            weight: vb.t("qkv_proj.weight"),
            bias: Some(vb.t("qkv_proj.bias")),
        };
        //let o_proj = linear(num_heads * head_dim, cfg.hidden_size, vb.pp("o_proj"))?;
        let o_proj = Linear {
            weight: vb.t("qkv_proj.weight"),
            bias: Some(vb.t("qkv_proj.bias")),
        };
        Ok(Self {
            qkv_proj,
            o_proj,
            rotary_emb,
            kv_cache: None,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor, ZyxError> {
        let [b_sz, q_len, _] = xs.shape()[..] else { panic!() };

        let qkv = self.qkv_proj.forward(xs)?;
        let query_pos = self.num_heads * self.head_dim;
        let query_states = qkv.narrow(-1, 0, query_pos)?;
        let key_states = qkv.narrow(-1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            -1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape([b_sz, q_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape([b_sz, q_len, self.num_kv_heads, self.head_dim])?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat([prev_k, &key_states], 2)?;
                let value_states = Tensor::cat([prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = repeat_kv(key_states, self.num_kv_groups)?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = query_states.matmul(key_states.transpose(2, 3)?)? * scale;

            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights + mask,
            };
            let attn_weights = attn_weights.softmax([-1])?;
            attn_weights.matmul(&value_states)?
        };
        self.o_proj.forward(attn_output
            .transpose(1, 2)?
            .reshape([b_sz, q_len, ])?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug)]
struct MLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: String,
    i_size: usize,
}

impl MLP {
    fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        //let hidden_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size;
        //let gate_up_proj = linear(hidden_size, 2 * i_size, vb.pp("gate_up_proj"))?;
        let gate_up_proj = Linear {
            weight: vb.t("gate_up_proj.weight"),
            bias: Some(vb.t("gate_up_proj.bias")),
        };
        //let down_proj = linear(i_size, hidden_size, vb.pp("down_proj"))?;
        let down_proj = Linear {
            weight: vb.t("down_proj.weight"),
            bias: Some(vb.t("down_proj.bias")),
        };
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: cfg.hidden_act.clone(),
            i_size,
        })
    }
}

impl MLP {
    fn forward(&self, xs: impl Into<Tensor>) -> Result<Tensor, ZyxError> {
        let xs = xs.into();
        let up_states = self.gate_up_proj.forward(xs)?;
        let gate = up_states.narrow(-1, 0, self.i_size)?;
        let up_states = up_states.narrow(-1, self.i_size, self.i_size)?;
        let up_states = up_states * match self.act_fn.as_str() {
            "relu" => gate.relu(),
            _ => panic!(),
        };
        self.down_proj.forward(up_states)
    }
}

#[derive(Debug)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl DecoderLayer {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let self_attn = Attention::new(rotary_emb, cfg, &mut vb.g("self_attn"))?;
        let mlp = MLP::new(cfg, &mut vb.g("mlp"))?;
        //let input_layernorm = RMSNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let input_layernorm = RMSNorm {
            scale: vb.t("input_layernorm.scale"),
            eps: cfg.rms_norm_eps,
        };
        /*let post_attention_layernorm = RMSNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;*/
        let post_attention_layernorm = RMSNorm {
            scale: vb.t("post_attention_layernorm.scale"),
            eps: cfg.rms_norm_eps,
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor, ZyxError> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs + residual;
        let residual = &xs;
        let xs = self.mlp.forward(self.post_attention_layernorm.forward(&xs)?)?;
        Ok(residual + xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug)]
pub struct Model {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    lm_head: Linear,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: &mut HashMap<String, Tensor>) -> Result<Self, ZyxError> {
        let mut vb_m = vb.g("model");
        //let embed_tokens = cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let embed_tokens = Embedding {
            vocab_size: cfg.vocab_size,
            embed_size: cfg.hidden_size,
            weight: vb_m.t("embed_tokens.weight"),
            arange: vb_m.t("embed_tokens.arange"),
        };
        let rotary_emb = Arc::new(RotaryEmbedding::new(cfg.dtype, cfg)?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut vb_l = vb_m.g("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, &mut vb_l.g(&format!("{layer_idx}")))?;
            layers.push(layer)
        }
        //let norm = RMSNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let norm = RMSNorm {
            scale: vb_m.t("norm.scale"),
            eps: cfg.rms_norm_eps,
        };
        //let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        let lm_head = Linear {
            weight: vb.t("lm_head.weight"),
            bias: Some(vb.t("lm_head.bias")),
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            dtype: cfg.dtype,
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor, ZyxError> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from(mask).reshape([tgt_len, tgt_len])?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros([tgt_len, seqlen_offset], DType::F32);
            Tensor::cat([&mask0, &mask], -1)?
        } else {
            mask
        };
        Ok(mask.expand([b_size, 1, tgt_len, tgt_len + seqlen_offset])?.cast(self.dtype))
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor, ZyxError> {
        let [b_size, seq_len] = input_ids.shape()[..] else { panic!() };
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(b_size, seq_len, seqlen_offset)?;
            Some(mask)
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?
        }
        self.lm_head.forward(self.norm.forward(xs.narrow(1, seq_len - 1, 1)?)?)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

//use anyhow::{Error as E, Result};
//use clap::{Parser, ValueEnum};
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
        let tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?;
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
        print!("{prompt}");
        std::io::stdout().flush()?;
        let start_gen = std::time::Instant::now();
        let mut pos = 0;
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
            let input = Tensor::from(ctxt).unsqueeze(0)?;
            let logits = self.model.forward(&input)?;
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token {
                if let Some(t) = self.tokenizer.decode_rest()? {
                    print!("{t}");
                    std::io::stdout().flush()?;
                }
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            pos += context_size;
        }
        let dt = start_gen.elapsed();
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "1")]
    V1,
    #[value(name = "1.5")]
    V1_5,
    #[value(name = "2")]
    V2,
    #[value(name = "3")]
    V3,
    #[value(name = "3-medium")]
    V3Medium,
    #[value(name = "2-old")]
    V2Old,
    PuffinPhiV2,
    PhiHermes,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

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

    #[arg(long, default_value = "2")]
    model: WhichModel,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    weight_file: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    quantized: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
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
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model_id) => model_id.to_string(),
        None => {
            if args.quantized {
                "lmz/candle-quantized-phi".to_string()
            } else {
                match args.model {
                    WhichModel::V1 => "microsoft/phi-1".to_string(),
                    WhichModel::V1_5 => "microsoft/phi-1_5".to_string(),
                    WhichModel::V2 | WhichModel::V2Old => "microsoft/phi-2".to_string(),
                    WhichModel::V3 => "microsoft/Phi-3-mini-4k-instruct".to_string(),
                    WhichModel::V3Medium => "microsoft/Phi-3-medium-4k-instruct".to_string(),
                    WhichModel::PuffinPhiV2 | WhichModel::PhiHermes => {
                        "lmz/candle-quantized-phi".to_string()
                    }
                }
            }
        }
    };
    let revision = match args.revision {
        Some(rev) => rev.to_string(),
        None => {
            if args.quantized {
                "main".to_string()
            } else {
                match args.model {
                    WhichModel::V1 => "refs/pr/8".to_string(),
                    WhichModel::V1_5 => "refs/pr/73".to_string(),
                    WhichModel::V2Old => "834565c23f9b28b96ccbeabe614dd906b6db551a".to_string(),
                    WhichModel::V2
                    | WhichModel::V3
                    | WhichModel::V3Medium
                    | WhichModel::PuffinPhiV2
                    | WhichModel::PhiHermes => "main".to_string(),
                }
            }
        }
    };
    let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let tokenizer_filename = match args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => match args.model {
            WhichModel::V1
            | WhichModel::V1_5
            | WhichModel::V2
            | WhichModel::V2Old
            | WhichModel::V3
            | WhichModel::V3Medium => repo.get("tokenizer.json")?,
            WhichModel::PuffinPhiV2 | WhichModel::PhiHermes => {
                repo.get("tokenizer-puffin-phi-v2.json")?
            }
        },
    };
    let filenames = match args.weight_file {
        Some(weight_file) => vec![std::path::PathBuf::from(weight_file)],
        None => {
            if args.quantized {
                match args.model {
                    WhichModel::V1 => vec![repo.get("model-v1-q4k.gguf")?],
                    WhichModel::V1_5 => vec![repo.get("model-q4k.gguf")?],
                    WhichModel::V2 | WhichModel::V2Old => vec![repo.get("model-v2-q4k.gguf")?],
                    WhichModel::PuffinPhiV2 => vec![repo.get("model-puffin-phi-v2-q4k.gguf")?],
                    WhichModel::PhiHermes => vec![repo.get("model-phi-hermes-1_3B-q4k.gguf")?],
                    WhichModel::V3 | WhichModel::V3Medium => anyhow::bail!(
                        "use the quantized or quantized-phi examples for quantized phi-v3"
                    ),
                }
            } else {
                match args.model {
                    WhichModel::V1 | WhichModel::V1_5 => vec![repo.get("model.safetensors")?],
                    WhichModel::V2 | WhichModel::V2Old | WhichModel::V3 | WhichModel::V3Medium => {
                        candle_examples::hub_load_safetensors(
                            &repo,
                            "model.safetensors.index.json",
                        )?
                    }
                    WhichModel::PuffinPhiV2 => vec![repo.get("model-puffin-phi-v2.safetensors")?],
                    WhichModel::PhiHermes => vec![repo.get("model-phi-hermes-1_3B.safetensors")?],
                }
            }
        }
    };
    println!("retrieved the files in {:?}", start.elapsed());
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let start = std::time::Instant::now();
    let config = || match args.model {
        WhichModel::V1 => Config::v1(),
        WhichModel::V1_5 => Config::v1_5(),
        WhichModel::V2 | WhichModel::V2Old => Config::v2(),
        WhichModel::PuffinPhiV2 => Config::puffin_phi_v2(),
        WhichModel::PhiHermes => Config::phi_hermes_1_3b(),
        WhichModel::V3 | WhichModel::V3Medium => {
            panic!("use the quantized or quantized-phi examples for quantized phi-v3")
        }
    };
    let device = candle_examples::device(args.cpu)?;
    let model = if args.quantized {
        let config = config();
        let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
            &filenames[0],
            &device,
        )?;
        let model = match args.model {
            WhichModel::V2 | WhichModel::V2Old => QMixFormer::new_v2(&config, vb)?,
            _ => QMixFormer::new(&config, vb)?,
        };
        Model::Quantized(model)
    } else {
        let dtype = match args.dtype {
            Some(dtype) => std::str::FromStr::from_str(&dtype)?,
            None => {
                if args.model == WhichModel::V3 || args.model == WhichModel::V3Medium {
                    device.bf16_default_to_f32()
                } else {
                    DType::F32
                }
            }
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        match args.model {
            WhichModel::V1 | WhichModel::V1_5 | WhichModel::V2 => {
                let config_filename = repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: PhiConfig = serde_json::from_str(&config)?;
                let phi = Phi::new(&config, vb)?;
                Model::Phi(phi)
            }
            WhichModel::V3 | WhichModel::V3Medium => {
                let config_filename = repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: Phi3Config = serde_json::from_str(&config)?;
                let phi3 = Phi3::new(&config, vb)?;
                Model::Phi3(phi3)
            }
            WhichModel::V2Old => {
                let config = config();
                Model::MixFormer(MixFormer::new_v2(&config, vb)?)
            }
            WhichModel::PhiHermes | WhichModel::PuffinPhiV2 => {
                let config = config();
                Model::MixFormer(MixFormer::new(&config, vb)?)
            }
        }
    };
    println!("loaded the model in {:?}", start.elapsed());

    match (args.prompt, args.mmlu_dir) {
        (None, None) | (Some(_), Some(_)) => {
            anyhow::bail!("exactly one of --prompt and --mmlu-dir must be specified")
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
                &device,
            );
            pipeline.run(&prompt, args.sample_len)?;
        }
        (None, Some(mmlu_dir)) => mmlu(model, tokenizer, &device, mmlu_dir)?,
    }
    Ok(())
}

fn mmlu<P: AsRef<std::path::Path>>(
    mut model: Model,
    tokenizer: Tokenizer,
    device: &Device,
    mmlu_dir: P,
) -> anyhow::Result<()> {
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
            let tokens = tokenizer.encode(prompt.as_str(), true).map_err(E::msg)?;
            let tokens = tokens.get_ids().to_vec();
            let input = Tensor::new(tokens, device)?.unsqueeze(0)?;
            let logits = match &mut model {
                Model::MixFormer(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
                Model::Phi3(m) => {
                    m.clear_kv_cache();
                    m.forward(&input, 0)?
                }
                Model::Quantized(m) => {
                    m.clear_kv_cache();
                    m.forward(&input)?
                }
            };
            let logits = logits.squeeze(0)?.to_dtype(DType::F32)?;
            let logits_v: Vec<f32> = logits.to_vec1()?;
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
