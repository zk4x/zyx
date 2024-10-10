//use super::with_tracing::{linear_no_bias as linear, Linear, RmsNorm};
//use candle::{DType, Device, IndexOp, Result, Tensor, D};
//use candle_nn::{embedding, Embedding, Module, VarBuilder};
use zyx::{DType, Tensor, ZyxError};
use zyx_nn::{Embedding, Linear, RMSNorm};
use std::{collections::HashMap, f32::consts::PI};

pub const DEFAULT_MAX_SEQ_LEN: usize = 4096;
const DT: DType = DType::F32;

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum LlamaEosToks {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: Option<bool>,
}

impl LlamaConfig {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaConfig {
    pub fn into_config(self, use_flash_attn: bool) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads(),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            use_flash_attn,
            bos_token_id: self.bos_token_id,
            eos_token_id: self.eos_token_id,
            rope_scaling: self.rope_scaling,
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<LlamaEosToks>,
    pub rope_scaling: Option<Llama3RopeConfig>,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn config_7b_v1(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }

    pub fn config_7b_v2(use_flash_attn: bool) -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 11008,
            vocab_size: 32000,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            use_flash_attn,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            bos_token_id: None,
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: DEFAULT_MAX_SEQ_LEN,
            tie_word_embeddings: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cache {
    masks: HashMap<usize, Tensor>,
    pub use_kv_cache: bool,
    kvs: Vec<Option<(Tensor, Tensor)>>,
    cos: Tensor,
    sin: Tensor,
}

fn calculate_default_inv_freq(cfg: &Config) -> Vec<f32> {
    let head_dim = cfg.hidden_size / cfg.num_attention_heads;
    (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / head_dim as f32))
        .collect()
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config) -> Result<Self, ZyxError> {
        // precompute freqs_cis
        let theta = match &config.rope_scaling {
            None
            | Some(Llama3RopeConfig {
                rope_type: Llama3RopeType::Default,
                ..
            }) => calculate_default_inv_freq(config),
            Some(rope_scaling) => {
                let low_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.low_freq_factor;
                let high_freq_wavelen = rope_scaling.original_max_position_embeddings as f32
                    / rope_scaling.high_freq_factor;

                calculate_default_inv_freq(config)
                    .into_iter()
                    .map(|freq| {
                        let wavelen = 2. * PI / freq;
                        if wavelen < high_freq_wavelen {
                            freq
                        } else if wavelen > low_freq_wavelen {
                            freq / rope_scaling.factor
                        } else {
                            let smooth = (rope_scaling.original_max_position_embeddings as f32
                                / wavelen
                                - rope_scaling.low_freq_factor)
                                / (rope_scaling.high_freq_factor - rope_scaling.low_freq_factor);
                            (1. - smooth) * freq / rope_scaling.factor + smooth * freq
                        }
                    })
                    .collect::<Vec<_>>()
            }
        };

        let theta = Tensor::from(theta);

        let idx_theta = Tensor::arange(0, config.max_position_embeddings as u32, 1)?
            .cast(DType::F32)
            .reshape((config.max_position_embeddings, 1))?
            .matmul(&theta.reshape((1, theta.numel()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let cos = idx_theta.cos().cast(dtype);
        let sin = idx_theta.sin().cast(dtype);
        Ok(Self {
            masks: HashMap::new(),
            use_kv_cache,
            kvs: vec![None; config.num_hidden_layers],
            cos,
            sin,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor, ZyxError> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from(&mask).reshape([t, t])?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }
}

#[derive(Debug)]
struct CausalSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,
    max_position_embeddings: usize,
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize, cache: &Cache) -> Result<Tensor, ZyxError> {
        todo!();
        /*let [_b_sz, _, seq_len, _hidden_size] = x.shape()[..] else { panic!() };
        let cos = cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = cache.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope(x, &cos, &sin)*/
    }

    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, ZyxError> {
        let [b_sz, seq_len, hidden_size] = x.shape()[..] else { panic!() };
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape([b_sz, seq_len, self.num_attention_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape([b_sz, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;
        let mut v = v
            .reshape([b_sz, seq_len, self.num_key_value_heads, self.head_dim])?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos, cache)?;
        let mut k = self.apply_rotary_emb(&k, index_pos, cache)?;

        if cache.use_kv_cache {
            if let Some((cache_k, cache_v)) = &cache.kvs[block_idx] {
                k = Tensor::cat([cache_k, &k], 2)?;
                v = Tensor::cat([cache_v, &v], 2)?;
                let k_seq_len = k.shape()[1];
                if k_seq_len > self.max_position_embeddings {
                    k = k
                        .narrow(
                            D::Minus1,
                            k_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * self.max_position_embeddings {
                    v = v
                        .narrow(
                            D::Minus1,
                            v_seq_len - self.max_position_embeddings,
                            self.max_position_embeddings,
                        )?
                }
            }
            cache.kvs[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let in_dtype = q.dtype();
        let q = q.cast(DType::F32);
        let k = k.cast(DType::F32);
        let v = v.cast(DType::F32);
        let att = q.matmul(k.t())? / (self.head_dim as f64).sqrt();
        let att = if seq_len == 1 {
            att
        } else {
            let mask = cache.mask(seq_len)?.expand(att.shape())?;
            masked_fill(&att, &mask, f32::NEG_INFINITY)?
        };
        let att = att.softmax([-1])?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(v)?.cast(in_dtype);
        let y = y.transpose(1, 2)?.reshape([b_sz, seq_len, hidden_size])?;
        let y = self.o_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor, ZyxError> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
        if self.num_key_value_heads == 1 {
            Ok(x)
        } else {
            let [b_sz, n_kv_head, seq_len, head_dim] = x.shape()[..] else { panic!() };
            // shape b_sz, n_kv_head * n_rep, seq_len, head_dim
            x.repeat([b_sz, n_kv_head, seq_len, head_dim, n_rep])
        }
    }

    fn load(vb: &mut HashMap<String, Tensor>, cfg: &Config) -> Result<Self, ZyxError> {
        //let size_in = cfg.hidden_size;
        //let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
        //let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
        let q_proj = Linear { weight: vb.remove("q_proj").unwrap(), bias: None };
        let k_proj = Linear { weight: vb.remove("k_proj").unwrap(), bias: None };
        let v_proj = Linear { weight: vb.remove("v_proj").unwrap(), bias: None };
        let o_proj = Linear { weight: vb.remove("o_proj").unwrap(), bias: None };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            use_flash_attn: cfg.use_flash_attn,
            max_position_embeddings: cfg.max_position_embeddings,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor, ZyxError> {
    let on_true = Tensor::from(on_true).expand(mask.shape())?;
    let m = mask.where_(on_true, on_false)?;
    Ok(m)
}

#[derive(Debug)]
struct MLP {
    c_fc1: Linear,
    c_fc2: Linear,
    c_proj: Linear,
}

impl MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor, ZyxError> {
        let x = self.c_fc1.forward(x)?.swish() * self.c_fc2.forward(x)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: &mut HashMap<String, Tensor>, cfg: &Config) -> Result<Self, ZyxError> {
        //let h_size = cfg.hidden_size;
        //let i_size = cfg.intermediate_size;
        let c_fc1 = Linear { weight: vb.remove("gate_proj").unwrap(), bias: None };
        let c_fc2 = Linear { weight: vb.remove("up_proj").unwrap(), bias: None };
        let c_proj = Linear { weight: vb.remove("down_proj").unwrap(), bias: None };
        Ok(Self {
            c_fc1,
            c_fc2,
            c_proj,
        })
    }
}

#[derive(Debug)]
struct Block {
    rms_1: RMSNorm,
    attn: CausalSelfAttention,
    rms_2: RMSNorm,
    mlp: MLP,
}

impl Block {
    fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, ZyxError> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = self.attn.forward(&x, index_pos, block_idx, cache)? + residual;
        let residual = &x;
        let x = self.mlp.forward(&self.rms_2.forward(&x)?)? + residual;
        Ok(x)
    }

    fn load(vb: &mut HashMap<String, Tensor>, cfg: &Config) -> Result<Self, ZyxError> {
        //let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        //let mlp = MLP::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RMSNorm { scale: vb.remove("input_layernorm").unwrap(), eps: cfg.rms_norm_eps };
        let rms_2 = RMSNorm { scale: vb.remove("post_attention_layernorm").unwrap(), eps: cfg.rms_norm_eps };
        Ok(Self {
            rms_1,
            attn: todo!(),
            rms_2,
            mlp: todo!(),
        })
    }
}

#[derive(Debug)]
pub struct Llama {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RMSNorm,
    lm_head: Linear,
}

impl Llama {
    // required by LLaVA
    pub fn embed(&self, x: &Tensor) -> Result<Tensor, ZyxError> {
        self.wte.forward(x)
    }
    // required by LLaVA
    pub fn forward_input_embed(
        &self,
        input_embed: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor, ZyxError> {
        let [_, seq_len, _] = input_embed.shape()[..] else { panic!() };
        let mut x = input_embed.clone();
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.get((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits.cast(DType::F32))
    }

    pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor, ZyxError> {
        let [_b_sz, seq_len] = x.shape()[..] else { panic!() };
        let mut x = self.wte.forward(x)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx, cache)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.get((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        Ok(logits.cast(DType::F32))
    }

    pub fn load(vb: HashMap<String, Tensor>, cfg: &Config) -> Result<Self, ZyxError> {
        let wte = Embedding { vocab_size: cfg.vocab_size, embed_size: cfg.hidden_size, weight: vb.remove("model.embed_tokens").unwrap(), arange: Tensor::arange(0, cfg.vocab_size as i64, 1)?.reshape([1, 1, cfg.vocab_size, 1])?.cast(DT) };
        let lm_head = if cfg.tie_word_embeddings {
            Linear { weight: wte.weight.clone(), bias: None }
        } else {
            Linear { weight: vb.remove("lm_head").unwrap(), bias: None }
        };
        let ln_f = RMSNorm {
            scale: vb.remove("model.norm").unwrap(),
            eps: cfg.rms_norm_eps,
        };
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| Block::load(vb.remove(&format!("model.layers.{i}")).unwrap(), cfg).unwrap())
            .collect();

        Ok(Self {
            wte,
            blocks,
            ln_f,
            lm_head,
        })
    }
}
