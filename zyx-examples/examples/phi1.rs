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
        let query_states = qkv.narrow(D::Minus1, 0, query_pos)?;
        let key_states = qkv.narrow(D::Minus1, query_pos, self.num_kv_heads * self.head_dim)?;
        let value_states = qkv.narrow(
            D::Minus1,
            query_pos + self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
        )?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat([prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
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
        let gate = up_states.narrow(D::Minus1, 0, self.i_size)?;
        let up_states = up_states.narrow(D::Minus1, self.i_size, self.i_size)?;
        let up_states = (up_states * gate.apply(&self.act_fn))?;
        up_states.apply(&self.down_proj)
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
        xs.narrow(1, seq_len - 1, 1)?
            .apply(&self.norm)?
            .apply(&self.lm_head)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

fn main() {}
