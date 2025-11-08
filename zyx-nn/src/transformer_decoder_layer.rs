use crate::{Dropout, LayerNorm, Linear, MultiheadAttention};
use zyx::{Tensor, ZyxError};

pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    dropout: Dropout,
    d_model: i64,
    // store flags
    norm_first: bool,
    activation: fn(&Tensor) -> Tensor,
}

impl TransformerDecoderLayer {
    pub fn new(
        d_model: i64,
        nhead: i64,
        dim_feedforward: i64,
        dropout_prob: f64,
        activation: fn(&Tensor) -> Tensor,
        layer_norm_eps: f64,
        norm_first: bool,
    ) -> Result<Self, ZyxError> {
        let self_attn = MultiheadAttention::new(d_model, nhead)?;
        let cross_attn = MultiheadAttention::new(d_model, nhead)?;
        let linear1 = Linear::new(d_model, dim_feedforward)?;
        let linear2 = Linear::new(dim_feedforward, d_model)?;
        let norm1 = LayerNorm::new(vec![d_model], layer_norm_eps)?;
        let norm2 = LayerNorm::new(vec![d_model], layer_norm_eps)?;
        let norm3 = LayerNorm::new(vec![d_model], layer_norm_eps)?;
        let dropout = Dropout::new(dropout_prob);

        Ok(Self {
            self_attn,
            cross_attn,
            linear1,
            linear2,
            norm1,
            norm2,
            norm3,
            dropout,
            d_model,
            norm_first,
            activation,
        })
    }

    pub fn forward(
        &self,
        tgt: &Tensor,
        memory: Option<&Tensor>,
        tgt_mask: Option<&Tensor>,
        memory_mask: Option<&Tensor>,
        tgt_key_padding_mask: Option<&Tensor>,
        memory_key_padding_mask: Option<&Tensor>,
        tgt_is_causal: bool,
        memory_is_causal: bool,
    ) -> Result<Tensor, ZyxError> {
        // Implementation note: zyx‑nn interfaces may differ; this is illustrative.

        let mut x = tgt.shallow_clone();

        if self.norm_first {
            // pre‑norm variant
            x = self.norm1.forward(&x)?;
        }

        // 1) Self‑attention
        let attn_output =
            self.self_attn
                .forward(&x, &x, &x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)?;
        let x2 = &attn_output + &x;
        let x2 = self.dropout.forward(&x2)?;
        x = if self.norm_first {
            x2
        } else {
            self.norm1.forward(&x2)?
        };

        // 2) Cross‑attention (if memory is provided)
        if let Some(mem) = memory {
            if self.norm_first {
                x = self.norm2.forward(&x)?;
            }
            let cross_output = self.cross_attn.forward(
                &x,
                mem,
                mem,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )?;
            let x3 = &cross_output + &x;
            let x3 = self.dropout.forward(&x3)?;
            x = if self.norm_first {
                x3
            } else {
                self.norm2.forward(&x3)?
            };
        }

        // 3) Feed‑forward network
        if self.norm_first {
            x = self.norm3.forward(&x)?;
        }
        let ff = (self.activation)(&self.linear1.forward(&x)?);
        let ff2 = self.linear2.forward(&ff)?;
        let x4 = ff2 + &x;
        let x4 = self.dropout.forward(&x4)?;
        x = if self.norm_first {
            x4
        } else {
            self.norm3.forward(&x4)?
        };

        Ok(x)
    }
}
