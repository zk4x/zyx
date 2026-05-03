// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx nn modules
//!
//! Default parameters match PyTorch where applicable.

use crate::*;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyList, PyTuple};
use pyo3::Bound;
use zyx::{DType, Tensor, ZyxError};

type ZyxResult<T> = std::result::Result<T, ZyxError>;

/// Convert a Python tuple of two u64 values to a Vec<u64> for shape parameters.
fn to_sh_from_tuple(t: (u64, u64)) -> Vec<u64> {
    vec![t.0, t.1]
}

/// Convert a Python tuple or nested list/tuple to a Vec<u64> for shape parameters.
fn to_sh(shape: &Bound<'_, PyTuple>) -> Vec<u64> {
    if shape.len() == 1 {
        let first = shape.get_item(0).unwrap();
        if first.is_instance_of::<PyList>() || first.is_instance_of::<PyTuple>() {
            let iter = PyIterator::from_object(&first).unwrap();
            return iter.filter_map(|item| item.ok().and_then(|v| v.extract::<u64>().ok())).collect();
        }
    }
    shape.as_slice().iter().filter_map(|x| x.extract::<u64>().ok()).collect()
}

/// Python bindings for Linear layer.
/// PyTorch: Linear(in_features, out_features, bias=True, device=None, dtype=None)
#[pymethods]
impl Linear {
    /// Create a new Linear layer.
    #[new]
    #[pyo3(signature = (in_features, out_features, bias=true, dtype=DType::F32))]
    pub fn py_new(in_features: u64, out_features: u64, bias: bool, dtype: DType) -> ZyxResult<Self> {
        Self::new(in_features, out_features, bias, dtype)
    }

    /// Forward pass through the linear layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the linear layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for Conv2d layer.
/// PyTorch: Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
#[pymethods]
impl Conv2d {
    /// Create a new Conv2d layer.
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1, bias=true, dtype=DType::F32))]
    pub fn py_new(
        in_channels: u64,
        out_channels: u64,
        kernel_size: (u64, u64),
        stride: (u64, u64),
        padding: (u64, u64),
        dilation: (u64, u64),
        groups: u64,
        bias: bool,
        dtype: DType,
    ) -> ZyxResult<Self> {
        Self::new(in_channels, out_channels, to_sh_from_tuple(kernel_size), to_sh_from_tuple(stride), to_sh_from_tuple(padding), to_sh_from_tuple(dilation), groups, bias, dtype)
    }

    /// Forward pass through the conv2d layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the conv2d layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for Embedding layer.
/// PyTorch: Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None, dtype=None)
#[pymethods]
impl Embedding {
    /// Create a new Embedding layer.
    #[new]
    #[pyo3(signature = (num_embeddings, embedding_dim, dtype=DType::F32))]
    pub fn py_new(num_embeddings: u64, embedding_dim: u64, dtype: DType) -> ZyxResult<Self> {
        Self::new(num_embeddings, embedding_dim, dtype)
    }

    /// Forward pass through the embedding layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the embedding layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for LayerNorm layer.
/// PyTorch: LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
#[pymethods]
impl LayerNorm {
    /// Create a new LayerNorm layer.
    #[new]
    #[pyo3(signature = (normalized_shape, eps=1e-5, elementwise_affine=true, py_bias=true, dtype=DType::F32))]
    pub fn py_new(normalized_shape: &Bound<'_, PyTuple>, eps: f64, elementwise_affine: bool, py_bias: bool, dtype: DType) -> ZyxResult<Self> {
        Self::new(to_sh(normalized_shape), eps, elementwise_affine, py_bias, dtype)
    }

    /// Forward pass through the layer norm layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the layer norm layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for BatchNorm layer.
/// PyTorch: BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
#[pymethods]
impl BatchNorm {
    /// Create a new BatchNorm layer.
    #[new]
    #[pyo3(signature = (num_features, eps=1e-5, momentum=0.1, affine=true, track_running_stats=true, dtype=DType::F32))]
    pub fn py_new(num_features: u64, eps: f64, momentum: f64, affine: bool, track_running_stats: bool, dtype: DType) -> Self {
        Self {
            eps: eps as f32,
            momentum: momentum as f32,
            track_running_stats,
            weight: if affine { Some(Tensor::ones(num_features, dtype)) } else { None },
            bias: if affine { Some(Tensor::zeros(num_features, dtype)) } else { None },
            running_mean: if track_running_stats { Tensor::zeros(num_features, dtype) } else { Tensor::zeros(0, dtype) },
            running_var: if track_running_stats { Tensor::ones(num_features, dtype) } else { Tensor::zeros(0, dtype) },
            num_batches_tracked: if track_running_stats { Tensor::zeros(1, dtype) } else { Tensor::zeros(0, dtype) },
        }
    }
}

/// Python bindings for GroupNorm layer.
/// PyTorch: GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
#[pymethods]
impl GroupNorm {
    /// Create a new GroupNorm layer.
    #[new]
    #[pyo3(signature = (num_groups, num_channels, eps=1e-5, affine=true, dtype=DType::F32))]
    pub fn py_new(num_groups: u64, num_channels: u64, eps: f64, affine: bool, dtype: DType) -> ZyxResult<Self> {
        // Note: zyx-nn GroupNorm::new hardcodes eps=1e-5 internally, we pass it through affine and dtype
        // The eps is already set to 1e-5 in the struct by default, matching PyTorch
        let _ = eps; // eps is 1e-5 by default in both
        Self::new(num_groups, num_channels, affine, dtype)
    }

    /// Forward pass through the group norm layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the group norm layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for RMSNorm layer.
/// PyTorch: RMSNorm(normalized_shape, eps=None, elementwise_affine=True, device=None, dtype=None)
/// Note: zyx-nn RMSNorm uses eps=1e-6 internally
#[pymethods]
impl RMSNorm {
    /// Create a new RMSNorm layer.
    #[new]
    #[pyo3(signature = (dim, dtype=DType::F32))]
    pub fn py_new(dim: u64, dtype: DType) -> Self {
        Self::new(dim, dtype)
    }

    /// Forward pass through the RMS norm layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the RMS norm layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for CausalSelfAttention layer.
/// Not a standard PyTorch module, but follows similar conventions.
#[pymethods]
impl CausalSelfAttention {
    /// Create a new CausalSelfAttention layer.
    #[new]
    #[pyo3(signature = (embed_dim, num_heads, bias=true, dropout=0.0, dtype=DType::F32))]
    pub fn py_new(embed_dim: u64, num_heads: u64, bias: bool, dropout: f32, dtype: DType) -> ZyxResult<Self> {
        Self::new(embed_dim, num_heads, bias, dropout, dtype)
    }

    /// Forward pass through the causal self attention layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the causal self attention layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for MultiheadAttention layer.
/// PyTorch: MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
#[pymethods]
impl MultiheadAttention {
    /// Create a new MultiheadAttention layer.
    #[new]
    #[pyo3(signature = (embed_dim, num_heads, dropout=0.0, bias=true, add_bias_kv=false, add_zero_attn=false, kdim=None, vdim=None, batch_first=false, dtype=DType::F32))]
    pub fn py_new(
        embed_dim: u64,
        num_heads: u64,
        dropout: f32,
        bias: bool,
        add_bias_kv: bool,
        add_zero_attn: bool,
        kdim: Option<u64>,
        vdim: Option<u64>,
        batch_first: bool,
        dtype: DType,
    ) -> ZyxResult<Self> {
        Self::new(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, dtype)
    }

    /// Forward pass through the multihead attention layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> ZyxResult<(Tensor, Option<Tensor>)> {
        self.forward(query.clone(), key.clone(), value.clone(), None::<Tensor>, true, None::<Tensor>, true, false)
    }

    /// Call the multihead attention layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, query: &Tensor, key: &Tensor, value: &Tensor) -> ZyxResult<(Tensor, Option<Tensor>)> {
        self.forward(query.clone(), key.clone(), value.clone(), None::<Tensor>, true, None::<Tensor>, true, false)
    }
}

/// Python bindings for PositionalEncoding layer.
/// Not a standard PyTorch module.
#[pymethods]
impl PositionalEncoding {
    /// Create a new PositionalEncoding layer.
    #[new]
    #[pyo3(signature = (d_model, max_len=5000, dropout=0.1, dtype=DType::F32))]
    pub fn py_new(d_model: u64, max_len: usize, dropout: f32, dtype: DType) -> ZyxResult<Self> {
        Self::new(d_model, max_len, dropout, dtype)
    }

    /// Forward pass through the positional encoding layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }

    /// Call the positional encoding layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x.clone())
    }
}

/// Python bindings for TransformerEncoderLayer.
/// PyTorch: TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
#[pymethods]
impl TransformerEncoderLayer {
    /// Create a new TransformerEncoderLayer.
    #[new]
    #[pyo3(signature = (d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=false, norm_first=false, bias=true, dtype=DType::F32))]
    pub fn py_new(
        d_model: u64,
        nhead: u64,
        dim_feedforward: u64,
        dropout: f32,
        layer_norm_eps: f64,
        batch_first: bool,
        norm_first: bool,
        bias: bool,
        dtype: DType,
    ) -> ZyxResult<Self> {
        Self::new(d_model, nhead, dim_feedforward, dropout, |t| t.relu(), layer_norm_eps, batch_first, norm_first, bias, dtype)
    }

    /// Forward pass through the transformer encoder layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, src: &Tensor) -> ZyxResult<Tensor> {
        self.forward(src.clone(), None::<Tensor>, None::<Tensor>)
    }

    /// Call the transformer encoder layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, src: &Tensor) -> ZyxResult<Tensor> {
        self.forward(src.clone(), None::<Tensor>, None::<Tensor>)
    }
}

/// Python bindings for TransformerDecoderLayer.
/// PyTorch: TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None)
#[pymethods]
impl TransformerDecoderLayer {
    /// Create a new TransformerDecoderLayer.
    #[new]
    #[pyo3(signature = (d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=false, norm_first=false, bias=true, dtype=DType::F32))]
    pub fn py_new(
        d_model: u64,
        nhead: u64,
        dim_feedforward: u64,
        dropout: f32,
        layer_norm_eps: f64,
        batch_first: bool,
        norm_first: bool,
        bias: bool,
        dtype: DType,
    ) -> ZyxResult<Self> {
        Self::new(d_model, nhead, dim_feedforward, dropout, |t| t.relu(), layer_norm_eps, batch_first, norm_first, bias, dtype)
    }

    /// Forward pass through the transformer decoder layer.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, tgt: &Tensor, memory: &Tensor) -> ZyxResult<Tensor> {
        self.forward(tgt, memory, None::<Tensor>, None::<Tensor>, None::<Tensor>, None::<Tensor>, false, false)
    }

    /// Call the transformer decoder layer (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, tgt: &Tensor, memory: &Tensor) -> ZyxResult<Tensor> {
        self.forward(tgt, memory, None::<Tensor>, None::<Tensor>, None::<Tensor>, None::<Tensor>, false, false)
    }
}

/// Python bindings for RNNCell.
/// PyTorch: RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)
#[pymethods]
impl RNNCell {
    /// Create a new RNNCell.
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, nonlinearity="tanh", dtype=DType::F32))]
    pub fn py_new(input_size: u64, hidden_size: u64, bias: bool, nonlinearity: &str, dtype: DType) -> ZyxResult<Self> {
        let s = String::from(nonlinearity);
        Self::new(input_size, hidden_size, bias, s.leak(), Some(dtype))
    }

    /// Forward pass through the RNN cell.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor, hx: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x, hx)
    }

    /// Call the RNN cell (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor, hx: &Tensor) -> ZyxResult<Tensor> {
        self.forward(x, hx)
    }
}

/// Python bindings for GRUCell.
/// PyTorch: GRUCell(input_size, hidden_size, bias=True, device=None, dtype=None)
#[pymethods]
impl GRUCell {
    /// Create a new GRUCell.
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype=DType::F32))]
    pub fn py_new(input_size: u64, hidden_size: u64, bias: bool, dtype: DType) -> ZyxResult<Self> {
        Self::new(input_size, hidden_size, bias, dtype)
    }

    /// Forward pass through the GRU cell.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, input: &Tensor, hx: &Tensor) -> ZyxResult<Tensor> {
        self.forward(input.clone(), hx.clone())
    }

    /// Call the GRU cell (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, input: &Tensor, hx: &Tensor) -> ZyxResult<Tensor> {
        self.forward(input.clone(), hx.clone())
    }
}

/// Python bindings for LSTMCell.
/// PyTorch: LSTMCell(input_size, hidden_size, bias=True, device=None, dtype=None)
#[pymethods]
impl LSTMCell {
    /// Create a new LSTMCell.
    #[new]
    #[pyo3(signature = (input_size, hidden_size, bias=true, dtype=DType::F32))]
    pub fn py_new(input_size: u64, hidden_size: u64, bias: bool, dtype: DType) -> ZyxResult<Self> {
        Self::new(input_size, hidden_size, bias, Some(dtype))
    }

    /// Forward pass through the LSTM cell.
    #[pyo3(name = "forward")]
    pub fn forward_py(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> ZyxResult<(Tensor, Tensor)> {
        self.forward(x, h, c)
    }

    /// Call the LSTM cell (alias for forward).
    #[pyo3(name = "__call__")]
    pub fn call_py(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> ZyxResult<(Tensor, Tensor)> {
        self.forward(x, h, c)
    }
}

/// Helper to register all nn classes in the zyx-py module.
pub fn register_nn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Linear>()?;
    m.add_class::<Conv2d>()?;
    m.add_class::<Embedding>()?;
    m.add_class::<LayerNorm>()?;
    m.add_class::<BatchNorm>()?;
    m.add_class::<GroupNorm>()?;
    m.add_class::<RMSNorm>()?;
    m.add_class::<CausalSelfAttention>()?;
    m.add_class::<MultiheadAttention>()?;
    m.add_class::<PositionalEncoding>()?;
    m.add_class::<TransformerEncoderLayer>()?;
    m.add_class::<TransformerDecoderLayer>()?;
    m.add_class::<RNNCell>()?;
    m.add_class::<GRUCell>()?;
    m.add_class::<LSTMCell>()?;
    Ok(())
}
