//! nn modules for zyx ML library

#![forbid(unsafe_code)]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![forbid(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
//#![forbid(rustdoc::missing_doc_code_examples)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]

extern crate alloc;

pub use zyx_derive::Module;

mod linear;
pub use linear::Linear;

// Normalization layers
mod layer_norm;
pub use layer_norm::LayerNorm;

mod batch_norm;
pub use batch_norm::BatchNorm;

mod group_norm;
pub use group_norm::GroupNorm;

mod rms_norm;
pub use rms_norm::RMSNorm;

// Recurrent layers
mod rnn_cell;
pub use rnn_cell::RNNCell;

mod causal_self_attention;
pub use causal_self_attention::CausalSelfAttention;

mod multi_head_attention;
pub use multi_head_attention::MultiheadAttention;

mod feed_forward;
pub use feed_forward::FeedForward;

mod embedding;
pub use embedding::Embedding;

mod conv2d;
pub use conv2d::Conv2d;

mod activation;
pub use activation::Activation;

mod positional_encoding;
pub use positional_encoding::PositionalEncoding;

mod transformer_encoder_layer;
pub use transformer_encoder_layer::TransformerEncoderLayer;

mod transformer_decoder_layer;
pub use transformer_decoder_layer::TransformerDecoderLayer;
