// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Optimizers for zyx ML library

#![forbid(unsafe_code)]
#![forbid(rustdoc::broken_intra_doc_links)]
#![forbid(rustdoc::private_intra_doc_links)]
#![forbid(missing_docs)]
#![forbid(rustdoc::missing_crate_level_docs)]
#![forbid(rustdoc::private_doc_tests)]
#![forbid(rustdoc::invalid_codeblock_attributes)]
#![forbid(rustdoc::invalid_html_tags)]
#![forbid(rustdoc::invalid_rust_codeblocks)]
#![forbid(rustdoc::bare_urls)]
#![forbid(rustdoc::unescaped_backticks)]
#![forbid(rustdoc::redundant_explicit_links)]

mod adam;
pub use adam::Adam;

mod adamw;
pub use adamw::AdamW;

mod rmsprop;
pub use rmsprop::RMSprop;

mod sgd;
pub use sgd::SGD;

#[cfg(feature = "py")]
mod py_bindings;
#[cfg(feature = "py")]
pub use py_bindings::register_optimizers;
