//! Optimizers for zyx ML library

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

mod sgd;
pub use sgd::SGD;

mod adam;
pub use adam::Adam;
