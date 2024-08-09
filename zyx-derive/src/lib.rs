//! # zyx-derive
//!
//! This crate contains procedural macros for zyx.
//!
//! Macro Module automatically implements IntoIterator<Item = &Tensor>
//! for your module, so that you can use it in backpropagation and save it to disk.
//! ```rust
//! use zyx::Tensor;
//! use zyx_derive::Module;
//!
//! #[derive(Module)]
//! struct MyNet {
//!     b: Tensor,
//!     w: Tensor,
//! }
//!
//! impl MyNet {
//!     fn forward(&self, x: &Tensor) -> Tensor {
//!         x.dot(&self.w) + &self.b
//!     }
//! }
//! ```
//!
//! For README, quick tutorial and source code, please visit `<https://www.github.com/zk4x/zyx>`.
//!
//! For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).
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

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DataStruct, DeriveInput};

/// # Procedural macro Module
///
/// Implements IntoIterator<Item = &Tensor> and IntoIterator<Item = &mut Tensor> for your struct.
///
/// This allows saving, loading, backpropagation and updating your modules.
#[proc_macro_derive(Module)]
pub fn into_iterator_item_tensor(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let mut field_iterators = quote! {
        trait __MarkerTraitRef<'a> {
            fn __iterate_by_ref(&self, res: &mut Vec<&'a zyx::Tensor>) {}
        }

        struct __MarkerStructRef<T: Copy>(T);

        impl<'a, T: IntoIterator<Item = &'a zyx::Tensor> + Copy> __MarkerStructRef<T> {
            fn __iterate_by_ref(&self, res: &mut Vec<&'a zyx::Tensor>) {
                res.extend(self.0.into_iter());
            }
        }

        impl<'a, T: Copy> __MarkerTraitRef<'a> for __MarkerStructRef<T>{}

        let mut res = Vec::<&zyx::Tensor>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_ty: &syn::Type = &field.ty;
            use std::string::ToString;
            if quote! { #field_ty }.to_string() == "Tensor" {
                field_iterators = quote! {
                    #field_iterators
                    res.push(&self.#field_name);
                }
            } else {
                field_iterators = quote! {
                    #field_iterators
                    __MarkerStructRef::<&#field_ty>::__iterate_by_ref(&__MarkerStructRef(&self.#field_name), &mut res);
                };
            }
        }
    }

    let expanded = quote! {
        impl<'a> IntoIterator for &'a #struct_name {
            type Item = &'a zyx::Tensor;
            type IntoIter = std::vec::IntoIter<&'a zyx::Tensor>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    let mut field_iterators = quote! {
        trait __MarkerTraitMut<'a>: Sized {
            fn __iterate_by_mut(mut self, res: &mut std::vec::Vec<&'a mut zyx::Tensor>) {}
        }

        struct __MarkerStructMut<T>(T);

        impl<'a, T: IntoIterator<Item = &'a mut zyx::Tensor>> __MarkerStructMut<T> {
            fn __iterate_by_mut(mut self, res: &mut std::vec::Vec<&'a mut zyx::Tensor>) {
                res.extend(self.0.into_iter());
            }
        }

        impl<'a, T> __MarkerTraitMut<'a> for __MarkerStructMut<T>{}

        let mut res = Vec::<&mut zyx::Tensor>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_ty: &syn::Type = &field.ty;
            use std::string::ToString;
            if quote! { #field_ty }.to_string() == "Tensor" {
                field_iterators = quote! {
                    #field_iterators
                    res.push(&mut self.#field_name);
                }
            } else {
                field_iterators = quote! {
                    #field_iterators
                    __MarkerStructMut::<&#field_ty>::__iterate_by_mut(__MarkerStructMut(&mut self.#field_name), &mut res);
                };
            }
        }
    }

    let expanded = quote! {
        #expanded

        impl<'a> IntoIterator for &'a mut #struct_name {
            type Item = &'a mut zyx::Tensor;
            type IntoIter = std::vec::IntoIter<&'a mut zyx::Tensor>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    TokenStream::from(expanded)
}
