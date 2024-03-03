//! # zyx-derive
//!
//! This crate contains procedural macros for zyx.
//!
//! Macro Module automatically implements IntoIterator<Item = &Tensor>
//! for your module, so that you can use it in backpropagation and save it to disk.
//! ```rust
//! use zyx_core::backend::Backend;
//! use zyx_core::tensor::Tensor;
//! use zyx_derive::Module;
//!
//! #[derive(Module)]
//! struct MyNet<B: Backend> {
//!     b: Tensor<B>,
//!     w: Tensor<B>,
//! }
//!
//! impl<B: Backend> MyNet<B> {
//!     fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
//!         x.dot(&self.w) + &self.b
//!     }
//! }
//! ```
//!
//! For README, quick tutorial and source code, please visit [https://www.github.com/zk4x/zyx].
//!
//! For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).
#![no_std]
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

extern crate proc_macro;
use proc_macro::{TokenStream};
use quote::{quote};
use syn::{Data, DataStruct, DeriveInput, parse_macro_input};

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
        trait __MarkerTraitRef<'a, B: zyx_core::backend::Backend + 'a> {
            fn __iterate_by_ref(&self, res: &mut Vec<&'a zyx_core::tensor::Tensor<B>>) {}
        }

        struct __MarkerStructRef<T: Copy>(T);

        impl<'a, B: zyx_core::backend::Backend + 'a, T: IntoIterator<Item = &'a zyx_core::tensor::Tensor<B>> + Copy> __MarkerStructRef<T> {
            fn __iterate_by_ref(&self, res: &mut Vec<&'a zyx_core::tensor::Tensor<B>>) {
                res.extend(self.0.into_iter());
            }
        }

        impl<'a, B: zyx_core::backend::Backend + 'a, T: Copy> __MarkerTraitRef<'a, B> for __MarkerStructRef<T>{}

        let mut res = Vec::<&zyx_core::tensor::Tensor<_>>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_ty: &syn::Type = &field.ty;
            // TODO check if field is tensor, or implement IntoIterator<Item = &Tensor> for &Tensor
            field_iterators = quote! {
                #field_iterators
                __MarkerStructRef::<&#field_ty>::__iterate_by_ref(&__MarkerStructRef(&self.#field_name), &mut res);
            };
        }
    }

    let expanded = quote! {
        impl<'a, B: zyx_core::backend::Backend> IntoIterator for &'a #struct_name<B> {
            type Item = &'a zyx_core::tensor::Tensor<B>;
            type IntoIter = std::vec::IntoIter<&'a zyx_core::tensor::Tensor<B>>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    let mut field_iterators = quote! {
        trait __MarkerTraitMut<'a, B: zyx_core::backend::Backend + 'a>: Sized {
            fn __iterate_by_mut(mut self, res: &mut Vec<&'a mut zyx_core::tensor::Tensor<B>>) {}
        }

        struct __MarkerStructMut<T>(T);

        impl<'a, B: zyx_core::backend::Backend + 'a, T: IntoIterator<Item = &'a mut zyx_core::tensor::Tensor<B>>> __MarkerStructMut<T> {
            fn __iterate_by_mut(mut self, res: &mut Vec<&'a mut zyx_core::tensor::Tensor<B>>) {
                res.extend(self.0.into_iter());
            }
        }

        impl<'a, B: zyx_core::backend::Backend + 'a, T> __MarkerTraitMut<'a, B> for __MarkerStructMut<T>{}

        let mut res = Vec::<&mut zyx_core::tensor::Tensor<_>>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_ty: &syn::Type = &field.ty;
            field_iterators = quote! {
                #field_iterators
                __MarkerStructMut::<&#field_ty>::__iterate_by_mut(__MarkerStructMut(&mut self.#field_name), &mut res);
            };
        }
    }

    let expanded = quote! {
        #expanded

        impl<'a, B: zyx_core::backend::Backend> IntoIterator for &'a mut #struct_name<B> {
            type Item = &'a mut zyx_core::tensor::Tensor<B>;
            type IntoIter = std::vec::IntoIter<&'a mut zyx_core::tensor::Tensor<B>>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    TokenStream::from(expanded)
}
