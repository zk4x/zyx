extern crate proc_macro;
use proc_macro::{TokenStream};
use std::any::Any;
use quote::{quote};
use syn::{Data, DataStruct, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(Module)]
pub fn into_iterator_for_a(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let struct_name = &input.ident;

    let mut field_iterators = quote! {
        trait __InnerMarkerTrait<'a, B: zyx_core::backend::Backend + 'a> {
            fn __is_trait_inner_method(&self, res: &mut Vec<&'a zyx_core::tensor::Tensor<B>>) {}
        }

        struct __TraitTest<T: Copy>(T);

        impl<'a, B: zyx_core::backend::Backend + 'a, T: IntoIterator<Item = &'a zyx_core::tensor::Tensor<B>> + Copy> __TraitTest<T> {
            fn __is_trait_inner_method(&self, res: &mut Vec<&'a zyx_core::tensor::Tensor<B>>) {
                res.extend((&self.0).into_iter());
            }
        }

        impl<'a, B: zyx_core::backend::Backend + 'a, T: Copy> __InnerMarkerTrait<'a, B> for __TraitTest<T>{}

        let mut res = Vec::new();
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
                __TraitTest::<&#field_ty>::__is_trait_inner_method(&__TraitTest(&self.#field_name), &mut res);
            };
        }
    }

    let expanded = quote! {
        impl<'a, B: zyx_core::backend::Backend> IntoIterator for &'a #struct_name<B> {
            type Item = &'a zyx_core::tensor::Tensor<B>;
            type IntoIter = vec::IntoIter<&'a zyx_core::tensor::Tensor<B>>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    TokenStream::from(expanded)
}
