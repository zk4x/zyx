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
        macro_rules! is_trait{
            ($name:ty, $trait_name:path)=>{{
                trait __InnerMarkerTrait{
                    fn __is_trait_inner_method()->bool{
                        false
                    }
                }
                struct __TraitTest<T>(T);
                impl<T:$trait_name> __TraitTest<T> {
                    fn __is_trait_inner_method()->bool{
                        true
                    }
                }
                impl<T> __InnerMarkerTrait for __TraitTest<T>{}
                __TraitTest::<$name>::__is_trait_inner_method()
            }}
        }

        let mut res = Vec::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };

            field_iterators = quote! {
                #field_iterators
                if is_trait!(#field_name, IntoIterator) {
                    res.extend(self.#field_name.into_iter());
                }
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
