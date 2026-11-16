// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
//!         x.dot(&self.w).unwrap() + &self.b
//!     }
//! }
//! ```
//!
//! For README, quick tutorial and source code, please visit `<https://www.github.com/zk4x/zyx>`.
//!
//! For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).
#![forbid(unsafe_code)]
#![doc = include_str!("../README.md")]
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
use syn::ext::IdentExt;
use syn::{parse_macro_input, Data, DataStruct, DeriveInput};

/// Implements FromIterator<Item = (String, Tensor)> and Module for your struct.
///
/// This allows saving, loading, backpropagation and updating your modules.
///
/// Recognised field attributes:
/// - `#[no_param]` — marks a field as a non-trainable hyperparameter (or
///   state buffer). The parameter iteration skips it entirely.
///
/// Field types:
/// - `Tensor` / `Option<Tensor>` — pushed as a parameter.
/// - `Vec<T>` where `T: Module` (and `T` is not `Tensor`) — walked element by
///   element, labelled `field.{index}`. A `Vec<T>` field of a type that is
///   neither `Tensor` nor `Module` is a compile error, never a silent skip.
#[proc_macro_derive(Module, attributes(no_param))]
pub fn module_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive_module(&input)
}

fn derive_module(input: &DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // iter_tensors (immutable)
    let mut field_iterators = quote! {
        trait __MarkerTraitRef: Sized {
            fn __iterate_by_ref(self, res: &mut Vec<(String, &zyx::Tensor)>, label: &str) {}
        }

        struct __MarkerStructRef<T>(T);

        impl<'a, T: zyx::Module> __MarkerStructRef<&'a T> {
            fn __iterate_by_ref(self, res: &mut Vec<(String, &'a zyx::Tensor)>, label: &str) {
                res.extend(self.0.iter_tensors().map(|(k, t)|  (format!("{label}.{k}"), t)));
            }
        }

        impl<'a, T> __MarkerTraitRef for __MarkerStructRef<&'a T>{}

        let mut res = Vec::<(String, &zyx::Tensor)>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_name_str = field_name.to_string();

            let field_ty: &syn::Type = &field.ty;
            let no_param = has_no_param_attr(&field.attrs);

            use std::string::ToString;
            if no_param {
                // Skip non-trainable hyperparameter entirely.
            } else if quote! { #field_ty }.to_string() == "Tensor" {
                field_iterators = quote! {
                    #field_iterators
                    res.push((#field_name_str.to_string(), &self.#field_name));
                }
            } else if quote! { #field_ty }.to_string() == "Option < Tensor >" {
                field_iterators = quote! {
                    #field_iterators
                    if let Some(tensor) = &self.#field_name {
                        res.push((#field_name_str.to_string(), tensor));
                    }
                }
            } else if let Some(elem) = vec_elem_type(field_ty) {
                field_iterators = quote! {
                    #field_iterators
                    for (index, elem) in self.#field_name.iter().enumerate() {
                        __MarkerStructRef::<&#elem>::__iterate_by_ref(
                            __MarkerStructRef(elem),
                            &mut res,
                            &format!("{}.{}", #field_name_str, index),
                        );
                    }
                }
            } else {
                field_iterators = quote! {
                    #field_iterators
                    __MarkerStructRef::<&#field_ty>::__iterate_by_ref(__MarkerStructRef(&self.#field_name), &mut res, #field_name_str);
                };
            }
        }
    }

    // iter_tensors_mut
    let mut mut_field_iterators = quote! {
        trait __MarkerTraitRef: Sized {
            fn __iterate_by_ref(mut self, res: &mut Vec<(String, &mut zyx::Tensor)>, label: &str) {}
        }

        struct __MarkerStructRef<T>(T);

        impl<'a, T: zyx::Module> __MarkerStructRef<&'a mut T> {
            fn __iterate_by_ref(mut self, res: &mut Vec<(String, &'a mut zyx::Tensor)>, label: &str) {
                res.extend(self.0.iter_tensors_mut().map(|(k, t)|  (format!("{label}.{k}"), t)));
            }
        }

        impl<'a, T> __MarkerTraitRef for __MarkerStructRef<&'a mut T>{}

        let mut res = Vec::<(String, &mut zyx::Tensor)>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_name_str = field_name.to_string();

            let field_ty: &syn::Type = &field.ty;
            let no_param = has_no_param_attr(&field.attrs);

            use std::string::ToString;
            if no_param {
                // Skip non-trainable hyperparameter entirely.
            } else if quote! { #field_ty }.to_string() == "Tensor" {
                mut_field_iterators = quote! {
                    #mut_field_iterators
                    res.push((#field_name_str.to_string(), &mut self.#field_name));
                }
            } else if quote! { #field_ty }.to_string() == "Option < Tensor >" {
                mut_field_iterators = quote! {
                    #mut_field_iterators
                    if let Some(tensor) = &mut self.#field_name {
                        res.push((#field_name_str.to_string(), tensor));
                    }
                }
            } else if let Some(elem) = vec_elem_type(field_ty) {
                mut_field_iterators = quote! {
                    #mut_field_iterators
                    for (index, elem) in self.#field_name.iter_mut().enumerate() {
                        __MarkerStructRef::<&mut #elem>::__iterate_by_ref(
                            __MarkerStructRef(elem),
                            &mut res,
                            &format!("{}.{}", #field_name_str, index),
                        );
                    }
                }
            } else {
                mut_field_iterators = quote! {
                    #mut_field_iterators
                    __MarkerStructRef::<&mut #field_ty>::__iterate_by_ref(__MarkerStructRef(&mut self.#field_name), &mut res, #field_name_str);
                };
            }
        }
    }

    let expanded = quote! {
        impl #impl_generics zyx::Module for #struct_name #ty_generics #where_clause {
            fn iter<'a>(&'a self) -> impl Iterator<Item = &'a zyx::Tensor> {
                self.into_iter()
            }

            fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut zyx::Tensor> {
                self.into_iter()
            }

            fn iter_tensors<'a>(&'a self) -> impl Iterator<Item = (String, &'a zyx::Tensor)> {
                #field_iterators
                res.into_iter()
            }

            fn iter_tensors_mut<'a>(&'a mut self) -> impl Iterator<Item = (String, &'a mut zyx::Tensor)> {
                #mut_field_iterators
                res.into_iter()
            }
        }
    };

    // Implementation of IntoIterator<Item = &Tensor>
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
            let no_param = has_no_param_attr(&field.attrs);
            use std::string::ToString;
            if no_param {
                // Skip non-trainable hyperparameter entirely.
            } else if quote! { #field_ty }.to_string() == "Tensor" {
                field_iterators = quote! {
                    #field_iterators
                    res.push(&self.#field_name);
                }
            } else if vec_elem_type(field_ty).is_some() {
                field_iterators = quote! {
                    #field_iterators
                    for elem in self.#field_name.iter() {
                        res.extend(::core::iter::IntoIterator::into_iter(elem));
                    }
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
        #expanded

        impl<'a> IntoIterator for &'a #struct_name #ty_generics {
            type Item = &'a zyx::Tensor;
            type IntoIter = std::vec::IntoIter<&'a zyx::Tensor>;

            fn into_iter(self) -> Self::IntoIter {
                #field_iterators
                res.into_iter()
            }
        }
    };

    // Implementation of IntoIterator<Item = &mut Tensor>
    let mut field_iterators = quote! {
        trait MarkerTraitMut<'a>: Sized {
            fn iterate_by_mut(mut self, res: &mut Vec<&'a mut zyx::Tensor>) {}
        }

        struct MarkerStructMut<T>(T);

        impl<'a, T: IntoIterator<Item = &'a mut zyx::Tensor>> MarkerStructMut<T> {
            fn iterate_by_mut(mut self, res: &mut Vec<&'a mut zyx::Tensor>) {
                res.extend(self.0.into_iter());
            }
        }

        impl<'a, T> MarkerTraitMut<'a> for MarkerStructMut<T>{}

        let mut res = Vec::<&mut zyx::Tensor>::new();
    };

    if let Data::Struct(DataStruct { fields, .. }) = &input.data {
        for field in fields.iter() {
            let field_name = match &field.ident {
                Some(ident) => ident,
                None => panic!("Unnamed fields are not supported"),
            };
            let field_ty: &syn::Type = &field.ty;
            let no_param = has_no_param_attr(&field.attrs);
            use std::string::ToString;
            if no_param {
                // Skip non-trainable hyperparameter entirely.
            } else if quote! { #field_ty }.to_string() == "Tensor" {
                field_iterators = quote! {
                    #field_iterators
                    res.push(&mut self.#field_name);
                }
            } else if vec_elem_type(field_ty).is_some() {
                field_iterators = quote! {
                    #field_iterators
                    for elem in self.#field_name.iter_mut() {
                        res.extend(::core::iter::IntoIterator::into_iter(elem));
                    }
                }
            } else {
                field_iterators = quote! {
                    #field_iterators
                    MarkerStructMut::<&mut #field_ty>::iterate_by_mut(MarkerStructMut(&mut self.#field_name), &mut res);
                };
            }
        }
    }

    let expanded = quote! {
        #expanded

        impl<'a> IntoIterator for &'a mut #struct_name #ty_generics {
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

/// Returns the element type when `ty` is `Vec<Elem>` with `Elem` a module
/// (anything but `Tensor`, which has its own `Module` impl on `Vec<Tensor>`).
fn vec_elem_type(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(type_path) = ty else { return None };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Vec" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else { return None };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    let elem = types.next()?;
    if types.next().is_some() || quote! { #elem }.to_string() == "Tensor" {
        return None;
    }
    Some(elem)
}

/// Returns true if any of the field's attributes is `#[no_param]`, which marks a
/// `Tensor` / `Option<Tensor>` field as a non-trainable hyperparameter (or
/// state buffer) that the `Module` derive's parameter iteration should skip.
fn has_no_param_attr(attrs: &[syn::Attribute]) -> bool {
    attrs.iter().any(|a| a.path().is_ident("no_param"))
}

/// Derives GGUF metadata parsing for a hyperparameter struct.
///
/// The derive generates two constructors on the struct:
///
/// - `from_gguf(metadata, <skipped fields...>)` reads
///   `general.architecture` and delegates with `"<arch>."` as the prefix.
/// - `from_gguf_with_prefix(metadata, prefix, <skipped fields...>)` resolves
///   every key as `format!("{prefix}{suffix}")`. `prefix` is used verbatim, so
///   it must include the trailing dot; this is the entry a parent struct's
///   `#[gguf(group = ...)]` field calls.
///
/// Field attributes:
/// - `#[gguf("suffix")]` — metadata key suffix, relative to the prefix. A
///   field without the attribute uses its own name.
/// - `#[gguf(skip)]` — not in the metadata; the value comes from a trailing
///   constructor parameter, in declaration order.
/// - `#[gguf(group = "rope.scaling", gate = "rope.scaling.type")]` — an
///   `Option<Nested>` field whose `Nested` also derives `gguf`. The field is
///   `None` when the gate key is absent and
///   `Some(Nested::from_gguf_with_prefix(metadata, "{prefix}{group}."))`
///   otherwise; the gate is relative to the same prefix.
///
/// `default = <value>` on any of the above sets the field's value in the
/// generated `impl Default` (any expression: `128256`, `1e-5`, `false`,
/// `Some(8)`, `None`). A field without one uses
/// `::core::default::Default::default()` (0 / 0.0 / false / `None`); a
/// `#[gguf(skip)]` field must declare one, since it has no metadata key.
/// The derive always emits `impl Default`, so a hand-written one must be
/// removed.
///
/// Field types: `i64` (`int`), `Option<i64>` (`opt_int`), `f64` (`float`),
/// `Option<f64>` (`opt_float`), or a group field.
#[proc_macro_derive(gguf, attributes(gguf))]
pub fn gguf_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    derive_gguf(&input)
}

/// One field's `#[gguf(...)]` specification.
enum GgufField {
    /// Metadata key suffix; the field keeps its own value.
    Key {
        /// Key suffix, relative to the prefix.
        suffix: String,
        /// `Default` value, when the field declares one.
        default: Option<syn::Expr>,
    },
    /// Value comes from a constructor parameter.
    Skip {
        /// `Default` value; mandatory, a skipped field has no metadata key.
        default: syn::Expr,
    },
    /// `Option<Nested>` built from a key group, gated on a key's presence.
    Group {
        /// Key group, relative to the prefix (`"rope.scaling"`).
        group: String,
        /// Key whose presence turns the group on (`"rope.scaling.type"`).
        gate: String,
        /// `Default` value, when the field declares one.
        default: Option<syn::Expr>,
    },
}

/// One comma-separated item inside `#[gguf(...)]`: `"key"`, `skip`,
/// `group = ".."`, `gate = ".."` or `default = <value>`.
enum GgufArg {
    /// Bare metadata key suffix.
    Key(syn::LitStr),
    /// Value used by the generated `Default`.
    Default(syn::Expr),
    /// Value comes from a constructor parameter.
    Skip,
    /// Key group prefix.
    Group(syn::LitStr),
    /// Key whose presence turns a group on.
    Gate(syn::LitStr),
}

impl syn::parse::Parse for GgufArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(syn::LitStr) {
            return Ok(Self::Key(input.parse()?));
        }
        let name: syn::Ident = input.call(syn::Ident::parse_any)?;
        if name == "skip" && !input.peek(syn::Token![=]) {
            return Ok(Self::Skip);
        }
        input.parse::<syn::Token![=]>()?;
        if name == "default" {
            return Ok(Self::Default(input.parse()?));
        }
        if name == "group" {
            return Ok(Self::Group(input.parse()?));
        }
        if name == "gate" {
            return Ok(Self::Gate(input.parse()?));
        }
        Err(syn::Error::new(
            name.span(),
            "unknown `gguf` attribute; expected a key string, `skip`, `group = \"...\"`, `gate = \"...\"` or `default = <value>`",
        ))
    }
}

/// Reads a field's `#[gguf(...)]` attribute, defaulting to the field name.
fn gguf_field(attrs: &[syn::Attribute], name: &syn::Ident) -> syn::Result<GgufField> {
    let field_name = name.to_string();
    let Some(attr) = attrs.iter().find(|a| a.path().is_ident("gguf")) else {
        return Ok(GgufField::Key {
            suffix: field_name,
            default: None,
        });
    };
    if let syn::Meta::Path(_) = &attr.meta {
        return Ok(GgufField::Key {
            suffix: field_name,
            default: None,
        });
    }
    let args = attr.parse_args_with(syn::punctuated::Punctuated::<GgufArg, syn::Token![,]>::parse_terminated)?;
    let mut suffix: Option<String> = None;
    let mut default: Option<syn::Expr> = None;
    let mut skip = false;
    let mut group: Option<String> = None;
    let mut gate: Option<String> = None;
    for arg in args {
        match arg {
            GgufArg::Key(lit) => {
                if suffix.is_some() {
                    return Err(syn::Error::new_spanned(lit, "duplicate `gguf` key"));
                }
                suffix = Some(lit.value());
            }
            GgufArg::Default(expr) => {
                if default.is_some() {
                    return Err(syn::Error::new_spanned(expr, "duplicate `gguf` default"));
                }
                default = Some(expr);
            }
            GgufArg::Skip => skip = true,
            GgufArg::Group(lit) => {
                if group.is_some() {
                    return Err(syn::Error::new_spanned(lit, "duplicate `gguf` group"));
                }
                group = Some(lit.value());
            }
            GgufArg::Gate(lit) => {
                if gate.is_some() {
                    return Err(syn::Error::new_spanned(lit, "duplicate `gguf` gate"));
                }
                gate = Some(lit.value());
            }
        }
    }
    match (skip, group, gate) {
        (true, None, None) => match default {
            Some(default) => Ok(GgufField::Skip { default }),
            None => Err(syn::Error::new_spanned(
                attr,
                "`#[gguf(skip)]` needs `default = <value>`; a skipped field has no metadata key",
            )),
        },
        (true, _, _) => Err(syn::Error::new_spanned(attr, "`skip` cannot be combined with `group`/`gate`")),
        (false, Some(group), Some(gate)) => Ok(GgufField::Group { group, gate, default }),
        (false, Some(_), None) | (false, None, Some(_)) => {
            Err(syn::Error::new_spanned(attr, "`group` and `gate` are used together"))
        }
        (false, None, None) => Ok(GgufField::Key {
            suffix: suffix.unwrap_or(field_name),
            default,
        }),
    }
}

/// Returns the element type when `ty` is `Option<Elem>`.
fn option_inner(ty: &syn::Type) -> Option<&syn::Type> {
    let syn::Type::Path(type_path) = ty else { return None };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else { return None };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        syn::GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    types.next()
}

/// The `Self { field: .. }` entry of the generated `Default`: the declared
/// value, or `Default::default()` (0 / 0.0 / false / `None`) when there is none.
fn gguf_default_init(name: &syn::Ident, default: Option<&syn::Expr>) -> proc_macro2::TokenStream {
    match default {
        Some(expr) => quote!(#name: #expr),
        None => quote!(#name: ::core::default::Default::default()),
    }
}

fn derive_gguf(input: &DeriveInput) -> TokenStream {
    let struct_name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(DataStruct { fields, .. }) => fields,
        _ => return syn::Error::new_spanned(struct_name, "`gguf` can only be derived for a struct").to_compile_error().into(),
    };

    let mut specs = Vec::new();
    for field in fields.iter() {
        let Some(name) = &field.ident else {
            return syn::Error::new_spanned(struct_name, "`gguf` needs a struct with named fields").to_compile_error().into();
        };
        match gguf_field(&field.attrs, name) {
            Ok(spec) => specs.push((name, &field.ty, spec)),
            Err(error) => return error.to_compile_error().into(),
        }
    }

    let mut inits = Vec::new();
    let mut default_inits = Vec::new();
    let mut skip_params = Vec::new();
    let mut skip_args = Vec::new();
    for (name, ty, spec) in &specs {
        match spec {
            GgufField::Skip { default } => {
                skip_params.push(quote!(#name: #ty));
                skip_args.push(quote!(#name));
                inits.push(quote!(#name));
                default_inits.push(gguf_default_init(name, Some(default)));
            }
            GgufField::Key { suffix, default } => {
                let accessor = match quote!(#ty).to_string().as_str() {
                    "i64" => quote!(metadata.int(&key)?),
                    "Option < i64 >" => quote!(metadata.opt_int(&key)?),
                    "f64" => quote!(metadata.float(&key)?),
                    "Option < f64 >" => quote!(metadata.opt_float(&key)?),
                    _ => {
                        return syn::Error::new_spanned(
                            ty,
                            "`gguf` field must be `i64`, `Option<i64>`, `f64` or `Option<f64>`; \
                             use `#[gguf(group = \"..\", gate = \"..\")]` for a nested struct or `#[gguf(skip)]`",
                        )
                        .to_compile_error()
                        .into();
                    }
                };
                inits.push(quote! {
                    #name: {
                        let key = format!("{prefix}{}", #suffix);
                        #accessor
                    }
                });
                default_inits.push(gguf_default_init(name, default.as_ref()));
            }
            GgufField::Group { group, gate, default } => {
                let Some(inner) = option_inner(ty) else {
                    return syn::Error::new_spanned(ty, "a `gguf` group field must be `Option<Nested>`").to_compile_error().into();
                };
                inits.push(quote! {
                    #name: if metadata.contains_key(&format!("{prefix}{}", #gate)) {
                        Some(#inner::from_gguf_with_prefix(metadata, &format!("{prefix}{}.", #group))?)
                    } else {
                        None
                    }
                });
                default_inits.push(gguf_default_init(name, default.as_ref()));
            }
        }
    }

    // `impl Default` bounds every type parameter, the way `#[derive(Default)]` does.
    let mut default_generics = input.generics.clone();
    for param in default_generics.type_params_mut() {
        param.bounds.push(syn::parse_quote!(::core::default::Default));
    }
    let (default_impl_generics, _, default_where_clause) = default_generics.split_for_impl();

    let expanded = quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #[doc = "Builds the struct from GGUF metadata, keying every field on `general.architecture`. Skipped fields are the extra parameters, in declaration order."]
            pub fn from_gguf(
                metadata: &::zyx::GGUFMetadata
                #(, #skip_params)*
            ) -> ::core::result::Result<Self, ::zyx::ZyxError> {
                let prefix = format!("{}.", metadata.str("general.architecture")?);
                Self::from_gguf_with_prefix(metadata, &prefix #(, #skip_args)*)
            }

            #[doc = "Builds the struct from GGUF metadata under an explicit key prefix. `prefix` is used verbatim and must include the trailing dot (`\"llama.\"`). Skipped fields are the extra parameters, in declaration order."]
            pub fn from_gguf_with_prefix(
                metadata: &::zyx::GGUFMetadata,
                prefix: &str
                #(, #skip_params)*
            ) -> ::core::result::Result<Self, ::zyx::ZyxError> {
                Ok(Self { #(#inits),* })
            }
        }

        impl #default_impl_generics ::core::default::Default for #struct_name #ty_generics #default_where_clause {
            fn default() -> Self {
                Self { #(#default_inits),* }
            }
        }
    };
    expanded.into()
}

