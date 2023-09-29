//! # Parameters

extern crate alloc;
use alloc::{boxed::Box, vec::Vec};

use crate::{node_id::NodeId, tensor::Tensor, OutOfMemoryError};

/// # `IntoParameters`
// TODO make sure that input does not contain duplicates!
#[allow(clippy::module_name_repetitions)]
pub trait IntoParameters<'p> {
    /// Convert this value into parameters
    fn into_parameters(self) -> Parameters<'p>;

    /// Get parameters as vec
    fn into_vec(self) -> Vec<&'p mut Tensor>
    where
        Self: Sized,
    {
        self.into_parameters().params
    }

    /// Set gradient to zero for all parameters
    fn zero_grad(self)
    where
        Self: Sized,
    {
        for parameter in self.into_parameters().params {
            parameter.zero_grad();
        }
    }

    /// Realize all parameters
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    fn realize(self) -> Result<(), OutOfMemoryError>
    where
        Self: Sized,
    {
        let params = self.into_parameters().params;
        if !params.is_empty() {
            let nodes: Box<[NodeId]> = params
                .iter()
                .map(|tensor| NodeId::new(tensor.data().id()))
                .collect();
            params[0].context().realize(&nodes)?;
        }
        Ok(())
    }

    /// Realize gradients of all parameters
    /// # Errors
    /// Returns [`OutOfMemoryError`] if backend failed to allocate
    /// necessary memory for result and intermediate tensors.
    fn realize_grads(self) -> Result<(), OutOfMemoryError>
    where
        Self: Sized,
    {
        let params = self.into_parameters().params;
        if !params.is_empty() {
            let nodes: Box<[NodeId]> = params
                .iter()
                .filter_map(|tensor| tensor.grad().map(|x| NodeId::new(x.id())))
                .collect();
            params[0].context().realize(&nodes)?;
        }
        Ok(())
    }

    /// Get number of all scalars in all parameters
    fn numel(self) -> usize
    where
        Self: Sized,
    {
        self.into_parameters()
            .params
            .iter()
            .fold(0, |init, p| init + p.shape().numel())
    }

    /// Get number of all parameters
    fn len(&self) -> usize
    where
        Self: Sized,
    {
        self.len()
    }

    /// Is number of parameters zero?
    fn is_empty(&self) -> bool
    where
        Self: Sized,
    {
        self.len() == 0
    }

    /// Save all parameters into file.
    /// All parameters must be realized before calling this function, otherwise it will panic.
    /// # Errors
    /// Returns io erorr if there was problem writing file to filesystem.
    #[cfg(feature = "io")]
    fn save(self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error>
    where
        Self: Sized,
    {
        use crate::dtype::DType;
        use core::fmt::Write as CoreFmtWrite;
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        let mut params = self.into_vec();
        let mut header = alloc::string::String::from("{");
        let mut begin = 0;
        for tensor in &mut params {
            let dtype = tensor.dtype();
            if let Some(label) = tensor.label() {
                write!(header, "\"{label}\":{{").unwrap();
            } else {
                write!(header, "\"{}\":{{", tensor.id()).unwrap();
            }
            write!(header, "\"dtype\":\"{}\",", dtype.safetensors()).unwrap();
            write!(header, "\"shape\":{},", tensor.shape().safetensors()).unwrap();
            let size = tensor.numel() * dtype.byte_size();
            write!(header, "\"data_offsets\":[{},{}]", begin, begin + size).unwrap();
            begin += size;
            write!(header, "}},").unwrap();
        }
        header.pop();
        write!(header, "}}").unwrap();
        let header_bytes = header.as_bytes();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(header_bytes)?;
        for tensor in params {
            match tensor.dtype() {
                DType::F32 => {
                    let vec = tensor.to_vec().unwrap();
                    let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
                    for x in vec {
                        bytes.extend(x.to_le_bytes());
                    }
                    f.write_all(&bytes)?;
                }
                DType::I32 => {
                    let vec = tensor.to_vec_i32().unwrap();
                    let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
                    for x in vec {
                        bytes.extend(x.to_le_bytes());
                    }
                    f.write_all(&bytes)?;
                }
            };
        }
        Ok(())
    }

    /// Load all parameters from file
    /// # Errors
    /// Returns io error if there was io erorr or parsing error.
    #[cfg(feature = "io")]
    fn load(self, path: impl AsRef<std::path::Path>) -> Result<(), std::io::Error>
    where
        Self: Sized,
    {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut header_len = [0u8; 8];
        f.read_exact(&mut header_len)?;
        let mut header = alloc::vec![0u8; usize::try_from(u64::from_le_bytes(header_len)).unwrap()];
        f.read_exact(&mut header)?;
        let header = core::str::from_utf8(&header)
            .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        let mut text = alloc::string::String::with_capacity(10);
        let mut begin_str = false;
        let mut params = self.into_vec();
        let mut i = 0;
        let graph = params[0].graph.clone();
        for x in header.chars() {
            if ['"', '[', ']'].contains(&x) {
                if begin_str {
                    //std::println!("{text}");
                    if i % 7 == 0 {
                        params[i / 7].set_label(&text);
                    /*} else if i % 7 == 2 {
                        // TODO assert dtype
                    } else if i % 7 == 4 {
                        // TODO assert shape
                        //params[i/7].shape() == text;*/
                    } else if i % 7 == 6 {
                        // TODO assert offsets
                        use crate::dtype::DType;
                        let shape = params[i / 7].shape().clone();
                        let n = shape.numel();
                        let mut buf = alloc::vec![0u8; n*4];
                        f.read_exact(&mut buf)?;
                        let t = Tensor {
                            data: match params[i / 7].dtype() {
                                DType::F32 => {
                                    let vec = buf
                                        .chunks_exact(4)
                                        .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                        .collect();
                                    graph
                                        .borrow_mut()
                                        .push(crate::graph::Node::StoreF32(vec, shape))
                                }
                                DType::I32 => {
                                    let vec = buf
                                        .chunks_exact(4)
                                        .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                        .collect();
                                    graph
                                        .borrow_mut()
                                        .push(crate::graph::Node::StoreI32(vec, shape))
                                }
                            },
                            grad: None,
                            graph: graph.clone(),
                        };
                        params[i / 7].set_data(t);
                    }
                    i += 1;
                    text.clear();
                    begin_str = false;
                } else {
                    text.clear();
                    begin_str = true;
                }
            } else {
                text.push(x);
            }
        }
        Ok(())
    }
}

/*#[test]
fn test_st() {
    use crate::context::Context;

    let mut ctx = Context::new();
    //let mut x = ctx.tensor([[3, 4, 2], [4, 2, 3]]);
    //let mut y = ctx.tensor([[3., 4., 2.], [4., 2., 6.]]);
    let mut x = ctx.zeros_i32((2, 3));
    let mut y = ctx.zeros((2, 3));
    //(&mut x, &mut y).realize().unwrap();
    //(&mut x, &mut y).save("model2.safetensors").unwrap();
    (&mut x, &mut y).load("model2.safetensors").unwrap();
    (&mut x, &mut y).realize().unwrap();
    std::println!("{}\n{}", x, y);
    panic!()
}*/

/// # Parameters
#[derive(Debug)]
pub struct Parameters<'p> {
    params: Vec<&'p mut Tensor>,
}

impl<'p> IntoIterator for Parameters<'p> {
    type Item = &'p mut Tensor;
    type IntoIter = <Vec<&'p mut Tensor> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<'p> Parameters<'p> {
    /// Join with other parameters
    #[must_use]
    pub fn join(self, other: impl IntoParameters<'p>) -> Parameters<'p> {
        Parameters {
            params: self
                .params
                .into_iter()
                .chain(other.into_parameters().params)
                .collect(),
        }
    }
}

impl<'p, M: crate::nn::Module> IntoParameters<'p> for &'p mut M {
    fn into_parameters(self) -> Parameters<'p> {
        self.parameters()
    }
}

impl<'p> IntoParameters<'p> for Parameters<'p> {
    fn into_parameters(self) -> Parameters<'p> {
        self
    }
}

impl<'p> IntoParameters<'p> for &'p mut Tensor {
    fn into_parameters(self) -> Parameters<'p> {
        Parameters {
            params: alloc::vec![self],
        }
    }
}

impl<'p, const N: usize> IntoParameters<'p> for [&'p mut Tensor; N] {
    fn into_parameters(self) -> Parameters<'p> {
        Parameters {
            params: self.into(),
        }
    }
}

impl<'p> IntoParameters<'p> for Vec<&'p mut Tensor> {
    fn into_parameters(self) -> Parameters<'p> {
        Parameters { params: self }
    }
}

impl<'p> IntoParameters<'p> for (&'p mut Tensor, &'p mut Tensor) {
    fn into_parameters(self) -> Parameters<'p> {
        Parameters {
            params: alloc::vec![self.0, self.1],
        }
    }
}
