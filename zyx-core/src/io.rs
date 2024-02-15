use crate::{backend::Backend, dtype::DType, error::ZyxError, shape::Shape, tensor::Tensor};
use alloc::{vec::Vec, string::String};
use std::fs::File;
use std::path::Path;
use std::io::{Read, Write};
use core::fmt::Write as CoreFmtWrite;

/// This trait is implemented automatically for all modules that implement
/// IntoIterator<Item = &mut Tensor>
pub trait ModuleIO {
    /// Save self into path
    fn save(self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
    /// Load self from path
    fn load(self, path: impl AsRef<Path>) -> Result<(), ZyxError>;
}

impl<'a, B: Backend + 'a, Tensors: IntoIterator<Item = &'a mut Tensor<B>>> ModuleIO for Tensors {
    fn save(self, path: impl AsRef<Path>) -> Result<(), ZyxError> {
        save(self.into_iter().map(|x| &*x), path)
    }

    fn load(self, path: impl AsRef<Path>) -> Result<(), ZyxError> {
        let targets: Vec<&mut Tensor<B>> = self.into_iter().collect();
        let dev = targets[0].backend();
        let tensors = load(dev, path)?;
        for (x, y) in targets.into_iter().zip(tensors) {
            *x = y;
        }
        Ok(())
    }
}

/// Save all tensors into file.
/// All parameters must be realized before calling this function, otherwise it will panic.
/// # Errors
/// Returns io erorr if there was problem writing file to filesystem.
pub fn save<'a, B: Backend + 'a>(
    tensors: impl IntoIterator<Item = &'a Tensor<B>>,
    path: impl AsRef<Path>,
) -> Result<(), ZyxError> {
    let mut f = File::create(path)?;
    let mut header = String::from("{");
    let mut begin = 0;
    let tensors: Vec<&Tensor<B>> = tensors.into_iter().collect();
    for tensor in &tensors {
        let dtype = tensor.dtype();
        //if let Some(label) = tensor.label() {
        //write!(header, "\"{label}\":{{").unwrap();
        //} else {
        write!(header, "\"{}\":{{", tensor.id()).unwrap();
        //}
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
    for tensor in tensors {
        match tensor.dtype() {
            DType::F32 => {
                let vec = tensor.to_vec::<f32>()?;
                let mut bytes: Vec<u8> = Vec::with_capacity(vec.len() * 4);
                for x in vec {
                    bytes.extend(x.to_le_bytes());
                }
                f.write_all(&bytes)?;
            }
            DType::I32 => {
                let vec = tensor.to_vec::<i32>().unwrap();
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
pub fn load<B: Backend>(
    dev: B,
    path: impl AsRef<Path>,
) -> Result<impl Iterator<Item = Tensor<B>>, ZyxError> {
    let mut f = File::open(path)?;
    let mut header_len = [0u8; 8];
    f.read_exact(&mut header_len)?;
    let mut header = alloc::vec![0u8; usize::try_from(u64::from_le_bytes(header_len)).unwrap()];
    f.read_exact(&mut header)?;
    let header = core::str::from_utf8(&header)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
    let mut text = alloc::string::String::with_capacity(10);
    let mut begin_str = false;
    let mut i = 0;
    let mut tensors = Vec::new();
    let mut dtype = DType::F32;
    let mut shape: Shape = [1].into();
    for x in header.chars() {
        if ['"', '[', ']'].contains(&x) {
            if begin_str {
                //std::println!("{text}");
                if i % 7 == 0 {
                    //params[i / 7].set_label(&text);
                } else if i % 7 == 2 {
                    dtype = DType::from_safetensors(&text)?;
                } else if i % 7 == 4 {
                    shape = Shape::from_safetensors(&text)?;
                } else if i % 7 == 6 {
                    // TODO assert offsets
                    //std::println!("Offsets: {text}");
                    let offsets = text
                        .split(',')
                        .map(|offset| {
                            offset.parse::<usize>().map_err(|err| {
                                ZyxError::ParseError(alloc::format!(
                                    "Could not parse safetensors offset: {err}"
                                ))
                            })
                        })
                        .collect::<Result<Vec<usize>, ZyxError>>()?;
                    //std::println!("Offsets: {offsets:?}");
                    if offsets[tensors.len() + 1] != shape.numel() * dtype.byte_size() {
                        return Err(ZyxError::ParseError(
                            "Safetensors shapes and offsets are incorrect.".into(),
                        ));
                    }
                    let mut buf = alloc::vec![0u8; shape.numel()*dtype.byte_size()];
                    f.read_exact(&mut buf)?;
                    tensors.push(match dtype {
                        DType::F32 => {
                            let vec: Vec<f32> = buf
                                .chunks_exact(dtype.byte_size())
                                .map(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                .collect();
                            dev.tensor(vec).reshape(&shape)
                        }
                        DType::I32 => {
                            let vec: Vec<i32> = buf
                                .chunks_exact(dtype.byte_size())
                                .map(|x| i32::from_le_bytes([x[0], x[1], x[2], x[3]]))
                                .collect();
                            dev.tensor(vec).reshape(&shape)
                        }
                    });
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
    Ok(tensors.into_iter())
}
