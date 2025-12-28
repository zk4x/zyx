use std::{collections::HashMap, ffi::OsStr, fs::File, path::Path};

use crate::{DType, Map, RT, Tensor, ZyxError, shape::Dim};

/// Module trait
pub trait Module {
    /// Iterate over all tensors immutably
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Tensor>;

    /// Iterate over all tensors mutably
    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Tensor>;

    /// Iterate over tensors without consuming the module
    fn iter_tensors<'a>(&'a self) -> impl Iterator<Item = (String, &'a Tensor)>;

    /// From tensors
    fn iter_tensors_mut<'a>(&'a mut self) -> impl Iterator<Item = (String, &'a mut Tensor)>;

    /// Realize all tensors in the module
    fn realize<'a>(&'a self) -> Result<(), ZyxError> {
        Tensor::realize(self.iter())
    }

    /// Set parameters, removes them from params, skips parameters that are not found in params.
    fn set_params(&mut self, params: &mut HashMap<String, Tensor>) {
        for (label, tensor) in self.iter_tensors_mut() {
            if let Some(param) = params.remove(&label) {
                *tensor = param;
            }
        }
    }

    /// Save tensors or modules to a file determined by file extension.
    /// Currently only safetensors is supported format.
    ///
    /// # Errors
    ///
    /// Errors if tensors failed to realize or failed to save to disk.
    fn save(&self, path: impl AsRef<Path>) -> Result<(), ZyxError> {
        use std::fmt::Write;
        use std::io::Write as IOWrite;
        let mut f = File::create(path)?;
        let mut header = String::from("{");
        let mut begin = 0;
        for (label, tensor) in self.iter_tensors() {
            let dtype = tensor.dtype();
            write!(header, "\"{label}\":{{").unwrap();
            write!(header, "\"dtype\":\"{}\",", dtype.safetensors()).unwrap();
            let mut st_shape = format!("{:?}", tensor.shape());
            st_shape.retain(|c| !c.is_whitespace());
            write!(header, "\"shape\":{st_shape},").unwrap();
            let size = tensor.numel() * dtype.byte_size() as Dim;
            write!(header, "\"data_offsets\":[{},{}]", begin, begin + size).unwrap();
            begin += size;
            write!(header, "}},").unwrap();
        }
        header.pop();
        write!(header, "}}").unwrap();
        let header_bytes = header.as_bytes();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
        f.write_all(header_bytes)?;
        for tensor in self.iter() {
            f.write_all(&tensor.to_le_bytes()?)?;
        }
        Ok(())
    }
}

/// GGUF metadata
#[allow(unused)]
pub enum GGUFMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Box<[GGUFMetadataValue]>),
}

impl<S: std::hash::BuildHasher + Default> Module for HashMap<String, Tensor, S> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Tensor> {
        self.values()
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Tensor> {
        self.values_mut()
    }

    fn iter_tensors<'a>(&'a self) -> impl Iterator<Item = (String, &'a Tensor)> {
        self.iter().map(|(k, v): (&String, &Tensor)| (k.clone(), v))
    }

    fn iter_tensors_mut<'a>(&'a mut self) -> impl Iterator<Item = (String, &'a mut Tensor)> {
        self.iter_mut().map(|(k, v): (&String, &Tensor)| (k.clone(), v))
    }
}

impl Module for Vec<Tensor> {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Tensor> {
        self.into_iter()
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Tensor> {
        self.into_iter()
    }

    fn iter_tensors<'a>(&'a self) -> impl Iterator<Item = (String, &'a Tensor)> {
        self.iter().map(|t: &Tensor| (format!("{}", t.id()), t))
    }

    fn iter_tensors_mut<'a>(&'a mut self) -> impl Iterator<Item = (String, &'a mut Tensor)> {
        self.iter_mut().map(|t: &Tensor| (format!("{}", t.id()), t))
    }
}

impl<M0: Module, M1: Module> Module for (M0, M1) {
    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Tensor> {
        self.0.iter().chain(self.1.iter())
    }

    fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Tensor> {
        self.0.iter_mut().chain(self.1.iter_mut())
    }

    fn iter_tensors<'a>(&'a self) -> impl Iterator<Item = (String, &'a Tensor)> {
        self.0.iter_tensors().chain(self.1.iter_tensors())
    }

    fn iter_tensors_mut<'a>(&'a mut self) -> impl Iterator<Item = (String, &'a mut Tensor)> {
        self.0.iter_tensors_mut().chain(self.1.iter_tensors_mut())
    }
}

impl Tensor {
    /// Load module from path. This function will determine the filetype based on file extension.
    ///
    /// # Errors
    ///
    /// Errors if loading from disk failed or if loaded tensors could not be allocated to device.
    #[allow(clippy::missing_panics_doc)]
    pub fn load(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, ZyxError>
    where
        Self: Sized,
    {
        RT.lock().initialize_devices()?; // So that we load debug mask
        let e = path.as_ref().extension().and_then(OsStr::to_str).unwrap();
        match e {
            "safetensors" => Self::load_safetensors(path),
            "gguf" => Ok(Self::load_gguf(path)?.1),
            _ => panic!("Unknown file extension. Zyx currently supports only safetensors format."),
        }
    }

    /// Load gguf module from path
    /// First returned value is metadata, second returned value are named tensors
    /// # Errors
    /// read failure
    #[allow(clippy::missing_panics_doc)]
    #[allow(clippy::type_complexity)]
    pub fn load_gguf(
        path: impl AsRef<Path>,
    ) -> Result<(HashMap<String, GGUFMetadataValue>, HashMap<String, Tensor>), ZyxError> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)?;
        let mut magic = [0; 4];
        f.read_exact(&mut magic)?;
        if magic != [b'G', b'G', b'U', b'F'] {
            if magic == [b'F', b'U', b'G', b'G'] {
                return Err(ZyxError::parse_error("GGUF data seems to be stored in big endian order. Only little endian is supported for GGUF in zyx.".into()));
            }
            return Err(ZyxError::parse_error(
                format!("Unknown GGUF magic: {magic:?}. Please check your file.").into(),
            ));
        }
        let mut version = [0; 4];
        f.read_exact(&mut version)?;
        //println!("File size is {} bytes", f.metadata()?.len());
        let mut tensor_count = [0u8; 8];
        f.read_exact(&mut tensor_count)?;
        let tensor_count = u64::from_le_bytes(tensor_count);
        let mut metadata_kv_count = [0u8; 8];
        f.read_exact(&mut metadata_kv_count)?;
        let metadata_kv_count = usize::try_from(u64::from_le_bytes(metadata_kv_count))
            .map_err(|e| ZyxError::parse_error(format!("Failed to parse tensor count in GGUF file. {e}").into()))?;

        let mut metadata = HashMap::new();
        for _ in 0..metadata_kv_count {
            // First string key, (len u64, chars),
            let mut metadata_key_len = [0; 8];
            f.read_exact(&mut metadata_key_len)?;
            let metadata_key_len = u64::from_le_bytes(metadata_key_len);
            let mut metadata_key = String::with_capacity(usize::try_from(metadata_key_len).unwrap());
            f.read_exact(unsafe { metadata_key.as_bytes_mut() })?;

            // Then metadata value type.
            // Then we the value itself.
            let mut metadata_value_type = [0; 1];
            f.read_exact(&mut metadata_value_type)?;
            let metadata_value_type = u8::from_le_bytes(metadata_value_type);
            let metadata_value = match metadata_value_type {
                // uint8
                0 => {
                    let mut buf = [0; 1];
                    f.read_exact(&mut buf)?;
                    let v = u8::from_le_bytes(buf);
                    GGUFMetadataValue::Uint8(v)
                }
                // int8
                1 => {
                    let mut buf = [0; 1];
                    f.read_exact(&mut buf)?;
                    let v = i8::from_le_bytes(buf);
                    GGUFMetadataValue::Int8(v)
                }
                x => todo!("{x}"),
            };
            metadata.insert(metadata_key, metadata_value);
        }

        // First we read the whole description of tensors
        let mut tensor_header = Map::default();
        for _ in 0..tensor_count {
            // name
            let mut tensor_name_len = [0; 8];
            f.read_exact(&mut tensor_name_len)?;
            let tensor_name_len = u64::from_le_bytes(tensor_name_len);
            let mut tensor_name = String::with_capacity(usize::try_from(tensor_name_len).unwrap());
            f.read_exact(unsafe { tensor_name.as_bytes_mut() })?;

            // rank (number of dimensions)
            let mut rank = [0; 4];
            f.read_exact(&mut rank)?;
            let rank = u32::from_le_bytes(rank);

            // shape (NOTE there is no explicit check for endiannes here)
            let mut shape = vec![0; rank as usize * 8];
            f.read_exact(shape.as_mut_slice())?;
            let shape: Vec<Dim> = shape
                .chunks_exact(8)
                .map(|x| usize::try_from(u64::from_le_bytes([x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]])).unwrap())
                .collect();

            // dtype
            let mut dtype = [0; 4];
            f.read_exact(&mut dtype)?;
            let dtype = u32::from_le_bytes(dtype);
            let dtype = match dtype {
                0 => DType::F32,
                1 => DType::F16,
                24 => DType::I8,
                25 => DType::I16,
                26 => DType::I32,
                27 => DType::I64,
                28 => DType::F64,
                x => todo!("GGUF dtype {x} is not supported by zyx yet."),
            };

            // offset (position in file)
            let mut offset = [0; 8];
            f.read_exact(&mut offset)?;
            let offset = u64::from_le_bytes(offset);

            tensor_header.insert(tensor_name, (shape, dtype, offset));
        }

        let mut progress_bar = if RT.lock().debug.dev() {
            println!("Loading tensors from safetensors file");
            let bar = crate::prog_bar::ProgressBar::new(tensor_count);
            Some(bar)
        } else {
            None
        };

        let mut tensors = HashMap::new();
        for (name, (shape, dtype, offset)) in tensor_header {
            if let Some(progress_bar) = &mut progress_bar {
                progress_bar.inc(1, &format!("{name}, {shape:?}, {dtype}"));
            }
            tensors.insert(name, Tensor::from_path(shape, dtype, &path, offset)?);
        }
        Ok((metadata, tensors))
    }

    /// Load safetensors module from path
    ///
    /// # Errors
    /// Errors if path does not exist or IO failed for other reasons.
    pub fn load_safetensors(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>, ZyxError> {
        use std::io::Read;
        let mut f = std::fs::File::open(&path)?;
        //println!("File size is {} bytes", f.metadata()?.len());
        let mut header_len = [0u8; 8];
        f.read_exact(&mut header_len)?;
        let n = usize::try_from(u64::from_le_bytes(header_len)).map_err(|e| {
            ZyxError::parse_error(format!("Failed to parse header len in safetensors file. {e}").into())
        })?;
        let mut header = vec![0u8; n];
        f.read_exact(&mut header)?;
        let header =
            core::str::from_utf8(&header).map_err(|err| std::io::Error::new(std::io::ErrorKind::InvalidData, err))?;
        let mut text = String::with_capacity(10);
        let mut begin_str = false;
        let mut i = 0;
        let mut tensors = HashMap::default();
        let mut dtype = DType::F32;
        let mut shape = vec![1];
        let mut label = String::new();
        let mut metadata = true;
        let mut progress_bar = if RT.lock().debug.dev() {
            println!("Loading tensors from safetensors file");
            let bar = crate::prog_bar::ProgressBar::new(
                u64::try_from(header.chars().filter(|&c| c == '[').count()).unwrap() / 2,
            );
            Some(bar)
        } else {
            None
        };
        //let mmap = Arc::new(unsafe { memmap2::Mmap::map(&f)? });
        //let mut mptr = mmap.as_ptr();
        //mptr = mptr.wrapping_add(8 + header.len());
        let mut offset = (8 + header.len()) as u64;
        for x in header.chars() {
            // We skip metadata for now
            if metadata && text.starts_with("__metadata__") {
                if x == '}' {
                    text.clear();
                    begin_str = false;
                    metadata = false;
                }
                continue;
            }
            if ['"', '[', ']'].contains(&x) {
                if begin_str {
                    //std::println!("{text}");
                    if i % 7 == 0 {
                        #[allow(clippy::assigning_clones)]
                        {
                            label = text.clone();
                        }
                    } else if i % 7 == 2 {
                        dtype = DType::from_safetensors(&text)?;
                    } else if i % 7 == 4 {
                        shape = text
                            .split(',')
                            .map(|d| {
                                d.parse::<usize>().map_err(|err| {
                                    ZyxError::parse_error(format!("Cannot parse safetensors shape: {err}").into())
                                })
                            })
                            .collect::<Result<_, ZyxError>>()?;
                    } else if i % 7 == 6 {
                        // TODO assert offsets
                        //println!("Offsets: {text}");
                        let offsets = text
                            .split(',')
                            .map(|offset| {
                                offset.parse::<usize>().map_err(|err| {
                                    ZyxError::parse_error(format!("Could not parse safetensors offset: {err}").into())
                                })
                            })
                            .collect::<Result<Vec<_>, ZyxError>>()?;
                        //println!("Offsets: {offsets:?}");
                        let bytes = shape.iter().product::<Dim>() * dtype.byte_size() as Dim;
                        if offsets[1] - offsets[0] != bytes {
                            return Err(ZyxError::parse_error(
                                "Safetensors shapes and offsets are incorrect.".into(),
                            ));
                        }
                        if let Some(bar) = &mut progress_bar {
                            bar.inc(1, &format!("{label}, {shape:?}, {dtype:?}"));
                        }
                        let tensor = Tensor::from_path(shape.clone(), dtype, &path, offset)?;
                        offset += bytes as u64;
                        tensors.insert(label.clone(), tensor);
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
        Ok(tensors)
    }
}
