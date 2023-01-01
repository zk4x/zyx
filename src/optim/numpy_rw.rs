// Loading from and saving into numpy
// std is necessary for this
extern crate std;

#[test]
fn numpy_rw() -> Result<(), std::io::Error> {
    use crate::prelude::*;
    use crate::shape::Sh3;
    use crate::device::cpu::{self, Buffer};

    static TEST_FILE: &'static str = "numpy_rw_test.npy";

    let device = cpu::Device::default();

    let x: Buffer<'_, Sh3<3, 2, 4>> = device.uniform(-5., 5.);
    let data = x.to_vec();

    let mut x = x.with_grad();

    x.save_npz(TEST_FILE).unwrap();

    let x: Buffer<'_, Sh3<3, 2, 4>> = device.zeros();
    let mut x = x.with_grad();

    x.load_npz(TEST_FILE).unwrap();

    assert_eq!(x.to_vec(), data);

    // delete uneeded file
    std::fs::remove_file(TEST_FILE)?;

    Ok(())
}

static NUMPY_FILE_TYPE: &[u8] = b"\x93NUMPY";
static NUMPY_VERSION: &[u8] = &[1, 0];

/// Load and save files into numpy format
pub trait NumpyRW<P>
where
    P: AsRef<std::path::Path>,
{
    /// Save parameters into npz zip compressed numpy format
    fn save_npz(self, path: P) -> std::io::Result<()>;
    /// Load parameters from npz zip compressed numpy format
    fn load_npz(self, path: P) -> Result<(), NpyError>;
}

use crate::{tensor::Variable, ops::{HasShape, HasDType, FillWithSlice, IntoVec}, shape::Shape};
use std::{fs::File, io::{BufReader, Read, Write, BufWriter}, vec::Vec, string::String};

impl<S, P> NumpyRW<P> for &mut Variable<S>
where
    P: AsRef<std::path::Path>,
    S: HasShape + HasDType + FillWithSlice + IntoVec,
    S::T: NumpyDType,
{
    fn save_npz(self, path: P) -> std::io::Result<()> {
        let vec = self.data.to_vec();
        save_npy::<P, S::Sh, S::T>(path, vec)?;
        Ok(())
    }

    fn load_npz(self, path: P) -> Result<(), NpyError> {
        // TODO we need to make this work with npz files, not only npy
        let buf = load_npy::<P, S::Sh, S::T>(path)?;
        self.data.fill_with_slice(&buf);
        Ok(())
    }
}

#[derive(Debug)]
pub enum NpyError {
    /// Unrecognized file type.
    FileType([u8; 6]),
    // Unsupported numpy version.
    Version([u8; 2]),
    /// Unable to read file due to io error.
    StdIo(std::io::Error),
    /// Unable to convert numpy file header to utf-8 encoded [String].
    Utf8(std::string::FromUtf8Error),
    Parsing {
        expected: Vec<u8>,
        found: Vec<u8>,
        expected_str: String,
        found_str: String,
    },
    /// Incorrect alignment for reading files with given [Endianness](Endianness)
    Alignment,
}

impl std::fmt::Display for NpyError {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NpyError::FileType(num) => write!(fmt, "Could not determine the file type: {:?}", num),
            NpyError::Version(ver) => write!(fmt, "Incorrect numpy version: {:?}", ver),
            NpyError::StdIo(err) => write!(fmt, "{}", err),
            NpyError::Utf8(err) => write!(fmt, "{}", err),
            NpyError::Parsing {
                expected: _,
                found: _,
                expected_str,
                found_str,
            } => write!(
                fmt,
                "Could not parse the file, expected {} found {}",
                expected_str, found_str
            ),
            NpyError::Alignment => write!(fmt, "Incorrect endian alignment"),
        }
    }
}

impl std::error::Error for NpyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpyError::StdIo(err) => Some(err),
            NpyError::Utf8(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        Self::StdIo(e)
    }
}

impl From<std::string::FromUtf8Error> for NpyError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::Utf8(e)
    }
}

/// NumpyDType
///
/// Implemented for each numpy type.
/// Gives us each of this types represented as numpy string and number of bytes they take.
/// 
/// Enables us to read and write them with given endianness.
pub trait NumpyDType: Sized {
    const NUMPY_DTYPE_STR: &'static str;
    fn read_be<R: Read>(reader: &mut R) -> std::io::Result<Self>;
    fn read_le<R: Read>(reader: &mut R) -> std::io::Result<Self>;
    fn read_ne<R: Read>(reader: &mut R) -> std::io::Result<Self>;
    fn write_be<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
    fn write_le<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
    fn write_ne<W: Write>(&self, writer: &mut W) -> std::io::Result<()>;
}

impl NumpyDType for f32 {
    const NUMPY_DTYPE_STR: &'static str = "f4";
    fn read_be<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        Ok(Self::from_be_bytes(bytes))
    }

    fn read_le<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        Ok(Self::from_le_bytes(bytes))
    }

    fn read_ne<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        Ok(Self::from_ne_bytes(bytes))
    }

    fn write_be<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_be_bytes())
    }

    fn write_le<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_le_bytes())
    }

    fn write_ne<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&self.to_ne_bytes())
    }
}

fn save_npy<P, Sh, T>(path: P, vec: Vec<T>) -> Result<(), std::io::Error>
where
    P: AsRef<std::path::Path>,
    Sh: Shape,
    T: NumpyDType,
{
    let mut writer = BufWriter::new(File::create(path)?);

    // write header
    use std::string::ToString;
    let mut shape_as_string: String = Sh::array().into_iter().map(|d| d.to_string()).collect::<Vec<String>>().join(", ");
    // numpy shapes end with , if they have only one dimension.
    if Sh::RANK == 1 { 
        shape_as_string += ",";
    };
    let mut header: Vec<u8> = Vec::new();
    write!(
        &mut header,
        "{{'descr': '<{}', 'fortran_order': False, 'shape': ({}), }}",
        T::NUMPY_DTYPE_STR,
        shape_as_string,
    )?;

    // padding
    while (header.len() + 1) % 64 != 0 {
        header.write_all(b"\x20")?;
    }

    // new line termination
    header.write_all(b"\n")?;

    // header length
    assert!(header.len() < u16::MAX as usize);
    assert!(header.len() % 64 == 0);

    writer.write_all(NUMPY_FILE_TYPE)?;
    writer.write_all(NUMPY_VERSION)?;
    writer.write_all(&(header.len() as u16).to_le_bytes())?;
    writer.write_all(&header)?;

    for v in vec.iter() {
        v.write_le(&mut writer)?;
    }
    Ok(())
}

fn load_npy<P, Sh, T>(path: P) -> Result<Vec<T>, NpyError>
where
    P: AsRef<std::path::Path>,
    Sh: Shape,
    T: NumpyDType,
{
    // read the file using buffered reader
    let mut buf_reader = BufReader::new(File::open(path)?);

    fn check_input(buf: &[u8], i: usize, chars: &[u8]) -> Result<usize, NpyError> {
        for (offset, &c) in chars.iter().enumerate() {
            if buf[i + offset] != c {
                let expected = chars.to_vec();
                let found = buf[i..i + offset + 1].to_vec();
                let expected_str = String::from_utf8(expected.clone())?;
                let found_str = String::from_utf8(found.clone())?;
                return Err(NpyError::Parsing {
                    expected,
                    found,
                    expected_str,
                    found_str,
                });
            }
        }
        Ok(i + chars.len())
    }

    // get the type of the file, this is static string
    // marking the file as numpy file
    let mut file_type = [0; 6];
    buf_reader.read_exact(&mut file_type)?;
    if file_type != NUMPY_FILE_TYPE {
        return Err(NpyError::FileType(file_type));
    }
    // check if it is supported version
    let mut version = [0; 2];
    buf_reader.read_exact(&mut version)?;
    if version != NUMPY_VERSION {
        return Err(NpyError::Version(version));
    }
    // get length of the header
    let mut header_len_bytes = [0; 2];
    buf_reader.read_exact(&mut header_len_bytes)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    // read the whole header
    let mut header: Vec<u8> = std::vec![0; header_len as usize];
    buf_reader.read_exact(&mut header)?;
    // i is iterator over the numpy file header
    let mut i = check_input(&header, 0, b"{'descr': '")?;
    // get endianness of the file
    let endian = header[i];
    i += 1;
    // get dtype
    let i = check_input(&header, i, T::NUMPY_DTYPE_STR.as_bytes())?;
    let i = check_input(&header, i, b"', ")?;
    // check if fortran order is False (we support only row-major tensors)
    let i = check_input(&header, i, b"'fortran_order': False, ")?;
    // check if shape is correct
    let i = check_input(&header, i, b"'shape': (")?;
    // check if shape is correct
    use std::string::ToString;
    let mut shape_as_string: String = Sh::array().into_iter().map(|d| d.to_string()).collect::<Vec<String>>().join(", ");
    // numpy shapes end with , if they have only one dimension.
    if Sh::RANK == 1 { 
        shape_as_string += ",";
    };
    let i = check_input(&header, i, shape_as_string.as_bytes())?;
    // end of header
    check_input(&header, i, b"), }")?;

    // rest of the file is actual data with given endianness,
    // so load it into a buffer
    let mut buf = Vec::with_capacity(Sh::NUMEL);
    match endian {
        b'>' => 
            for _ in 0..Sh::NUMEL {
                buf.push(T::read_be(&mut buf_reader)?);
            },
        b'<' =>
            for _ in 0..Sh::NUMEL {
                buf.push(T::read_le(&mut buf_reader)?);
            },
        b'=' =>
            for _ in 0..Sh::NUMEL {
                buf.push(T::read_ne(&mut buf_reader)?);
            },
        _ => return Err(NpyError::Alignment),
    };

    // and return the buffer
    Ok(buf)
}
