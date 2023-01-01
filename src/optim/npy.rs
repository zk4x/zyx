// Loading from and saving into numpy
// std is necessary for this
extern crate std;

#[test]
fn read_file() {
    use crate::prelude::*;
    use crate::shape::Sh1;
    use crate::device::cpu::{self, Buffer};

    let device = cpu::Device::default();

    let x: Buffer<'_, Sh1<3>> = device.zeros();
    let mut x = x.with_grad();

    x.load_npz("file.npy").unwrap();

    std::println!("{}", x);

    panic!()
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
use std::{fs::File, io::{BufReader, Read, Write}, vec::Vec, string::String};

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

fn load_npy<P, Sh, T>(path: P) -> Result<Vec<T>, NpyError>
where
    P: AsRef<std::path::Path>,
    Sh: Shape,
    T: NumpyDType,
{
    // read the file using buffered reader
    let mut buf_reader = BufReader::new(File::open(path)?);

    // get the type of the file, this is static string
    // marking the file as numpy file
    let mut file_type = [0; 6];
    buf_reader.read_exact(&mut file_type)?;
    if file_type != NUMPY_FILE_TYPE {
        return Err(NpyError::FileTypeError(file_type));
    }
    // check if it is supported version
    let mut version = [0; 2];
    buf_reader.read_exact(&mut version)?;
    if version != NUMPY_VERSION {
        return Err(NpyError::VersionError(version));
    }
    // get length of the header
    let mut header_len_bytes = [0; 2];
    buf_reader.read_exact(&mut header_len_bytes)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    // read the whole header
    let mut header: Vec<u8> = std::vec![0; header_len as usize];
    buf_reader.read_exact(&mut header)?;
    // i is iterator over the numpy file header
    let mut i = expect(&header, 0, b"{'descr': '")?;
    // get endianness of the file
    let endian = match header[i] {
        b'>' => Endianness::Big,
        b'<' => Endianness::Little,
        b'=' => Endianness::Native,
        _ => return Err(NpyError::AlignmentError),
    };
    i += 1;
    // get dtype
    let i = expect(&header, i, T::NUMPY_DTYPE_STR.as_bytes())?;
    let i = expect(&header, i, b"', ")?;
    // check if fortran order is False (we support only row-major tensors)
    let i = expect(&header, i, b"'fortran_order': False, ")?;
    // check if shape is correct
    let i = expect(&header, i, b"'shape': (")?;
    use std::string::ToString;
    let mut shape_as_string: String = Sh::array().into_iter().map(|d| d.to_string()).collect::<Vec<String>>().join(", ");
    // numpy shapes end with , if they have only one dimension.
    if Sh::RANK == 1 { 
        shape_as_string += ",";
    };
    let i = expect(&header, i, shape_as_string.as_bytes())?;
    // end of header
    expect(&header, i, b"), }")?;

    // rest of the file is actual data with given endianness,
    // so load it into a buffer
    let mut buf = Vec::with_capacity(Sh::NUMEL);
    for _ in 0..Sh::NUMEL {
        buf.push(T::read_endian(&mut buf_reader, endian)?);
    }

    // and return the buffer
    Ok(buf)
}

fn expect(buf: &[u8], i: usize, chars: &[u8]) -> Result<usize, NpyError> {
    for (offset, &c) in chars.iter().enumerate() {
        if buf[i + offset] != c {
            let expected = chars.to_vec();
            let found = buf[i..i + offset + 1].to_vec();
            let expected_str = String::from_utf8(expected.clone())?;
            let found_str = String::from_utf8(found.clone())?;
            return Err(NpyError::ParsingError {
                expected,
                found,
                expected_str,
                found_str,
            });
        }
    }
    Ok(i + chars.len())
}

#[derive(Debug)]
pub enum NpyError {
    /// Unrecognized file type.
    FileTypeError([u8; 6]),

    // Unsupported numpy version.
    VersionError([u8; 2]),

    /// Unable to read file due to io error.
    StdIoError(std::io::Error),

    /// Unable to convert numpy file header to utf-8 encoded [String].
    Utf8Error(std::string::FromUtf8Error),

    ParsingError {
        expected: Vec<u8>,
        found: Vec<u8>,
        expected_str: String,
        found_str: String,
    },

    /// Incorrect alignment for reading files with given [Endianness](Endianness)
    AlignmentError,
}

impl std::fmt::Display for NpyError {
    fn fmt(&self, fmt: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NpyError::FileTypeError(num) => write!(fmt, "Could not determine the file type: {:?}", num),
            NpyError::VersionError(ver) => write!(fmt, "Incorrect numpy version: {:?}", ver),
            NpyError::StdIoError(err) => write!(fmt, "{}", err),
            NpyError::Utf8Error(err) => write!(fmt, "{}", err),
            NpyError::ParsingError {
                expected: _,
                found: _,
                expected_str,
                found_str,
            } => write!(
                fmt,
                "Could not parse the file, expected {} found {}",
                expected_str, found_str
            ),
            NpyError::AlignmentError => write!(fmt, "Incorrect endian alignment"),
        }
    }
}

impl std::error::Error for NpyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NpyError::StdIoError(err) => Some(err),
            NpyError::Utf8Error(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for NpyError {
    fn from(e: std::io::Error) -> Self {
        Self::StdIoError(e)
    }
}

impl From<std::string::FromUtf8Error> for NpyError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::Utf8Error(e)
    }
}

/// Endianness types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Endianness {
    Big,
    Little,
    Native,
}

/// NumpyDType
///
/// Implemented for each numpy type.
/// Gives us each of this types represented as numpy string and number of bytes they take.
/// 
/// Enables us to read and write them with given endianness.
pub trait NumpyDType: Sized {
    const NUMPY_DTYPE_STR: &'static str;

    fn read_endian<R: Read>(reader: &mut R, endian: Endianness) -> std::io::Result<Self>;

    fn write_endian<W: Write>(&self, writer: &mut W, endian: Endianness) -> std::io::Result<()>;
}

impl NumpyDType for f32 {
    const NUMPY_DTYPE_STR: &'static str = "f4";

    fn read_endian<R: Read>(reader: &mut R, endian: Endianness) -> std::io::Result<Self> {
        let mut bytes = [0; 4];
        reader.read_exact(&mut bytes)?;
        Ok(match endian {
            Endianness::Big => Self::from_be_bytes(bytes),
            Endianness::Little => Self::from_le_bytes(bytes),
            Endianness::Native => Self::from_ne_bytes(bytes),
        })
    }

    fn write_endian<W: Write>(&self, writer: &mut W, endian: Endianness) -> std::io::Result<()> {
        match endian {
            Endianness::Big => writer.write_all(&self.to_be_bytes()),
            Endianness::Little => writer.write_all(&self.to_le_bytes()),
            Endianness::Native => writer.write_all(&self.to_ne_bytes()),
        }
    }
}

impl NumpyDType for f64 {
    const NUMPY_DTYPE_STR: &'static str = "f8";

    fn read_endian<R: Read>(reader: &mut R, endian: Endianness) -> std::io::Result<Self> {
        let mut bytes = [0; 8];
        reader.read_exact(&mut bytes)?;
        Ok(match endian {
            Endianness::Big => Self::from_be_bytes(bytes),
            Endianness::Little => Self::from_le_bytes(bytes),
            Endianness::Native => Self::from_ne_bytes(bytes),
        })
    }

    fn write_endian<W: Write>(&self, writer: &mut W, endian: Endianness) -> std::io::Result<()> {
        match endian {
            Endianness::Big => writer.write_all(&self.to_be_bytes()),
            Endianness::Little => writer.write_all(&self.to_le_bytes()),
            Endianness::Native => writer.write_all(&self.to_ne_bytes()),
        }
    }
}

// Attemps to load the data from a `.npy` file at `path`
    /*pub fn load_from_npy<P: AsRef<Path>>(&mut self, path: P) -> Result<(), NpyError> {
        let mut f = BufReader::new(File::open(path)?);
        self.read_from(&mut f)
    }

    /// Saves the tensor to a `.npy` file located at `path`
    pub fn save_to_npy<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut f = BufWriter::new(File::create(path)?);
        self.write_to(&mut f)
    }

    pub(crate) fn read_from<R: Read>(&mut self, r: &mut R) -> Result<(), NpyError> {
        let endian = read_header::<R, E>(r, self.shape().concrete().into_iter().collect())?;
        let numel = self.shape().num_elements();
        let mut buf = Vec::with_capacity(numel);
        for _ in 0..numel {
            buf.push(E::read_endian(r, endian)?);
        }
        D::copy_from(self, &buf);
        Ok(())
    }

    pub(crate) fn write_to<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let endian = Endianness::Little;
        write_header::<W, E>(w, endian, self.shape().concrete().into_iter().collect())?;
        let numel = self.shape().num_elements();
        let mut buf = std::vec![Default::default(); numel];
        D::copy_into(self, &mut buf);
        for v in buf.iter() {
            v.write_endian(w, endian)?;
        }
        Ok(())
    }

fn write_header<W: Write, E: NumpyDtype>(
    w: &mut W,
    endian: Endianness,
    shape: Vec<usize>,
) -> io::Result<()> {
    let shape_str = to_shape_str(shape);

    let mut header: Vec<u8> = Vec::new();
    write!(
        &mut header,
        "{{'descr': '{}{}', 'fortran_order': False, 'shape': ({}), }}",
        match endian {
            Endianness::Big => '>',
            Endianness::Little => '<',
            Endianness::Native => '=',
        },
        E::NUMPY_DTYPE_STR,
        shape_str,
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

    w.write_all(MAGIC_NUMBER)?; // magic number
    w.write_all(VERSION)?; // version major & minor
    w.write_all(&(header.len() as u16).to_le_bytes())?;
    w.write_all(&header)?;
    Ok(())
}

fn read_header<R: Read, E: NumpyDtype>(r: &mut R, shape: Vec<usize>) -> Result<Endianness, NpyError> {
    let mut magic = [0; 6];
    r.read_exact(&mut magic)?;
    if magic != MAGIC_NUMBER {
        return Err(NpyError::InvalidMagicNumber(magic));
    }

    let mut version = [0; 2];
    r.read_exact(&mut version)?;
    if version != VERSION {
        return Err(NpyError::InvalidVersion(version));
    }

    let mut header_len_bytes = [0; 2];
    r.read_exact(&mut header_len_bytes)?;
    let header_len = u16::from_le_bytes(header_len_bytes);

    let mut header: Vec<u8> = std::vec![0; header_len as usize];
    r.read_exact(&mut header)?;

    let mut i = 0;
    i = expect(&header, i, b"{'descr': '")?;

    let endian = match header[i] {
        b'>' => Endianness::Big,
        b'<' => Endianness::Little,
        b'=' => Endianness::Native,
        _ => return Err(NpyError::InvalidAlignment),
    };
    i += 1;

    i = expect(&header, i, E::NUMPY_DTYPE_STR.as_bytes())?;
    i = expect(&header, i, b"', ")?;

    // fortran order
    i = expect(&header, i, b"'fortran_order': False, ")?;

    // shape
    i = expect(&header, i, b"'shape': (")?;
    let shape_str = to_shape_str(shape);
    i = expect(&header, i, shape_str.as_bytes())?;
    expect(&header, i, b"), }")?;

    Ok(endian)
}*/
