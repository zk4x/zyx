use crate::dtype::DType;
use alloc::boxed::Box;
use core::fmt::{Debug, Display, Formatter};

/// ZyxError
#[derive(Debug)]
pub enum ZyxError {
    /// Error returned by backend
    BackendError(&'static str),
    /// Compilation error
    CompileError(Box<dyn Debug>),
    /// Unexpected dtype found
    InvalidDType {
        /// Expected dtype
        expected: DType,
        /// Found dtype
        found: DType,
    },
    /// Index out of bounds
    IndexOutOfBounds {
        /// Passed index
        index: usize,
        /// Actual length
        len: usize,
    },
    /// IO error when writing to or reading from disk
    #[cfg(feature = "std")]
    IOError(std::io::Error),
    /// Parse error
    ParseError(alloc::string::String),
}

impl Display for ZyxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ZyxError::BackendError(err) => f.write_fmt(format_args!("{err:?}")),
            ZyxError::CompileError(err) => f.write_fmt(format_args!(
                "compiled backend could not compile this program:\n{err:?}"
            )),
            ZyxError::InvalidDType { expected, found } => f.write_fmt(format_args!(
                "invalid dtype: expected {expected:?} but found {found:?}."
            )),
            ZyxError::IndexOutOfBounds { index, len } => f.write_fmt(format_args!(
                "range out of bounds: the index is {index}, but the len is {len}"
            )),
            ZyxError::IOError(err) => f.write_fmt(format_args!("{err}")),
            ZyxError::ParseError(err) => f.write_fmt(format_args!("{err}")),
        }
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for ZyxError {
    fn from(err: std::io::Error) -> Self {
        Self::IOError(err)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ZyxError {}
