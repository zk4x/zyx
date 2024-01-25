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
}

impl Display for ZyxError {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ZyxError::BackendError(err) => f.write_fmt(format_args!("{err:?}")),
            ZyxError::CompileError(err) => f.write_fmt(format_args!(
                "Compiled backend could not compile this program:\n{err:?}"
            )),
            ZyxError::InvalidDType { expected, found } => f.write_fmt(format_args!(
                "InvalidDType: Expected {expected:?} but found {found:?}."
            )),
            ZyxError::IndexOutOfBounds { index, len } => f.write_fmt(format_args!(
                "Range out of bounds: The index is {index}, but the len is {len}"
            )),
        }
    }
}
