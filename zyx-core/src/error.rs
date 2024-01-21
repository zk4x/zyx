use core::fmt::{Debug, Display, Formatter};
use crate::dtype::DType;

/// ZyxError
#[derive(Debug)]
pub enum ZyxError<E> {
    /// Error returned by backend
    BackendError(E),
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
    }
}

impl<E: Debug> Display for ZyxError<E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            ZyxError::BackendError(err) => f.write_fmt(format_args!("{err:?}")),
            ZyxError::InvalidDType { expected, found } => f.write_fmt(format_args!("InvalidDType: Expected {expected:?} but found {found:?}.")),
            ZyxError::IndexOutOfBounds { index, len } => f.write_fmt(format_args!("Range out of bounds: The index is {index}, but the len is {len}")),
        }
    }
}
