//! `DType` and constant

use half::{bf16, f16};
use std::fmt::{Debug, Display};

use crate::{Scalar, ZyxError};

/// Represents the data type used for operations.
#[cfg_attr(feature = "py", pyo3::pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 16 bit bfloat data type.
    BF16,
    /// 16 bit float data type.
    F16,
    /// 32 bit float data type.
    F32,
    /// 64 bit float data type.
    F64,
    /// 8 bit unsigned integer data type.
    U8,
    /// 16 bit unsigned integer data type.
    U16,
    /// 32 bit unsigned integer data type.
    U32,
    /// 64 bit unsigned integer data type.
    U64,
    /// 8 bit signed integer data type.
    I8,
    /// 16 bit signed integer data type.
    I16,
    /// 32 bit signed integer data type.
    I32,
    /// 64 bit signed integer data type.
    I64,
    /// 8 bit boolean data type.
    Bool,
}

impl Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::U64 => "U64",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Bool => "Bool",
        })
    }
}

impl DType {
    /// Is this dtype floating point?
    #[must_use]
    pub const fn is_float(self) -> bool {
        match self {
            Self::BF16 | Self::F16 | Self::F32 | Self::F64 => true,
            Self::U8
            | Self::U16
            | Self::U32
            | Self::U64
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64
            | Self::Bool => false,
        }
    }

    #[must_use]
    pub(super) const fn is_shiftable(self) -> bool {
        match self {
            Self::BF16
            | Self::F16
            | Self::F32
            | Self::F64
            | DType::Bool
            | Self::I8
            | Self::I16
            | Self::I32
            | Self::I64 => false,
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => true,
        }
    }

    // TODO remove this in favor of bit_size, since we need to support quantized dtypes
    /// Get the size of this dtype in bytes
    #[must_use]
    pub const fn byte_size(&self) -> u8 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::BF16 | Self::F16 | Self::I16 | Self::U16 => 2,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }

    /// Get the size of this dtype in bits
    #[must_use]
    pub const fn bit_size(&self) -> u8 {
        match self {
            Self::U8 | Self::I8 | Self::Bool => 8,
            Self::BF16 | Self::F16 | Self::I16 | Self::U16 => 16,
            Self::F32 | Self::I32 | Self::U32 => 32,
            Self::F64 | Self::I64 | Self::U64 => 64,
        }
    }

    #[must_use]
    pub(super) const fn zero_constant(self) -> Constant {
        match self {
            Self::BF16 => Constant::BF16(bf16::ZERO.to_bits()),
            Self::F16 => Constant::F16(f16::ZERO.to_bits()),
            Self::F32 => Constant::F32(0f32.to_bits()),
            Self::F64 => Constant::F64(0f64.to_bits()),
            Self::U8 => Constant::U8(0),
            Self::U16 => Constant::U16(0),
            Self::U32 => Constant::U32(0),
            Self::I8 => Constant::I8(0),
            Self::I16 => Constant::I16(0),
            Self::I32 => Constant::I32(0),
            Self::I64 => Constant::I64(0),
            Self::U64 => Constant::U64(0),
            Self::Bool => Constant::Bool(false),
        }
    }

    #[must_use]
    pub(super) const fn one_constant(self) -> Constant {
        match self {
            Self::BF16 => Constant::BF16(bf16::ONE.to_bits()),
            Self::F16 => Constant::F16(f16::ONE.to_bits()),
            Self::F32 => Constant::F32(1f32.to_bits()),
            Self::F64 => Constant::F64(1f64.to_bits()),
            Self::U8 => Constant::U8(1),
            Self::U16 => Constant::U16(1),
            Self::U32 => Constant::U32(1),
            Self::I8 => Constant::I8(1),
            Self::I16 => Constant::I16(1),
            Self::I32 => Constant::I32(1),
            Self::I64 => Constant::I64(1),
            Self::U64 => Constant::U64(1),
            Self::Bool => Constant::Bool(true),
        }
    }

    #[must_use]
    pub(super) const fn min_constant(self) -> Constant {
        match self {
            Self::BF16 => Constant::BF16(bf16::MIN.to_bits()),
            Self::F16 => Constant::F16(f16::MIN.to_bits()),
            Self::F32 => Constant::F32(f32::MIN.to_bits()),
            Self::F64 => Constant::F64(f64::MIN.to_bits()),
            Self::U8 => Constant::U8(u8::MIN),
            Self::U16 => Constant::U16(u16::MIN),
            Self::U32 => Constant::U32(u32::MIN),
            Self::I8 => Constant::I8(i8::MIN),
            Self::I16 => Constant::I16(i16::MIN),
            Self::I32 => Constant::I32(i32::MIN),
            Self::I64 => Constant::I64(i64::MIN),
            Self::U64 => Constant::U64(u64::MIN),
            Self::Bool => Constant::Bool(false),
        }
    }

    #[must_use]
    pub(super) const fn safetensors(&self) -> &str {
        match self {
            Self::BF16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::U8 => "U8",
            Self::U16 => "U16",
            Self::U32 => "U32",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::U64 => "U64",
            Self::Bool => "BOOL",
        }
    }

    pub(super) fn from_safetensors(text: &str) -> Result<Self, ZyxError> {
        Ok(match text {
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "U8" => Self::U8,
            "U32" => Self::U32,
            "I8" => Self::I8,
            "I16" => Self::I16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "U64" => Self::U64,
            "BOOL" => Self::Bool,
            _ => {
                return Err(ZyxError::ParseError(format!(
                    "Could not parse dtype {text}"
                )))
            }
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constant {
    BF16(u16),
    F16(u16),
    F32(u32),
    F64(u64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Constant {
    pub(crate) fn new<T: Scalar>(x: T) -> Self {
        use core::mem::transmute_copy as t;
        match T::dtype() {
            DType::BF16 => Self::BF16(unsafe { t(&x) }),
            DType::F16 => Self::F16(unsafe { t(&x) }),
            DType::F32 => Self::F32(unsafe { t(&x) }),
            DType::F64 => Self::F64(unsafe { t(&x) }),
            DType::U8 => Self::U8(unsafe { t(&x) }),
            DType::U16 => Self::U16(unsafe { t(&x) }),
            DType::U32 => Self::U32(unsafe { t(&x) }),
            DType::U64 => Self::U64(unsafe { t(&x) }),
            DType::I8 => Self::I8(unsafe { t(&x) }),
            DType::I16 => Self::I16(unsafe { t(&x) }),
            DType::I32 => Self::I32(unsafe { t(&x) }),
            DType::I64 => Self::I64(unsafe { t(&x) }),
            DType::Bool => Self::Bool(unsafe { t(&x) }),
        }
    }

    pub(crate) fn from_bytes(bytes: &[u8], dtype: DType) -> Self {
        match dtype {
            DType::BF16 => Self::BF16(bf16::from_ne_bytes([bytes[0], bytes[1]]).to_bits()),
            DType::F16 => Self::F16(f16::from_ne_bytes([bytes[0], bytes[1]]).to_bits()),
            DType::F32 => Self::F32(f32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]).to_bits()),
            DType::F64 => Self::F64(f64::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]]).to_bits()),
            DType::U8 => Self::U8(u8::from_ne_bytes([bytes[0]])),
            DType::U16 => Self::U16(u16::from_ne_bytes([bytes[0], bytes[1]])),
            DType::U32 => Self::U32(u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            DType::U64 => Self::U64(u64::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])),
            DType::I8 => Self::I8(i8::from_ne_bytes([bytes[0]])),
            DType::I16 => Self::I16(i16::from_ne_bytes([bytes[0], bytes[1]])),
            DType::I32 => Self::I32(i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            DType::I64 => Self::I64(i64::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]])),
            DType::Bool => Self::Bool(bytes[0] != 0),
        }
    }

    pub(crate) const fn dtype(&self) -> DType {
        match self {
            Self::BF16(_) => DType::BF16,
            Self::F16(_) => DType::F16,
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::U8(_) => DType::U8,
            Self::U16(_) => DType::U16,
            Self::U32(_) => DType::U32,
            Self::U64(_) => DType::U64,
            Self::I8(_) => DType::I8,
            Self::I16(_) => DType::I16,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
        }
    }

    pub(crate) fn is_zero(&self) -> bool {
        match *self {
            Constant::BF16(x) => bf16::from_bits(x) == bf16::ZERO,
            Constant::F16(x) => f16::from_bits(x) == f16::ZERO,
            Constant::F32(x) => f32::from_bits(x) == 0f32,
            Constant::F64(x) => f64::from_bits(x) == 0f64,
            Constant::U8(x) => x == 0,
            Constant::U16(x) => x == 0,
            Constant::U32(x) => x == 0,
            Constant::U64(x) => x == 0,
            Constant::I8(x) => x == 0,
            Constant::I16(x) => x == 0,
            Constant::I32(x) => x == 0,
            Constant::I64(x) => x == 0,
            Constant::Bool(x) => !x,
        }
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn is_one(&self) -> bool {
        match *self {
            Constant::BF16(x) => bf16::from_bits(x) == bf16::ONE,
            Constant::F16(x) => f16::from_bits(x) == f16::ONE,
            Constant::F32(x) => f32::from_bits(x) == 1f32,
            Constant::F64(x) => f64::from_bits(x) == 1f64,
            Constant::U8(x) => x == 1,
            Constant::U16(x) => x == 1,
            Constant::U32(x) => x == 1,
            Constant::U64(x) => x == 1,
            Constant::I8(x) => x == 1,
            Constant::I16(x) => x == 1,
            Constant::I32(x) => x == 1,
            Constant::I64(x) => x == 1,
            Constant::Bool(x) => x,
        }
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn is_two(&self) -> bool {
        match *self {
            Constant::BF16(x) => bf16::from_bits(x) == bf16::ONE + bf16::ONE,
            Constant::F16(x) => f16::from_bits(x) == f16::ONE + f16::ONE,
            Constant::F32(x) => f32::from_bits(x) == 2f32,
            Constant::F64(x) => f64::from_bits(x) == 2f64,
            Constant::U8(x) => x == 2,
            Constant::U16(x) => x == 2,
            Constant::U32(x) => x == 2,
            Constant::U64(x) => x == 2,
            Constant::I8(x) => x == 2,
            Constant::I16(x) => x == 2,
            Constant::I32(x) => x == 2,
            Constant::I64(x) => x == 2,
            Constant::Bool(_) => false,
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BF16(value) => f.write_fmt(format_args!("{}", bf16::from_bits(*value))),
            Self::F16(value) => f.write_fmt(format_args!("{}", f16::from_bits(*value))),
            Self::F32(value) => f.write_fmt(format_args!("{}", f32::from_bits(*value))),
            Self::F64(value) => f.write_fmt(format_args!("{}", f64::from_bits(*value))),
            Self::U8(value) => f.write_fmt(format_args!("{value}")),
            Self::U16(value) => f.write_fmt(format_args!("{value}")),
            Self::U64(value) => f.write_fmt(format_args!("{value}")),
            Self::U32(value) => f.write_fmt(format_args!("{value}")),
            Self::I8(value) => f.write_fmt(format_args!("{value}")),
            Self::I16(value) => f.write_fmt(format_args!("{value}")),
            Self::I32(value) => f.write_fmt(format_args!("{value}")),
            Self::I64(value) => f.write_fmt(format_args!("{value}")),
            Self::Bool(value) => f.write_fmt(format_args!("{value}")),
        }
    }
}

impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}
