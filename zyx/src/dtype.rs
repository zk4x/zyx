//! `DType` and constant

use crate::{
    Scalar, ZyxError,
    graph::{BOp, UOp},
};
use half::{bf16, f16};
use std::fmt::{Debug, Display};

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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Constant {
    BF16([u8; 2]), // le bytes
    F16([u8; 2]),  // le bytes
    F32([u8; 4]),  // le bytes
    F64([u8; 8]),  // le bytes
    U8(u8),
    U16(u16),
    U32(u32),
    U64([u8; 8]), // le bytes
    I8(i8),
    I16(i16),
    I32(i32),
    I64([u8; 8]), // le bytes
    Bool(bool),
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

    /*#[must_use]
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
    }*/

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
            Self::BF16 => Constant::BF16(bf16::ZERO.to_le_bytes()),
            Self::F16 => Constant::F16(f16::ZERO.to_le_bytes()),
            Self::F32 => Constant::F32(0f32.to_le_bytes()),
            Self::F64 => Constant::F64(0f64.to_le_bytes()),
            Self::U8 => Constant::U8(0),
            Self::U16 => Constant::U16(0),
            Self::U32 => Constant::U32(0),
            Self::I8 => Constant::I8(0),
            Self::I16 => Constant::I16(0),
            Self::I32 => Constant::I32(0),
            Self::I64 => Constant::I64(0i64.to_le_bytes()),
            Self::U64 => Constant::U64(0u64.to_le_bytes()),
            Self::Bool => Constant::Bool(false),
        }
    }

    #[must_use]
    pub(super) const fn one_constant(self) -> Constant {
        match self {
            Self::BF16 => Constant::BF16(bf16::ONE.to_le_bytes()),
            Self::F16 => Constant::F16(f16::ONE.to_le_bytes()),
            Self::F32 => Constant::F32(1f32.to_le_bytes()),
            Self::F64 => Constant::F64(1f64.to_le_bytes()),
            Self::U8 => Constant::U8(1),
            Self::U16 => Constant::U16(1),
            Self::U32 => Constant::U32(1),
            Self::I8 => Constant::I8(1),
            Self::I16 => Constant::I16(1),
            Self::I32 => Constant::I32(1),
            Self::I64 => Constant::I64(1i64.to_le_bytes()),
            Self::U64 => Constant::U64(1u64.to_le_bytes()),
            Self::Bool => Constant::Bool(true),
        }
    }

    #[must_use]
    pub(super) const fn min_constant(self) -> Constant {
        match self {
            Self::BF16 => Constant::BF16(bf16::MIN.to_le_bytes()),
            Self::F16 => Constant::F16(f16::MIN.to_le_bytes()),
            Self::F32 => Constant::F32(f32::MIN.to_le_bytes()),
            Self::F64 => Constant::F64(f64::MIN.to_le_bytes()),
            Self::U8 => Constant::U8(u8::MIN),
            Self::U16 => Constant::U16(u16::MIN),
            Self::U32 => Constant::U32(u32::MIN),
            Self::I8 => Constant::I8(i8::MIN),
            Self::I16 => Constant::I16(i16::MIN),
            Self::I32 => Constant::I32(i32::MIN),
            Self::I64 => Constant::I64(i64::MIN.to_le_bytes()),
            Self::U64 => Constant::U64(u64::MIN.to_le_bytes()),
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
                return Err(ZyxError::ParseError(
                    format!("Could not parse dtype {text}").into(),
                ));
            }
        })
    }
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

    pub(crate) fn from_le_bytes(bytes: &[u8], dtype: DType) -> Self {
        match dtype {
            DType::BF16 => Self::BF16([bytes[0], bytes[1]]),
            DType::F16 => Self::F16([bytes[0], bytes[1]]),
            DType::F32 => Self::F32([bytes[0], bytes[1], bytes[2], bytes[3]]),
            DType::F64 => Self::F64([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            DType::U8 => Self::U8(u8::from_le_bytes([bytes[0]])),
            DType::U16 => Self::U16(u16::from_le_bytes([bytes[0], bytes[1]])),
            DType::U32 => Self::U32(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            DType::U64 => Self::U64([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
            DType::I8 => Self::I8(i8::from_le_bytes([bytes[0]])),
            DType::I16 => Self::I16(i16::from_le_bytes([bytes[0], bytes[1]])),
            DType::I32 => Self::I32(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            DType::I64 => Self::I64([
                bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
            ]),
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
            Constant::BF16(x) => bf16::from_le_bytes(x) == bf16::ZERO,
            Constant::F16(x) => f16::from_le_bytes(x) == f16::ZERO,
            Constant::F32(x) => f32::from_le_bytes(x) == 0f32,
            Constant::F64(x) => f64::from_le_bytes(x) == 0f64,
            Constant::U8(x) => x == 0,
            Constant::U16(x) => x == 0,
            Constant::U32(x) => x == 0,
            Constant::U64(x) => u64::from_le_bytes(x) == 0,
            Constant::I8(x) => x == 0,
            Constant::I16(x) => x == 0,
            Constant::I32(x) => x == 0,
            Constant::I64(x) => i64::from_le_bytes(x) == 0,
            Constant::Bool(x) => !x,
        }
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn is_one(&self) -> bool {
        match *self {
            Constant::BF16(x) => bf16::from_le_bytes(x) == bf16::ONE,
            Constant::F16(x) => f16::from_le_bytes(x) == f16::ONE,
            Constant::F32(x) => f32::from_le_bytes(x) == 1f32,
            Constant::F64(x) => f64::from_le_bytes(x) == 1f64,
            Constant::U8(x) => x == 1,
            Constant::U16(x) => x == 1,
            Constant::U32(x) => x == 1,
            Constant::U64(x) => u64::from_le_bytes(x) == 1,
            Constant::I8(x) => x == 1,
            Constant::I16(x) => x == 1,
            Constant::I32(x) => x == 1,
            Constant::I64(x) => i64::from_le_bytes(x) == 1,
            Constant::Bool(x) => x,
        }
    }

    #[allow(clippy::float_cmp)]
    pub(crate) fn is_two(&self) -> bool {
        match *self {
            Constant::BF16(x) => bf16::from_le_bytes(x) == bf16::ONE + bf16::ONE,
            Constant::F16(x) => f16::from_le_bytes(x) == f16::ONE + f16::ONE,
            Constant::F32(x) => f32::from_le_bytes(x) == 2f32,
            Constant::F64(x) => f64::from_le_bytes(x) == 2f64,
            Constant::U8(x) => x == 2,
            Constant::U16(x) => x == 2,
            Constant::U32(x) => x == 2,
            Constant::U64(x) => u64::from_le_bytes(x) == 2,
            Constant::I8(x) => x == 2,
            Constant::I16(x) => x == 2,
            Constant::I32(x) => x == 2,
            Constant::I64(x) => i64::from_le_bytes(x) == 2,
            Constant::Bool(_) => false,
        }
    }

    pub(super) fn cast(self, dtype: DType) -> Constant {
        match self {
            Constant::BF16(x) => half::bf16::from_le_bytes(x).cast_dtype(dtype),
            Constant::F16(x) => half::f16::from_le_bytes(x).cast_dtype(dtype),
            Constant::F32(x) => f32::from_le_bytes(x).cast_dtype(dtype),
            Constant::F64(x) => f64::from_le_bytes(x).cast_dtype(dtype),
            Constant::U8(x) => x.cast_dtype(dtype),
            Constant::I8(x) => x.cast_dtype(dtype),
            Constant::I16(x) => x.cast_dtype(dtype),
            Constant::U16(x) => x.cast_dtype(dtype),
            Constant::U32(x) => x.cast_dtype(dtype),
            Constant::U64(x) => u64::from_le_bytes(x).cast_dtype(dtype),
            Constant::I32(x) => x.cast_dtype(dtype),
            Constant::I64(x) => i64::from_le_bytes(x).cast_dtype(dtype),
            Constant::Bool(x) => x.cast_dtype(dtype),
        }
    }

    pub(super) fn unary(self, uop: UOp) -> Constant {
        use crate::Float;
        fn unary_func<T: Scalar>(x: T, uop: UOp) -> T {
            match uop {
                UOp::Exp2 | UOp::Log2 | UOp::Reciprocal | UOp::Sqrt | UOp::Sin | UOp::Cos => {
                    unreachable!()
                }
                UOp::ReLU => x.relu(),
                UOp::Neg => x.neg(),
                UOp::Not => x.not(),
            }
        }
        fn unary_func_float<T: Float>(x: T, uop: UOp) -> T {
            match uop {
                UOp::ReLU => x.relu(),
                UOp::Neg => x.neg(),
                UOp::Exp2 => x.exp2(),
                UOp::Log2 => x.log2(),
                UOp::Reciprocal => x.reciprocal(),
                UOp::Sqrt => x.sqrt(),
                UOp::Sin => x.sin(),
                UOp::Cos => x.cos(),
                UOp::Not => x.not(),
            }
        }
        match self {
            Constant::BF16(x) => {
                Constant::BF16(unary_func_float(half::bf16::from_le_bytes(x), uop).to_le_bytes())
            }
            Constant::F16(x) => {
                Constant::F16(unary_func_float(half::f16::from_le_bytes(x), uop).to_le_bytes())
            }
            Constant::F32(x) => {
                Constant::F32(unary_func_float(f32::from_le_bytes(x), uop).to_le_bytes())
            }
            Constant::F64(x) => {
                Constant::F64(unary_func_float(f64::from_le_bytes(x), uop).to_le_bytes())
            }
            Constant::U8(x) => Constant::U8(unary_func(x, uop)),
            Constant::U16(x) => Constant::U16(unary_func(x, uop)),
            Constant::U32(x) => Constant::U32(unary_func(x, uop)),
            Constant::U64(x) => Constant::U64(unary_func(u64::from_le_bytes(x), uop).to_le_bytes()),
            Constant::I8(x) => Constant::I8(unary_func(x, uop)),
            Constant::I16(x) => Constant::I16(unary_func(x, uop)),
            Constant::I32(x) => Constant::I32(unary_func(x, uop)),
            Constant::I64(x) => Constant::I64(unary_func(i64::from_le_bytes(x), uop).to_le_bytes()),
            Constant::Bool(x) => Constant::Bool(unary_func(x, uop)),
        }
    }

    // TODO binary constant evaluation
    // Assumes both constants are the same dtype
    pub(super) fn binary(x: Constant, y: Constant, bop: BOp) -> Constant {
        fn binary_func<T: Scalar>(x: T, y: T, bop: BOp) -> Constant {
            match bop {
                BOp::Add => Constant::new(x.add(y)),
                BOp::Sub => Constant::new(x.sub(y)),
                BOp::Mul => Constant::new(x.mul(y)),
                BOp::Div => Constant::new(x.div(y)),
                BOp::Pow => Constant::new(x.pow(y)),
                BOp::Mod => Constant::new(x.mod_(y)),
                BOp::Max => Constant::new(x.max(y)),
                BOp::Cmplt => Constant::new(x.cmplt(y)),
                BOp::Cmpgt => Constant::new(x.cmpgt(y)),
                BOp::Or => Constant::new(x.or(y)),
                BOp::And => Constant::new(x.and(y)),
                BOp::NotEq => Constant::new(x.noteq(y)),
                BOp::BitXor => Constant::new(x.bitxor(y)),
                BOp::BitOr => Constant::new(x.bitor(y)),
                BOp::BitAnd => Constant::new(x.bitand(y)),
                BOp::BitShiftLeft => Constant::new(x.bitshiftleft(y)),
                BOp::BitShiftRight => Constant::new(x.bitshiftright(y)),
            }
        }
        debug_assert_eq!(x.dtype(), y.dtype());
        match x {
            Constant::BF16(x) => {
                let Constant::BF16(y) = y else { unreachable!() };
                binary_func(bf16::from_le_bytes(x), bf16::from_le_bytes(y), bop)
            }
            Constant::F16(x) => {
                let Constant::F16(y) = y else { unreachable!() };
                binary_func(f16::from_le_bytes(x), f16::from_le_bytes(y), bop)
            }
            Constant::F32(x) => {
                let Constant::F32(y) = y else { unreachable!() };
                binary_func(f32::from_le_bytes(x), f32::from_le_bytes(y), bop)
            }
            Constant::F64(x) => {
                let Constant::F64(y) = y else { unreachable!() };
                binary_func(f64::from_le_bytes(x), f64::from_le_bytes(y), bop)
            }
            Constant::U8(x) => {
                let Constant::U8(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U16(x) => {
                let Constant::U16(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U32(x) => {
                let Constant::U32(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::U64(x) => {
                let Constant::U64(y) = y else { unreachable!() };
                binary_func(u64::from_le_bytes(x), u64::from_le_bytes(y), bop)
            }
            Constant::I8(x) => {
                let Constant::I8(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I16(x) => {
                let Constant::I16(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I32(x) => {
                let Constant::I32(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
            Constant::I64(x) => {
                let Constant::I64(y) = y else { unreachable!() };
                binary_func(i64::from_le_bytes(x), i64::from_le_bytes(y), bop)
            }
            Constant::Bool(x) => {
                let Constant::Bool(y) = y else { unreachable!() };
                binary_func(x, y, bop)
            }
        }
    }
}

trait CastDType: Scalar {
    fn cast_dtype(self, dtype: DType) -> Constant {
        match dtype {
            DType::BF16 => Constant::BF16(self.cast::<half::bf16>().to_le_bytes()),
            DType::F16 => Constant::F16(self.cast::<half::f16>().to_le_bytes()),
            DType::F32 => Constant::F32(self.cast::<f32>().to_le_bytes()),
            DType::F64 => Constant::F64(self.cast::<f64>().to_le_bytes()),
            DType::U8 => Constant::U8(self.cast()),
            DType::U16 => Constant::U16(self.cast()),
            DType::U32 => Constant::U32(self.cast()),
            DType::U64 => Constant::U64(self.cast::<u64>().to_le_bytes()),
            DType::I8 => Constant::I8(self.cast()),
            DType::I16 => Constant::I16(self.cast()),
            DType::I32 => Constant::I32(self.cast()),
            DType::I64 => Constant::I64(self.cast::<i64>().to_le_bytes()),
            DType::Bool => Constant::Bool(self.cast()),
        }
    }
}

impl<T: Scalar> CastDType for T {}

impl Display for Constant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::BF16(value) => f.write_fmt(format_args!("{}", bf16::from_le_bytes(*value))),
            Self::F16(value) => f.write_fmt(format_args!("{}", f16::from_le_bytes(*value))),
            Self::F32(value) => f.write_fmt(format_args!("{}", f32::from_le_bytes(*value))),
            Self::F64(value) => f.write_fmt(format_args!("{}", f64::from_le_bytes(*value))),
            Self::U8(value) => f.write_fmt(format_args!("{value}")),
            Self::U16(value) => f.write_fmt(format_args!("{value}")),
            &Self::U64(value) => f.write_fmt(format_args!("{}", u64::from_le_bytes(value))),
            Self::U32(value) => f.write_fmt(format_args!("{value}")),
            Self::I8(value) => f.write_fmt(format_args!("{value}")),
            Self::I16(value) => f.write_fmt(format_args!("{value}")),
            Self::I32(value) => f.write_fmt(format_args!("{value}")),
            &Self::I64(value) => f.write_fmt(format_args!("{}", i64::from_le_bytes(value))),
            Self::Bool(value) => f.write_fmt(format_args!("{value}")),
        }
    }
}

impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{self}"))
    }
}
