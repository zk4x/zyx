//! DType and constant

use core::fmt::Display;
#[cfg(feature = "half")]
use half::{bf16, f16};

use crate::{Scalar, ZyxError};

/// Represents the data type used for operations.
#[cfg_attr(feature = "py", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    /// 16 bit bfloat data type.
    #[cfg(feature = "half")]
    BF16,
    /// 8 bit float data type, 4 bit exponent, 3 bit mantissa.
    F8,
    /// 16 bit float data type.
    #[cfg(feature = "half")]
    F16,
    /// 32 bit float data type.
    F32,
    /// 64 bit float data type.
    F64,
    #[cfg(feature = "complex")]
    /// 32 bit complex float data type.
    CF32,
    #[cfg(feature = "complex")]
    /// 64 bit complex float data type.
    CF64,
    /// 8 bit unsigned integer data type.
    U8,
    /// 32 bit unsigned integer data type.
    U32,
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
        return f.write_str(match self {
            #[cfg(feature = "half")]
            DType::BF16 => "BF16",
            DType::F8 => "F8",
            #[cfg(feature = "half")]
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            #[cfg(feature = "complex")]
            DType::CF32 => "CF32",
            #[cfg(feature = "complex")]
            DType::CF64 => "CF64",
            DType::U8 => "U8",
            DType::U32 => "U32",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::Bool => "Bool",
        });
    }
}

impl DType {
    /// Is this dtype floating point?
    pub fn is_float(&self) -> bool {
        match self {
            DType::F8 | DType::F32 | DType::F64 => true,
            DType::U8 | DType::U32 | DType::I8 | DType::I16 | DType::I32 | DType::I64 | DType::Bool => false,
        }
    }

    /// Get the size of this dtype in bytes
    pub fn byte_size(&self) -> usize {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => 2,
            DType::F8 => 1,
            #[cfg(feature = "half")]
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            #[cfg(feature = "complex")]
            DType::CF32 => 8,
            #[cfg(feature = "complex")]
            DType::CF64 => 16,
            DType::U8 => 1,
            DType::U32 => 1,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        };
    }

    pub(super) fn zero_constant(&self) -> Constant {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => Constant::BF16(unsafe { core::mem::transmute(bf16::ZERO) }),
            DType::F8 => todo!(), //Constant::F8(unsafe { core::mem::transmute(0f32) }),
            #[cfg(feature = "half")]
            DType::F16 => Constant::F16(unsafe { core::mem::transmute(f16::ZERO) }),
            DType::F32 => Constant::F32(unsafe { core::mem::transmute(0f32) }),
            DType::F64 => Constant::F64(unsafe { core::mem::transmute(0f64) }),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => Constant::U8(0),
            DType::U32 => Constant::U32(0),
            DType::I8 => Constant::I8(0),
            DType::I16 => Constant::I16(0),
            DType::I32 => Constant::I32(0),
            DType::I64 => Constant::I64(0),
            DType::Bool => Constant::Bool(false),
        };
    }

    pub(super) fn min_constant(&self) -> Constant {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => Constant::BF16(unsafe { core::mem::transmute(bf16::MIN) }),
            DType::F8 => Constant::F8(255),
            #[cfg(feature = "half")]
            DType::F16 => Constant::F16(unsafe { core::mem::transmute(f16::MIN) }),
            DType::F32 => Constant::F32(unsafe { core::mem::transmute(f32::MIN) }),
            DType::F64 => Constant::F64(unsafe { core::mem::transmute(f64::MIN) }),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => Constant::U8(u8::MIN),
            DType::U32 => Constant::U32(u32::MIN),
            DType::I8 => Constant::I8(i8::MIN),
            DType::I16 => Constant::I16(i16::MIN),
            DType::I32 => Constant::I32(i32::MIN),
            DType::I64 => Constant::I64(i64::MIN),
            DType::Bool => Constant::Bool(false),
        };
    }

    pub(super) fn safetensors(&self) -> &str {
        match self {
            DType::F8 => "F8",
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::U8 => "U8",
            DType::U32 => "U32",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::Bool => "BOOL",
        }
    }

    pub(super) fn from_safetensors(text: &str) -> Result<Self, ZyxError> {
        Ok(match text {
            "F8" => DType::F8,
            "F32" => DType::F32,
            "F64" => DType::F64,
            "U8" => DType::U8,
            "U32" => DType::U32,
            "I8" => DType::I8,
            "I16" => DType::I16,
            "I32" => DType::I32,
            "I64" => DType::I64,
            "BOOL" => DType::Bool,
            _ => {
                return Err(ZyxError::ParseError(format!(
                    "Could not parse dtype {text}"
                )))
            }
        })
    }
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Constant {
    #[cfg(feature = "half")]
    BF16(u16),
    F8(u8),
    #[cfg(feature = "half")]
    F16(u16),
    F32(u32),
    F64(u64),
    #[cfg(feature = "complex")]
    CF32(u32, u32),
    #[cfg(feature = "complex")]
    CF64(u64, u64),
    U8(u8),
    U32(u32),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    Bool(bool),
}

impl Constant {
    pub(crate) fn new<T: Scalar>(x: T) -> Constant {
        use core::mem::transmute_copy as t;
        match T::dtype() {
            #[cfg(feature = "half")]
            DType::BF16 => Constant::BF16(unsafe { t(&x) }),
            DType::F8 => Constant::F8(unsafe { t(&x) }),
            #[cfg(feature = "half")]
            DType::F16 => Constant::F16(unsafe { t(&x) }),
            DType::F32 => Constant::F32(unsafe { t(&x) }),
            DType::F64 => Constant::F64(unsafe { t(&x) }),
            #[cfg(feature = "complex")]
            DType::CF32 => {
                let x: num_complex::Complex<f32> = x.cast();
                unsafe { Constant::CF32(t(&x.re), t(&x.re)) }
            }
            #[cfg(feature = "complex")]
            DType::CF64 => {
                let x: num_complex::Complex<f64> = x.cast();
                unsafe { Constant::CF64(t(&x.re), t(&x.re)) }
            }
            DType::U8 => Constant::U8(unsafe { t(&x) }),
            DType::U32 => Constant::U32(unsafe { t(&x) }),
            DType::I8 => Constant::I8(unsafe { t(&x) }),
            DType::I16 => Constant::I16(unsafe { t(&x) }),
            //DType::U32 => Constant::U32(unsafe {t(&x)}),
            DType::I32 => Constant::I32(unsafe { t(&x) }),
            DType::I64 => Constant::I64(unsafe { t(&x) }),
            DType::Bool => Constant::Bool(unsafe { t(&x) }),
        }
    }

    pub(crate) fn dtype(&self) -> DType {
        match self {
            #[cfg(feature = "half")]
            Constant::BF16(_) => DType::BF16,
            Constant::F8(_) => DType::F8,
            #[cfg(feature = "half")]
            Constant::F16(_) => DType::F16,
            Constant::F32(_) => DType::F32,
            Constant::F64(_) => DType::F64,
            #[cfg(feature = "complex")]
            Constant::CF32(..) => DType::CF32,
            #[cfg(feature = "complex")]
            Constant::CF64(..) => DType::CF64,
            Constant::U8(_) => DType::U8,
            Constant::U32(_) => DType::U32,
            Constant::I8(_) => DType::I8,
            Constant::I16(_) => panic!(),
            Constant::I32(_) => DType::I32,
            Constant::I64(_) => DType::I64,
            Constant::Bool(_) => DType::Bool,
        }
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use core::mem::transmute as t;
        match self {
            #[cfg(feature = "half")]
            Constant::BF16(value) => {
                return f.write_fmt(format_args!("{}", unsafe { t::<_, bf16>(*value) }));
            }
            Constant::F8(value) => {
                return f.write_fmt(format_args!("{}", todo!()));
            }
            #[cfg(feature = "half")]
            Constant::F16(value) => {
                return f.write_fmt(format_args!("{}", unsafe { t::<_, f16>(*value) }));
            }
            Constant::F32(value) => {
                return f.write_fmt(format_args!("{}", unsafe { t::<_, f32>(*value) }));
            }
            Constant::F64(value) => {
                return f.write_fmt(format_args!("{}", unsafe { t::<_, f64>(*value) }));
            }
            #[cfg(feature = "complex")]
            Constant::CF32(re, im) => {
                return unsafe {
                    f.write_fmt(format_args!("{}+{}i", t::<_, f32>(*re), t::<_, f32>(*im)))
                };
            }
            #[cfg(feature = "complex")]
            Constant::CF64(re, im) => {
                return unsafe {
                    f.write_fmt(format_args!("{}+{}i", t::<_, f64>(*re), t::<_, f64>(*im)))
                };
            }
            Constant::U8(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::U32(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::I8(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::I16(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::I32(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::I64(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
            Constant::Bool(value) => {
                return f.write_fmt(format_args!("{}", value));
            }
        }
    }
}
