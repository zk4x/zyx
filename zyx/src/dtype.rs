#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Constant {
    #[cfg(feature = "half")]
    BF16(u16),
    #[cfg(feature = "half")]
    F16(u16),
    F32(u32),
    F64(u64),
    #[cfg(feature = "complex")]
    CF32(u64),
    #[cfg(feature = "complex")]
    CF64(u128),
    U8(u8),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    #[cfg(feature = "half")]
    BF16,
    #[cfg(feature = "half")]
    F16,
    F32,
    F64,
    #[cfg(feature = "complex")]
    CF32,
    #[cfg(feature = "complex")]
    CF64,
    U8,
    I8,
    I16,
    I32,
    I64,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        return f.write_str(match self {
            #[cfg(feature = "half")]
            DType::BF16 => "BF16",
            #[cfg(feature = "half")]
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            #[cfg(feature = "complex")]
            DType::CF32 => "CF32",
            #[cfg(feature = "complex")]
            DType::CF64 => "CF64",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
        });
    }
}

impl DType {
    pub(super) fn byte_size(&self) -> usize {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => 2,
            #[cfg(feature = "half")]
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            #[cfg(feature = "complex")]
            DType::CF32 => 8,
            #[cfg(feature = "complex")]
            DType::CF64 => 16,
            DType::U8 => 1,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        };
    }

    pub(super) fn zero_constant(&self) -> Constant {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => Constant::BF16(unsafe { core::mem::transmute(bf16::ZERO) }),
            #[cfg(feature = "half")]
            DType::F16 => Constant::F16(unsafe { core::mem::transmute(f16::ZERO) }),
            DType::F32 => Constant::F32(unsafe { core::mem::transmute(0f32) }),
            DType::F64 => Constant::F64(unsafe { core::mem::transmute(0f64) }),
            #[cfg(feature = "complex")]
            DType::CF32 => todo!(),
            #[cfg(feature = "complex")]
            DType::CF64 => todo!(),
            DType::U8 => Constant::U8(0),
            DType::I8 => Constant::I8(0),
            DType::I16 => Constant::I16(0),
            DType::I32 => Constant::I32(0),
            DType::I64 => Constant::I64(0),
        };
    }

    pub(super) fn min_constant(&self) -> Constant {
        todo!()
    }
}
