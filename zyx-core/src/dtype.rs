/// DType of tensor
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    /// 32 bit floating point type
    F32,
    /// 64 bit floating point type
    F64,
    /// 32 bit integer type
    I32,
}

impl DType {
    /// Get the size of DType in bytes
    pub fn byte_size(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    /// Check if self is floating point dtype
    pub fn is_floating(self) -> bool {
        match self {
            Self::F32 | Self::F64 => true,
            Self::I32 => false,
        }
    }

    /// Min value as string
    pub fn min_value_str(self) -> &'static str {
        match self {
            Self::F32 => "-3.40282347E+38f",
            Self::F64 => "-1.7976931348623157E+308",
            Self::I32 => "-2147483648",
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn safetensors(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F64 => "F64",
            Self::I32 => "I32",
        }
    }

    #[cfg(feature = "std")]
    pub(crate) fn from_safetensors(dtype: &str) -> Result<Self, crate::error::ZyxError> {
        match dtype {
            "F32" => Ok(Self::F32),
            "F64" => Ok(Self::F64),
            "I32" => Ok(Self::I32),
            _ => Err(crate::error::ZyxError::ParseError(alloc::format!(
                "Could not parse safetensors dtype {dtype}"
            ))),
        }
    }
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> Result<(), core::fmt::Error> {
        f.write_fmt(format_args!("{self:?}"))
    }
}
