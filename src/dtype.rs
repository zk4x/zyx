//! # `DType`
//!
//! See [`DType`].

/// # `DType`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32 bit floating point
    F32,
    /// 32 bit integer
    I32,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DType::F32 => f.write_str("f32"),
            DType::I32 => f.write_str("i32"),
        }
    }
}

impl DType {
    #[cfg(feature = "opencl")]
    pub(crate) fn cl_type(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::I32 => "int",
        }
    }

    #[cfg(feature = "io")]
    pub(crate) fn safetensors(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::I32 => "I32",
        }
    }

    #[cfg(any(feature = "opencl", feature = "io", feature = "debug1"))]
    pub(crate) fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
        }
    }
}
