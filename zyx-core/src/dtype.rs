/// DType of tensor
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    /// 32 bit floating point type
    F32,
    /// 32 bit integer type
    I32,
}

impl DType {
    /// Get the size of DType in bytes
    pub fn byte_size(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
        }
    }

    /// Check if self is floating point dtype
    pub fn is_floating(self) -> bool {
        match self {
            Self::F32 => true,
            Self::I32 => false,
        }
    }
}
