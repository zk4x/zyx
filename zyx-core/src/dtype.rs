/// DType
#[derive(Clone, Copy)]
pub enum DType {
    F32,
    I32,
}

impl DType {
    /// Get the size of DType in bytes
    pub fn byte_size(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
        }
    }
}
