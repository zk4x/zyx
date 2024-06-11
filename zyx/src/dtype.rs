#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    CF32,
    CF64,
    U8,
    I8,
    I16,
    I32,
    I64,
}

impl core::fmt::Display for DType {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::CF32 => "CF32",
            DType::CF64 => "CF64",
            DType::U8 => "U8",
            DType::I8 => "I8",
            DType::I16 => "I16",
            DType::I32 => "I32",
            DType::I64 => "I64",
        })
    }
}

impl DType {
    pub(super) fn byte_size(&self) -> usize {
        match self {
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
            DType::CF32 => 8,
            DType::CF64 => 16,
            DType::U8 => 1,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }
}
